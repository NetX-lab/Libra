"""Support code for Heterogeneous engine."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from RL_Framework.engine.rollout_engine import VLLMRolloutEngine
from RL_Framework.infra.scheduling.base import (
    BaseScheduler,
    SchedulingResult,
)
from RL_Framework.infra.scheduling.factory import SchedulerFactory
from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import CMLFQCostScheduler
from RL_Framework.engine.cmlfq_backend import CMLFQGenerationBackend

logger = logging.getLogger(__name__)


class HeterogeneousRolloutEngine:
    """Heterogeneous rollout engine implementation."""

    def __init__(
        self,
        model_path: str = "",
        scheduler: BaseScheduler | None = None,
    ):
        self.model_path = model_path

        if scheduler is None:
            from RL_Framework.infra.scheduling.length_aware import LengthAwareScheduler
            scheduler = LengthAwareScheduler()
        self.scheduler: BaseScheduler = scheduler


        self.engines: list[VLLMRolloutEngine] = []
        self.instance_configs: list[dict[str, Any]] = []

        # round-robin fallback
        self._rr_counter = 0
        self._lock = threading.RLock()


        self._pending_futures: dict[str, list[asyncio.Future]] = {}
        self._cmlfq_backend = CMLFQGenerationBackend()

        # Engine-owned telemetry. Whole-engine replacement stops these
        # resources before creating new ones; they are not global singletons.
        self._metrics_poller: Any = None
        # Engine-layer cross-rank load publisher/aggregator (same
        # lifetime rules as the poller).
        self._shared_state: Any = None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _setup_metrics_polling(self, hetero_cfg: Any) -> None:
        """Create/replace/stop the /metrics poller per scheduling config."""
        from RL_Framework.infra.scheduling.base import MetricsFeedbackConfig
        from RL_Framework.infra.scheduling.metrics_feed import VLLMMetricsPoller

        feedback = MetricsFeedbackConfig.from_scheduling(
            getattr(hetero_cfg, "scheduling", None)
        )
        if not feedback.enabled:
            self._stop_metrics_poller()
            return
        self._stop_metrics_poller()
        poller = VLLMMetricsPoller(
            interval_s=feedback.poll_interval_s,
            timeout_s=feedback.request_timeout_s,
            ttl_s=feedback.staleness_ttl_s,
        )
        self._metrics_poller = poller
        self._refresh_metrics_endpoints()
        poller.start()
        if self.scheduler is not None:
            self.scheduler.attach_metrics_feed(poller)

    def _refresh_metrics_endpoints(self) -> None:
        if self._metrics_poller is None:
            return
        urls = {
            cfg["instance_id"]: f"http://{cfg['host']}:{cfg['port']}/metrics"
            for cfg in self.instance_configs
        }
        self._metrics_poller.set_endpoints(urls)

    def _calibrate_capacities_from_metrics(self) -> None:
        """Retain fresh profiled capacities per instance, never max by TP.

        Same-TP endpoints can have different budgets. The scheduler uses
        fresh snapshots first, then this last profiled value, then the
        explicitly configured/analytic per-TP estimate.
        """
        if self._metrics_poller is None or self.scheduler is None:
            return
        calibrated: dict[str, int] = {}
        for cfg, snap in (
            (cfg, snap)
            for cfg in self.instance_configs
            for snap in [self._metrics_poller.get(cfg["instance_id"])]
            if snap is not None
        ):
            capacity = int(getattr(snap, "kv_capacity_tokens", -1))
            if capacity > 0:
                calibrated[cfg["instance_id"]] = capacity
        with self.scheduler._lock:
            for handle in self.scheduler._instances:
                if handle.instance_id in calibrated:
                    handle.kv_capacity_tokens = calibrated[handle.instance_id]
        if calibrated:
            logger.info("[MetricsFeedback] profiled capacity by instance: %s", calibrated)

    def _stop_metrics_poller(self) -> None:
        if self._metrics_poller is not None:
            try:
                self._metrics_poller.stop()
            except Exception as exc:
                logger.warning("Failed stopping metrics poller: %s", exc)
            self._metrics_poller = None
            self.scheduler.attach_metrics_feed(None)

    def _setup_shared_state(self, hetero_cfg: Any) -> None:
        """Create the cross-rank load state once per engine lifetime."""
        sched = getattr(hetero_cfg, "scheduling", None)
        directory = str(getattr(sched, "shared_load_dir", "") or "")
        if not directory:
            return
        if hasattr(self.scheduler, "get_request_route"):
            logger.warning("C-MLFQ uses cmlfq_shared_load_dir, ignoring shared_load_dir")
            return
        from RL_Framework.infra.scheduling.shared_token_state import (
            SharedTokenLoadState,
        )

        self._shared_state = SharedTokenLoadState(
            directory=directory,
            ttl_s=float(getattr(sched, "shared_load_ttl_s", 30.0)),
            heartbeat_interval_s=float(
                getattr(sched, "shared_load_heartbeat_s", 10.0)
            ),
            cache_ttl_s=float(getattr(sched, "shared_load_cache_ttl_s", 1.0)),
        )
        if self.scheduler is not None:
            self.scheduler.attach_shared_state(self._shared_state)
        logger.info(
            "Attached cross-rank shared load state: dir=%s writer=%s",
            directory,
            self._shared_state.writer_id,
        )

    def _reattach_shared_state(self) -> None:
        """Wire the existing shared state into a freshly built scheduler.

        The new scheduler's local counters start at zero, so this rank's
        published totals must be zeroed to match (other ranks untouched).
        """
        if self._shared_state is None or self.scheduler is None:
            return
        try:
            self._shared_state.reset()
        except Exception as exc:
            logger.warning("Shared-state reset on reattach failed: %s", exc)
        self.scheduler.attach_shared_state(self._shared_state)

    def _close_shared_state(self) -> None:
        if self._shared_state is not None:
            try:
                self._shared_state.close()
            except Exception as exc:
                logger.warning("Failed closing shared load state: %s", exc)
            self._shared_state = None

    def metrics_snapshots(self) -> dict[str, Any]:
        if self._metrics_poller is None:
            return {}
        return self._metrics_poller.snapshots()

    def add_instance(
        self,
        instance_id: str,
        host: str,
        port: int,
        tp_degree: int,
        gpu_ids: list[int] | None = None,
    ):
        """Add instance."""
        engine = VLLMRolloutEngine(
            host=host,
            port=port,
            model_path=self.model_path,
        )
        idx = len(self.engines)
        self.engines.append(engine)
        self.instance_configs.append({
            "instance_id": instance_id,
            "host": host,
            "port": port,
            "tp_degree": tp_degree,
            "gpu_ids": gpu_ids or [],
        })
        self._refresh_metrics_endpoints()


        self.scheduler.register_instance(
            index=idx,
            instance_id=instance_id,
            tp_degree=tp_degree,
        )
        self._configure_cost_runtime()
        logger.info(
            f"Added heterogeneous instance {instance_id}: TP={tp_degree}, "
            f"address={host}:{port}, GPUs={gpu_ids}"
        )

    @property
    def num_instances(self) -> int:
        return len(self.engines)

    @property
    def instance_urls(self) -> list[str]:
        return [e.base_url for e in self.engines]

    @property
    def tp_list(self) -> list[int]:
        """Tp list."""
        return [cfg["tp_degree"] for cfg in self.instance_configs]

    def reconfigure_from_plan(self, plan: Any, config: Any):
        """Apply a GlobalResourcePlan to the logical rollout topology.

        This updates the engine's instance table and recreates scheduler state.
        The actual vLLM processes must already be reachable at the configured
        ports/hosts; launch scripts can use the same plan metadata to elastically
        start or stop workers before this method is called.
        """
        with self._lock:
            if any(h.active_requests for h in self.scheduler._instances) or any(
                not f.done() for futures in self._pending_futures.values() for f in futures
            ):
                raise RuntimeError("Drain rollout requests before reconfiguring the engine")
            for engine in self.engines:
                if hasattr(engine, "close_sync"):
                    engine.close_sync()

            hetero = config.heterogeneous_rollout
            scheduler_type = getattr(hetero.scheduling, "scheduler_type", "length_aware")
            scheduler = SchedulerFactory.create(
                scheduler_type=scheduler_type,
                hetero_config=hetero,
            )
            old_scheduler = self.scheduler
            if old_scheduler is not None and old_scheduler is not scheduler:
                try:
                    scheduler.import_learned_state(
                        old_scheduler.export_learned_state()
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to carry scheduler state across reconfigure: %s", exc
                    )

            self.scheduler = scheduler
            self.engines = []
            self.instance_configs = []
            self._rr_counter = 0
            self._pending_futures.clear()
            self._configure_cost_model(config)

            base_port = hetero.vllm_base_port
            global_host = hetero.vllm_host
            for i, inst_cfg in enumerate(hetero.instances):
                host = inst_cfg.host or global_host
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                self.add_instance(
                    instance_id=inst_cfg.instance_id or f"grp_tp{inst_cfg.tp}_{i}",
                    host=host,
                    port=int(inst_cfg.port or (base_port + i)),
                    tp_degree=inst_cfg.tp,
                    gpu_ids=inst_cfg.gpus,
                )

            logger.info(
                "[GlobalResourcePlanner] applied rollout plan: TP layout=%s, instances=%s",
                self.tp_list,
                self.num_instances,
            )

            self._setup_metrics_polling(hetero)
            self._reattach_shared_state()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def wait_for_ready(self, timeout: float = 300.0):
        """Wait for ready."""
        for i, engine in enumerate(self.engines):
            cfg = self.instance_configs[i]
            logger.info(
                f"Waiting for heterogeneous instance {cfg['instance_id']} "
                f"(TP={cfg['tp_degree']}) to become ready..."
            )
            engine.wait_for_ready(timeout=timeout)

        logger.info(
            f"All {self.num_instances} heterogeneous instances are ready: "
            f"TP layout={self.tp_list}"
        )
        # Give the metrics poller a moment to capture its first snapshots,
        # then calibrate KV capacities against vLLM's own profiling.
        if self._metrics_poller is not None:
            import time as _time

            deadline = _time.time() + 15.0
            while _time.time() < deadline:
                if len(self.metrics_snapshots()) >= self.num_instances:
                    break
                _time.sleep(0.5)
            self._calibrate_capacities_from_metrics()

    def wait_until_idle(self, timeout: float = 3600.0, poll_interval: float = 0.5):
        """Block until no rollout requests are active before reconfiguration."""
        deadline = time.time() + max(0.0, timeout)
        while True:
            with self._lock:
                handles = list(getattr(self.scheduler, "_instances", []))
                active = sum(int(getattr(handle, "active_requests", 0)) for handle in handles)
                pending = sum(
                    1
                    for futures in self._pending_futures.values()
                    for future in futures
                    if not future.done()
                )
            if active == 0 and pending == 0:
                return
            if time.time() >= deadline:
                raise TimeoutError(
                    "timed out draining heterogeneous rollout engine "
                    f"(active_requests={active}, pending_futures={pending})"
                )
            time.sleep(max(0.05, poll_interval))

    async def close(self):
        """Close."""
        self._stop_metrics_poller()
        self._close_shared_state()
        await self._cmlfq_backend.close()
        for engine in self.engines:
            await engine.close()

    def close_sync(self):
        """Synchronous close used when the engine is replaced at rebind."""
        self._stop_metrics_poller()
        self._close_shared_state()
        for engine in self.engines:
            if hasattr(engine, "close_sync"):
                engine.close_sync()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def notify_epoch_start(self, epoch: int):
        """Notify epoch start."""
        self.scheduler.on_epoch_start(epoch)

    def notify_epoch_end(self, epoch: int):
        """Notify epoch end."""
        self.scheduler.on_epoch_end(epoch)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        input_tokens: int = 0,
        prompt_id: str = "",
        n_samples: int = 1,
        epoch: int = -1,
        request_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate."""

        if isinstance(self.scheduler, CMLFQCostScheduler):
            return await self._generate_cost_routed(
                prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, n=n, input_tokens=input_tokens, prompt_id=prompt_id,
                n_samples=n_samples, epoch=epoch, request_id=request_id, **kwargs,
            )

        if input_tokens <= 0:
            # Cheap chars-based fallback only: every bundled workflow now
            # passes its exact token count, so this path serves unknown
            # future callers. ~4 chars/token is a better prior than the
            # legacy //3 (which overestimated English ~40%).
            input_tokens = max(1, int(len(prompt) / 4))


        with self._lock:
            route_scheduler = self.scheduler
            requested_cmlfq_route = bool(
                request_id and hasattr(self.scheduler, "get_request_route")
            )
            result = (
                self.scheduler.get_request_route(request_id)
                if requested_cmlfq_route
                else None
            )
            cmlfq_managed = result is not None
            if result is None:
                result = self.scheduler.schedule(
                    input_tokens=input_tokens,
                    prompt_id=prompt_id,
                    n_samples=n_samples,
                    epoch=epoch,
                    max_new_tokens=max_new_tokens,
                )


        if result.pending:
            result = await self._wait_for_scout(
                prompt_id=prompt_id,
                input_tokens=input_tokens,
                n_samples=n_samples,
                epoch=epoch,
                max_new_tokens=max_new_tokens,
            )

        with self._lock:
            scheduled_by_scheduler = result.instance_index >= 0
            if not scheduled_by_scheduler:
                if hasattr(route_scheduler, "finish_request"):
                    raise RuntimeError(f"C-MLFQ routing failed: {result.reason}")
                with route_scheduler._lock:
                    ready = route_scheduler._selectable([
                        h for h in route_scheduler._instances if h.is_ready
                    ])
                if not ready:
                    raise RuntimeError(f"No ready rollout instance: {result.reason}")
                idx = ready[self._rr_counter % len(ready)].index
                self._rr_counter += 1
                result.is_fallback = True
                logger.warning(
                    f"Scheduling failed ({result.reason}), falling back to instance {idx}"
                )
            else:
                idx = result.instance_index
            engine = self.engines[idx]
            instance_config = dict(self.instance_configs[idx])
            # The scheduler never accounted a fallback route (its schedule()
            # call failed), so debit it here. Otherwise the completion path
            # would dec a request that was never inc'd and silently steal
            # tokens from whichever request is still in flight on idx.
            if not scheduled_by_scheduler and not cmlfq_managed:
                with_signal = getattr(
                    self.scheduler, "_record_route", None
                )
                if callable(with_signal):
                    with self.scheduler._lock:
                        result.reserved_tokens = with_signal(
                            self.scheduler.get_instance_handle(idx),
                            input_tokens,
                            category=result.category or "any",
                            prompt_id=prompt_id,
                            max_new_tokens=max_new_tokens,
                        )
                else:
                    handle = self.scheduler.get_instance_handle(idx)
                    if handle is not None:
                        handle.inc_active()
        output_tokens = 0

        try:
            gen_result = await engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                **kwargs,
            )
            output_tokens = len(gen_result.get("tokens", []))


            gen_result["_schedule_info"] = {
                "instance_index": idx,
                "instance_id": instance_config["instance_id"],
                "tp_degree": instance_config["tp_degree"],
                "category": result.category,
                "is_fallback": result.is_fallback,
                "reason": result.reason,
                "prompt_id": prompt_id,
                "request_id": request_id or result.request_id,
            }

            return gen_result
        finally:

            if not cmlfq_managed:
                if result.request_id and hasattr(
                    route_scheduler, "finish_request"
                ):
                    with self._lock:
                        route_scheduler.finish_request(
                            result.request_id,
                            output_tokens,
                        )
                else:
                    with self._lock:
                        completion_kwargs = {}
                        if result.reserved_tokens is not None:
                            completion_kwargs["reserved_tokens"] = result.reserved_tokens
                        route_scheduler.on_request_done(
                            instance_index=idx,
                            prompt_id=prompt_id,
                            final_bucket=result.category,
                            output_tokens=output_tokens,
                            **completion_kwargs,
                        )

    def _configure_cost_runtime(self):
        if isinstance(self.scheduler, CMLFQCostScheduler):
            self.scheduler.configure_runtime(
                self.instance_configs,
                lambda src, dst: self._cmlfq_backend.supports_transfer(
                    self.instance_configs[src], self.instance_configs[dst],
                ),
            )

    def _configure_cost_model(self, config):
        if not isinstance(self.scheduler, CMLFQCostScheduler):
            return
        from RL_Framework.infra.cost_model.model import CostModel

        # The same configured analytic rollout model as the global planner.
        self.scheduler.routing_costs.rollout_model = CostModel(
            hardware=config.hardware, model_arch=config.model_arch,
            profiling=config.profiling,
        ).rollout_model
        sched = config.heterogeneous_rollout.scheduling
        self._cmlfq_backend = CMLFQGenerationBackend(
            mode=getattr(sched, "cmlfq_kv_backend", "recompute"),
            transfer_tp_pairs=getattr(sched, "cmlfq_kv_transfer_tp_pairs", []),
        )
        self._configure_cost_runtime()

    async def _generate_cost_routed(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
        n: int, input_tokens: int, prompt_id: str, n_samples: int, epoch: int,
        request_id: str, **kwargs,
    ) -> dict[str, Any]:
        if n != 1 or n_samples != 1:
            raise ValueError("cmlfq_cost requires one sample per trajectory")
        scheduler = self.scheduler
        managed = bool(request_id)
        if managed and not scheduler.has_request(request_id):
            raise ValueError(f"Unknown CMLFQ request: {request_id}")
        if not managed:
            route = scheduler.schedule(
                input_tokens=max(1, input_tokens or len(prompt) // 3),
                prompt_id=prompt_id, epoch=epoch,
            )
            if route.instance_index < 0:
                raise RuntimeError("No available CMLFQ instance")
            request_id = route.request_id
        decision = None
        output_tokens = 0
        try:
            decision = scheduler.reserve_generation(
                request_id, input_tokens if input_tokens > 0 else None,
            )
            outcome = await self._cmlfq_backend.generate(
                source=self.engines[decision.source_instance_index],
                target=self.engines[decision.target_instance_index],
                prompt=prompt, path=decision.execution_path,
                request_id=request_id,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                n=n, **kwargs,
            )
            scheduler.commit_generation(request_id, decision)
            route = scheduler.get_request_route(request_id)
            output = outcome.output
            output_tokens = len(output.get("tokens", []))
            output["_schedule_info"] = {
                "instance_index": route.instance_index,
                "instance_id": self.instance_configs[route.instance_index]["instance_id"],
                "tp_degree": route.tp_degree, "category": route.category,
                "is_fallback": bool(outcome.fallback_reason),
                "reason": decision.reason, "prompt_id": route.prompt_id,
                "request_id": request_id, "execution_path": outcome.path,
                "planned_execution_path": decision.execution_path,
                "fallback_reason": outcome.fallback_reason,
                "estimated_decode_seconds": decision.decode_seconds,
                "estimated_migration_seconds": decision.migration_seconds,
            }
            return output
        except BaseException:
            # Do not roll back another coroutine's reservation if this one
            # failed to reserve (e.g. a duplicate concurrent generation).
            if decision is not None:
                scheduler.rollback_generation(request_id)
            if not scheduler.has_request(request_id):
                self._cmlfq_backend.release_request(request_id)
            raise
        finally:
            if not managed:
                self._cmlfq_backend.release_request(request_id)
                scheduler.finish_request(request_id, output_tokens)

    async def _wait_for_scout(
        self,
        prompt_id: str,
        input_tokens: int,
        n_samples: int,
        epoch: int,
        timeout: float = 60.0,
        max_new_tokens: int = 0,
    ) -> SchedulingResult:
        """Wait for scout."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_futures.setdefault(prompt_id, []).append(future)


        from RL_Framework.infra.scheduling.la_mlfq import WaitingRequest
        if hasattr(self.scheduler, "scout_manager"):
            wr = WaitingRequest(
                prompt_id=prompt_id,
                input_tokens=input_tokens,
                n_samples=n_samples,
                epoch=epoch,
                sample_index=-1,
                future=future,
                max_new_tokens=max_new_tokens,
            )
            self.scheduler.scout_manager.add_waiting(wr)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            if isinstance(result, SchedulingResult):
                return result
        except asyncio.TimeoutError:
            logger.warning(
                f"Timed out waiting for scout: prompt={prompt_id}, "
                f"using default routing"
            )
        except Exception as e:
            logger.warning(
                f"Error while waiting for scout: prompt={prompt_id}, error={e}, "
                f"using default routing"
            )
        finally:
            futures = self._pending_futures.get(prompt_id, [])
            if future in futures:
                futures.remove(future)
            if not futures:
                self._pending_futures.pop(prompt_id, None)


        return self.scheduler.schedule(
            input_tokens=input_tokens,
            prompt_id="",
            n_samples=1,
            epoch=epoch,
            max_new_tokens=max_new_tokens,
        )

    async def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        input_tokens_list: list[int] | None = None,
        prompt_ids: list[str] | None = None,
        n_samples: int = 1,
        epoch: int = -1,
    ) -> list[dict[str, Any]]:
        """Generate batch."""
        if input_tokens_list is None:
            input_tokens_list = [0] * len(prompts)
        if prompt_ids is None:
            prompt_ids = [""] * len(prompts)

        tasks = [
            self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                input_tokens=tokens,
                prompt_id=pid,
                n_samples=n_samples,
                epoch=epoch,
            )
            for prompt, tokens, pid in zip(prompts, input_tokens_list, prompt_ids)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(f"Prompt {i} Generation failed: {result}")
                continue
            valid_results.append(result)

        return valid_results

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def check_migration(self, request_id: str, generated_tokens: int):
        """Check migration."""
        if hasattr(self.scheduler, "check_and_migrate"):
            return self.scheduler.check_and_migrate(request_id, generated_tokens)
        return None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_tool_return(
        self,
        request_id: str,
        tool_result: Any,
        generated_tokens: int = 0,
    ) -> Any:
        """On tool return."""
        if hasattr(self.scheduler, "on_tool_return"):
            return self.scheduler.on_tool_return(
                request_id, tool_result, generated_tokens
            )
        return None

    def begin_cmlfq_request(
        self,
        prompt_id: str,
        input_tokens: int,
        epoch: int = -1,
    ) -> str:
        """Begin cmlfq request."""
        if not hasattr(self.scheduler, "get_request_route"):
            return ""
        result = self.scheduler.schedule(
            input_tokens=input_tokens,
            prompt_id=prompt_id,
            n_samples=1,
            epoch=epoch,
        )
        return result.request_id

    def route_cmlfq_tool_return(
        self,
        request_id: str,
        tool_result: Any,
        generated_tokens: int,
    ) -> Any:
        """Route cmlfq tool return."""
        decision = self.on_tool_return(
            request_id=request_id,
            tool_result=tool_result,
            generated_tokens=generated_tokens,
        )
        if decision is not None and decision.should_migrate:
            self.execute_cmlfq_migration(request_id, decision)
        return decision

    def finish_cmlfq_request(self, request_id: str, total_output_tokens: int):
        """Finish cmlfq request."""
        self._cmlfq_backend.release_request(request_id)
        if request_id and hasattr(self.scheduler, "finish_request"):
            self.scheduler.finish_request(request_id, total_output_tokens)

    def cancel_cmlfq_request(self, request_id: str):
        """Cancel cmlfq request."""
        self._cmlfq_backend.release_request(request_id)
        if request_id and hasattr(self.scheduler, "cancel_request"):
            self.scheduler.cancel_request(request_id)

    def execute_cmlfq_migration(
        self,
        request_id: str,
        decision: Any,
    ) -> Any:
        """Execute cmlfq migration."""
        if hasattr(self.scheduler, "execute_migration"):
            return self.scheduler.execute_migration(request_id, decision)
        return None

    def get_cmlfq_tree_stats(self) -> dict:
        """Get cmlfq tree stats."""
        if hasattr(self.scheduler, "prefix_tree"):
            return self.scheduler.prefix_tree.get_stats()
        return {}

    def update_cmlfq_tree(self, trajectories: list[Any]):
        """Update cmlfq tree."""
        if hasattr(self.scheduler, "update_tree_from_trajectories"):
            self.scheduler.update_tree_from_trajectories(trajectories)

    def rebuild_cmlfq_tree(self, trajectories: list[Any]):
        """Rebuild cmlfq tree."""
        if hasattr(self.scheduler, "rebuild_tree"):
            self.scheduler.rebuild_tree(trajectories)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def print_stats(self):
        """Print stats."""
        self.scheduler.print_stats()

    def get_cluster_info(self) -> dict[str, Any]:
        """Get cluster info."""
        return {
            "num_instances": self.num_instances,
            "tp_list": self.tp_list,
            "scheduler_type": self.scheduler.name,
            "instances": [
                {
                    "instance_id": cfg["instance_id"],
                    "tp_degree": cfg["tp_degree"],
                    "url": self.engines[i].base_url,
                    "gpu_ids": cfg["gpu_ids"],
                }
                for i, cfg in enumerate(self.instance_configs)
            ],
            "scheduler_stats": self.scheduler.get_stats().to_dict(),
        }

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config,
        carry_scheduler_state_from: "BaseScheduler | None" = None,
    ) -> "HeterogeneousRolloutEngine":
        """From config.

        ``carry_scheduler_state_from`` transfers learned scheduler state
        (output-length EMA, history tables) across a rebind. Weight sync
        rebinds rebuild the engine every sync step; without the carry the
        EMA would reset to its prior every step and never converge.
        """
        hetero = config.heterogeneous_rollout


        scheduler_type = getattr(hetero.scheduling, "scheduler_type", "length_aware")
        scheduler = SchedulerFactory.create(
            scheduler_type=scheduler_type,
            hetero_config=hetero,
        )
        if carry_scheduler_state_from is not None:
            try:
                scheduler.import_learned_state(
                    carry_scheduler_state_from.export_learned_state()
                )
                logger.info(
                    "Carried learned scheduler state (%s -> %s) across rebind",
                    type(carry_scheduler_state_from).__name__,
                    type(scheduler).__name__,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to carry scheduler state across rebind: %s", exc
                )


        engine = cls(
            model_path=config.model_path,
            scheduler=scheduler,
        )
        engine._configure_cost_model(config)


        base_port = hetero.vllm_base_port
        global_host = hetero.vllm_host


        import os
        env_hosts = os.environ.get("HETERO_INSTANCE_HOSTS", "")
        instance_hosts = [h.strip() for h in env_hosts.split(",") if h.strip()] if env_hosts else []

        for i, inst_cfg in enumerate(hetero.instances):
            instance_id = inst_cfg.instance_id or f"hetero_tp{inst_cfg.tp}_{i}"
            port = int(inst_cfg.port or (base_port + i))
            gpu_ids = inst_cfg.gpus


            if inst_cfg.host:
                host = inst_cfg.host
            elif i < len(instance_hosts):
                host = instance_hosts[i]
            else:
                host = global_host


            if host == "0.0.0.0":
                host = "127.0.0.1"

            engine.add_instance(
                instance_id=instance_id,
                host=host,
                port=port,
                tp_degree=inst_cfg.tp,
                gpu_ids=gpu_ids,
            )

        logger.info(
            f"Created heterogeneous engine from configuration: {engine.num_instances} instances, "
            f"TP layout={engine.tp_list}, scheduler={scheduler_type}"
        )
        engine._setup_metrics_polling(hetero)
        engine._setup_shared_state(hetero)
        return engine

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def __repr__(self):
        return (
            f"HeterogeneousRolloutEngine("
            f"instances={self.num_instances}, "
            f"tp_list={self.tp_list}, "
            f"scheduler={self.scheduler.name})"
        )

    def __del__(self):
        for engine in self.engines:
            try:
                engine.__del__()
            except Exception:
                pass

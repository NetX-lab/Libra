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

logger = logging.getLogger(__name__)


class HeterogeneousRolloutEngine:
    """Heterogeneous rollout engine implementation."""

    def __init__(
        self,
        model_path: str = "",
        scheduler: BaseScheduler | None = None,
        request_timeout: float = 600.0,
    ):
        self.model_path = model_path
        self.request_timeout = max(1.0, float(request_timeout))

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

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

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
            request_timeout=self.request_timeout,
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


        self.scheduler.register_instance(
            index=idx,
            instance_id=instance_id,
            tp_degree=tp_degree,
        )
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
            for engine in self.engines:
                if hasattr(engine, "close_sync"):
                    engine.close_sync()

            hetero = config.heterogeneous_rollout
            scheduler_type = getattr(hetero.scheduling, "scheduler_type", "length_aware")
            scheduler = SchedulerFactory.create(
                scheduler_type=scheduler_type,
                hetero_config=hetero,
            )

            self.scheduler = scheduler
            self.engines = []
            self.instance_configs = []
            self._rr_counter = 0
            self._pending_futures.clear()

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
        for engine in self.engines:
            await engine.close()

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

        if input_tokens <= 0:
            input_tokens = max(1, len(prompt) // 3)


        with self._lock:
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
                )


        if result.pending:
            result = await self._wait_for_scout(
                prompt_id=prompt_id,
                input_tokens=input_tokens,
                n_samples=n_samples,
                epoch=epoch,
            )

        with self._lock:
            if result.instance_index < 0:
                idx = self._rr_counter % max(1, self.num_instances)
                self._rr_counter += 1
                logger.warning(
                    f"Scheduling failed ({result.reason}), falling back to instance {idx}"
                )
            else:
                idx = result.instance_index
            engine = self.engines[idx]
            instance_config = dict(self.instance_configs[idx])
        output_tokens = 0

        try:
            effective_max_new_tokens = int(max_new_tokens)
            bucket_limit = getattr(
                self.scheduler,
                "get_bucket_max_tokens",
                lambda _bucket: 0,
            )(result.category)
            if cmlfq_managed and bucket_limit > 0:
                # Treat the bucket limit as an MLFQ time quantum.  A 30K
                # episode may start on TP1, but one short turn cannot occupy
                # that worker for the entire episode before a tool-return
                # migration opportunity becomes available.
                effective_max_new_tokens = min(
                    effective_max_new_tokens,
                    int(bucket_limit),
                )
                remaining_quantum = getattr(
                    self.scheduler,
                    "get_request_remaining_quantum",
                    lambda _request_id: 0,
                )(request_id)
                if remaining_quantum > 0:
                    effective_max_new_tokens = min(
                        effective_max_new_tokens,
                        int(remaining_quantum),
                    )

            gen_result = await engine.generate(
                prompt=prompt,
                max_new_tokens=effective_max_new_tokens,
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
                "requested_max_new_tokens": int(max_new_tokens),
                "effective_max_new_tokens": effective_max_new_tokens,
            }

            return gen_result
        finally:

            if not cmlfq_managed:
                if result.request_id and hasattr(
                    self.scheduler, "finish_request"
                ):
                    with self._lock:
                        self.scheduler.finish_request(
                            result.request_id,
                            output_tokens,
                        )
                else:
                    with self._lock:
                        self.scheduler.on_request_done(
                            instance_index=idx,
                            prompt_id=prompt_id,
                            final_bucket=result.category,
                            output_tokens=output_tokens,
                        )

    async def _wait_for_scout(
        self,
        prompt_id: str,
        input_tokens: int,
        n_samples: int,
        epoch: int,
        timeout: float = 60.0,
    ) -> SchedulingResult:
        """Wait for scout."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()


        from RL_Framework.infra.scheduling.la_mlfq import WaitingRequest
        if hasattr(self.scheduler, "scout_manager"):
            wr = WaitingRequest(
                prompt_id=prompt_id,
                input_tokens=input_tokens,
                n_samples=n_samples,
                epoch=epoch,
                sample_index=-1,
                future=future,
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


        return self.scheduler.schedule(
            input_tokens=input_tokens,
            prompt_id="",
            n_samples=1,
            epoch=epoch,
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
            if isinstance(result, Exception):
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
        if request_id and hasattr(self.scheduler, "finish_request"):
            self.scheduler.finish_request(request_id, total_output_tokens)

    def cancel_cmlfq_request(self, request_id: str):
        """Cancel cmlfq request."""
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
    def from_config(cls, config) -> "HeterogeneousRolloutEngine":
        """From config."""
        hetero = config.heterogeneous_rollout


        scheduler_type = getattr(hetero.scheduling, "scheduler_type", "length_aware")
        scheduler = SchedulerFactory.create(
            scheduler_type=scheduler_type,
            hetero_config=hetero,
        )


        engine = cls(
            model_path=config.model_path,
            scheduler=scheduler,
            request_timeout=float(
                getattr(hetero.scheduling, "request_timeout", 600.0)
            ),
        )


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

"""Runtime executor for Global Resource Planner decisions."""

from __future__ import annotations

import os
import json
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    CoreRuntimeCapabilities,
    GlobalResourcePlan,
    GlobalResourcePlanner,
    PlannerDecision,
)

if TYPE_CHECKING:  # pragma: no cover
    from RL_Framework.infra.elastic.hybrid_pool import ElasticHybridPool


@dataclass
class ManagedRolloutProcess:
    """Subprocess owned by the runtime elastic executor."""

    instance_id: str
    command: str
    pid: int
    gpus: list[int]
    port: int
    host: str = "127.0.0.1"
    tp: int = 1
    adopted: bool = False
    log_path: str = ""


@dataclass
class ManagedHybridWorkerProcess:
    """Hybrid training worker process owned by the runtime executor."""

    worker_id: str
    target_core_id: str
    command: str
    pid: int
    snapshot_path: str
    host: str = ""
    gpus: list[int] = field(default_factory=list)
    replica_id: str = ""
    replica_rank: int = 0
    replica_world_size: int = 1


@dataclass
class RuntimeReconfigurationResult:
    """Outcome of applying one planner decision at runtime."""

    applied: bool
    reason: str
    actions: list[str] = field(default_factory=list)
    started_processes: list[ManagedRolloutProcess] = field(default_factory=list)
    stopped_processes: list[ManagedRolloutProcess] = field(default_factory=list)
    started_hybrid_workers: list[ManagedHybridWorkerProcess] = field(default_factory=list)
    training_actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "actions": list(self.actions),
            "started_processes": [vars(p) for p in self.started_processes],
            "stopped_processes": [vars(p) for p in self.stopped_processes],
            "started_hybrid_workers": [vars(p) for p in self.started_hybrid_workers],
            "training_actions": list(self.training_actions),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class PreparedRuntimeTransaction:
    """Immutable eligibility snapshot for one runtime boundary decision."""

    decision: PlannerDecision
    result: RuntimeReconfigurationResult
    should_execute: bool
    reason: str
    candidate_plan: GlobalResourcePlan | None
    effective_planner_signal: Any | None
    current_train_gpus: int
    rollout_runtime_enabled: bool
    strategy: str
    cluster_swap_enabled: bool
    drain_before_reconfigure: bool
    should_pause_dispatcher: bool
    drain_required: bool
    coord_id: str
    resource_reconfiguration: bool = True


class RuntimeElasticExecutor:
    """Apply planner decisions to rollout and elastic training control planes."""

    _TRAIN_DISTRIBUTED_ENV_KEYS = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    )

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        """Persist a small control-plane record without exposing partial JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def __init__(
        self,
        *,
        config: AsyncRLConfig,
        planner: GlobalResourcePlanner,
        train_engine: Any | None = None,
        rollout_engine: Any | None = None,
        dispatcher: Any | None = None,
        elastic_pool: "ElasticHybridPool | None" = None,
        membership_run_id: str | None = None,
    ):
        self.config = config
        self.planner = planner
        self.train_engine = train_engine
        self.rollout_engine = rollout_engine
        self.dispatcher = dispatcher
        self.elastic_pool = elastic_pool
        self._processes: dict[str, subprocess.Popen] = {}
        self._process_meta: dict[str, ManagedRolloutProcess] = {}
        self._hybrid_worker_processes: dict[str, subprocess.Popen] = {}
        self._hybrid_worker_meta: dict[str, ManagedHybridWorkerProcess] = {}
        self._hybrid_launch_threads: list[threading.Thread] = []
        self._pending_hybrid_joins: dict[str, Any] = {}
        self._pending_hybrid_join_started_at: dict[str, float] = {}
        self._pending_hybrid_join_lock = threading.RLock()
        self._active_elastic_signal: dict[str, Any] | None = None
        self.gradient_server = None
        self.membership_run_id = membership_run_id or uuid.uuid4().hex
        self._membership_publication_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._closing = threading.Event()
        self._closed = False
        self.runtime_run_id = ""
        self._peer_drain_request_coord_id = ""
        self._peer_drain_completed_coord_id = ""
        self._pending_activation_lock = threading.Lock()
        self._pending_activations: dict[str, dict[str, Any]] = {}
        self._snapshot_manifests: dict[int, str] = {}
        self._leased_rollout_processes: dict[str, ManagedRolloutProcess] = {}

    @contextmanager
    def _hybrid_resource_guard(self, handle=None):
        """Commit managed side effects atomically with shutdown/cancellation.

        Preparation and readiness waits stay outside this gate. Once close has
        acquired it, no callback can launch, register or update worker metadata.
        """
        from RL_Framework.infra.elastic.hybrid_pool import JoinCancelledError

        with self._lifecycle_lock:
            if self._closing.is_set():
                raise JoinCancelledError("runtime executor is closing")
            guard = self.elastic_pool.activation_guard(handle) if self.elastic_pool is not None else nullcontext()
            with guard:
                yield

    def prepare_transaction(self, decision: PlannerDecision) -> PreparedRuntimeTransaction:
        """Freeze runtime eligibility without launching, stopping, or waiting."""
        planner_cfg = self.config.global_resource_planner
        plan = decision.candidate_plan
        resource_permission = getattr(decision, "resource_should_reconfigure", None)
        resource_reconfiguration = (
            bool(decision.should_reconfigure)
            if resource_permission is None else bool(resource_permission)
        )
        if not resource_reconfiguration:
            # EHP changes membership within the existing allocation. Its demand
            # signal does not authorize installing a rejected GRP candidate.
            plan = getattr(decision, "current_plan", None)
        should_execute = bool(decision.should_reconfigure and plan is not None)
        reason = str(decision.reason)
        current_train_gpus = int(self.config.train_gpus)
        capabilities = CoreRuntimeCapabilities.from_engine(self.train_engine)
        if should_execute and resource_reconfiguration and capabilities is not None:
            rejection = capabilities.incompatibility(plan)
            if rejection:
                # Fence stale/manual decisions before publishing a peer drain,
                # pausing generation or closing live gradient servers.
                signal = getattr(decision, "elastic_hybrid_signal", None)
                plan = getattr(decision, "current_plan", None)
                resource_reconfiguration = False
                should_execute = bool(plan is not None and signal is not None and signal.should_adjust)
                reason = "runtime_candidate_rejected:" + rejection
        rollout_runtime_enabled = bool(
            getattr(planner_cfg, "apply_to_runtime", True)
            or getattr(planner_cfg, "runtime_manage_rollout_processes", False)
        )
        strategy = getattr(
            planner_cfg,
            "runtime_rollout_reconfigure_strategy",
            "diff",
        )
        cluster_swap_enabled = self._cluster_swap_enabled(strategy)
        drain_before_reconfigure = bool(
            getattr(planner_cfg, "runtime_drain_before_reconfigure", True)
        )
        should_pause_dispatcher = not (
            cluster_swap_enabled and not drain_before_reconfigure
        )

        if should_execute and not getattr(
            planner_cfg,
            "runtime_dynamic_reconfiguration_enabled",
            True,
        ):
            should_execute = False
            reason = "runtime_dynamic_reconfiguration_disabled"

        if (
            should_execute
            and resource_reconfiguration
            and plan is not None
            and getattr(planner_cfg, "runtime_manage_rollout_processes", False)
        ):
            # Metadata adoption is part of the eligibility snapshot. It does not
            # launch, stop, wait for, or reconfigure any worker.
            self._adopt_existing_rollout_processes()
            if self._runtime_plan_already_applied(
                plan,
                current_train_gpus=current_train_gpus,
            ):
                should_execute = False
                reason = "runtime_plan_already_applied"

        drain_required = bool(
            should_execute
            and drain_before_reconfigure
            and getattr(planner_cfg, "runtime_coordinate_reconfiguration_ranks", True)
            and int(os.environ.get("WORLD_SIZE", "1") or "1") > 1
            and int(os.environ.get("RANK", "0") or "0") == 0
        )
        coord_id = uuid.uuid4().hex if should_execute else ""
        result = RuntimeReconfigurationResult(applied=False, reason=reason)
        return PreparedRuntimeTransaction(
            decision=decision,
            result=result,
            should_execute=should_execute,
            reason=reason,
            candidate_plan=plan,
            effective_planner_signal=getattr(decision, "elastic_hybrid_signal", None),
            current_train_gpus=current_train_gpus,
            rollout_runtime_enabled=rollout_runtime_enabled,
            strategy=strategy,
            cluster_swap_enabled=cluster_swap_enabled,
            drain_before_reconfigure=drain_before_reconfigure,
            should_pause_dispatcher=should_pause_dispatcher,
            drain_required=drain_required,
            coord_id=coord_id,
            resource_reconfiguration=resource_reconfiguration,
        )

    def execute(
        self,
        decision: PlannerDecision,
        *,
        before_execute: Callable[[str, bool], None] | None = None,
        request_published: Callable[[str], None] | None = None,
    ) -> RuntimeReconfigurationResult:
        """Compatibility entry point for callers without boundary orchestration."""
        prepared = self.prepare_transaction(decision)
        try:
            if prepared.should_execute:
                self._active_runtime_coord_id = prepared.coord_id
                if prepared.drain_required:
                    self.publish_peer_rank_drain_request(prepared)
                    if request_published is not None:
                        request_published(prepared.coord_id)
                if before_execute is not None:
                    before_execute(prepared.coord_id, prepared.drain_required)
                if prepared.drain_required:
                    self.wait_for_peer_rank_drain(prepared)
        except Exception as exc:
            self.abort_prepared_transaction(prepared, exc)
            raise
        return self.execute_prepared(prepared)

    def abort_prepared_transaction(
        self,
        prepared: PreparedRuntimeTransaction,
        error: BaseException,
    ) -> None:
        if prepared.should_execute and prepared.candidate_plan is not None:
            self._write_peer_rank_reconfiguration_state(
                prepared.candidate_plan,
                phase="aborted",
                coord_id=prepared.coord_id,
                error=str(error),
            )

    def execute_prepared(
        self,
        prepared: PreparedRuntimeTransaction,
    ) -> RuntimeReconfigurationResult:
        """Perform maintenance and execute one previously frozen transaction."""
        self._active_runtime_coord_id = prepared.coord_id
        self._runtime_transaction_started = prepared.should_execute
        try:
            return self._execute_prepared(prepared)
        except Exception as exc:
            self.abort_prepared_transaction(prepared, exc)
            raise

    def _execute_prepared(
        self,
        prepared: PreparedRuntimeTransaction,
    ) -> RuntimeReconfigurationResult:
        decision = prepared.decision
        result = prepared.result
        self._reconcile_pending_hybrid_joins(result)
        self.accept_planner_signal(
            prepared.effective_planner_signal,
            result=result,
        )
        if not prepared.should_execute or prepared.candidate_plan is None:
            return result

        plan = prepared.candidate_plan
        planner_cfg = self.config.global_resource_planner
        if not prepared.resource_reconfiguration:
            return self._execute_membership_only(prepared)
        if getattr(self, "_lent_rollout", None):
            self._release_all_hybrid_workers(reason="resource_repartition", result=result)
            from RL_Framework.infra.elastic.rollout_lending import return_rollout
            result.started_processes.extend(return_rollout(self))
        rollout_runtime_enabled = prepared.rollout_runtime_enabled
        current_train_gpus = prepared.current_train_gpus
        strategy = prepared.strategy
        cluster_swap_enabled = prepared.cluster_swap_enabled
        drain_before_reconfigure = prepared.drain_before_reconfigure
        should_pause_dispatcher = prepared.should_pause_dispatcher
        config_applied = False
        prewarmed_rollout = False
        prewarmed_stopped: list[ManagedRolloutProcess] = []
        prewarmed_started: list[ManagedRolloutProcess] = []
        coordinated_peers = bool(
            prepared.drain_required
            and self._peer_drain_completed_coord_id == prepared.coord_id
        )
        if (
            getattr(planner_cfg, "runtime_manage_rollout_processes", False)
            and strategy == "prewarm"
            and not cluster_swap_enabled
        ):
            self._adopt_existing_rollout_processes()
            self._apply_plan_to_runtime_config(plan)
            config_applied = True
            result.actions.append("apply_config")
            capacity_issue = self._prewarm_capacity_issue(plan)
            if capacity_issue:
                fallback_strategy = getattr(
                    planner_cfg,
                    "runtime_prewarm_no_spare_fallback_strategy",
                    "restart_all",
                )
                result.actions.append(
                    f"prewarm_unavailable:{capacity_issue}:fallback_{fallback_strategy}"
                )
                strategy = fallback_strategy
            else:
                prewarmed_stopped, prewarmed_started = (
                    self._prewarm_rollout_processes(plan)
                )
                prewarmed_rollout = True
                result.actions.append("prewarm_rollout_processes")

        if (
            cluster_swap_enabled
            and getattr(planner_cfg, "runtime_manage_rollout_processes", False)
            and not config_applied
        ):
            self._adopt_existing_rollout_processes()
            self._apply_plan_to_runtime_config(plan)
            config_applied = True
            result.actions.append("apply_config")
            self._write_rollout_manifest(plan, phase="planned")

        if (
            cluster_swap_enabled
            and config_applied
            and not coordinated_peers
            and getattr(
                planner_cfg,
                "runtime_coordinate_reconfiguration_ranks",
                True,
            )
        ):
            self._coordinate_peer_rank_drain(plan, result)
            coordinated_peers = True

        if (
            getattr(planner_cfg, "runtime_reconfigure_training", False)
            and not getattr(planner_cfg, "runtime_training_pool_plan_only", True)
            and self._hybrid_training_prewarm_enabled
            and not cluster_swap_enabled
        ):
            self._ensure_elastic_pool()
            result.training_actions.extend(
                self._prewarm_hybrid_training_workers(
                    current_train_gpus,
                    plan,
                    desired_hybrid_workers=self._desired_hybrid_workers(
                        current_train_gpus,
                        plan,
                    ),
                )
            )

        paused = False
        if (
            should_pause_dispatcher
            and self.dispatcher is not None
            and hasattr(self.dispatcher, "pause")
        ):
            self.dispatcher.pause()
            paused = True
            result.actions.append("dispatcher_pause")
            if (
                not coordinated_peers
                and getattr(planner_cfg, "runtime_manage_rollout_processes", False)
                and config_applied
            ):
                self._coordinate_peer_rank_drain(plan, result)
                coordinated_peers = True
            if drain_before_reconfigure:
                if hasattr(self.dispatcher, "wait_until_idle"):
                    self.dispatcher.wait_until_idle(
                        timeout=float(
                            getattr(planner_cfg, "runtime_drain_timeout_s", 3600.0)
                        )
                    )
                    result.actions.append("dispatcher_drain")
        if (
            drain_before_reconfigure
            and self.rollout_engine is not None
            and hasattr(self.rollout_engine, "wait_until_idle")
        ):
            self.rollout_engine.wait_until_idle(
                timeout=float(
                    getattr(planner_cfg, "runtime_drain_timeout_s", 3600.0)
                )
            )
            result.actions.append("rollout_engine_drain")

        try:
            if getattr(planner_cfg, "runtime_manage_rollout_processes", False):
                self._adopt_existing_rollout_processes()

            cluster_swap_applied = False
            if cluster_swap_enabled:
                print(
                    "[RuntimeElasticExecutor] cluster_swap_execute_begin "
                    f"current_train_gpus={current_train_gpus} "
                    f"target_train_gpus={int(getattr(planner_cfg, 'runtime_training_pool_target_gpus', 0) or int(plan.train_gpus))} "
                    f"rollout_tp={plan.rollout_tp_list}",
                    flush=True,
                )
                if not config_applied:
                    self._apply_plan_to_runtime_config(plan)
                    config_applied = True
                    result.actions.append("apply_config")
                    self._write_rollout_manifest(plan, phase="planned")
                if not coordinated_peers:
                    self._coordinate_peer_rank_drain(plan, result)
                    coordinated_peers = True
                self._cluster_swap_rollout_training_pools(
                    current_train_gpus,
                    plan,
                    result,
                )
                cluster_swap_applied = True
                print(
                    "[RuntimeElasticExecutor] cluster_swap_execute_complete "
                    f"actions={','.join(result.actions)} "
                    f"training_actions={','.join(result.training_actions) or 'none'}",
                    flush=True,
                )

            if (
                not cluster_swap_applied
                and getattr(planner_cfg, "runtime_reconfigure_training", False)
                and not self._hybrid_worker_launch_enabled
            ):
                if getattr(planner_cfg, "runtime_training_pool_plan_only", True):
                    result.training_actions.extend(
                        self._plan_training_pool_reconfiguration(
                            current_train_gpus,
                            plan,
                        )
                    )
                else:
                    self._ensure_elastic_pool()
                    result.training_actions.extend(
                    self._reconfigure_training_pool(current_train_gpus, plan, result)
                    )

            if (
                not prewarmed_rollout
                and not config_applied
                and rollout_runtime_enabled
            ):
                self._apply_plan_to_runtime_config(plan)
                result.actions.append("apply_config")
            elif not prewarmed_rollout and not config_applied:
                # The planner decision may still drive the Elastic Hybrid Pool,
                # but externally managed rollout endpoints must remain exactly
                # as launched.  Mutating the config here makes the trainer
                # rebind to planned-yet-nonexistent ports even though
                # apply_to_runtime is explicitly disabled.
                result.actions.append("preserve_external_rollout_config")

            self._write_rollout_manifest(plan, phase="planned")
            if not coordinated_peers:
                self._coordinate_peer_rank_drain(plan, result)

            rollout_reconfigured = bool(
                cluster_swap_applied
                and getattr(self, "_rollout_engine_reconfigured", False)
            )
            if prewarmed_rollout:
                stopped, started = self._finish_prewarmed_rollout_cutover(plan)
                rollout_reconfigured = True
                result.stopped_processes.extend(prewarmed_stopped + stopped)
                result.started_processes.extend(prewarmed_started + started)
                result.actions.append("reconfigure_rollout_processes:prewarm")
            elif (
                not cluster_swap_applied
                and getattr(planner_cfg, "runtime_manage_rollout_processes", False)
            ):
                stopped, started = self._reconfigure_rollout_processes(
                    plan,
                    strategy_override=strategy,
                )
                rollout_reconfigured = bool(
                    getattr(self, "_rollout_engine_reconfigured", False)
                )
                result.stopped_processes.extend(stopped)
                result.started_processes.extend(started)
                result.actions.append(f"reconfigure_rollout_processes:{strategy}")

            if (
                not rollout_reconfigured
                and not cluster_swap_applied
                and getattr(planner_cfg, "apply_to_runtime", True)
                and self.rollout_engine is not None
                and hasattr(self.rollout_engine, "reconfigure_from_plan")
            ):
                self._reconfigure_rollout_engine(plan, wait_ready=bool(
                    getattr(planner_cfg, "runtime_manage_rollout_processes", False)
                ))
                result.actions.append("reconfigure_rollout_engine")
                if getattr(planner_cfg, "runtime_manage_rollout_processes", False):
                    result.actions.append("wait_rollout_ready")

            if getattr(self, "trainer", None) is not None and rollout_runtime_enabled:
                # Recreated services start from the HF checkpoint. Publish the
                # current optimizer state before any client can route work.
                self._publish_current_rollout_weights()
            if (not cluster_swap_applied and self._hybrid_worker_launch_enabled
                    and getattr(planner_cfg, "runtime_reconfigure_training", False)
                    and not getattr(planner_cfg, "runtime_training_pool_plan_only", True)):
                from RL_Framework.infra.elastic.rollout_lending import lend_rollout
                from RL_Framework.infra.elastic.rollout_lifecycle import apply_availability, unavailable_rollouts
                self._ensure_elastic_pool()
                desired = self._desired_hybrid_workers(int(self.config.train_gpus), plan)
                if desired:
                    self._cluster_swap_training_slots = lend_rollout(self, desired)
                result.training_actions.extend(self._reconfigure_training_pool(int(self.config.train_gpus), plan, result))
                if getattr(self, "trainer", None) is not None:
                    apply_availability(self.trainer, unavailable_rollouts(self))
            self._write_rollout_manifest(plan, phase="applied")
            self._write_peer_rank_reconfiguration_state(plan, phase="applied")

            result.applied = True
            return result
        except Exception as exc:
            result.errors.append(str(exc))
            raise
        finally:
            if paused and result.applied and self.dispatcher is not None and hasattr(self.dispatcher, "resume"):
                if getattr(self, "trainer", None) is not None:
                    self.trainer._resume_rollout_pipeline()
                else:
                    self.dispatcher.resume()
                result.actions.append("dispatcher_resume")

    def _capture_core_snapshot(self) -> str:
        from RL_Framework.infra.elastic.core_reconfiguration import dispatch_core_command
        revision = int(getattr(self.train_engine, "optimizer_revision", 0))
        if revision in self._snapshot_manifests:
            return self._snapshot_manifests[revision]
        receipts = dispatch_core_command(self, "capture")
        lanes: dict[str, dict[str, str]] = {}
        for receipt in receipts:
            if int(receipt["optimizer_revision"]) != revision:
                raise RuntimeError("core snapshot ranks disagree on optimizer revision")
            lane = receipt["lane"]
            core_id = f"dp{int(lane['data_parallel_rank'])}"
            index = (int(lane.get("tensor_parallel_rank", 0))
                     + int(lane.get("context_parallel_rank", 0)) * int(self.config.train_tp_size)
                     + int(lane.get("pipeline_parallel_rank", 0)) * int(self.config.train_tp_size)
                     * int(getattr(self.config, "train_cp_size", 1)))
            lanes.setdefault(core_id, {})[str(index)] = receipt["snapshot_uri"]
        width = self._elastic_replica_size_gpus()
        if (set(lanes) != {f"dp{i}" for i in range(int(self.config.train_dp_size))}
                or any(set(value) != {str(i) for i in range(width)} for value in lanes.values())):
            raise RuntimeError("core snapshot is missing model-parallel lanes")
        # DistributedOptimizer gathers each TP lane's complete Adam/master
        # state on its DP-zero rank. DP followers have identical parameters but
        # no complete optimizer payload; every joining replica reads DP zero.
        if int(getattr(self.config, "train_ep_size", 1)) == 1:
            lanes = {core: dict(lanes["dp0"]) for core in lanes}
        manifest = {
            "format": "libra-ehp-lanes-v1", "optimizer_revision": revision,
            "snapshot_id": uuid.uuid4().hex, "lanes": lanes,
        }
        path = self._coordination_dir() / f"ehp_state_{revision}_{manifest['snapshot_id']}.json"
        self._write_json_atomic(path, manifest)
        self._snapshot_manifests[revision] = str(path)
        return str(path)

    def _queue_activation(self, replica_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
        """Request activation at a core step boundary, never from a loader thread."""
        request = {"receipt": receipt, "event": threading.Event()}
        with self._hybrid_resource_guard():
            with self._pending_activation_lock:
                self._pending_activations[replica_id] = request
        timeout = float(getattr(self.config.global_resource_planner, "elastic_hybrid_join_timeout_s", 180))
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while not request["event"].wait(0.05):
            if self._closing.is_set():
                raise RuntimeError("EHP activation cancelled during cleanup")
            if deadline is not None and time.monotonic() >= deadline:
                with self._pending_activation_lock:
                    self._pending_activations.pop(replica_id, None)
                raise TimeoutError(f"EHP activation missed a training boundary: {replica_id}")
        if "error" in request:
            raise RuntimeError(request["error"])
        return request["receipt"]

    def has_pending_activations(self) -> bool:
        with self._pending_activation_lock:
            return bool(self._pending_activations)

    def _finish_pending_activations(self) -> None:
        import torch
        with self._pending_activation_lock:
            pending = dict(self._pending_activations)
        if not pending:
            return
        revision = int(self.train_engine.optimizer_revision)
        path = self._capture_core_snapshot()
        task_dir = Path(self.config.global_resource_planner.hybrid_worker_task_dir)
        try:
            for replica_id, request in pending.items():
                if int(request["receipt"]["optimizer_revision"]) != revision:
                    task = task_dir / f"{replica_id}.install_{uuid.uuid4().hex}.pt"
                    temporary = task.with_suffix(".tmp")
                    # This file contains only an installation command and URI.
                    torch.save({"install_state": True, "snapshot_path": path}, temporary)
                    temporary.replace(task)
                receipt = self._wait_hybrid_worker_ready(
                    replica_id, state_version=revision,
                    replica_world_size=self._elastic_replica_size_gpus(), snapshot_path=path,
                )
                if int(self.train_engine.optimizer_revision) != revision:
                    raise RuntimeError("optimizer advanced during the activation boundary")
                request["receipt"] = receipt
                request["event"].set()
                handle = self._pending_hybrid_joins.get(replica_id)
                if handle is not None:
                    timeout = float(getattr(self.config.global_resource_planner, "elastic_hybrid_join_timeout_s", 180))
                    handle.result(timeout=timeout if timeout > 0 else None)
            self._reconcile_pending_hybrid_joins()
        except Exception as exc:
            for request in pending.values():
                request["error"] = str(exc)
                request["event"].set()
            raise
        finally:
            with self._pending_activation_lock:
                for replica_id in pending:
                    self._pending_activations.pop(replica_id, None)
        # Once all loaders have acknowledged, their tensors are owned by the
        # workers and the transient core copies can be released collectively.
        if not self._pending_hybrid_joins:
            from RL_Framework.infra.elastic.core_reconfiguration import dispatch_core_command
            dispatch_core_command(self, "release", revisions=self._snapshot_manifests)
            for manifest in self._snapshot_manifests.values():
                Path(manifest).unlink(missing_ok=True)
            self._snapshot_manifests.clear()

    def _execute_membership_only(self, prepared):
        from RL_Framework.infra.elastic.rollout_lending import lend_rollout, return_rollout
        plan, result = prepared.candidate_plan, prepared.result
        cfg = self.config.global_resource_planner
        paused = self.dispatcher is not None
        if paused:
            self.dispatcher.pause()
            self.dispatcher.wait_until_idle(timeout=float(getattr(cfg, "runtime_drain_timeout_s", 3600)))
        try:
            if getattr(cfg, "runtime_reconfigure_training", False):
                if getattr(cfg, "runtime_training_pool_plan_only", True):
                    result.training_actions.extend(self._plan_training_pool_reconfiguration(prepared.current_train_gpus, plan))
                else:
                    self._ensure_elastic_pool()
                    desired = self._desired_hybrid_workers(prepared.current_train_gpus, plan)
                    if self._hybrid_worker_launch_enabled and desired:
                        self._cluster_swap_training_slots = lend_rollout(self, desired)
                    result.training_actions.extend(self._reconfigure_training_pool(prepared.current_train_gpus, plan, result))
                    self._finish_pending_activations()
                    if self._hybrid_worker_launch_enabled and not desired:
                        restored = return_rollout(self)
                        if restored:
                            self._publish_current_rollout_weights()
            from RL_Framework.infra.elastic.rollout_lifecycle import apply_availability, unavailable_rollouts
            trainer = getattr(self, "trainer", None)
            if trainer is not None:
                apply_availability(trainer, unavailable_rollouts(self))
            self._write_peer_rank_reconfiguration_state(plan, phase="applied")
            result.actions.append("elastic_membership_only")
            result.applied = True
            return result
        finally:
            if paused and result.applied:
                if getattr(self, "trainer", None) is not None:
                    self.trainer._resume_rollout_pipeline()
                elif not getattr(self, "_lent_rollout", None):
                    self.dispatcher.resume()

    def _publish_current_rollout_weights(self):
        """Publish while all core ranks are parked and admission stays closed."""
        from RL_Framework.infra.elastic.core_reconfiguration import dispatch_core_command
        targets = [{"instance_id": inst.instance_id, "tp": int(inst.tp),
                    "gpus": list(inst.gpus), "host": inst.host, "port": inst.port}
                   for inst in self.config.heterogeneous_rollout.instances]
        dispatch_core_command(self, "publish", target={"instances": targets})

    def _reconfigure_training_pool(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
        result: RuntimeReconfigurationResult,
    ) -> list[str]:
        if self.elastic_pool is None:
            return ["training_resize_requested_without_elastic_pool"]

        actions: list[str] = []
        replica_size = self._elastic_replica_size_gpus()
        desired_hybrid_workers = self._desired_hybrid_workers(current_train_gpus, plan)
        desired_replicas = desired_hybrid_workers // replica_size
        core_ids, _target_core, candidates = self._training_reconfiguration_targets(
            current_train_gpus,
            plan,
        )
        snapshot = self.elastic_pool.snapshot()
        replica_snapshot = (
            self.elastic_pool.replica_snapshot()
            if hasattr(self.elastic_pool, "replica_snapshot")
            else {}
        )
        active_replica_ids = [
            replica_id
            for replica_id, members in replica_snapshot.items()
            if members
            and all(
                self._role_name(worker.role)
                in {"hybrid_training", "hybrid_joining"}
                for worker in members
            )
        ]
        active_replicas = len(active_replica_ids)
        if not replica_snapshot and replica_size == 1:
            active_replicas = sum(
                1
                for worker in snapshot.values()
                if self._role_name(worker.role)
                in {"hybrid_training", "hybrid_joining"}
            )
        delta = desired_replicas - active_replicas

        if delta > 0:
            if self._hybrid_worker_launch_enabled:
                # Only confirmed loans/cluster swaps transfer physical GPU
                # ownership. A failed loan must not fall back to live rollout
                # slots or start a snapshot/join with an empty allocation.
                by_host: dict[str, list[str]] = {}
                physical = set()
                for worker_id in candidates:
                    slot = self._cluster_swap_training_slot(worker_id)
                    gpus = slot.get("gpus", [])
                    if len(gpus) != 1:
                        continue
                    host = str(slot.get("host", ""))
                    address = (host, int(gpus[0]))
                    if address in physical:
                        continue
                    physical.add(address)
                    by_host.setdefault(host, []).append(worker_id)
                groups = [
                    group for workers in by_host.values()
                    for group in self._candidate_replica_groups(workers, replica_size)
                ]
                if groups:
                    self._capture_core_snapshot()
            else:
                groups = self._candidate_replica_groups(candidates, replica_size)
            for offset, (replica_id, members) in enumerate(groups[:delta]):
                target_core = core_ids[(active_replicas + offset) % len(core_ids)]
                member_ranks = {worker_id: rank for rank, worker_id in enumerate(members)}

                def activate_replica(
                    _replica_id: str,
                    _members: tuple[str, ...],
                    target_core_id: str,
                    state_version: int,
                ) -> dict[str, Any]:
                    slots = [self._cluster_swap_training_slot(wid) for wid in _members]
                    hosts = {str(slot.get("host", "")) for slot in slots}
                    if len(hosts) > 1:
                        raise RuntimeError(
                            "a complete DP Replica must launch as one process group "
                            f"on one host; replica={_replica_id}, hosts={sorted(hosts)}"
                        )
                    gpus = [int(gpu) for slot in slots for gpu in slot.get("gpus", [])]
                    if len(gpus) != len(_members):
                        raise RuntimeError(
                            f"Replica { _replica_id } has {len(gpus)} GPUs for {len(_members)} members"
                        )
                    snapshot_path = self._snapshot_path_for_version(state_version)
                    command = self._build_hybrid_worker_command(
                        worker_id=_replica_id,
                        target_core_id=target_core_id,
                        snapshot_path=snapshot_path,
                        host=next(iter(hosts), ""),
                        gpus=gpus,
                        replica_id=_replica_id,
                        replica_rank=0,
                        replica_world_size=len(_members),
                    )
                    with self._hybrid_resource_guard():
                        task_dir = Path(self.config.global_resource_planner.hybrid_worker_task_dir)
                        for ready in task_dir.glob(f"{_replica_id}.rank_*.ready"):
                            ready.unlink(missing_ok=True)
                        (task_dir / f"{_replica_id}.ready").unlink(missing_ok=True)
                        host = next(iter(hosts), "")
                        if self._hybrid_worker_remote_control_enabled:
                            meta = self._request_remote_hybrid_worker(
                                worker_id=_replica_id, target_core_id=target_core_id,
                                snapshot_path=snapshot_path, command=command,
                                slot={"host": host, "gpus": gpus},
                            )
                            pid = meta.pid
                            proc = None
                        else:
                            proc = subprocess.Popen(command, shell=True, env=os.environ.copy(), start_new_session=True)
                            self._hybrid_worker_processes[_replica_id] = proc
                            pid = proc.pid
                        for rank, worker_id in enumerate(_members):
                            if proc is not None:
                                self._hybrid_worker_processes[worker_id] = proc
                            self._hybrid_worker_meta[worker_id] = ManagedHybridWorkerProcess(
                                worker_id=worker_id,
                                target_core_id=target_core_id,
                                command=command,
                                pid=pid,
                                snapshot_path=snapshot_path,
                                host=host,
                                gpus=gpus,
                                replica_id=_replica_id,
                                replica_rank=rank,
                                replica_world_size=len(_members),
                            )
                    receipt = self._wait_hybrid_worker_ready(
                        _replica_id, state_version=state_version,
                        replica_world_size=len(_members), snapshot_path=snapshot_path,
                        target_core_id=target_core_id,
                    )
                    return self._queue_activation(_replica_id, receipt)

                with self._hybrid_resource_guard():
                    handle = self.elastic_pool.join_replica(
                        replica_id,
                        members,
                        target_core,
                        replica_activation_barrier=(
                            activate_replica if self._hybrid_worker_launch_enabled else None
                        ),
                        timeout_s=float(
                            getattr(
                                self.config.global_resource_planner,
                                "elastic_hybrid_join_timeout_s",
                                180.0,
                            )
                        ),
                    )
                    with self._pending_hybrid_join_lock:
                        self._pending_hybrid_joins[replica_id] = handle
                        self._pending_hybrid_join_started_at[replica_id] = time.time()
                actions.append(
                    f"join_replica_submitted:{replica_id}[{','.join(members)}]"
                    f"->{target_core}:"
                    f"generation={handle.generation}"
                )
                if len(members) == 1:
                    actions.append(
                        f"join_training_submitted:{members[0]}->{target_core}:"
                        f"generation={handle.generation}"
                    )
                    # Preserve the historical singleton-worker action for
                    # control-plane consumers while replicas use the richer
                    # membership-qualified action below.
                    actions.append(f"join_training:{members[0]}->{target_core}")
                # Keep the old prefix for log consumers while making the DP
                # replica membership explicit.
                actions.append(
                    f"join_replica_training:{replica_id}[{','.join(members)}]->{target_core}"
                )
            if delta > len(groups):
                actions.append(f"join_replica_capacity_shortfall:{delta - len(groups)}")
        elif delta < 0:
            for replica_id in active_replica_ids[: abs(delta)]:
                with self._pending_hybrid_join_lock:
                    pending = self._pending_hybrid_joins.pop(replica_id, None)
                    self._pending_hybrid_join_started_at.pop(replica_id, None)
                if pending is not None:
                    pending.cancel()
                members = tuple(
                    worker.worker_id for worker in replica_snapshot[replica_id]
                )
                try:
                    self.elastic_pool.release_replica_to_rollout(replica_id)
                except (KeyError, RuntimeError):
                    # A cancellation may already have completed the rollback.
                    pass
                self._publish_hybrid_membership()
                for worker_id in members:
                    self._stop_hybrid_worker_process(worker_id, result)
                actions.append(
                    f"release_replica_to_rollout:{replica_id}[{','.join(members)}]"
                )
            if not replica_snapshot and replica_size == 1:
                active_workers = [
                    worker_id
                    for worker_id, worker in snapshot.items()
                    if self._role_name(worker.role)
                    in {"hybrid_training", "hybrid_joining"}
                ]
                for worker_id in active_workers[: abs(delta)]:
                    with self._pending_hybrid_join_lock:
                        pending = self._pending_hybrid_joins.pop(worker_id, None)
                        self._pending_hybrid_join_started_at.pop(worker_id, None)
                    if pending is not None:
                        pending.cancel()
                    try:
                        self.elastic_pool.release_to_rollout(worker_id)
                    except (KeyError, RuntimeError):
                        continue
                    self._publish_hybrid_membership()
                    self._stop_hybrid_worker_process(worker_id, result)
                    actions.append(f"release_to_rollout:{worker_id}")
        else:
            actions.append(
                f"training_gpus_unchanged:{current_train_gpus + desired_hybrid_workers}:"
                f"active_hybrid_replicas={active_replicas}"
            )
            actions.append(
                f"training_gpus_unchanged:{current_train_gpus + desired_hybrid_workers}:"
                f"active_hybrids={active_replicas}"
            )

        return actions

    def _desired_hybrid_workers(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan | None = None,
    ) -> int:
        """Return the GRP-commanded elastic GPU count in full-replica units."""
        cfg = self.config.global_resource_planner
        replica_size = self._elastic_replica_size_gpus()
        signal = self._active_elastic_signal
        if signal is not None:
            expires_step = int(signal.get("expires_step", 0) or 0)
            current_step = int(
                getattr(self.planner.latest_runtime_metrics, "step", 0) or 0
            )
            if not expires_step or current_step <= expires_step:
                desired = max(
                    0,
                    int(
                        signal.get(
                            "desired_replicas",
                            signal.get("desired_workers", 0),
                        )
                    ),
                )
                return min(desired, self._elastic_replica_capacity()) * replica_size
        if getattr(cfg, "elastic_hybrid_require_planner_signal", False):
            return 0
        target_train_gpus = int(
            getattr(cfg, "runtime_training_pool_target_gpus", 0) or 0
        )
        if target_train_gpus <= 0 and plan is not None:
            # Preserve the generic executor contract for callers that do not
            # require planner-only EHP control.  Production configs set
            # elastic_hybrid_require_planner_signal=true, so this fallback is
            # unreachable in the strict experiment arms.
            target_train_gpus = int(getattr(plan, "train_gpus", 0) or 0)
        desired = max(0, target_train_gpus - int(current_train_gpus))
        return min(
            desired // replica_size,
            self._elastic_replica_capacity(),
        ) * replica_size

    def _elastic_replica_size_gpus(self) -> int:
        cfg = self.config.global_resource_planner
        topology_width = (
            max(1, int(getattr(self.config, "train_tp_size", 1) or 1))
            * max(1, int(getattr(self.config, "train_ep_size", 1) or 1))
            * max(1, int(getattr(self.config, "train_pp_size", 1) or 1))
            * max(1, int(getattr(self.config, "train_cp_size", 1) or 1))
        )
        configured = int(
            getattr(cfg, "elastic_hybrid_replica_size_gpus", 0) or 0
        )
        if configured > 0:
            if configured != topology_width:
                raise ValueError(
                    "elastic_hybrid_replica_size_gpus must equal one complete "
                    f"DP Replica topology width ({topology_width}); got {configured}"
                )
            return configured
        return topology_width

    def _elastic_replica_capacity(self) -> int:
        cfg = self.config.global_resource_planner
        reserve = max(
            0,
            int(getattr(cfg, "elastic_hybrid_min_rollout_gpus", 0) or 0),
        )
        return max(
            0,
            (int(getattr(self.config, "rollout_gpus", 0) or 0) - reserve)
            // self._elastic_replica_size_gpus(),
        )

    @staticmethod
    def _candidate_replica_groups(
        candidates: list[str], replica_size: int
    ) -> list[tuple[str, tuple[str, ...]]]:
        groups: list[tuple[str, tuple[str, ...]]] = []
        for start in range(0, len(candidates), replica_size):
            members = tuple(candidates[start:start + replica_size])
            if len(members) != replica_size:
                break
            groups.append((f"ehp_replica_{members[0]}", members))
        return groups

    def _publish_hybrid_membership(self) -> dict[str, dict[str, Any]]:
        """Publish current membership, including retained detach versions.

        Serializing snapshot acquisition and publication prevents a delayed
        writer from overwriting a release with an older training record.
        """
        if self.elastic_pool is None:
            return {}
        directory = Path(getattr(
            self.config.global_resource_planner,
            "hybrid_worker_task_dir", "./logs/elastic_training_tasks",
        )) / "membership"
        with self._membership_publication_lock:
            records = self.elastic_pool.membership_snapshot()
            for replica_id, record in records.items():
                path = directory / f"{replica_id}.json"
                try:
                    self._write_json_atomic(path, {**record, "run_id": self.membership_run_id})
                except OSError as exc:
                    raise RuntimeError(
                        f"failed to publish EHP membership for {replica_id}: {path}"
                    ) from exc
            return records

    def hybrid_runtime_state(self) -> dict[str, Any]:
        """Expose EHP state to GRP without sharing mutable pool internals."""
        self._reconcile_pending_hybrid_joins()
        self._expire_stale_elastic_signal()
        active = 0
        joining = 0
        domain_state: dict[str, Any] = {}
        if self.elastic_pool is not None:
            records = self._publish_hybrid_membership()
            active = sum(record["role"] == "hybrid_training" for record in records.values())
            joining = sum(record["role"] == "hybrid_joining" for record in records.values())
            gradient_domain = getattr(self.elastic_pool, "gradient_domain", None)
            if gradient_domain is not None and hasattr(
                gradient_domain, "membership_state"
            ):
                domain_state = dict(gradient_domain.membership_state())
        with self._pending_hybrid_join_lock:
            pending = len(self._pending_hybrid_joins)
        return {
            "active_hybrid_replicas": active,
            "joining_hybrid_replicas": joining,
            "pending_hybrid_replica_joins": pending,
            # Compatibility aliases now carry logical replica counts.
            "active_hybrid_workers": active,
            "joining_hybrid_workers": joining,
            "pending_hybrid_joins": pending,
            "replica_size_gpus": self._elastic_replica_size_gpus(),
            "communication_domain": domain_state,
            "planner_signal": dict(self._active_elastic_signal or {}),
        }

    def accept_planner_signal(
        self,
        signal: Any | None,
        *,
        result: RuntimeReconfigurationResult | None = None,
    ) -> None:
        """Record a versioned GRP command and apply safety-critical releases.

        Join commands are enacted by :meth:`execute`, where a candidate plan is
        available.  Release commands are safe and urgent: they cancel pending
        activation and return active workers to rollout even if the resource
        topology itself does not otherwise need reconfiguration.
        """
        if signal is None:
            return
        payload = signal.to_dict() if hasattr(signal, "to_dict") else dict(signal)
        previous_step = int((self._active_elastic_signal or {}).get("step", -1))
        signal_step = int(payload.get("step", -1))
        if signal_step < previous_step:
            return
        self._active_elastic_signal = payload
        print(
            "[RuntimeElasticExecutor] GRP_EHP_SIGNAL "
            f"step={signal_step} action={payload.get('action', 'hold')} "
            f"desired_replicas={int(payload.get('desired_replicas', payload.get('desired_workers', 0)) or 0)} "
            f"current_replicas={int(payload.get('current_replicas', payload.get('current_workers', 0)) or 0)} "
            f"capacity_replicas={int(payload.get('capacity_replicas', payload.get('max_workers', 0)) or 0)} "
            f"replica_size_gpus={int(payload.get('replica_size_gpus', self._elastic_replica_size_gpus()) or 0)} "
            f"reason={payload.get('reason', '')} "
            f"expires_step={int(payload.get('expires_step', 0) or 0)}",
            flush=True,
        )
        if int(
            payload.get("desired_replicas", payload.get("desired_workers", 0)) or 0
        ) <= 0:
            self._release_all_hybrid_workers(
                reason=f"grp_{payload.get('reason', 'release')}",
                result=result,
            )

    def _expire_stale_elastic_signal(self) -> None:
        signal = self._active_elastic_signal
        if not signal:
            return
        expires_step = int(signal.get("expires_step", 0) or 0)
        current_step = int(
            getattr(self.planner.latest_runtime_metrics, "step", 0) or 0
        )
        if expires_step and current_step > expires_step:
            self._release_all_hybrid_workers(reason="planner_signal_expired")
            print(
                "[RuntimeElasticExecutor] GRP_EHP_SIGNAL_EXPIRED "
                f"signal_step={int(signal.get('step', -1))} "
                f"expires_step={expires_step} current_step={current_step}",
                flush=True,
            )
            self._active_elastic_signal = None

    def _release_all_hybrid_workers(
        self,
        *,
        reason: str,
        result: RuntimeReconfigurationResult | None = None,
    ) -> None:
        with self._pending_activation_lock:
            for request in self._pending_activations.values():
                request["error"] = f"EHP activation cancelled: {reason}"
                request["event"].set()
            self._pending_activations.clear()
        with self._pending_hybrid_join_lock:
            pending_items = list(self._pending_hybrid_joins.items())
            self._pending_hybrid_joins.clear()
            self._pending_hybrid_join_started_at.clear()
        for replica_id, handle in pending_items:
            handle.cancel()
            self._publish_hybrid_membership()
            members = tuple(getattr(handle, "member_worker_ids", (replica_id,)))
            for worker_id in members:
                self._stop_hybrid_worker_process(worker_id, result)
            if result is not None:
                result.training_actions.append(
                    f"join_replica_cancelled:{replica_id}:{reason}"
                )

        if self.elastic_pool is None:
            return
        replica_snapshot = (
            self.elastic_pool.replica_snapshot()
            if hasattr(self.elastic_pool, "replica_snapshot")
            else {}
        )
        for replica_id, members in replica_snapshot.items():
            if not any(
                self._role_name(worker.role)
                in {"hybrid_training", "hybrid_joining"}
                for worker in members
            ):
                continue
            try:
                self.elastic_pool.release_replica_to_rollout(replica_id)
            except (KeyError, RuntimeError):
                continue
            self._publish_hybrid_membership()
            for worker in members:
                self._stop_hybrid_worker_process(worker.worker_id, result)
            if result is not None:
                result.training_actions.append(
                    f"release_replica_to_rollout:{replica_id}:{reason}"
                )
        if not replica_snapshot:
            for worker_id, worker in self.elastic_pool.snapshot().items():
                if self._role_name(worker.role) not in {
                    "hybrid_training",
                    "hybrid_joining",
                }:
                    continue
                try:
                    self.elastic_pool.release_to_rollout(worker_id)
                except RuntimeError:
                    continue
                self._publish_hybrid_membership()
                self._stop_hybrid_worker_process(worker_id, result)

    def _reconcile_pending_hybrid_joins(
        self,
        result: RuntimeReconfigurationResult | None = None,
    ) -> None:
        """Collect completed background joins without blocking the train loop."""
        with self._pending_hybrid_join_lock:
            pending_items = list(self._pending_hybrid_joins.items())
        for replica_id, handle in pending_items:
            if not handle.done():
                started = self._pending_hybrid_join_started_at.get(replica_id, time.time())
                timeout = float(
                    getattr(
                        self.config.global_resource_planner,
                        "elastic_hybrid_join_timeout_s",
                        180.0,
                    )
                )
                if timeout > 0 and time.time() - started > timeout:
                    handle.cancel()
                    self._publish_hybrid_membership()
                    with self._pending_hybrid_join_lock:
                        self._pending_hybrid_joins.pop(replica_id, None)
                        self._pending_hybrid_join_started_at.pop(replica_id, None)
                    for worker_id in tuple(
                        getattr(handle, "member_worker_ids", (replica_id,))
                    ):
                        self._stop_hybrid_worker_process(worker_id, result)
                    if result is not None:
                        result.training_actions.append(
                            f"join_replica_timeout_cancelled:{replica_id}"
                        )
                continue
            try:
                joined = handle.result(timeout=0)
                workers = joined if isinstance(joined, tuple) else (joined,)
                if result is not None:
                    result.training_actions.append(
                        f"join_replica_active:{replica_id}:members="
                        f"{','.join(worker.worker_id for worker in workers)}:"
                        f"generation={handle.generation}"
                    )
            except Exception as exc:
                self._publish_hybrid_membership()
                for worker_id in tuple(
                    getattr(handle, "member_worker_ids", (replica_id,))
                ):
                    self._stop_hybrid_worker_process(worker_id, result)
                if result is not None:
                    result.errors.append(f"hybrid_replica_join_failed:{replica_id}:{exc}")
            finally:
                with self._pending_hybrid_join_lock:
                    self._pending_hybrid_joins.pop(replica_id, None)
                    self._pending_hybrid_join_started_at.pop(replica_id, None)

    def _activate_hybrid_worker(
        self,
        worker_id: str,
        target_core_id: str,
        state_version: int,
        *,
        replica_id: str = "",
        replica_rank: int = 0,
        replica_world_size: int = 1,
    ) -> dict[str, Any]:
        """Activation barrier executed by the pool's background join thread."""
        from RL_Framework.infra.elastic.readiness import clear_ready_receipts
        snapshot_path = self._snapshot_path_for_version(state_version)
        task_dir = Path(self.config.global_resource_planner.hybrid_worker_task_dir)
        slot = self._cluster_swap_training_slot(worker_id)
        if not slot and self._hybrid_worker_remote_control_enabled:
            slot = self._remote_hybrid_worker_slot(worker_id)
        if not self._is_hybrid_worker_running(worker_id):
            clear_ready_receipts(task_dir, worker_id)
            command = self._build_hybrid_worker_command(
                worker_id=worker_id,
                target_core_id=target_core_id,
                snapshot_path=snapshot_path,
                idle=False,
                host=str(slot.get("host", "")),
                gpus=[int(gpu) for gpu in slot.get("gpus", [])],
                replica_id=replica_id or worker_id,
                replica_rank=replica_rank,
                replica_world_size=replica_world_size,
            )
            with self._hybrid_resource_guard():
                if self._hybrid_worker_remote_control_enabled:
                    meta = self._request_remote_hybrid_worker(
                        worker_id=worker_id,
                        target_core_id=target_core_id,
                        snapshot_path=snapshot_path,
                        command=command,
                        slot=slot,
                    )
                    meta.replica_id = replica_id or worker_id
                    meta.replica_rank = replica_rank
                    meta.replica_world_size = replica_world_size
                    self._hybrid_worker_meta[worker_id] = meta
                else:
                    proc = subprocess.Popen(command, shell=True, env=os.environ.copy(), start_new_session=True)
                    self._hybrid_worker_processes[worker_id] = proc
                    self._hybrid_worker_meta[worker_id] = ManagedHybridWorkerProcess(
                        worker_id=worker_id,
                        target_core_id=target_core_id,
                        command=command,
                        pid=proc.pid,
                        snapshot_path=snapshot_path,
                        host=str(slot.get("host", "")),
                        gpus=[int(gpu) for gpu in slot.get("gpus", [])],
                        replica_id=replica_id or worker_id,
                        replica_rank=replica_rank,
                        replica_world_size=replica_world_size,
                    )
        else:
            meta = self._hybrid_worker_meta.get(worker_id)
            if meta is not None and (
                meta.snapshot_path != snapshot_path
                or meta.replica_id != (replica_id or worker_id)
                or meta.replica_rank != replica_rank
                or meta.replica_world_size != replica_world_size
            ):
                # A prewarmed process may have loaded an older checkpoint. Relaunch
                # it in the background before the domain marks it active.
                with self._hybrid_resource_guard():
                    self._stop_hybrid_worker_process(worker_id)
                clear_ready_receipts(task_dir, worker_id)
                command = self._build_hybrid_worker_command(
                    worker_id=worker_id,
                    target_core_id=target_core_id,
                    snapshot_path=snapshot_path,
                    idle=False,
                    host=str(slot.get("host", "")),
                    gpus=[int(gpu) for gpu in slot.get("gpus", [])],
                    replica_id=replica_id or worker_id,
                    replica_rank=replica_rank,
                    replica_world_size=replica_world_size,
                )
                with self._hybrid_resource_guard():
                    if self._hybrid_worker_remote_control_enabled:
                        meta = self._request_remote_hybrid_worker(
                            worker_id=worker_id,
                            target_core_id=target_core_id,
                            snapshot_path=snapshot_path,
                            command=command,
                            slot=slot,
                        )
                        meta.replica_id = replica_id or worker_id
                        meta.replica_rank = replica_rank
                        meta.replica_world_size = replica_world_size
                        self._hybrid_worker_meta[worker_id] = meta
                    else:
                        proc = subprocess.Popen(command, shell=True, env=os.environ.copy(), start_new_session=True)
                        self._hybrid_worker_processes[worker_id] = proc
                        self._hybrid_worker_meta[worker_id] = ManagedHybridWorkerProcess(
                            worker_id=worker_id,
                            target_core_id=target_core_id,
                            command=command,
                            pid=proc.pid,
                            snapshot_path=snapshot_path,
                            replica_id=replica_id or worker_id,
                            replica_rank=replica_rank,
                            replica_world_size=replica_world_size,
                        )
        receipt = self._wait_hybrid_worker_ready(
            worker_id, state_version=state_version, replica_world_size=replica_world_size,
            snapshot_path=snapshot_path, target_core_id=target_core_id,
        )
        return self._queue_activation(replica_id or worker_id, receipt)

    @property
    def _hybrid_worker_remote_control_enabled(self) -> bool:
        return bool(
            getattr(
                self.config.global_resource_planner,
                "hybrid_worker_remote_control_enabled",
                False,
            )
        )

    def _hybrid_worker_control_dir(self) -> Path:
        cfg = self.config.global_resource_planner
        return Path(
            getattr(cfg, "hybrid_worker_remote_control_dir", "")
            or getattr(cfg, "hybrid_worker_task_dir", "./logs/elastic_training_tasks")
        )

    def _remote_hybrid_worker_slot(self, worker_id: str) -> dict[str, Any]:
        instances = list(
            getattr(self.config.heterogeneous_rollout, "instances", []) or []
        )
        try:
            index = int(worker_id.removeprefix("rollout"))
        except ValueError:
            index = 0
        if not instances:
            raise RuntimeError("remote EHP launch requires rollout instances")
        physical_slots = sorted(
            {
                (str(getattr(instance, "host", "")), int(gpu))
                for instance in instances
                for gpu in getattr(instance, "gpus", [])
            }
        )
        if index >= len(physical_slots):
            raise RuntimeError(
                f"hybrid worker {worker_id} exceeds physical rollout slots: "
                f"{len(physical_slots)}"
            )
        host, gpu = physical_slots[index]
        return {
            "host": host,
            "gpus": [gpu],
        }

    def _request_remote_hybrid_worker(
        self,
        *,
        worker_id: str,
        target_core_id: str,
        snapshot_path: str,
        command: str,
        slot: dict[str, Any],
    ) -> ManagedHybridWorkerProcess:
        with self._hybrid_resource_guard():
            return self._request_remote_hybrid_worker_locked(
                worker_id=worker_id, target_core_id=target_core_id,
                snapshot_path=snapshot_path, command=command, slot=slot,
            )

    def _request_remote_hybrid_worker_locked(
        self,
        *,
        worker_id: str,
        target_core_id: str,
        snapshot_path: str,
        command: str,
        slot: dict[str, Any],
    ) -> ManagedHybridWorkerProcess:
        host = str(slot.get("host", ""))
        if not host:
            raise RuntimeError(f"remote EHP worker {worker_id} has no host")
        control_dir = self._hybrid_worker_control_dir()
        control_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("started", "error"):
            (control_dir / f".launch_{worker_id}.{suffix}").unlink(missing_ok=True)
        request = control_dir / f".launch_{worker_id}.json"
        tmp = request.with_suffix(request.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "target_core_id": target_core_id,
                    "snapshot_path": snapshot_path,
                    "host": host,
                    "gpus": [int(gpu) for gpu in slot.get("gpus", [])],
                    "command": command,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(request)
        meta = ManagedHybridWorkerProcess(
            worker_id=worker_id,
            target_core_id=target_core_id,
            command=command,
            pid=-1,
            snapshot_path=snapshot_path,
            host=host,
            gpus=[int(gpu) for gpu in slot.get("gpus", [])],
        )
        self._hybrid_worker_meta[worker_id] = meta
        print(
            "[RuntimeElasticExecutor] remote_hybrid_launch_requested "
            f"worker={worker_id} host={host} gpus={slot.get('gpus', [])}",
            flush=True,
        )
        return meta

    def _cluster_swap_enabled(self, strategy: str | None = None) -> bool:
        cfg = self.config.global_resource_planner
        return bool(
            getattr(cfg, "runtime_cluster_swap_enabled", False)
            or (strategy or getattr(cfg, "runtime_rollout_reconfigure_strategy", ""))
            == "cluster_swap"
        )

    def _runtime_plan_already_applied(
        self,
        plan: GlobalResourcePlan,
        *,
        current_train_gpus: int,
    ) -> bool:
        planner_cfg = self.config.global_resource_planner
        if (
            getattr(planner_cfg, "runtime_reconfigure_training", False)
            and int(getattr(plan, "train_gpus", current_train_gpus)) != current_train_gpus
        ):
            return False
        target_train_gpus = current_train_gpus + self._desired_hybrid_workers(
            current_train_gpus,
            plan,
        )
        reconfigure_training = bool(
            getattr(planner_cfg, "runtime_reconfigure_training", False)
        )
        if reconfigure_training and target_train_gpus != current_train_gpus:
            return False

        desired = {
            meta.instance_id: meta
            for meta in self._desired_rollout_processes(plan)
        }
        if set(desired) != set(self._process_meta):
            return False
        return all(
            self._same_rollout_process(self._process_meta[instance_id], wanted)
            for instance_id, wanted in desired.items()
        )

    def _cluster_swap_rollout_training_pools(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
        result: RuntimeReconfigurationResult,
    ) -> None:
        """Swap GPUs between rollout and training without spare prewarm GPUs."""

        planner_cfg = self.config.global_resource_planner
        reconfigure_training = bool(
            getattr(planner_cfg, "runtime_reconfigure_training", False)
        )
        plan_only = bool(
            getattr(planner_cfg, "runtime_training_pool_plan_only", True)
        )

        if reconfigure_training and not plan_only:
            self._ensure_elastic_pool()

        planned_train_gpus = int(getattr(plan, "train_gpus", current_train_gpus))
        target_train_gpus = (
            planned_train_gpus
            if planned_train_gpus < current_train_gpus
            else current_train_gpus + self._desired_hybrid_workers(
                current_train_gpus,
                plan,
            )
        )
        delta = target_train_gpus - current_train_gpus
        result.actions.append(
            f"cluster_swap_begin:train_delta={delta}:target_train_gpus={target_train_gpus}"
        )

        if reconfigure_training and plan_only:
            result.training_actions.extend(
                self._plan_training_pool_reconfiguration(current_train_gpus, plan)
            )
        elif reconfigure_training and delta <= 0:
            result.training_actions.extend(
                self._reconfigure_training_pool(current_train_gpus, plan, result)
            )

        if getattr(planner_cfg, "runtime_manage_rollout_processes", False):
            stopped, started = self._reconfigure_rollout_processes(
                plan,
                strategy_override="cluster_swap",
            )
            result.stopped_processes.extend(stopped)
            result.started_processes.extend(started)
            result.actions.append("reconfigure_rollout_processes:cluster_swap")
            if delta > 0:
                assigned = self._assign_cluster_swap_training_slots(
                    stopped,
                    current_train_gpus,
                    plan,
                )
                if assigned:
                    result.training_actions.append(
                        "cluster_swap_training_slots:"
                        + ",".join(
                            f"{worker_id}@{slot['host']}:{slot['gpus']}"
                            for worker_id, slot in assigned.items()
                        )
                    )
        elif (
            getattr(planner_cfg, "apply_to_runtime", True)
            and self.rollout_engine is not None
            and hasattr(self.rollout_engine, "reconfigure_from_plan")
        ):
            self._reconfigure_rollout_engine(plan, wait_ready=False)
            self._rollout_engine_reconfigured = True
            result.actions.append("reconfigure_rollout_engine:cluster_swap")

        if reconfigure_training and not plan_only and delta > 0:
            result.training_actions.extend(
                self._reconfigure_training_pool(current_train_gpus, plan, result)
            )

        result.actions.append("cluster_swap_complete")

    def _assign_cluster_swap_training_slots(
        self,
        stopped_rollout: list[ManagedRolloutProcess],
        current_train_gpus: int,
        plan: GlobalResourcePlan,
    ) -> dict[str, dict[str, Any]]:
        target_train_gpus = current_train_gpus + self._desired_hybrid_workers(
            current_train_gpus,
            plan,
        )
        delta = max(0, target_train_gpus - current_train_gpus)
        if delta <= 0 or not stopped_rollout:
            self._cluster_swap_training_slots = {}
            return {}

        _core_ids, _target_core, candidates = self._training_reconfiguration_targets(
            current_train_gpus,
            plan,
        )
        desired_rollout_slots = {
            (meta.host, int(gpu))
            for meta in self._desired_rollout_processes(plan)
            for gpu in meta.gpus
        }
        slots: list[dict[str, Any]] = []
        for meta in stopped_rollout:
            for gpu in meta.gpus:
                if (meta.host, int(gpu)) in desired_rollout_slots:
                    continue
                slots.append({"host": meta.host, "gpus": [int(gpu)]})

        assigned: dict[str, dict[str, Any]] = {}
        for worker_id, slot in zip(candidates[:delta], slots[:delta]):
            assigned[str(worker_id)] = slot
        self._cluster_swap_training_slots = assigned
        return assigned

    def _cluster_swap_training_slot(self, worker_id: str) -> dict[str, Any]:
        slots = getattr(self, "_cluster_swap_training_slots", {})
        return dict(slots.get(worker_id, {}))

    def _training_reconfiguration_targets(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
    ) -> tuple[list[str], str, list[str]]:
        if self.elastic_pool is not None:
            snapshot = self.elastic_pool.snapshot()
            core_ids = [
                wid for wid, worker in snapshot.items()
                if self._role_name(worker.role) == "core_training"
            ]
            target_core = core_ids[0] if core_ids else next(iter(snapshot))
            candidates = [
                wid for wid, worker in snapshot.items()
                if self._role_name(worker.role) in {"core_rollout", "hybrid_rollout"}
            ]
            return core_ids, target_core, candidates

        dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
        core_ids = [f"dp{i}" for i in range(dp)]
        target_core = core_ids[0]
        target_train_gpus = current_train_gpus + self._desired_hybrid_workers(
            current_train_gpus,
            plan,
        )
        delta = max(0, target_train_gpus - current_train_gpus)
        rollout_workers = [
            f"rollout{i}" for i in range(max(delta, int(self.config.rollout_gpus)))
        ]
        return core_ids, target_core, rollout_workers

    def _plan_training_pool_reconfiguration(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
    ) -> list[str]:
        """Record a training-pool plan without mutating Megatron process groups.

        Megatron-Core cannot safely change the live distributed optimizer or
        gradient-domain membership in the middle of a running process group.
        This mode validates the online planner and the Elastic Hybrid Pool
        control-plane decision while keeping the training update path fixed.
        A deployment with real external hybrid workers can disable
        runtime_training_pool_plan_only to attach the gradient domain.
        """

        planner_cfg = self.config.global_resource_planner
        target_train_gpus = current_train_gpus + self._desired_hybrid_workers(
            current_train_gpus,
            plan,
        )
        delta = target_train_gpus - current_train_gpus
        if delta == 0:
            return [f"plan_training_gpus_unchanged:{target_train_gpus}"]

        current_dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
        core_ids = [f"dp{i}" for i in range(current_dp)]
        replica_size = self._elastic_replica_size_gpus()

        if delta > 0:
            rollout_workers = [
                f"rollout{i}" for i in range(max(0, int(self.config.rollout_gpus)))
            ]
            actions = ["training_pool_plan_only"]
            desired_replicas = delta // replica_size
            groups = self._candidate_replica_groups(rollout_workers, replica_size)
            for index, (replica_id, members) in enumerate(groups[:desired_replicas]):
                target_core = core_ids[index % len(core_ids)]
                actions.append(
                    f"plan_join_replica:{replica_id}[{','.join(members)}]->{target_core}"
                )
                actions.append(
                    f"plan_join_training:{replica_id}[{','.join(members)}]->{target_core}"
                )
            if desired_replicas > len(groups):
                actions.append(
                    f"plan_join_replica_shortfall:{desired_replicas - len(groups)}"
                )
            return actions

        release_count = abs(delta) // replica_size
        return [
            "training_pool_plan_only",
            *[
                f"plan_release_replica_to_rollout:ehp_replica_{i}"
                for i in range(release_count)
            ],
        ]

    def _apply_plan_to_runtime_config(self, plan: GlobalResourcePlan) -> None:
        """Apply rollout changes while preserving fixed Megatron topology if requested."""

        planner_cfg = self.config.global_resource_planner
        preserve_training = bool(
            getattr(planner_cfg, "runtime_training_pool_only", True)
            and getattr(planner_cfg, "runtime_reconfigure_training", False)
        )
        if not preserve_training:
            self.planner.apply_plan_to_config(plan, self.config)
            return

        training_fields = {
            "n_total_gpus": self.config.n_total_gpus,
            "train_gpus": self.config.train_gpus,
            "train_tp_size": self.config.train_tp_size,
            "train_ep_size": self.config.train_ep_size,
            "train_cp_size": self.config.train_cp_size,
            "tp_size": self.config.tp_size,
            "train_pp_size": self.config.train_pp_size,
            "train_dp_size": self.config.train_dp_size,
            "micro_batch_size": self.config.micro_batch_size,
        }
        self.planner.apply_plan_to_config(plan, self.config)
        for field_name, value in training_fields.items():
            setattr(self.config, field_name, value)

    def _prewarm_capacity_issue(self, plan: GlobalResourcePlan) -> str:
        """Return a reason string when the target warm pool cannot fit."""

        desired = self._desired_rollout_processes(plan)
        current_by_id = dict(self._process_meta)
        starting: list[ManagedRolloutProcess] = []
        for wanted in desired:
            current = current_by_id.get(wanted.instance_id)
            if current is None or not self._same_rollout_process(current, wanted):
                starting.append(wanted)
        if not starting:
            return ""

        per_host_capacity: dict[str, int] = {}
        per_host_current: dict[str, int] = {}
        per_host_starting: dict[str, int] = {}

        for meta in list(current_by_id.values()) + desired:
            if not meta.gpus:
                continue
            host = meta.host or "127.0.0.1"
            per_host_capacity[host] = max(
                per_host_capacity.get(host, 0),
                max(int(gpu) for gpu in meta.gpus) + 1,
                len(meta.gpus),
            )
        hetero = self.config.heterogeneous_rollout
        if getattr(hetero, "available_gpus", None):
            host = hetero.vllm_host or "127.0.0.1"
            if host == "0.0.0.0":
                host = "127.0.0.1"
            available = [int(gpu) for gpu in hetero.available_gpus]
            per_host_capacity[host] = max(
                per_host_capacity.get(host, 0),
                max(available) + 1,
                len(available),
            )

        for meta in current_by_id.values():
            host = meta.host or "127.0.0.1"
            per_host_current[host] = per_host_current.get(host, 0) + max(
                int(meta.tp),
                len(meta.gpus),
            )

        for meta in starting:
            host = meta.host or "127.0.0.1"
            per_host_starting[host] = per_host_starting.get(host, 0) + max(
                int(meta.tp),
                len(meta.gpus),
            )

        for host, starting_gpus in per_host_starting.items():
            capacity = per_host_capacity.get(host, 0)
            demand = per_host_current.get(host, 0) + starting_gpus
            if capacity > 0 and demand > capacity:
                return f"no_spare_gpu_capacity:{host}:{demand}>{capacity}"

        return ""

    def _coordination_dir(self) -> Path:
        planner_cfg = self.config.global_resource_planner
        configured = getattr(planner_cfg, "runtime_reconfiguration_coord_dir", "")
        if configured:
            return Path(configured)
        return Path(self.config.log_dir) / "runtime_reconfiguration"

    def _current_coordination_id(self, plan: GlobalResourcePlan) -> str:
        return getattr(self, "_active_runtime_coord_id", "") or uuid.uuid4().hex

    def _write_peer_rank_reconfiguration_state(
        self,
        plan: GlobalResourcePlan,
        *,
        phase: str,
        coord_id: str | None = None,
        error: str = "",
        elastic_signal: Any | None = None,
    ) -> Path:
        coord_dir = self._coordination_dir()
        coord_dir.mkdir(parents=True, exist_ok=True)
        if coord_id is None:
            coord_id = getattr(self, "_active_runtime_coord_id", "")
        payload = {
            "coord_id": coord_id,
            "run_id": self.runtime_run_id,
            "phase": phase,
            "unavailable_rollout_ids": [meta.instance_id for meta in
                (getattr(self, "_lent_rollout", None) or {}).get("engines", [])],
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
            "updated_at": time.time(),
            "plan": plan.to_dict(),
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "tp": int(inst.tp),
                    "gpus": list(inst.gpus),
                    "host": inst.host or self.config.heterogeneous_rollout.vllm_host,
                    "port": int(inst.port or (self.config.heterogeneous_rollout.vllm_base_port + idx)),
                }
                for idx, inst in enumerate(self.config.heterogeneous_rollout.instances)
            ],
            "training": self._peer_training_reconfiguration_state(
                plan,
                elastic_signal=elastic_signal,
            ),
            "error": error,
        }
        path = coord_dir / f"{phase}.json"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
        return path

    def _peer_training_reconfiguration_state(
        self,
        plan: GlobalResourcePlan,
        *,
        elastic_signal: Any | None = None,
    ) -> dict[str, Any]:
        planner_cfg = self.config.global_resource_planner
        if not getattr(planner_cfg, "runtime_reconfigure_training", False):
            return {"enabled": False}

        current_train_gpus = int(getattr(self.config, "train_gpus", 0) or 0)
        desired_hybrid_workers = self._desired_hybrid_workers(current_train_gpus, plan)
        target_train_gpus = current_train_gpus + desired_hybrid_workers
        dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
        core_ids = [f"dp{i}" for i in range(dp)]
        rollout_workers = [
            f"rollout{i}" for i in range(max(0, int(self.config.rollout_gpus)))
        ]
        hybrid_targets: dict[str, str] = {}
        if desired_hybrid_workers > 0:
            replica_size = self._elastic_replica_size_gpus()
            groups = self._candidate_replica_groups(rollout_workers, replica_size)
            desired_replicas = desired_hybrid_workers // replica_size
            hybrid_targets = {
                replica_id: core_ids[index % len(core_ids)]
                for index, (replica_id, _members) in enumerate(
                    groups[:desired_replicas]
                )
            }
        active_hybrid_ids: list[str] = []
        if self.elastic_pool is not None:
            membership = self.elastic_pool.gradient_domain.membership_state()
            actual_targets = {
                str(worker_id): str(target_core)
                for worker_id, target_core in (
                    membership.get("hybrid_targets") or {}
                ).items()
            }
            if actual_targets:
                hybrid_targets = actual_targets
            active_hybrid_ids = [
                str(worker_id)
                for worker_id in membership.get("active_replica_ids", ())
                if str(worker_id) in hybrid_targets
            ]
        if elastic_signal is None:
            signal_payload = dict(self._active_elastic_signal or {})
        elif hasattr(elastic_signal, "to_dict"):
            signal_payload = dict(elastic_signal.to_dict())
        else:
            signal_payload = dict(elastic_signal)
        return {
            "enabled": True,
            "plan_only": bool(
                getattr(planner_cfg, "runtime_training_pool_plan_only", True)
            ),
            "target_train_gpus": target_train_gpus,
            "current_train_gpus": current_train_gpus,
            "core_replica_ids": core_ids,
            "hybrid_targets": hybrid_targets,
            "active_hybrid_ids": active_hybrid_ids,
            "replica_size_gpus": self._elastic_replica_size_gpus(),
            "elastic_hybrid_signal": signal_payload,
            # Compatibility summary for older readers. Peers use the explicit
            # list above so a submitted join cannot become active before its
            # worker crosses the activation barrier.
            "activate_hybrids": bool(active_hybrid_ids),
        }

    def publish_peer_rank_drain_request(
        self,
        prepared: PreparedRuntimeTransaction,
    ) -> None:
        """Atomically expose a prepared drain request before its boundary decision."""
        if not prepared.should_execute or not prepared.drain_required:
            raise RuntimeError("peer drain request requires a draining transaction")
        if prepared.candidate_plan is None or not prepared.coord_id:
            raise RuntimeError("prepared drain transaction is incomplete")
        self._active_runtime_coord_id = prepared.coord_id
        self._publish_peer_rank_drain_request(
            prepared.candidate_plan,
            prepared.result,
            coord_id=prepared.coord_id,
            elastic_signal=prepared.effective_planner_signal,
        )

    def wait_for_peer_rank_drain(
        self,
        prepared: PreparedRuntimeTransaction,
    ) -> None:
        """Wait for ready records after request and boundary publication."""
        if not prepared.should_execute or not prepared.drain_required:
            raise RuntimeError("peer ready wait requires a draining transaction")
        if prepared.candidate_plan is None or not prepared.coord_id:
            raise RuntimeError("prepared drain transaction is incomplete")
        self._active_runtime_coord_id = prepared.coord_id
        self._coordinate_peer_rank_drain(
            prepared.candidate_plan,
            prepared.result,
        )

    def _publish_peer_rank_drain_request(
        self,
        plan: GlobalResourcePlan,
        result: RuntimeReconfigurationResult,
        *,
        coord_id: str,
        elastic_signal: Any | None = None,
    ) -> None:
        if self._peer_drain_request_coord_id == coord_id:
            return
        coord_dir = self._coordination_dir()
        ready_dir = coord_dir / "ready"
        ready_dir.mkdir(parents=True, exist_ok=True)
        for old_ready in ready_dir.glob("rank_*.json"):
            old_ready.unlink(missing_ok=True)
        for old_state in ("request.json", "applied.json", "aborted.json"):
            (coord_dir / old_state).unlink(missing_ok=True)
        self._write_peer_rank_reconfiguration_state(
            plan,
            phase="request",
            coord_id=coord_id,
            elastic_signal=elastic_signal,
        )
        self._peer_drain_request_coord_id = coord_id
        self._peer_drain_completed_coord_id = ""
        result.actions.append("peer_reconfig_request")

    def _coordinate_peer_rank_drain(
        self,
        plan: GlobalResourcePlan,
        result: RuntimeReconfigurationResult,
    ) -> None:
        if int(os.environ.get("WORLD_SIZE", "1") or "1") <= 1:
            return
        if int(os.environ.get("RANK", "0") or "0") != 0:
            return
        if not getattr(
            self.config.global_resource_planner,
            "runtime_coordinate_reconfiguration_ranks",
            True,
        ):
            return
        if not bool(
            getattr(
                self.config.global_resource_planner,
                "runtime_drain_before_reconfigure",
                True,
            )
        ):
            result.actions.append("peer_reconfig_drain_skipped")
            return

        coord_id = self._current_coordination_id(plan)
        self._active_runtime_coord_id = coord_id
        if self._peer_drain_completed_coord_id == coord_id:
            return
        coord_dir = self._coordination_dir()
        ready_dir = coord_dir / "ready"
        if self._peer_drain_request_coord_id != coord_id:
            self._publish_peer_rank_drain_request(
                plan,
                result,
                coord_id=coord_id,
            )

        timeout = float(
            getattr(
                self.config.global_resource_planner,
                "runtime_drain_timeout_s",
                3600.0,
            )
        )
        deadline = time.time() + timeout if timeout > 0 else float("inf")
        expected_ranks = self._expected_peer_ready_ranks()
        expected = [ready_dir / f"rank_{rank}.json" for rank in expected_ranks]
        job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", ""))

        def ready_matches(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return False
            if str(payload.get("coord_id", "")) != coord_id:
                return False
            if self.runtime_run_id and payload.get("run_id") != self.runtime_run_id:
                return False
            ready_job_id = str(payload.get("job_id", ""))
            return not job_id or ready_job_id == job_id

        while time.time() < deadline:
            if all(ready_matches(path) for path in expected):
                result.actions.append(
                    "peer_reconfig_drain:"
                    + ",".join(str(rank) for rank in expected_ranks)
                )
                self._peer_drain_completed_coord_id = coord_id
                return
            time.sleep(0.5)
        missing = [str(path) for path in expected if not ready_matches(path)]
        raise TimeoutError(
            "timed out waiting for peer ranks to drain before runtime reconfiguration: "
            + ", ".join(missing)
        )

    def _expected_peer_ready_ranks(self) -> list[int]:
        """Return ranks that own rollout clients and must drain before a cutover."""
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size <= 1:
            return []
        planner_cfg = self.config.global_resource_planner
        if (getattr(planner_cfg, "runtime_reconfigure_training", False)
                or not getattr(planner_cfg, "runtime_coordinate_batch_source_only", True)):
            return list(range(1, world_size))

        model_parallel = (
            max(1, int(getattr(self.config, "train_tp_size", 1) or 1))
            * max(1, int(getattr(self.config, "train_pp_size", 1) or 1))
        )
        if getattr(self.config, "train_backend", "") == "megatron_core":
            model_parallel *= max(
                1, int(getattr(self.config, "train_cp_size", 1) or 1)
            )
        if model_parallel <= 1:
            return list(range(1, world_size))

        source_ranks = [
            rank for rank in range(world_size)
            if rank % model_parallel == 0
        ]
        return [rank for rank in source_ranks if rank != 0]

    def _ensure_elastic_pool(self):
        with self._hybrid_resource_guard():
            self._initialize_elastic_pool()

    def _initialize_elastic_pool(self):
        if self.elastic_pool is not None:
            self._ensure_gradient_server()
            self._attach_train_engine_gradient_domain()
            return
        if self.train_engine is None:
            return

        from RL_Framework.infra.elastic.hybrid_pool import (
            ElasticHybridPool,
            GradientCommunicationDomains,
            InterReplicaGradientDomain,
        )

        self._ensure_gradient_server()
        if hasattr(self.train_engine, "get_elastic_core_replica_ids"):
            core_ids = list(self.train_engine.get_elastic_core_replica_ids())
        else:
            dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
            core_ids = [f"dp{i}" for i in range(dp)]

        rollout_workers = [
            f"rollout{i}" for i in range(max(0, int(self.config.rollout_gpus)))
        ]

        def fetch_snapshot(worker_id: str, target_core_id: str) -> int:
            if self._hybrid_worker_launch_enabled:
                if not self._snapshot_manifests:
                    raise RuntimeError("EHP join has no collective live-state snapshot at this boundary")
                return max(self._snapshot_manifests)
            if hasattr(self.train_engine, "capture_elastic_state_snapshot"):
                return int(
                    self.train_engine.capture_elastic_state_snapshot(
                        worker_id,
                        target_core_id,
                    )
                )
            return int(getattr(self.train_engine, "current_version", 0))

        decoupled = bool(
            getattr(
                self.config.global_resource_planner,
                "decouple_communication_domains",
                True,
            )
        )
        core_group = None
        hybrid_group = None
        if hasattr(self.train_engine, "get_elastic_core_process_group"):
            core_group = self.train_engine.get_elastic_core_process_group()
        if decoupled and hasattr(
            self.train_engine, "get_elastic_gradient_process_group"
        ):
            hybrid_group = self.train_engine.get_elastic_gradient_process_group()
        gradient_domain = InterReplicaGradientDomain(
            core_replica_ids=core_ids,
            communication_domains=GradientCommunicationDomains(
                core_process_group=core_group,
                hybrid_process_group=hybrid_group,
                decoupled=decoupled,
            ),
            require_isolated_process_group=bool(
                getattr(
                    self.config.global_resource_planner,
                    "elastic_hybrid_require_isolated_ccl",
                    False,
                )
            ),
            replica_world_size=self._elastic_replica_size_gpus(),
            local_replica_rank=(
                int(getattr(self.train_engine, "rank", os.environ.get("RANK", 0)) or 0)
                % self._elastic_replica_size_gpus()
            ),
        )
        gradient_domain.validate_communication_isolation()
        mode = "decoupled" if decoupled else "coupled"

        self.elastic_pool = ElasticHybridPool(
            core_train_workers=core_ids,
            core_rollout_workers=rollout_workers,
            gradient_domain=gradient_domain,
            snapshot_fetcher=fetch_snapshot,
            zero_sync_steps=1,
        )
        print(
            "[RuntimeElasticExecutor] elastic_gradient_domain "
            f"mode={mode} core_replicas={core_ids}",
            flush=True,
        )
        self._attach_train_engine_gradient_domain()

    @property
    def _hybrid_worker_launch_enabled(self) -> bool:
        return bool(
            getattr(
                self.config.global_resource_planner,
                "hybrid_worker_launch_enabled",
                False,
            )
        )

    @property
    def _hybrid_training_prewarm_enabled(self) -> bool:
        cfg = self.config.global_resource_planner
        return bool(
            getattr(cfg, "hybrid_training_prewarm_enabled", False)
            and getattr(cfg, "hybrid_worker_launch_enabled", False)
        )

    def _prewarm_hybrid_training_workers(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
        desired_hybrid_workers: int | None = None,
    ) -> list[str]:
        if self.elastic_pool is None:
            return []

        if desired_hybrid_workers is None:
            desired_hybrid_workers = self._desired_hybrid_workers(current_train_gpus, plan)
        snapshot = self.elastic_pool.snapshot()
        active = sum(
            1
            for worker in snapshot.values()
            if self._role_name(worker.role) in {"hybrid_training", "hybrid_joining"}
        )
        delta = max(0, int(desired_hybrid_workers) - active)
        if delta <= 0:
            return []

        _core_ids, target_core, candidates = self._training_reconfiguration_targets(
            current_train_gpus,
            plan,
        )
        worker_ids = self._hybrid_training_prewarm_worker_ids(candidates[:delta])
        actions: list[str] = []
        for worker_id in worker_ids[:delta]:
            if self._is_hybrid_worker_running(worker_id):
                actions.append(
                    f"prewarmed_hybrid_worker_already_running:{worker_id}->{target_core}"
                )
                continue
            snapshot_path = self._prewarm_snapshot_path(worker_id, target_core)
            command = self._build_hybrid_worker_command(
                worker_id=worker_id,
                target_core_id=target_core,
                snapshot_path=snapshot_path,
                idle=True,
            )
            meta = self._launch_prewarmed_hybrid_worker(
                worker_id=worker_id,
                target_core_id=target_core,
                command=command,
                snapshot_path=snapshot_path,
            )
            actions.append(
                f"prewarm_hybrid_worker:{meta.worker_id}->{meta.target_core_id}"
            )
        return actions

    def _hybrid_training_prewarm_worker_ids(
        self,
        candidates: list[str],
    ) -> list[str]:
        cfg = self.config.global_resource_planner
        configured = [
            str(worker_id)
            for worker_id in getattr(cfg, "hybrid_training_prewarm_worker_ids", [])
        ]
        if configured:
            return configured
        count = int(getattr(cfg, "hybrid_training_prewarm_count", 0) or 0)
        if count > 0:
            return candidates[:count]
        return candidates

    def _prewarm_snapshot_path(self, worker_id: str, target_core_id: str) -> str:
        if self.train_engine is not None and hasattr(
            self.train_engine,
            "capture_elastic_state_snapshot",
        ):
            version = int(
                self.train_engine.capture_elastic_state_snapshot(
                    worker_id,
                    target_core_id,
                )
            )
            return self._snapshot_path_for_version(version)
        version = int(getattr(self.train_engine, "current_version", 0) or 0)
        return self._snapshot_path_for_version(version)

    def _is_hybrid_worker_running(self, worker_id: str) -> bool:
        proc = self._hybrid_worker_processes.get(worker_id)
        if proc is None:
            return worker_id in self._hybrid_worker_meta
        return proc.poll() is None

    def _launch_prewarmed_hybrid_worker(
        self,
        *,
        worker_id: str,
        target_core_id: str,
        command: str,
        snapshot_path: str,
    ) -> ManagedHybridWorkerProcess:
        with self._hybrid_resource_guard():
            env = os.environ.copy()
            proc = subprocess.Popen(command, shell=True, env=env, start_new_session=True)
            meta = ManagedHybridWorkerProcess(
                worker_id=worker_id,
                target_core_id=target_core_id,
                command=command,
                pid=proc.pid,
                snapshot_path=snapshot_path,
            )
            self._hybrid_worker_processes[worker_id] = proc
            self._hybrid_worker_meta[worker_id] = meta
            return meta

    def _adopt_prewarmed_hybrid_worker(
        self,
        *,
        worker_id: str,
        target_core_id: str,
        handle: Any,
        result: RuntimeReconfigurationResult,
    ) -> bool:
        if not self._hybrid_training_prewarm_enabled:
            return False
        if not self._is_hybrid_worker_running(worker_id):
            return False
        worker = handle.result(timeout=None)
        with self._hybrid_resource_guard(handle):
            meta = self._hybrid_worker_meta.get(worker_id)
            if meta is None:
                meta = ManagedHybridWorkerProcess(
                    worker_id=worker_id,
                    target_core_id=target_core_id,
                    command="prewarmed_external_worker",
                    pid=-1,
                    snapshot_path=self._snapshot_path_for_version(worker.state_version),
                    host=str(self._cluster_swap_training_slot(worker_id).get("host", "")),
                    gpus=list(self._cluster_swap_training_slot(worker_id).get("gpus", [])),
                )
            else:
                meta.target_core_id = target_core_id
                meta.snapshot_path = self._snapshot_path_for_version(worker.state_version)
            self._hybrid_worker_meta[worker_id] = meta
            result.started_hybrid_workers.append(meta)
            return True

    def _stop_hybrid_worker_process(
        self,
        worker_id: str,
        result: RuntimeReconfigurationResult | None = None,
    ) -> None:
        with self._lifecycle_lock:
            self._stop_hybrid_worker_process_locked(worker_id, result)

    def _stop_hybrid_worker_process_locked(
        self, worker_id: str, result: RuntimeReconfigurationResult | None = None,
    ) -> None:
        proc = self._hybrid_worker_processes.get(worker_id)
        meta = self._hybrid_worker_meta.get(worker_id)
        if proc is None:
            if meta is not None and self._hybrid_worker_remote_control_enabled:
                from RL_Framework.infra.elastic.rollout_lending import wait_hybrid_stopped
                control_dir = self._hybrid_worker_control_dir()
                control_dir.mkdir(parents=True, exist_ok=True)
                launch_id = meta.worker_id
                replica = meta.replica_id
                if replica and replica in self._hybrid_worker_meta:
                    candidate = self._hybrid_worker_meta[replica]
                    if candidate.worker_id == replica:
                        launch_id, meta = replica, candidate
                (control_dir / f".launch_{launch_id}.json").unlink(missing_ok=True)
                request_id = getattr(meta, "stop_request_id", None) or uuid.uuid4().hex
                meta.stop_request_id = request_id
                outstanding = False
                for suffix in ("json", "running", "done"):
                    try:
                        record = json.loads((control_dir / f".stop_{launch_id}.{suffix}").read_text(encoding="utf-8"))
                        outstanding |= record.get("request_id") == request_id
                    except FileNotFoundError:
                        pass
                if not outstanding:
                    self._write_json_atomic(control_dir / f".stop_{launch_id}.json", {
                        "worker_id": launch_id, "host": meta.host, "pid": meta.pid,
                        "request_id": request_id,
                    })
                acknowledgement = wait_hybrid_stopped(self, launch_id, request_id)
                if acknowledgement.get("request_id") != request_id or acknowledgement.get("exited") is not True:
                    raise RuntimeError("Hybrid stop did not acknowledge the owned request")
                # Retain every ownership record until the controller proves the
                # entire recorded worker process group exited.
                for alias, owned in list(self._hybrid_worker_meta.items()):
                    if owned is meta or (owned.worker_id == launch_id and owned.pid == meta.pid) or (
                        owned.replica_id == launch_id and owned.host == meta.host
                    ):
                        self._hybrid_worker_meta.pop(alias, None)
            elif meta is not None:
                raise RuntimeError(f"Hybrid worker {worker_id} has metadata but no owned process or remote controller")
            if result is not None and meta is not None:
                result.training_actions.append(f"stop_hybrid_worker_meta:{worker_id}")
            return
        from RL_Framework.infra.elastic.process_lifecycle import stop_local_worker
        stop_local_worker(proc)
        for alias, owned in list(self._hybrid_worker_processes.items()):
            if owned is proc:
                self._hybrid_worker_processes.pop(alias, None)
                self._hybrid_worker_meta.pop(alias, None)
        if result is not None:
            result.training_actions.append(f"stop_hybrid_worker:{worker_id}")

    def _ensure_gradient_server(self):
        with self._hybrid_resource_guard():
            self._initialize_gradient_server()

    def _initialize_gradient_server(self):
        if self.gradient_server is not None:
            return
        if self.train_engine is None:
            return
        cfg = self.config.global_resource_planner
        on_payload = getattr(
            self.train_engine,
            "enqueue_hybrid_gradient_payload",
            lambda payload: None,
        )
        backend = getattr(cfg, "gradient_transport_backend", "tcp")
        if backend == "native_rdma":
            from RL_Framework.infra.elastic.native_rdma import (
                NativeRDMAConfig,
                NativeRDMAGradientServer,
            )

            self.gradient_server = NativeRDMAGradientServer(
                host=getattr(cfg, "gradient_server_host", "127.0.0.1"),
                port=int(getattr(cfg, "gradient_server_port", 0)),
                authkey=getattr(cfg, "gradient_server_authkey", ""),
                on_payload=on_payload,
                rdma_config=NativeRDMAConfig(
                    device=getattr(cfg, "native_rdma_device", "mlx5_0"),
                    gid_index=int(getattr(cfg, "native_rdma_gid_index", 0)),
                    ib_port=int(getattr(cfg, "native_rdma_ib_port", 1)),
                    max_bytes=int(getattr(cfg, "native_rdma_max_bytes", 67108864)),
                ),
                work_dir=getattr(cfg, "hybrid_worker_task_dir", "./logs/elastic_training_tasks"),
            )
        else:
            from RL_Framework.infra.elastic.gradient_ipc import ElasticGradientServer

            self.gradient_server = ElasticGradientServer(
                host=getattr(cfg, "gradient_server_host", "127.0.0.1"),
                port=int(getattr(cfg, "gradient_server_port", 0)),
                authkey=getattr(cfg, "gradient_server_authkey", ""),
                on_payload=on_payload,
            )
        self.gradient_server.start()

    def _attach_train_engine_gradient_domain(self):
        if self.train_engine is None or self.elastic_pool is None:
            return
        if hasattr(self.train_engine, "set_elastic_gradient_domain"):
            self.train_engine.set_elastic_gradient_domain(
                self.elastic_pool.gradient_domain
            )

    def _launch_hybrid_worker_after_join(
        self,
        handle: Any,
        result: RuntimeReconfigurationResult | None = None,
    ) -> ManagedHybridWorkerProcess | None:
        try:
            worker = handle.result(timeout=None)
            snapshot_path = self._snapshot_path_for_version(worker.state_version)
            slot = self._cluster_swap_training_slot(worker.worker_id)
            command = self._build_hybrid_worker_command(
                worker_id=worker.worker_id,
                target_core_id=handle.target_core_id,
                snapshot_path=snapshot_path,
                host=str(slot.get("host", "")),
                gpus=[int(gpu) for gpu in slot.get("gpus", [])],
            )
            with self._hybrid_resource_guard(handle):
                env = os.environ.copy()
                proc = subprocess.Popen(command, shell=True, env=env, start_new_session=True)
                meta = ManagedHybridWorkerProcess(
                    worker_id=worker.worker_id,
                    target_core_id=handle.target_core_id,
                    command=command,
                    pid=proc.pid,
                    snapshot_path=snapshot_path,
                    host=str(slot.get("host", "")),
                    gpus=[int(gpu) for gpu in slot.get("gpus", [])],
                )
                self._hybrid_worker_processes[worker.worker_id] = proc
                self._hybrid_worker_meta[worker.worker_id] = meta
                if result is not None:
                    result.started_hybrid_workers.append(meta)
                return meta
        except Exception as exc:
            if result is not None:
                result.errors.append(f"hybrid_worker_launch_failed:{exc}")
            return None

    def _wait_hybrid_worker_ready(
        self, worker_id: str, *, state_version: int | None = None,
        replica_world_size: int | None = None, snapshot_path: str | None = None,
        target_core_id: str | None = None,
    ) -> dict[str, Any]:
        from RL_Framework.infra.elastic.readiness import read_ready_receipts, snapshot_contract
        task_dir = Path(getattr(self.config.global_resource_planner, "hybrid_worker_task_dir", "./logs/elastic_training_tasks"))
        meta = self._hybrid_worker_meta.get(worker_id)
        if meta is None:
            meta = next((value for value in self._hybrid_worker_meta.values() if value.replica_id == worker_id), None)
        if snapshot_path is None and meta is not None:
            snapshot_path = meta.snapshot_path
        if target_core_id is None and meta is not None:
            target_core_id = meta.target_core_id
        width = int(replica_world_size if replica_world_size is not None else getattr(meta, "replica_world_size", 1))
        if width <= 0:
            raise ValueError("Hybrid readiness requires positive model-lane coverage")
        contract = snapshot_contract(snapshot_path, state_version=state_version, target_core_id=target_core_id)
        control_dir = self._hybrid_worker_control_dir()
        error_path = control_dir / f".launch_{worker_id}.error"
        started_path = control_dir / f".launch_{worker_id}.started"
        timeout = float(getattr(self.config.global_resource_planner, "hybrid_worker_ready_timeout_s", 60.))
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while True:
            if self._closing.is_set():
                from RL_Framework.infra.elastic.hybrid_pool import JoinCancelledError
                raise JoinCancelledError(f"Hybrid readiness cancelled during cleanup: {worker_id}")
            with self._hybrid_resource_guard():
                if error_path.exists():
                    raise RuntimeError(f"remote hybrid worker {worker_id} launch failed: {error_path.read_text(encoding='utf-8')[-1000:]}")
                proc = self._hybrid_worker_processes.get(worker_id)
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(f"hybrid worker {worker_id} exited before ready")
                receipt = read_ready_receipts(task_dir, worker_id, width, contract)
                if receipt is not None:
                    if meta is not None and started_path.exists():
                        started = json.loads(started_path.read_text(encoding="utf-8"))
                        if started.get("worker_id", worker_id) != worker_id:
                            raise RuntimeError("Hybrid launch controller returned a different worker identity")
                        meta.pid = int(started["pid"])
                    if meta is not None:
                        meta.snapshot_path = str(snapshot_path)
                    return receipt
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"hybrid worker {worker_id} did not acknowledge snapshot {contract['snapshot_id']} revision {contract['optimizer_revision']} on all {width} lanes after {timeout:.0f}s")
            self._closing.wait(.05)

    def _snapshot_path_for_version(self, version: int) -> str:
        if int(version) in self._snapshot_manifests:
            return self._snapshot_manifests[int(version)]
        if self.train_engine is not None and hasattr(
            self.train_engine,
            "get_elastic_state_snapshot_path",
        ):
            return str(self.train_engine.get_elastic_state_snapshot_path(version))
        state_dir = os.getenv(
            "ELASTIC_TRAINING_STATE_DIR",
            "./logs/elastic_training_state",
        )
        rank = int(getattr(self.train_engine, "rank", 0) if self.train_engine else 0)
        return str(Path(state_dir) / f"v{version}" / f"rank_{rank}.pt")

    def _build_hybrid_worker_command(
        self,
        *,
        worker_id: str,
        target_core_id: str,
        snapshot_path: str,
        idle: bool = False,
        host: str = "",
        gpus: list[int] | None = None,
        replica_id: str = "",
        replica_rank: int = 0,
        replica_world_size: int = 1,
    ) -> str:
        cfg = self.config.global_resource_planner
        if self._hybrid_worker_launch_enabled and hasattr(self.config, "to_yaml"):
            config_path = self._coordination_dir() / f"hybrid-config-{self._active_runtime_coord_id}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = config_path.with_name(f".{config_path.name}.{uuid.uuid4().hex}.tmp")
            self.config.to_yaml(str(temporary))
            temporary.replace(config_path)
            cfg.hybrid_worker_config_path = str(config_path)
        endpoint = self.gradient_server.endpoint
        gradient_host = getattr(cfg, "gradient_server_public_host", "") or endpoint.host
        rl_root = Path(__file__).resolve().parents[2]
        template = getattr(cfg, "hybrid_worker_command_template", "")
        assigned_gpus = [int(gpu) for gpu in (gpus or [])]
        cuda_visible_devices = ",".join(str(gpu) for gpu in assigned_gpus)
        values = {
            "python": shlex.quote(getattr(cfg, "hybrid_worker_python", sys.executable)),
            "rl_framework_path": shlex.quote(str(rl_root)),
            "gradient_transport_backend": shlex.quote(getattr(cfg, "gradient_transport_backend", "tcp")),
            "worker_id": shlex.quote(worker_id),
            "replica_id": shlex.quote(replica_id or worker_id),
            "replica_rank": int(replica_rank),
            "replica_world_size": int(replica_world_size),
            "target_core_id": shlex.quote(target_core_id),
            "snapshot_path": shlex.quote(snapshot_path),
            "host": shlex.quote(host),
            "gpus": cuda_visible_devices,
            "cuda_visible_devices": shlex.quote(cuda_visible_devices),
            "worker_mode": shlex.quote(getattr(cfg, "hybrid_worker_mode", "megatron_core")),
            "worker_config": shlex.quote(
                getattr(cfg, "hybrid_worker_config_path", "")
                or os.environ.get("RUNTIME_CONFIG", os.environ.get("CONFIG_PATH", ""))
            ),
            "gradient_endpoint_dir": shlex.quote(getattr(cfg, "hybrid_worker_endpoint_dir", getattr(cfg, "hybrid_worker_task_dir", "")) + "/core_endpoints"),
            "replica_gpus": len(assigned_gpus) or int(replica_world_size),
            "gradient_host": shlex.quote(gradient_host),
            "gradient_port": endpoint.port,
            "authkey": shlex.quote(endpoint.authkey),
            "task_dir": shlex.quote(getattr(cfg, "hybrid_worker_task_dir", "")),
            "idle_flag": "--idle" if idle else "",
            "native_rdma_device": shlex.quote(getattr(cfg, "native_rdma_device", "mlx5_0")),
            "native_rdma_gid_index": int(getattr(cfg, "native_rdma_gid_index", 0)),
            "native_rdma_ib_port": int(getattr(cfg, "native_rdma_ib_port", 1)),
            "native_rdma_max_bytes": int(getattr(cfg, "native_rdma_max_bytes", 67108864)),
        }
        if template:
            return template.format(**values)
        env_prefix = ""
        if cuda_visible_devices:
            env_prefix = f"CUDA_VISIBLE_DEVICES={values['cuda_visible_devices']} "
        worker_script = f"{values['rl_framework_path']}/scripts/elastic_hybrid_worker.py"
        launcher = ""
        if getattr(cfg, "hybrid_worker_mode", "megatron_core") == "megatron_core" and values["replica_gpus"] > 1:
            launcher = f"{values['python']} -m torch.distributed.run --standalone --nproc_per_node={values['replica_gpus']} "
        else:
            launcher = f"{values['python']} "
        return (
            f"{env_prefix}{launcher}{worker_script} "
            f"--worker-id {values['worker_id']} "
            f"--replica-id {values['replica_id']} "
            f"--replica-rank {values['replica_rank']} "
            f"--replica-world-size {values['replica_world_size']} "
            f"--target-core-id {values['target_core_id']} "
            f"--snapshot-path {values['snapshot_path']} "
            f"--gradient-host {values['gradient_host']} "
            f"--gradient-port {values['gradient_port']} "
            f"--worker-mode {values['worker_mode']} "
            f"--config {values['worker_config']} "
            f"--gradient-endpoint-dir {values['gradient_endpoint_dir']} "
            f"--authkey {values['authkey']} "
            f"--task-dir {values['task_dir']} "
            f"--gradient-transport {values['gradient_transport_backend']} "
            f"{values['idle_flag']} "
            f"--native-rdma-device {values['native_rdma_device']} "
            f"--native-rdma-gid-index {values['native_rdma_gid_index']} "
            f"--native-rdma-ib-port {values['native_rdma_ib_port']} "
            f"--native-rdma-max-bytes {values['native_rdma_max_bytes']}"
        )

    @staticmethod
    def _role_name(role: Any) -> str:
        return str(getattr(role, "value", role))

    def _stop_managed_rollout_processes(self) -> list[ManagedRolloutProcess]:
        stopped: list[ManagedRolloutProcess] = []
        for instance_id in list(self._process_meta):
            stopped.extend(self._stop_rollout_process(instance_id))
        return stopped

    def _adopt_existing_rollout_processes(self) -> None:
        """Track already configured rollout endpoints as externally managed.

        Slurm launchers often start vLLM outside the Python trainer.  Adopting
        those endpoints lets the diff planner keep unchanged slots without ever
        sending signals to processes it does not own.
        """
        cfgs = getattr(self.rollout_engine, "instance_configs", None)
        if not cfgs or self._process_meta:
            return
        if not getattr(
            self.config.global_resource_planner,
            "runtime_adopt_existing_rollout_processes",
            True,
        ):
            return
        registry = self._load_rollout_pid_registry()
        for idx, cfg in enumerate(cfgs):
            instance_id = str(cfg.get("instance_id") or f"adopted_{idx}")
            host = str(cfg.get("host") or self.config.heterogeneous_rollout.vllm_host)
            if host == "0.0.0.0":
                host = "127.0.0.1"
            registry_meta = registry.get(instance_id, {})
            meta = ManagedRolloutProcess(
                instance_id=instance_id,
                command="",
                pid=int(registry_meta.get("pid", -1)),
                gpus=list(registry_meta.get("gpus") or cfg.get("gpu_ids") or []),
                port=int(
                    registry_meta.get(
                        "port",
                        cfg.get(
                            "port",
                            self.config.heterogeneous_rollout.vllm_base_port + idx,
                        ),
                    )
                ),
                host=str(registry_meta.get("host") or host),
                tp=int(registry_meta.get("tp", cfg.get("tp_degree", 1))),
                adopted=True,
                log_path=str(registry_meta.get("log_path") or ""),
            )
            self._process_meta[instance_id] = meta

    def _load_rollout_pid_registry(self) -> dict[str, dict[str, Any]]:
        path = (
            getattr(
                self.config.global_resource_planner,
                "runtime_rollout_pid_registry_path",
                "",
            )
            or os.getenv("GRP_ROLLOUT_PID_REGISTRY_PATH", "")
        )
        if not path:
            return {}
        registry_path = Path(path)
        if not registry_path.exists():
            return {}
        registry: dict[str, dict[str, Any]] = {}
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = str(payload.get("instance_id") or "")
            if instance_id:
                registry[instance_id] = payload
        return registry

    def _reconfigure_rollout_processes(
        self,
        plan: GlobalResourcePlan,
        *,
        strategy_override: str | None = None,
    ) -> tuple[list[ManagedRolloutProcess], list[ManagedRolloutProcess]]:
        self._rollout_engine_reconfigured = False
        strategy = strategy_override or getattr(
            self.config.global_resource_planner,
            "runtime_rollout_reconfigure_strategy",
            "diff",
        )
        if strategy == "restart_all":
            stopped = self._stop_managed_rollout_processes()
            self._sleep_after_rollout_stop(stopped)
            started = self._start_rollout_processes(plan)
            self._reconfigure_rollout_engine(plan, wait_ready=True)
            self._rollout_engine_reconfigured = True
            return stopped, started
        if strategy == "blue_green":
            return self._blue_green_reconfigure_rollout_processes(plan)
        if strategy == "cluster_swap":
            return self._cluster_swap_reconfigure_rollout_processes(plan)

        # Keep diff semantics, but use a blue-green ordering.  Async rollout
        # requests may still be in flight even after dispatcher drain, so
        # stopping old vLLM endpoints before the logical cutover can disconnect
        # requests that already selected an old endpoint.
        return self._blue_green_reconfigure_rollout_processes(plan)

    def _cluster_swap_reconfigure_rollout_processes(
        self,
        plan: GlobalResourcePlan,
    ) -> tuple[list[ManagedRolloutProcess], list[ManagedRolloutProcess]]:
        """Reconfigure rollout using only GPUs handed between live clusters."""

        desired = {
            meta.instance_id: meta
            for meta in self._desired_rollout_processes(plan)
        }
        stopped: list[ManagedRolloutProcess] = []
        started: list[ManagedRolloutProcess] = []
        satisfied_current: set[str] = set()
        satisfied_desired: set[str] = set()

        for instance_id, current in list(self._process_meta.items()):
            wanted = desired.get(instance_id)
            if wanted is not None and self._same_rollout_process(current, wanted):
                satisfied_current.add(instance_id)
                satisfied_desired.add(instance_id)
                continue

        for wanted_id, wanted in desired.items():
            if wanted_id in satisfied_desired:
                continue
            match_id = next(
                (
                    current_id
                    for current_id, current in self._process_meta.items()
                    if current_id not in satisfied_current
                    and self._same_rollout_process(current, wanted)
                ),
                None,
            )
            if match_id is not None:
                self._adopt_rollout_process_identity(match_id, wanted)
                satisfied_current.add(wanted_id)
                satisfied_desired.add(wanted_id)

        for instance_id, current in list(self._process_meta.items()):
            if instance_id in satisfied_desired:
                continue
            wanted = desired.get(instance_id)
            if wanted is not None and self._same_rollout_process(current, wanted):
                continue
            stopped.extend(self._stop_rollout_process(instance_id))

        self._sleep_after_rollout_stop(stopped)

        for instance_id, wanted in desired.items():
            current = self._process_meta.get(instance_id)
            if current is not None and self._same_rollout_process(current, wanted):
                continue
            started_meta = self._start_one_rollout_process(wanted)
            started.append(started_meta)
            self._wait_rollout_process_start_ready(started_meta)

        self._reconfigure_rollout_engine(plan, wait_ready=bool(started))
        self._rollout_engine_reconfigured = True
        return stopped, started

    def _adopt_rollout_process_identity(
        self,
        old_instance_id: str,
        wanted: ManagedRolloutProcess,
    ) -> ManagedRolloutProcess:
        current = self._process_meta.pop(old_instance_id)
        proc = self._processes.pop(old_instance_id, None)
        adopted = ManagedRolloutProcess(
            instance_id=wanted.instance_id,
            command=wanted.command or current.command,
            pid=current.pid,
            gpus=list(current.gpus),
            port=current.port,
            host=current.host,
            tp=current.tp,
            adopted=current.adopted,
            log_path=current.log_path or wanted.log_path,
        )
        self._process_meta[wanted.instance_id] = adopted
        if proc is not None:
            self._processes[wanted.instance_id] = proc
        return adopted

    def _prewarm_rollout_processes(
        self,
        plan: GlobalResourcePlan,
    ) -> tuple[list[ManagedRolloutProcess], list[ManagedRolloutProcess]]:
        """Start the target rollout pool before pausing traffic.

        New or changed instances are launched on spare ports so they can load
        weights and pass health checks while existing endpoints keep serving.
        The actual traffic cutover is finished later by
        ``_finish_prewarmed_rollout_cutover`` inside the short paused window.
        """

        desired = [
            meta for meta in self._desired_rollout_processes(plan)
        ]
        current = dict(self._process_meta)
        current_ports = {meta.port for meta in current.values()}
        desired_ports = {meta.port for meta in desired}
        next_spare_port = max(
            current_ports
            | desired_ports
            | {int(self.config.heterogeneous_rollout.vllm_base_port)}
        ) + 1
        started: list[ManagedRolloutProcess] = []
        to_stop: list[str] = []

        for instance_id, meta in current.items():
            wanted = next(
                (item for item in desired if item.instance_id == instance_id),
                None,
            )
            if wanted is None or not self._same_rollout_process(meta, wanted):
                to_stop.append(instance_id)

        for wanted in desired:
            existing = current.get(wanted.instance_id)
            if existing is not None and self._same_rollout_process(existing, wanted):
                continue
            wanted.port = next_spare_port
            next_spare_port += 1
            self._set_config_instance_port(wanted.instance_id, wanted.port)
            wanted.command = self._command_for_instance_id(
                wanted.instance_id,
                wanted.port,
            )
            started_meta = self._start_one_rollout_process(wanted)
            self._wait_rollout_process_start_ready(started_meta)
            started.append(started_meta)

        self._prewarmed_rollout_to_stop = to_stop
        return [], started

    def _finish_prewarmed_rollout_cutover(
        self,
        plan: GlobalResourcePlan,
    ) -> tuple[list[ManagedRolloutProcess], list[ManagedRolloutProcess]]:
        stopped: list[ManagedRolloutProcess] = []
        self._reconfigure_rollout_engine(plan, wait_ready=True)
        self._rollout_engine_reconfigured = True
        to_stop = list(getattr(self, "_prewarmed_rollout_to_stop", []))
        if to_stop or getattr(self, "_blue_green_old_processes", []):
            self._sleep_before_rollout_stop()
        for instance_id in to_stop:
            stopped.extend(self._stop_rollout_process(instance_id))
        for old_meta, old_proc in list(
            getattr(self, "_blue_green_old_processes", [])
        ):
            stopped.extend(self._stop_process_meta(old_meta, old_proc))
        self._blue_green_old_processes = []
        self._prewarmed_rollout_to_stop = []
        return stopped, []

    def _blue_green_reconfigure_rollout_processes(
        self,
        plan: GlobalResourcePlan,
    ) -> tuple[list[ManagedRolloutProcess], list[ManagedRolloutProcess]]:
        """Hot-swap rollout processes by starting replacements before cutover."""

        desired = {
            meta.instance_id: meta
            for meta in self._desired_rollout_processes(plan)
        }
        stopped: list[ManagedRolloutProcess] = []
        started: list[ManagedRolloutProcess] = []
        to_stop: list[str] = []

        used_ports = {meta.port for meta in self._process_meta.values()}
        next_spare_port = max(
            used_ports | {int(self.config.heterogeneous_rollout.vllm_base_port)}
        ) + 1
        used_ports_by_instance = {
            meta.port: instance_id
            for instance_id, meta in self._process_meta.items()
        }

        for instance_id, current in list(self._process_meta.items()):
            wanted = desired.get(instance_id)
            if wanted is None:
                to_stop.append(instance_id)
                continue
            if self._same_rollout_process(current, wanted):
                continue
            if wanted.port in used_ports:
                wanted.port = next_spare_port
                next_spare_port += 1
                self._set_config_instance_port(instance_id, wanted.port)
                wanted.command = self._command_for_instance_id(instance_id, wanted.port)
            to_stop.append(instance_id)

        for instance_id, wanted in desired.items():
            current = self._process_meta.get(instance_id)
            if current is not None and self._same_rollout_process(current, wanted):
                continue
            owner = used_ports_by_instance.get(wanted.port)
            if owner is not None and owner != instance_id:
                wanted.port = next_spare_port
                next_spare_port += 1
                self._set_config_instance_port(instance_id, wanted.port)
                wanted.command = self._command_for_instance_id(instance_id, wanted.port)
                used_ports.add(wanted.port)
            started.append(self._start_one_rollout_process(wanted))
            self._wait_rollout_process_start_ready(wanted)

        self._reconfigure_rollout_engine(plan, wait_ready=True)
        self._rollout_engine_reconfigured = True
        if to_stop or getattr(self, "_blue_green_old_processes", []):
            self._sleep_before_rollout_stop()

        for instance_id in to_stop:
            current = self._process_meta.get(instance_id)
            if current is not None and any(p.instance_id == instance_id for p in started):
                # _start_one_rollout_process replaces metadata with the new process.
                # Stop only the old adopted/external process through its saved handle.
                continue
            stopped.extend(self._stop_rollout_process(instance_id))

        # Stop old same-id processes that were replaced before metadata changed.
        for old_meta, old_proc in list(
            getattr(self, "_blue_green_old_processes", [])
        ):
            stopped.extend(self._stop_process_meta(old_meta, old_proc))
        self._blue_green_old_processes = []
        return stopped, started

    def _start_rollout_processes(
        self,
        plan: GlobalResourcePlan,
    ) -> list[ManagedRolloutProcess]:
        started: list[ManagedRolloutProcess] = []
        for meta in self._desired_rollout_processes(plan):
            started_meta = self._start_one_rollout_process(meta)
            started.append(started_meta)
        for started_meta in started:
            self._wait_rollout_process_start_ready(started_meta)
        return started

    def _desired_rollout_processes(
        self,
        plan: GlobalResourcePlan,
    ) -> list[ManagedRolloutProcess]:
        del plan
        hetero = self.config.heterogeneous_rollout
        base_port = int(hetero.vllm_base_port)
        desired: list[ManagedRolloutProcess] = []
        for idx, inst in enumerate(hetero.instances):
            port = base_port + idx
            if getattr(inst, "port", 0):
                port = int(inst.port)
            host = inst.host or hetero.vllm_host
            if host == "0.0.0.0":
                host = "127.0.0.1"
            desired.append(ManagedRolloutProcess(
                instance_id=inst.instance_id or f"grp_tp{inst.tp}_{idx}",
                command=self._build_vllm_command(inst, port),
                pid=0,
                gpus=list(inst.gpus),
                port=port,
                host=host,
                tp=int(inst.tp),
                log_path=self._rollout_log_path(inst.instance_id or f"grp_tp{inst.tp}_{idx}"),
            ))
        return desired

    def _start_one_rollout_process(
        self,
        meta: ManagedRolloutProcess,
    ) -> ManagedRolloutProcess:
        old = self._process_meta.get(meta.instance_id)
        if old is not None and not self._same_rollout_process(old, meta):
            old_list = getattr(self, "_blue_green_old_processes", [])
            old_list.append((old, self._processes.get(meta.instance_id)))
            self._blue_green_old_processes = old_list
        env = self._rollout_launch_env(meta)
        stdout = None
        log_handle = None
        if meta.log_path:
            Path(meta.log_path).parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(meta.log_path, "a", encoding="utf-8")
            log_handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"starting {meta.instance_id} host={meta.host} port={meta.port} "
                f"tp={meta.tp} gpus={meta.gpus}\n{meta.command}\n"
            )
            log_handle.flush()
            stdout = log_handle
        try:
            from RL_Framework.infra.elastic.rollout_process import launch_rollout_process
            proc = launch_rollout_process(
                meta, env,
                stdout=stdout,
                control_dir=self.config.rollout_weight_sync_control_dir,
            )
        finally:
            if log_handle is not None:
                log_handle.close()
        started = ManagedRolloutProcess(
            instance_id=meta.instance_id,
            command=meta.command,
            pid=proc.pid,
            gpus=list(meta.gpus),
            port=meta.port,
            host=meta.host,
            tp=meta.tp,
            adopted=False,
            log_path=meta.log_path,
        )
        self._processes[started.instance_id] = proc
        self._process_meta[started.instance_id] = started
        return started

    def _rollout_launch_env(self, meta: ManagedRolloutProcess) -> dict[str, str]:
        env = os.environ.copy()
        for key in self._TRAIN_DISTRIBUTED_ENV_KEYS:
            env.pop(key, None)
        for key in tuple(env):
            if key.startswith('TORCHELASTIC_'):
                env.pop(key)
        devices = ",".join(str(gpu) for gpu in meta.gpus)
        env["PLANNED_CUDA_VISIBLE_DEVICES"] = devices
        env["CUDA_VISIBLE_DEVICES"] = devices
        env["RL_FRAMEWORK_ROLE"] = "runtime_rollout"
        env["LIBRA_INSTANCE_ID"] = meta.instance_id
        env["LIBRA_SERVICE_LABEL"] = meta.instance_id
        env["LIBRA_EXPECTED_TP_SIZE"] = str(meta.tp)
        env.setdefault("LIBRA_CHECK_PREFIX_CACHE", "1")
        env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
        backend = str(getattr(self.config.heterogeneous_rollout.scheduling, "cmlfq_kv_backend", "recompute"))
        if backend == "cpu_offload":
            directory = str(Path(self.config.rollout_weight_sync_control_dir) / "kv-store")
            env["LIBRA_KV_STORAGE_PATH"] = os.environ.get("LIBRA_KV_STORAGE_PATH", directory)
        return env

    def _stop_rollout_process(self, instance_id: str) -> list[ManagedRolloutProcess]:
        meta = self._process_meta.get(instance_id)
        if meta is None:
            return []
        proc = self._processes.get(instance_id)
        self._stop_process_meta(meta, proc)
        # Retain ownership on a failed stop: a retry must still be able to find
        # the live service, and its GPUs cannot yet be lent to training.
        self._processes.pop(instance_id, None)
        self._process_meta.pop(instance_id, None)
        return [meta]

    def _stop_process_meta(
        self,
        meta: ManagedRolloutProcess,
        proc: subprocess.Popen | None,
    ) -> list[ManagedRolloutProcess]:
        if meta.adopted or proc is None:
            template = getattr(self.config.global_resource_planner, "vllm_stop_command_template", "")
            if template:
                self._run_external_stop(meta)
            elif meta.pid > 0:
                self._terminate_external_pid(meta)
            else:
                raise RuntimeError(f"No process ownership or stop command for rollout {meta.instance_id}")
            return [meta]
        from RL_Framework.infra.elastic.rollout_process import stop_rollout_process
        timeout = float(self.config.global_resource_planner.vllm_stop_timeout_s)
        stop_rollout_process(proc, meta, timeout=timeout,
                             control_dir=self.config.rollout_weight_sync_control_dir)
        return [meta]

    @staticmethod
    def _same_rollout_process(
        a: ManagedRolloutProcess,
        b: ManagedRolloutProcess,
    ) -> bool:
        return (
            a.host == b.host
            and a.port == b.port
            and a.tp == b.tp
            and list(a.gpus) == list(b.gpus)
        )

    def _rollout_log_path(self, instance_id: str) -> str:
        log_dir = getattr(
            self.config.global_resource_planner,
            "runtime_rollout_log_dir",
            "",
        )
        if not log_dir:
            log_dir = os.path.join(self.config.log_dir, "runtime_rollout")
        return str(Path(log_dir) / f"{instance_id}.log")

    def _set_config_instance_port(self, instance_id: str, port: int) -> None:
        for inst in self.config.heterogeneous_rollout.instances:
            if (inst.instance_id or "") == instance_id:
                inst.port = int(port)
                return

    def _command_for_instance_id(self, instance_id: str, port: int) -> str:
        for inst in self.config.heterogeneous_rollout.instances:
            if (inst.instance_id or "") == instance_id:
                return self._build_vllm_command(inst, port)
        raise KeyError(instance_id)

    def _reconfigure_rollout_engine(
        self,
        plan: GlobalResourcePlan,
        *,
        wait_ready: bool,
    ) -> None:
        if (
            not getattr(self.config.global_resource_planner, "apply_to_runtime", True)
            or self.rollout_engine is None
            or not hasattr(self.rollout_engine, "reconfigure_from_plan")
        ):
            return
        self.rollout_engine.reconfigure_from_plan(plan, self.config)
        if wait_ready and hasattr(self.rollout_engine, "wait_for_ready"):
            self.rollout_engine.wait_for_ready(
                timeout=getattr(
                    self.config.global_resource_planner,
                    "vllm_ready_timeout_s",
                    300.0,
                )
            )

    def _run_external_stop(self, meta: ManagedRolloutProcess) -> None:
        template = getattr(
            self.config.global_resource_planner,
            "vllm_stop_command_template",
            "",
        )
        if not template:
            return
        values = {
            "instance_id": shlex.quote(meta.instance_id),
            "host": shlex.quote(meta.host),
            "port": meta.port,
            "tp": meta.tp,
            "gpus": ",".join(str(g) for g in meta.gpus),
            "pid": meta.pid,
        }
        command = template.format(**values)
        log_path = meta.log_path or self._rollout_log_path(meta.instance_id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"stopping {meta.instance_id} host={meta.host} port={meta.port} "
                f"tp={meta.tp} gpus={meta.gpus}\n{command}\n"
            )
            log_handle.flush()
            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    check=False,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    timeout=(float(getattr(self.config.global_resource_planner, "vllm_stop_timeout_s", 60.0)) or None),
                )
                log_handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"stop exit_code={completed.returncode}\n"
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"rollout stop failed for {meta.instance_id}: exit_code={completed.returncode}")
            except subprocess.TimeoutExpired as exc:
                log_handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] stop timeout\n"
                )

                raise TimeoutError(f"rollout stop timed out for {meta.instance_id}") from exc

    def _wait_rollout_process_start_ready(self, meta: ManagedRolloutProcess) -> None:
        if not getattr(
            self.config.global_resource_planner,
            "runtime_wait_rollout_process_start_ready",
            False,
        ):
            return
        timeout = float(
            getattr(
                self.config.global_resource_planner,
                "vllm_ready_timeout_s",
                300.0,
            )
        )
        url = f"http://{meta.host}:{meta.port}/health"
        deadline = time.time() + timeout
        log_path = meta.log_path or self._rollout_log_path(meta.instance_id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"waiting for health {url}\n"
            )
            log_handle.flush()
            while time.time() < deadline:
                proc = self._processes.get(meta.instance_id)
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"rollout process {meta.instance_id} exited before ready"
                    )
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as response:
                        if 200 <= int(response.status) < 300:
                            log_handle.write(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"health ready {url}\n"
                            )
                            return
                except Exception:
                    time.sleep(5.0)
            raise TimeoutError(
                f"rollout process {meta.instance_id} not ready after {timeout:.0f}s"
            )

    def _sleep_after_rollout_stop(self, stopped: list[ManagedRolloutProcess]) -> None:
        if not stopped:
            return
        grace_s = float(
            getattr(
                self.config.global_resource_planner,
                "runtime_post_rollout_stop_grace_s",
                0.0,
            )
        )
        if grace_s > 0:
            time.sleep(grace_s)

    def _sleep_before_rollout_stop(self) -> None:
        grace_s = float(
            getattr(
                self.config.global_resource_planner,
                "runtime_rollout_drain_grace_s",
                getattr(
                    self.config.global_resource_planner,
                    "runtime_post_rollout_stop_grace_s",
                    0.0,
                ),
            )
        )
        if grace_s > 0:
            time.sleep(grace_s)

    def _terminate_external_pid(self, meta: ManagedRolloutProcess) -> None:
        from RL_Framework.infra.elastic.rollout_process import stop_adopted_rollout
        stop_adopted_rollout(
            meta, env=self._rollout_launch_env(meta),
            timeout=float(self.config.global_resource_planner.vllm_stop_timeout_s),
            control_dir=self.config.rollout_weight_sync_control_dir,
        )

    def _write_rollout_manifest(
        self,
        plan: GlobalResourcePlan,
        *,
        phase: str,
    ) -> None:
        path = getattr(
            self.config.global_resource_planner,
            "runtime_rollout_manifest_path",
            "",
        )
        if not path:
            return
        payload = {
            "phase": phase,
            "run_id": self.runtime_run_id,
            "coord_id": getattr(self, "_active_runtime_coord_id", ""),
            "updated_at": time.time(),
            "plan": plan.to_dict(),
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "tp": inst.tp,
                    "gpus": list(inst.gpus),
                    "host": inst.host or self.config.heterogeneous_rollout.vllm_host,
                    "port": int(inst.port or (self.config.heterogeneous_rollout.vllm_base_port + idx)),
                }
                for idx, inst in enumerate(self.config.heterogeneous_rollout.instances)
            ],
            "managed": [vars(meta) for meta in self._process_meta.values()],
        }
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(manifest_path)

    def _build_vllm_command(self, inst: Any, port: int) -> str:
        planner_cfg = self.config.global_resource_planner
        template = getattr(planner_cfg, "vllm_launch_command_template", "")
        host = inst.host or self.config.heterogeneous_rollout.vllm_host
        max_model_len = self.config.heterogeneous_rollout.max_model_len or self.config.max_seq_length
        public_host = host if host != "0.0.0.0" else "127.0.0.1"
        module = os.environ.get("VLLM_SERVER_MODULE") or "RL_Framework.scripts.vllm_hot_reload_api_server"
        control = str(getattr(self.config, "rollout_weight_sync_control_dir", "") or
                      Path(self.config.log_dir) / "rollout_weight_sync")
        instance_id = str(inst.instance_id or f"tp{inst.tp}_{port}")
        devices = ",".join(str(gpu) for gpu in inst.gpus)
        values = dict(python=shlex.quote(sys.executable), instance_id=shlex.quote(instance_id),
            model_path=shlex.quote(self.config.model_path), initial_model=shlex.quote(self.config.model_path),
            host=shlex.quote(host), public_host=shlex.quote(public_host), port=port, tp=inst.tp,
            gpus=devices, devices=devices, device_visible_env="CUDA_VISIBLE_DEVICES",
            vllm_module=shlex.quote(module), max_model_len=max_model_len,
            gpu_memory_utilization=self.config.heterogeneous_rollout.gpu_memory_utilization,
            health_url=shlex.quote(f"http://{public_host}:{port}/health"),
            local_health_url=shlex.quote(f"http://127.0.0.1:{port}/health"),
            control_dir=shlex.quote(control), log_path=shlex.quote(self._rollout_log_path(instance_id)))
        if template:
            # Keep explicit site templates, but placement remains the launch
            # helper's responsibility for default local commands.
            return template.format(**values)
        if (len(inst.gpus) != int(inst.tp) or len(set(inst.gpus)) != len(inst.gpus)
                or any(int(gpu) < 0 for gpu in inst.gpus)):
            raise ValueError("Rollout TP workers must own distinct assigned node GPU IDs")
        server = [sys.executable, "-m", module, "--model", "__MODEL_PATH__",
            "--served-model-name", self.config.model_path, "--host", "0.0.0.0", "--port", str(port),
            "--tensor-parallel-size", str(inst.tp), "--max-model-len", str(max_model_len),
            "--dtype", "bfloat16", "--gpu-memory-utilization", str(values["gpu_memory_utilization"]),
            "--max-num-seqs", str(self.config.max_concurrent_rollouts), "--seed", str(self.config.seed),
            "--worker-extension-cls", "RL_Framework.vllm_hot_reload.InplaceReloadWorkerExtension",
            "--enforce-eager", "--trust-remote-code", "--disable-log-requests"]
        backend = str(getattr(self.config.heterogeneous_rollout.scheduling, "cmlfq_kv_backend", "recompute"))
        if backend == "cpu_offload":
            storage = os.environ.get("LIBRA_KV_STORAGE_PATH") or str(Path(control) / "kv-store")
            server += ["--kv-transfer-config", json.dumps(dict(
                kv_connector="LibraCPUOffloadConnector", kv_role="kv_both",
                kv_connector_module_path="RL_Framework.infra.scheduling.vllm_cpu_offload",
                kv_connector_extra_config=dict(storage_path=storage)), separators=(",", ":"))]
        supervisor = [sys.executable, "-m", "RL_Framework.scripts.restartable_vllm_server",
            "--instance-id", instance_id, "--control-dir", control,
            "--health-url", f"http://127.0.0.1:{port}/health", "--initial-model", self.config.model_path,
            "--ready-timeout", str(planner_cfg.vllm_ready_timeout_s),
            "--reload-method", "inplace", "--reload-strategy", "parallel", "--", *server]
        return shlex.join(["env", "CUDA_VISIBLE_DEVICES=" + devices, *supervisor])

    def close(self):
        """Fence late callbacks, then stop every resource already committed.

        This waits for launch/register critical sections, not arbitrary join
        Futures (which can be waiting on external snapshot/readiness work).
        A failed cleanup raises and retains ownership for a subsequent retry.
        """
        # Stop admitting new reservations even while an earlier launch commit
        # owns the gate. That commit must finish registration before we scan.
        self._closing.set()
        with self._lifecycle_lock:
            if self._closed:
                return
            errors = []
            if self.elastic_pool is not None:
                try:
                    self.elastic_pool.close()
                except Exception as exc:
                    errors.append(exc)
            with self._pending_hybrid_join_lock:
                self._pending_hybrid_joins.clear()
                self._pending_hybrid_join_started_at.clear()
            worker_ids = set(self._hybrid_worker_processes) | set(self._hybrid_worker_meta)
            for worker_id in sorted(worker_ids):
                try:
                    self._stop_hybrid_worker_process(worker_id)
                except Exception as exc:
                    errors.append(exc)
            try:
                self._stop_managed_rollout_processes()
            except Exception as exc:
                errors.append(exc)
            if self.gradient_server is not None:
                try:
                    self.gradient_server.close()
                    self.gradient_server = None
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError("runtime shutdown cleanup failed: " + "; ".join(map(str, errors))) from errors[0]
            self._closed = True

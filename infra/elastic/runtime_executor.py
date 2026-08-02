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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
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
    log_path: str = ""


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
    _SLURM_NESTED_STEP_ENV_KEYS = (
        "SLURM_STEP_ID",
        "SLURM_STEPID",
        "SLURM_STEP_NODELIST",
        "SLURM_STEP_NUM_NODES",
        "SLURM_STEP_NUM_TASKS",
        "SLURM_STEP_TASKS_PER_NODE",
        "SLURM_STEP_GPUS",
        "SLURM_STEP_GRES",
        "SLURM_STEP_LAUNCHER_PORT",
        "SLURM_STEP_RESV_PORTS",
        "SLURM_SRUN_COMM_HOST",
        "SLURM_SRUN_COMM_PORT",
        "SLURM_GTIDS",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
    )

    def __init__(
        self,
        *,
        config: AsyncRLConfig,
        planner: GlobalResourcePlanner,
        train_engine: Any | None = None,
        rollout_engine: Any | None = None,
        dispatcher: Any | None = None,
        elastic_pool: "ElasticHybridPool | None" = None,
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
        self.gradient_server = None

    def execute(self, decision: PlannerDecision) -> RuntimeReconfigurationResult:
        if not decision.should_reconfigure or decision.candidate_plan is None:
            return RuntimeReconfigurationResult(
                applied=False,
                reason=decision.reason,
            )

        plan = decision.candidate_plan
        result = RuntimeReconfigurationResult(applied=False, reason=decision.reason)
        planner_cfg = self.config.global_resource_planner
        if not getattr(planner_cfg, "runtime_dynamic_reconfiguration_enabled", True):
            return RuntimeReconfigurationResult(
                applied=False,
                reason="runtime_dynamic_reconfiguration_disabled",
            )
        core_train_gpus = int(self.config.train_gpus)
        current_train_gpus = self._current_effective_train_gpus(core_train_gpus)
        if self._requires_supervised_training_handoff(
            core_train_gpus,
            plan,
        ):
            handoff_path = self._request_training_handoff(
                plan,
                core_train_gpus,
            )
            return RuntimeReconfigurationResult(
                applied=True,
                reason="training_handoff_requested",
                actions=[
                    "training_handoff_requested",
                    f"training_handoff_request:{handoff_path}",
                ],
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
        if getattr(planner_cfg, "runtime_manage_rollout_processes", False):
            self._adopt_existing_rollout_processes()
            if self._runtime_plan_already_applied(
                plan,
                current_train_gpus=current_train_gpus,
            ):
                self._write_rollout_manifest(plan, phase="applied")
                self._write_peer_rank_reconfiguration_state(plan, phase="applied")
                return RuntimeReconfigurationResult(
                    applied=False,
                    reason="runtime_plan_already_applied",
                    actions=["runtime_plan_already_applied"],
                )
        config_applied = False
        prewarmed_rollout = False
        prewarmed_stopped: list[ManagedRolloutProcess] = []
        prewarmed_started: list[ManagedRolloutProcess] = []
        coordinated_peers = False
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
                self._prewarm_hybrid_training_workers(current_train_gpus, plan)
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
                getattr(planner_cfg, "runtime_manage_rollout_processes", False)
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

            if not prewarmed_rollout and not config_applied:
                self._apply_plan_to_runtime_config(plan)
                result.actions.append("apply_config")

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

            self._write_rollout_manifest(plan, phase="applied")
            self._write_peer_rank_reconfiguration_state(plan, phase="applied")

            result.applied = True
            return result
        except Exception as exc:
            result.errors.append(str(exc))
            self._write_peer_rank_reconfiguration_state(plan, phase="aborted", error=str(exc))
            raise
        finally:
            if paused and self.dispatcher is not None and hasattr(self.dispatcher, "resume"):
                self.dispatcher.resume()
                result.actions.append("dispatcher_resume")

    def _requires_supervised_training_handoff(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
    ) -> bool:
        """Return whether a live Megatron topology change needs parent supervision."""

        cfg = self.config.global_resource_planner
        if not getattr(cfg, "runtime_reconfigure_training", False):
            return False
        target = int(
            getattr(cfg, "runtime_training_pool_target_gpus", 0)
            or int(plan.train_gpus)
        )
        if target == int(current_train_gpus):
            return False
        mode = str(
            getattr(cfg, "runtime_training_resize_mode", "hybrid_nonblocking")
        ).lower()
        replica_gpus = self._hybrid_replica_gpus()
        can_use_hybrid = (
            mode == "hybrid_nonblocking"
            and target >= int(current_train_gpus)
            and (target - int(current_train_gpus)) % replica_gpus == 0
        )
        if can_use_hybrid:
            return False
        return bool(getattr(cfg, "runtime_training_handoff_enabled", False))

    def _hybrid_replica_gpus(self) -> int:
        configured = int(
            getattr(
                self.config.global_resource_planner,
                "hybrid_replica_gpus",
                0,
            )
            or 0
        )
        if configured > 0:
            return configured
        return max(
            1,
            int(getattr(self.config, "train_tp_size", 1) or 1)
            * int(getattr(self.config, "train_pp_size", 1) or 1)
            * int(getattr(self.config, "train_cp_size", 1) or 1),
        )

    def _current_effective_train_gpus(self, core_train_gpus: int) -> int:
        if self.elastic_pool is None:
            configured = int(
                getattr(
                    self.config.global_resource_planner,
                    "runtime_effective_train_gpus",
                    0,
                )
                or 0
            )
            return configured or int(core_train_gpus)
        snapshot = self.elastic_pool.snapshot()
        elastic_replicas = sum(
            1
            for worker in snapshot.values()
            if self._role_name(worker.role) in {"hybrid_training", "hybrid_joining"}
        )
        return int(core_train_gpus) + elastic_replicas * self._hybrid_replica_gpus()

    def _training_handoff_dir(self) -> Path:
        configured = str(
            getattr(
                self.config.global_resource_planner,
                "runtime_training_handoff_dir",
                "",
            )
            or ""
        )
        if configured:
            return Path(configured)
        return self._coordination_dir() / "training_handoff"

    def _request_training_handoff(
        self,
        plan: GlobalResourcePlan,
        current_train_gpus: int,
    ) -> Path:
        """Publish a checkpoint-and-restart request for the Slurm supervisor."""

        directory = self._training_handoff_dir()
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        payload = {
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
            "requested_at": time.time(),
            "current_train_gpus": int(current_train_gpus),
            "target_train_gpus": int(plan.train_gpus),
            "resume_step": None,
            "plan": plan.to_dict(),
        }
        tmp_path = request_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(request_path)
        print(
            "[RuntimeElasticExecutor] training_handoff_requested "
            f"path={request_path} current_train_gpus={current_train_gpus} "
            f"target_train_gpus={plan.train_gpus}",
            flush=True,
        )
        return request_path

    def _reconfigure_training_pool(
        self,
        current_train_gpus: int,
        plan: GlobalResourcePlan,
        result: RuntimeReconfigurationResult,
    ) -> list[str]:
        if self.elastic_pool is None:
            return ["training_resize_requested_without_elastic_pool"]

        actions: list[str] = []
        target_train_gpus = int(
            getattr(
                self.config.global_resource_planner,
                "runtime_training_pool_target_gpus",
                0,
            )
            or int(plan.train_gpus)
        )
        core_train_gpus = int(self.config.train_gpus)
        current_train_gpus = self._current_effective_train_gpus(core_train_gpus)
        if target_train_gpus < core_train_gpus:
            raise ValueError(
                "non-blocking elasticity cannot remove immutable core ranks: "
                f"target={target_train_gpus}, core={core_train_gpus}"
            )
        replica_gpus = self._hybrid_replica_gpus()
        delta = target_train_gpus - current_train_gpus
        if delta % replica_gpus != 0:
            raise ValueError(
                "training target must change by complete model-parallel replicas: "
                f"delta_gpus={delta}, replica_gpus={replica_gpus}"
            )
        replica_delta = delta // replica_gpus
        core_ids, target_core, candidates = self._training_reconfiguration_targets(
            current_train_gpus,
            plan,
        )
        snapshot = self.elastic_pool.snapshot()

        if replica_delta > 0:
            for index, worker_id in enumerate(candidates[:replica_delta]):
                target_core = core_ids[index % len(core_ids)] if core_ids else target_core
                handle = self.elastic_pool.join_training(worker_id, target_core)
                if self._adopt_prewarmed_hybrid_worker(
                    worker_id=worker_id,
                    target_core_id=target_core,
                    handle=handle,
                    result=result,
                ):
                    actions.append(
                        f"activate_prewarmed_hybrid_worker:{worker_id}->{target_core}"
                    )
                elif self._cluster_swap_enabled():
                    actions.append(f"cluster_swap_training_slot_ready:{worker_id}")
                elif (
                    self._hybrid_worker_launch_enabled
                    and getattr(self.elastic_pool, "join_preparer", None) is None
                ):
                    thread = threading.Thread(
                        target=self._launch_hybrid_worker_after_join,
                        args=(handle, result),
                        name=f"launch-{worker_id}",
                        daemon=True,
                    )
                    thread.start()
                    self._hybrid_launch_threads.append(thread)
                actions.append(f"join_training:{worker_id}->{target_core}")
        elif replica_delta < 0:
            candidates = [
                wid for wid, worker in snapshot.items()
                if self._role_name(worker.role) in {"hybrid_training", "hybrid_joining"}
            ]
            for worker_id in candidates[: abs(replica_delta)]:
                self.elastic_pool.release_to_rollout(worker_id)
                self._stop_hybrid_worker_process(worker_id, result)
                actions.append(f"release_to_rollout:{worker_id}")
        else:
            actions.append(f"training_gpus_unchanged:{target_train_gpus}")

        self.config.global_resource_planner.runtime_effective_train_gpus = (
            target_train_gpus
        )

        return actions

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
        target_train_gpus = int(
            getattr(planner_cfg, "runtime_training_pool_target_gpus", 0)
            or int(plan.train_gpus)
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

        target_train_gpus = int(
            getattr(planner_cfg, "runtime_training_pool_target_gpus", 0)
            or int(plan.train_gpus)
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
        target_train_gpus = int(
            getattr(
                self.config.global_resource_planner,
                "runtime_training_pool_target_gpus",
                0,
            )
            or int(plan.train_gpus)
        )
        delta = max(0, target_train_gpus - current_train_gpus)
        replica_gpus = self._hybrid_replica_gpus()
        replica_delta = delta // replica_gpus
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
        free_by_host: dict[str, list[int]] = {}
        for meta in stopped_rollout:
            for gpu in meta.gpus:
                if (meta.host, int(gpu)) in desired_rollout_slots:
                    continue
                free_by_host.setdefault(meta.host, []).append(int(gpu))

        slots: list[dict[str, Any]] = []
        for host, gpu_ids in free_by_host.items():
            gpu_ids.sort()
            while len(gpu_ids) >= replica_gpus:
                slots.append({"host": host, "gpus": gpu_ids[:replica_gpus]})
                del gpu_ids[:replica_gpus]

        assigned: dict[str, dict[str, Any]] = {}
        for worker_id, slot in zip(
            candidates[:replica_delta],
            slots[:replica_delta],
        ):
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
        target_train_gpus = int(
            getattr(
                self.config.global_resource_planner,
                "runtime_training_pool_target_gpus",
                0,
            )
            or int(plan.train_gpus)
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
        target_train_gpus = int(
            getattr(planner_cfg, "runtime_training_pool_target_gpus", 0)
            or int(plan.train_gpus)
        )
        delta = target_train_gpus - current_train_gpus
        if delta == 0:
            return [f"plan_training_gpus_unchanged:{target_train_gpus}"]

        current_dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
        core_ids = [f"dp{i}" for i in range(current_dp)]
        target_core = core_ids[0]

        if delta > 0:
            rollout_workers = [
                f"rollout{i}" for i in range(max(0, int(self.config.rollout_gpus)))
            ]
            actions = ["training_pool_plan_only"]
            for worker_id in rollout_workers[:delta]:
                actions.append(f"plan_join_training:{worker_id}->{target_core}")
            if delta > len(rollout_workers):
                actions.append(
                    f"plan_join_training_shortfall:{delta - len(rollout_workers)}"
                )
            return actions

        release_count = abs(delta)
        return [
            "training_pool_plan_only",
            *[
                f"plan_release_to_rollout:hybrid{i}"
                for i in range(release_count)
            ],
        ]

    def _apply_plan_to_runtime_config(self, plan: GlobalResourcePlan) -> None:
        """Apply rollout changes while preserving fixed Megatron topology if requested."""

        planner_cfg = self.config.global_resource_planner
        resize_mode = str(
            getattr(
                planner_cfg,
                "runtime_training_resize_mode",
                "hybrid_nonblocking",
            )
        ).lower()
        preserve_training = bool(
            getattr(planner_cfg, "runtime_reconfigure_training", False)
            and (
                getattr(planner_cfg, "runtime_training_pool_only", True)
                or resize_mode == "hybrid_nonblocking"
            )
        )
        if not preserve_training:
            self.planner.apply_plan_to_config(plan, self.config)
            return

        training_fields = {
            "n_total_gpus": self.config.n_total_gpus,
            "train_gpus": self.config.train_gpus,
            "train_tp_size": self.config.train_tp_size,
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
        return (
            f"rollout_{int(time.time())}_"
            f"{'_'.join(str(tp) for tp in plan.rollout_tp_list)}"
        )

    def _write_peer_rank_reconfiguration_state(
        self,
        plan: GlobalResourcePlan,
        *,
        phase: str,
        coord_id: str | None = None,
        error: str = "",
    ) -> Path:
        coord_dir = self._coordination_dir()
        coord_dir.mkdir(parents=True, exist_ok=True)
        if coord_id is None:
            coord_id = getattr(self, "_active_runtime_coord_id", "")
        payload = {
            "coord_id": coord_id,
            "phase": phase,
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
            "training": self._peer_training_reconfiguration_state(plan),
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
    ) -> dict[str, Any]:
        planner_cfg = self.config.global_resource_planner
        if not getattr(planner_cfg, "runtime_reconfigure_training", False):
            return {"enabled": False}

        current_train_gpus = int(getattr(self.config, "train_gpus", 0) or 0)
        target_train_gpus = int(
            getattr(planner_cfg, "runtime_training_pool_target_gpus", 0)
            or int(plan.train_gpus)
        )
        delta = target_train_gpus - current_train_gpus
        dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
        core_ids = [f"dp{i}" for i in range(dp)]
        target_core = core_ids[0]
        rollout_workers = [
            f"rollout{i}" for i in range(max(0, int(self.config.rollout_gpus)))
        ]
        hybrid_targets: dict[str, str] = {}
        if delta > 0:
            hybrid_targets = {
                worker_id: target_core
                for worker_id in rollout_workers[:delta]
            }
        return {
            "enabled": True,
            "plan_only": bool(
                getattr(planner_cfg, "runtime_training_pool_plan_only", True)
            ),
            "target_train_gpus": target_train_gpus,
            "current_train_gpus": current_train_gpus,
            "core_replica_ids": core_ids,
            "hybrid_targets": hybrid_targets,
            "activate_hybrids": bool(
                not getattr(planner_cfg, "runtime_training_pool_plan_only", True)
            ),
        }

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
        )
        result.actions.append("peer_reconfig_request")

        timeout = float(
            getattr(
                self.config.global_resource_planner,
                "runtime_drain_timeout_s",
                3600.0,
            )
        )
        deadline = time.time() + timeout
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
            ready_job_id = str(payload.get("job_id", ""))
            return not job_id or ready_job_id == job_id

        while time.time() < deadline:
            if all(ready_matches(path) for path in expected):
                result.actions.append(
                    "peer_reconfig_drain:"
                    + ",".join(str(rank) for rank in expected_ranks)
                )
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
        if not getattr(planner_cfg, "runtime_coordinate_batch_source_only", True):
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
        if self.elastic_pool is not None:
            self._ensure_gradient_server()
            self._attach_train_engine_gradient_domain()
            if hasattr(self.train_engine, "set_elastic_step_callback"):
                self.train_engine.set_elastic_step_callback(
                    self.elastic_pool.advance_training_boundary
                )
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
        if not decoupled and hasattr(self.train_engine, "get_elastic_core_process_group"):
            core_group = self.train_engine.get_elastic_core_process_group()
        gradient_domain = getattr(
            self.train_engine,
            "elastic_gradient_domain",
            None,
        ) or InterReplicaGradientDomain(
            core_replica_ids=core_ids,
            communication_domains=GradientCommunicationDomains(
                core_process_group=core_group,
                hybrid_process_group=None,
                decoupled=decoupled,
            ),
        )
        mode = "decoupled" if decoupled else "coupled"

        self.elastic_pool = ElasticHybridPool(
            core_train_workers=core_ids,
            core_rollout_workers=rollout_workers,
            gradient_domain=gradient_domain,
            snapshot_fetcher=fetch_snapshot,
            zero_sync_steps=int(
                getattr(
                    self.config.global_resource_planner,
                    "hybrid_zero_sync_steps",
                    1,
                )
            ),
            require_training_boundary=True,
            join_preparer=(
                self._prepare_hybrid_worker
                if self._hybrid_worker_launch_enabled
                else None
            ),
            state_aligner=(
                self._align_hybrid_worker_state
                if self._hybrid_worker_launch_enabled
                else None
            ),
            state_listener=self._write_hybrid_membership_state,
        )
        print(
            "[RuntimeElasticExecutor] elastic_gradient_domain "
            f"mode={mode} core_replicas={core_ids}",
            flush=True,
        )
        self._attach_train_engine_gradient_domain()
        if hasattr(self.train_engine, "set_elastic_step_callback"):
            self.train_engine.set_elastic_step_callback(
                self.elastic_pool.advance_training_boundary
            )

    def _hybrid_membership_dir(self) -> Path:
        return Path(
            getattr(
                self.config.global_resource_planner,
                "hybrid_worker_task_dir",
                "./logs/elastic_training_tasks",
            )
        ) / "membership"

    def _hybrid_bootstrap_lease_path(self, worker_id: str) -> Path:
        """Return the lease that protects a snapshot during worker bootstrap."""
        return (
            Path(self.config.global_resource_planner.hybrid_worker_task_dir)
            / "bootstrap_leases"
            / f"{worker_id}.json"
        )

    def _write_hybrid_bootstrap_lease(self, worker_id: str, version: int) -> None:
        path = self._hybrid_bootstrap_lease_path(worker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "worker_id": str(worker_id),
                    "snapshot_version": int(version),
                    "created_at": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    def _release_hybrid_bootstrap_lease(self, worker_id: str) -> None:
        self._hybrid_bootstrap_lease_path(worker_id).unlink(missing_ok=True)

    def _write_hybrid_membership_state(self, worker: Any) -> None:
        directory = self._hybrid_membership_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{worker.worker_id}.json"
        payload = {
            "worker_id": str(worker.worker_id),
            "role": self._role_name(worker.role),
            "target_core_id": str(worker.target_core_id or ""),
            "join_state": self._role_name(worker.join_state),
            "state_version": int(worker.state_version),
            "membership_epoch": int(worker.membership_epoch),
            "activate_after_step": worker.activate_after_step,
            "last_error": str(getattr(worker, "last_error", "")),
            "updated_at": float(worker.last_transition_ts),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _prepare_hybrid_worker(
        self,
        worker_id: str,
        target_core_id: str,
        version: int,
    ) -> None:
        """Launch and validate a joining replica while core ranks keep stepping."""
        if self._is_hybrid_worker_running(worker_id):
            return
        task_dir = Path(
            self.config.global_resource_planner.hybrid_worker_task_dir
        )
        (task_dir / f"{worker_id}.ready").unlink(missing_ok=True)
        # Membership state tracks the latest boundary for final alignment.  The
        # worker still needs this immutable bootstrap snapshot until it reports
        # ready, so protect it from the asynchronous checkpoint reaper.
        self._write_hybrid_bootstrap_lease(worker_id, version)
        slot = self._cluster_swap_training_slot(worker_id)
        command = self._build_hybrid_worker_command(
            worker_id=worker_id,
            target_core_id=target_core_id,
            snapshot_path=self._snapshot_path_for_version(version),
            host=str(slot.get("host", "")),
            gpus=[int(gpu) for gpu in slot.get("gpus", [])],
        )
        meta = self._launch_prewarmed_hybrid_worker(
            worker_id=worker_id,
            target_core_id=target_core_id,
            command=command,
            snapshot_path=self._snapshot_path_for_version(version),
        )
        meta.host = str(slot.get("host", ""))
        meta.gpus = [int(gpu) for gpu in slot.get("gpus", [])]
        try:
            self._wait_hybrid_worker_ready(worker_id)
        finally:
            self._release_hybrid_bootstrap_lease(worker_id)

    def _align_hybrid_worker_state(
        self,
        worker_id: str,
        target_core_id: str,
        version: int,
        membership_epoch: int,
    ) -> None:
        """Reload the post-zero-sync state before making a replica active."""
        del target_core_id
        cfg = self.config.global_resource_planner
        timeout = float(getattr(cfg, "hybrid_state_alignment_timeout_s", 300.0))
        snapshot_path = Path(self._snapshot_path_for_version(version))
        deadline = time.time() + max(timeout, 0.0)
        while not snapshot_path.exists() and time.time() < deadline:
            time.sleep(0.1)
        if not snapshot_path.exists():
            raise TimeoutError(
                f"elastic snapshot v{version} was not completed before alignment"
            )
        task_dir = Path(cfg.hybrid_worker_task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / f"{worker_id}.align_v{version}.pt"
        done_path = task_path.with_suffix(task_path.suffix + ".done")
        tmp_path = task_path.with_suffix(task_path.suffix + ".tmp")
        torch.save(
            {
                "reload_snapshot": str(snapshot_path),
                "state_version": int(version),
                "membership_epoch": int(membership_epoch),
            },
            tmp_path,
        )
        tmp_path.replace(task_path)
        while not done_path.exists() and time.time() < deadline:
            proc = self._hybrid_worker_processes.get(worker_id)
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"hybrid worker {worker_id} exited during state alignment"
                )
            time.sleep(0.1)
        if not done_path.exists():
            raise TimeoutError(
                f"hybrid worker {worker_id} did not align state v{version}"
            )

    def dispatch_hybrid_training_batch(
        self,
        *,
        worker_id: str,
        trajectories: list[dict[str, Any]],
        step: int,
    ) -> Path:
        """Publish a step-tagged batch to one active external replica."""
        if self.elastic_pool is None:
            raise RuntimeError("elastic pool is not initialized")
        worker = self.elastic_pool.snapshot()[worker_id]
        if self._role_name(worker.role) != "hybrid_training":
            raise RuntimeError(f"hybrid worker {worker_id} is not active")
        task_dir = Path(self.config.global_resource_planner.hybrid_worker_task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / f"{worker_id}.step_{step}.pt"
        tmp_path = task_path.with_suffix(task_path.suffix + ".tmp")
        torch.save(
            {
                "trajectories": trajectories,
                "step": int(step),
                "state_version": int(step),
                "membership_epoch": int(worker.membership_epoch),
                "snapshot_path": self._snapshot_path_for_version(int(step)),
            },
            tmp_path,
        )
        tmp_path.replace(task_path)
        return task_path

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
    ) -> list[str]:
        if self.elastic_pool is None:
            return []

        target_train_gpus = int(
            getattr(
                self.config.global_resource_planner,
                "runtime_training_pool_target_gpus",
                0,
            )
            or int(plan.train_gpus)
        )
        delta = target_train_gpus - current_train_gpus
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
        env = self._allocation_launch_env("hybrid_training")
        task_dir = Path(
            getattr(
                self.config.global_resource_planner,
                "hybrid_worker_task_dir",
                "./logs/elastic_training_tasks",
            )
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_dir / f"{worker_id}.launch.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                command,
                shell=True,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        meta = ManagedHybridWorkerProcess(
            worker_id=worker_id,
            target_core_id=target_core_id,
            command=command,
            pid=proc.pid,
            snapshot_path=snapshot_path,
            log_path=str(log_path),
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
        # Joining completes on a future core training boundary. Adoption must
        # not wait for that boundary on the rank executing the planner.
        del handle
        meta = self._hybrid_worker_meta.get(worker_id)
        if meta is None:
            version = int(getattr(self.train_engine, "current_version", 0) or 0)
            meta = ManagedHybridWorkerProcess(
                worker_id=worker_id,
                target_core_id=target_core_id,
                command="prewarmed_external_worker",
                pid=-1,
                snapshot_path=self._snapshot_path_for_version(version),
                host=str(self._cluster_swap_training_slot(worker_id).get("host", "")),
                gpus=list(self._cluster_swap_training_slot(worker_id).get("gpus", [])),
            )
        else:
            meta.target_core_id = target_core_id
        self._hybrid_worker_meta[worker_id] = meta
        result.started_hybrid_workers.append(meta)
        return True

    def _stop_hybrid_worker_process(
        self,
        worker_id: str,
        result: RuntimeReconfigurationResult | None = None,
    ) -> None:
        proc = self._hybrid_worker_processes.pop(worker_id, None)
        meta = self._hybrid_worker_meta.pop(worker_id, None)
        if proc is None:
            if result is not None and meta is not None:
                result.training_actions.append(f"stop_hybrid_worker_meta:{worker_id}")
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if result is not None:
            result.training_actions.append(f"stop_hybrid_worker:{worker_id}")

    def _ensure_gradient_server(self):
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
            env = self._allocation_launch_env("hybrid_training")
            proc = subprocess.Popen(command, shell=True, env=env)
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

    def _wait_hybrid_worker_ready(self, worker_id: str) -> None:
        task_dir = Path(
            getattr(
                self.config.global_resource_planner,
                "hybrid_worker_task_dir",
                "./logs/elastic_training_tasks",
            )
        )
        ready_path = task_dir / f"{worker_id}.ready"
        timeout = float(
            getattr(
                self.config.global_resource_planner,
                "hybrid_worker_ready_timeout_s",
                60.0,
            )
        )
        deadline = time.time() + max(timeout, 0.0)
        proc = self._hybrid_worker_processes.get(worker_id)
        while time.time() < deadline:
            if ready_path.exists():
                return
            if proc is not None and proc.poll() is not None:
                self._release_hybrid_bootstrap_lease(worker_id)
                raise RuntimeError(
                    f"hybrid worker {worker_id} exited before ready"
                )
            time.sleep(0.2)
        self._release_hybrid_bootstrap_lease(worker_id)
        raise TimeoutError(
            f"hybrid worker {worker_id} did not become ready after {timeout:.0f}s"
        )

    def _snapshot_path_for_version(self, version: int) -> str:
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
    ) -> str:
        cfg = self.config.global_resource_planner
        endpoint = self.gradient_server.endpoint
        gradient_host = getattr(cfg, "gradient_server_public_host", "") or endpoint.host
        rl_root = Path(__file__).resolve().parents[2]
        template = getattr(cfg, "hybrid_worker_command_template", "")
        assigned_gpus = [int(gpu) for gpu in (gpus or [])]
        replica_gpus = len(assigned_gpus) or self._hybrid_replica_gpus()
        cuda_visible_devices = ",".join(str(gpu) for gpu in assigned_gpus)
        worker_mode = str(getattr(cfg, "hybrid_worker_mode", "megatron_core"))
        worker_config = str(
            getattr(cfg, "hybrid_worker_config_path", "")
            or os.environ.get("RUNTIME_CONFIG", "")
        )
        if worker_mode == "megatron_core" and not worker_config and not template:
            raise ValueError(
                "hybrid_worker_config_path or RUNTIME_CONFIG is required for "
                "a physical Megatron-Core hybrid worker"
            )
        values = {
            "python": shlex.quote(getattr(cfg, "hybrid_worker_python", sys.executable)),
            "rl_framework_path": shlex.quote(str(rl_root)),
            "gradient_transport_backend": shlex.quote(getattr(cfg, "gradient_transport_backend", "tcp")),
            "worker_id": shlex.quote(worker_id),
            "target_core_id": shlex.quote(target_core_id),
            "snapshot_path": shlex.quote(snapshot_path),
            "host": shlex.quote(host),
            "gpus": cuda_visible_devices,
            "cuda_visible_devices": shlex.quote(cuda_visible_devices),
            "gradient_host": shlex.quote(gradient_host),
            "gradient_port": endpoint.port,
            "authkey": shlex.quote(endpoint.authkey),
            "task_dir": shlex.quote(getattr(cfg, "hybrid_worker_task_dir", "")),
            "worker_mode": shlex.quote(worker_mode),
            "worker_config": shlex.quote(worker_config),
            "gradient_endpoint_dir": shlex.quote(
                str(Path(getattr(cfg, "hybrid_worker_task_dir", "")) / "core_endpoints")
            ),
            "replica_gpus": replica_gpus,
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
        return (
            f"{env_prefix}{values['python']} -m torch.distributed.run --standalone "
            f"--nproc_per_node={values['replica_gpus']} "
            f"{values['rl_framework_path']}/scripts/elastic_hybrid_worker.py "
            f"--worker-id {values['worker_id']} "
            f"--target-core-id {values['target_core_id']} "
            f"--snapshot-path {values['snapshot_path']} "
            f"--gradient-host {values['gradient_host']} "
            f"--gradient-port {values['gradient_port']} "
            f"--authkey {values['authkey']} "
            f"--task-dir {values['task_dir']} "
            f"--worker-mode {values['worker_mode']} "
            f"--config {values['worker_config']} "
            f"--gradient-endpoint-dir {values['gradient_endpoint_dir']} "
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
        timeout = float(self.config.global_resource_planner.vllm_stop_timeout_s)
        for instance_id, meta in list(self._process_meta.items()):
            proc = self._processes.get(instance_id)
            if meta.adopted:
                if meta.pid > 0:
                    self._terminate_external_pid(meta)
                else:
                    self._run_external_stop(meta)
                self._processes.pop(instance_id, None)
                self._process_meta.pop(instance_id, None)
                stopped.append(meta)
                continue
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            stopped.append(meta)
            self._processes.pop(instance_id, None)
            self._process_meta.pop(instance_id, None)
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
            proc = subprocess.Popen(
                meta.command,
                shell=True,
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT if stdout is not None else None,
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
        env = self._allocation_launch_env("runtime_rollout")
        env["PLANNED_CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in meta.gpus)
        return env

    def _allocation_launch_env(self, role: str) -> dict[str, str]:
        """Build an environment for a sibling Slurm step in the parent job."""
        env = os.environ.copy()
        for key in (
            *self._TRAIN_DISTRIBUTED_ENV_KEYS,
            *self._SLURM_NESTED_STEP_ENV_KEYS,
        ):
            env.pop(key, None)
        if role == "hybrid_training":
            # A nested srun must establish its own GPU-to-rank mapping. Keeping
            # the parent Core step's visibility mask can make torchrun ranks on
            # another node bind against a stale cgroup device namespace.
            env.pop("CUDA_VISIBLE_DEVICES", None)
        env["RL_FRAMEWORK_ROLE"] = str(role)
        return env

    def _stop_rollout_process(self, instance_id: str) -> list[ManagedRolloutProcess]:
        meta = self._process_meta.get(instance_id)
        if meta is None:
            return []
        proc = self._processes.get(instance_id)
        if meta.adopted or proc is None:
            terminated = False
            if meta.pid > 0:
                terminated = self._terminate_external_pid(meta)
            if not terminated:
                self._run_external_stop(meta)
            self._process_meta.pop(instance_id, None)
            self._processes.pop(instance_id, None)
            return [meta]
        timeout = float(self.config.global_resource_planner.vllm_stop_timeout_s)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._processes.pop(instance_id, None)
        self._process_meta.pop(instance_id, None)
        return [meta]

    def _stop_process_meta(
        self,
        meta: ManagedRolloutProcess,
        proc: subprocess.Popen | None,
    ) -> list[ManagedRolloutProcess]:
        if meta.adopted or proc is None:
            terminated = False
            if meta.pid > 0:
                terminated = self._terminate_external_pid(meta)
            if not terminated:
                self._run_external_stop(meta)
            return [meta]
        timeout = float(self.config.global_resource_planner.vllm_stop_timeout_s)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
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
                    timeout=float(
                        getattr(
                            self.config.global_resource_planner,
                            "vllm_stop_timeout_s",
                            60.0,
                        )
                    ),
                )
                log_handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"stop exit_code={completed.returncode}\n"
                )
            except subprocess.TimeoutExpired:
                log_handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] stop timeout\n"
                )

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

    def _terminate_external_pid(self, meta: ManagedRolloutProcess) -> bool:
        timeout = float(
            getattr(
                self.config.global_resource_planner,
                "vllm_stop_timeout_s",
                30.0,
            )
        )
        log_path = meta.log_path or self._rollout_log_path(meta.instance_id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"terminating adopted launcher pid={meta.pid} "
                f"instance={meta.instance_id} host={meta.host} port={meta.port}\n"
            )
            log_handle.flush()
            try:
                os.kill(meta.pid, signal.SIGTERM)
            except ProcessLookupError:
                log_handle.write("launcher pid already exited\n")
                return True
            except PermissionError as exc:
                log_handle.write(f"launcher pid terminate permission error: {exc}\n")
                return False

            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    os.kill(meta.pid, 0)
                except ProcessLookupError:
                    log_handle.write("launcher pid exited after SIGTERM\n")
                    return True
                time.sleep(0.5)

            log_handle.write("launcher pid did not exit after SIGTERM; sending SIGKILL\n")
            try:
                os.kill(meta.pid, signal.SIGKILL)
            except ProcessLookupError:
                log_handle.write("launcher pid exited before SIGKILL\n")
            return True

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
        max_model_len = (
            self.config.heterogeneous_rollout.max_model_len
            or self.config.max_seq_length
        )
        public_host = host if host != "0.0.0.0" else "127.0.0.1"
        values = {
            "python": shlex.quote(sys.executable),
            "instance_id": shlex.quote(str(inst.instance_id or f"tp{inst.tp}_{port}")),
            "model_path": shlex.quote(self.config.model_path),
            "initial_model": shlex.quote(self.config.model_path),
            "host": shlex.quote(host),
            "public_host": shlex.quote(public_host),
            "port": port,
            "tp": inst.tp,
            "gpus": ",".join(str(g) for g in inst.gpus),
            "max_model_len": max_model_len,
            "gpu_memory_utilization": self.config.heterogeneous_rollout.gpu_memory_utilization,
            "health_url": shlex.quote(f"http://{public_host}:{port}/health"),
            "local_health_url": shlex.quote(f"http://127.0.0.1:{port}/health"),
            "control_dir": shlex.quote(
                getattr(
                    self.config,
                    "rollout_weight_sync_control_dir",
                    "",
                )
                or os.path.join(self.config.log_dir, "rollout_weight_sync")
            ),
            "log_path": shlex.quote(
                self._rollout_log_path(str(inst.instance_id or f"tp{inst.tp}_{port}"))
            ),
        }
        if template:
            return template.format(**values)
        return (
            "python3 -m vllm.entrypoints.openai.api_server "
            f"--model {values['model_path']} "
            f"--host {values['host']} "
            f"--port {port} "
            f"--tensor-parallel-size {inst.tp} "
            f"--max-model-len {max_model_len} "
            "--dtype bfloat16 "
            f"--gpu-memory-utilization {values['gpu_memory_utilization']} "
            "--enforce-eager "
            "--trust-remote-code"
        )

    def close(self):
        self._stop_managed_rollout_processes()
        for proc in list(self._hybrid_worker_processes.values()):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._hybrid_worker_processes.clear()
        self._hybrid_worker_meta.clear()
        if self.gradient_server is not None:
            self.gradient_server.close()
            self.gradient_server = None
        if self.elastic_pool is not None and hasattr(self.elastic_pool, "close"):
            self.elastic_pool.close()
        time.sleep(0)

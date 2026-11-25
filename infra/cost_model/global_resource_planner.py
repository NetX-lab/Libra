"""Global Resource Planner for cross-stage RL post-training resources."""

from __future__ import annotations

import logging
import itertools
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any, Sequence

import numpy as np

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.model import (
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
    TrainingCostModel,
    representative_global_batch,
)
from RL_Framework.infra.cost_model.optimizer import (
    OptimizationResult,
    TwoLevelNestedOptimizer,
)
from RL_Framework.infra.cost_model.resource_alloc import split_gpu_slots
from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel
from RL_Framework.infra.sync.staleness import can_fit_group_deadlines

logger = logging.getLogger(__name__)


def bounded_update_time(
    *, train_s: float, rollout_batch_s: float, groups_per_batch: int,
    horizon: int, max_age: int, revision: int, reservations: list[dict],
    publish_s: float = 0.0, initial_s: float = 0.0,
    hybrid_publish_s: float | None = None, active_hybrid_replicas: int = 0,
    rollout_call_s: float | None = None,
    publish_inflight_progress_s: float = 0.0,
    hybrid_publish_inflight_progress_s: float = 0.0,
    background_ready_s: float = 0.0, core_train_s: float | None = None,
    group_tail_s: float = 0.0,
    group_size: int = 1, max_concurrent_rollouts: int | None = None,
    pending_activation: bool = False, activation_s: float = 0.0,
    activation_ready_replicas: int = 0,
    activation_train_s: list[float] | None = None,
    activation_ready_times_s: list[float] | None = None,
    initial_rollout_batch_s: float | None = None,
) -> float:
    """Advance legal groups, bounded execution and Core work on one local clock.

    The input is a POST snapshot. The current publication opens admission, then
    each forecast update retires a batch before the next POST/publication. A
    member without measured remaining work uses one representative trace time;
    completed members never run again. This is a small planning approximation,
    and never changes runtime membership or substitutes for a real READY.

    Publication may advance an already running member to its current modeled
    generate boundary, within the calibrated inflight interval. It cannot start
    queued members or their next calls. The rest of publication is frozen. The
    single-call cost is separate from full-trace work and its terminal tool tail;
    absent overlap calibration retains the original fully frozen approximation.
    """
    group_size = max(1, int(group_size))
    capacity = max(1, int(max_concurrent_rollouts or groups_per_batch * group_size))
    window = groups_per_batch * (max_age + 1)
    member_work = max(0.0, float(rollout_batch_s) + float(group_tail_s))
    now = 0.0
    groups = []
    running, queued = [], []
    identities = itertools.count()

    def add_group(deadline, total, completed=0, already_running=0, remaining=None):
        unfinished = max(0, int(total) - int(completed))
        group = {"index": next(identities), "deadline": int(deadline), "remaining": unfinished}
        groups.append(group)
        jobs = [[member_work if remaining is None else max(0.0, float(remaining)), group]
                for _ in range(unfinished)]
        # Existing running members retain the execution slots they already own.
        live = min(unfinished, max(0, int(already_running)))
        running.extend(jobs[:live])
        queued.extend(jobs[live:])

    for reservation in reservations:
        total = int(reservation.get("members_total", group_size))
        completed = total if reservation.get("ready", False) else int(reservation.get("members_ready", 0))
        add_group(reservation.get("deadline_revision", revision + max_age), total,
                  completed, reservation.get("members_running", 0), reservation.get("remaining_s"))
    if not can_fit_group_deadlines((group["deadline"] for group in groups), revision, groups_per_batch):
        return float("inf")

    def retire_finished():
        finished = [job for job in running if job[0] <= 1e-9]
        for job in finished:
            job[1]["remaining"] -= 1
            running.remove(job)
        return bool(finished)

    def start_queued():
        while queued and len(running) < capacity:
            running.append(queued.pop(0))
        finished = retire_finished()
        if finished and queued:
            start_queued()

    def advance(seconds, *, generating=True, speed=1.0):
        nonlocal now
        end = now + max(0.0, float(seconds))
        if end <= now or not generating or speed <= 0.0:
            now = end
            return
        start_queued()
        while running and now < end:
            elapsed = min(end - now, min(job[0] for job in running) / speed)
            for job in running:
                job[0] = max(0.0, job[0] - elapsed * speed)
            now += elapsed
            start_queued()
        now = end

    def admit(promise):
        while len(groups) < window:
            add_group(promise + max_age, group_size)
        start_queued()

    work = float(core_train_s if core_train_s is not None else train_s)
    hybrid_active = active_hybrid_replicas > 0
    hybrid_publish = float(hybrid_publish_s if hybrid_publish_s is not None else publish_s)
    publication = hybrid_publish if hybrid_active else publish_s
    hybrid_inflight = hybrid_publish_inflight_progress_s
    inflight = hybrid_inflight if hybrid_active else publish_inflight_progress_s
    training_after_activation = list(activation_train_s or ([train_s]
        if pending_activation or background_ready_s > 0 else []))
    ready_times = sorted(activation_ready_times_s if activation_ready_times_s is not None else
                         [background_ready_s] * len(training_after_activation))

    def activate(count):
        nonlocal work, hybrid_active
        count = min(max(0, int(count)), len(training_after_activation))
        if count:
            advance(activation_s)
            hybrid_active = True
            for _ in range(count):
                work = training_after_activation.pop(0)
            del ready_times[:count]

    def publish():
        nonlocal now
        budget = max(0.0, min(float(publication), float(inflight)))
        if rollout_call_s is not None and rollout_call_s > 0 and budget:
            for job in running:
                generation_left = max(0.0, job[0] - float(group_tail_s))
                if not math.isfinite(generation_left):
                    continue
                # Normal work can have advanced partway through this call.
                call_left = generation_left % rollout_call_s
                if call_left <= 1e-9:
                    call_left = rollout_call_s
                job[0] -= min(budget, call_left, generation_left)
            retire_finished()
        now += max(0.0, float(publication))

    initial_speed = 1.0
    if initial_rollout_batch_s is not None:
        initial_work = float(initial_rollout_batch_s) + float(group_tail_s)
        initial_speed = (member_work / initial_work if initial_work > 0 and math.isfinite(initial_work)
                         else 0.0 if not math.isfinite(initial_work) else 1.0)
    advance(initial_s, speed=initial_speed)
    # Only the actual ready subset can activate at this POST, even if a load
    # duration estimate has elapsed for another joining replica.
    activate(activation_ready_replicas)
    publish()
    admit(revision)
    for update in range(revision, revision + horizon):
        while True:
            ready = [group for group in groups if not group["remaining"]]
            if len(ready) >= groups_per_batch:
                selected = sorted(ready, key=lambda group: group["deadline"])[:groups_per_batch]
                selected_ids = {group["index"] for group in selected}
                deferred = (group["deadline"] for group in groups if group["index"] not in selected_ids)
                if can_fit_group_deadlines(deferred, update + 1, groups_per_batch):
                    break
            start_queued()
            if not running or not math.isfinite(min(job[0] for job in running)):
                return float("inf")
            advance(min(job[0] for job in running))
        advance(work)
        publication = hybrid_publish if hybrid_active else publish_s
        inflight = hybrid_inflight if hybrid_active else publish_inflight_progress_s
        for group in selected:
            groups.remove(group)
        if update == revision + horizon - 1:
            break
        if ready_times:
            activate(sum(ready_at <= now for ready_at in ready_times))
        publish()
        admit(update + 1)
    return now


@dataclass(frozen=True)
class CoreRuntimeCapabilities:
    """Fixed Core layout on the already initialized physical ranks.

    Hybrid replicas use a separate process group and do not enlarge this rank
    set. No capability is supplied to startup GRP, which may choose its size.
    """
    world_size: int
    tp: int
    pp: int
    cp: int
    ep: int
    dp: int
    rebuild_tp_dp: bool = field(default=False, init=False)

    @classmethod
    def from_engine(cls, engine):
        world = getattr(engine, "world_size", None)
        if not isinstance(world, int) or world < 1:
            return None
        getter = getattr(engine, "get_parallel_state", None)
        state = getter() if callable(getter) else {}
        dimensions = {axis: int(state.get(f"train_{axis}",
            getattr(engine, f"train_{axis}_size", 1))) for axis in ("tp", "pp", "cp", "ep", "dp")}
        return cls(world_size=world, **dimensions)

    def incompatibility(self, plan):
        topology = plan.train_config
        if int(plan.train_gpus) != self.world_size:
            return "physical_core_rank_set_changed"
        for axis in ("pp", "cp", "ep"):
            if int(getattr(topology, axis, 1)) != getattr(self, axis):
                return f"core_{axis}_rebuild_unsupported"
        if not self.rebuild_tp_dp and (int(topology.tp) != self.tp
                or int(topology.dp) * int(getattr(topology, "ep", 1)) != self.dp):
            return "core_tp_dp_rebuild_unsupported"
        return ""

    def to_dict(self):
        return {"physical_rank_count": self.world_size, "tp": self.tp, "pp": self.pp,
            "cp": self.cp, "ep": self.ep, "dp": self.dp,
            "rebuild_tp_dp": self.rebuild_tp_dp,
            "physical_rank_resize": False, "source": "initialized_training_engine"}


@dataclass
class GlobalResourcePlan:
    """Concrete cross-stage plan selected by the planner."""

    train_config: TrainParallelConfig
    rollout_config: RolloutClusterConfig
    t_train: float
    t_rollout: float
    t_global: float
    n_total_gpus: int
    max_concurrent_rollouts: int
    expected_gain_s: float = 0.0
    expected_gain_ratio: float = 0.0
    reconfiguration_cost_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def train_gpus(self) -> int:
        return self.train_config.n_gpus

    @property
    def rollout_gpus(self) -> int:
        return self.rollout_config.n_gpus

    @property
    def rollout_tp_list(self) -> list[int]:
        return list(self.rollout_config.tp_list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": {
                "tp": self.train_config.tp,
                "ep": self.train_config.ep,
                "pp": self.train_config.pp,
                "dp": self.train_config.dp,
                "cp": self.train_config.cp,
                "b_micro": self.train_config.b_micro,
                "n_gpus": self.train_gpus,
            },
            "rollout": {
                "tp_list": self.rollout_tp_list,
                "n_gpus": self.rollout_gpus,
                "n_instances": self.rollout_config.n_instances,
            },
            "t_train": self.t_train,
            "t_rollout": self.t_rollout,
            "t_global": self.t_global,
            "n_total_gpus": self.n_total_gpus,
            "max_concurrent_rollouts": self.max_concurrent_rollouts,
            "expected_gain_s": self.expected_gain_s,
            "expected_gain_ratio": self.expected_gain_ratio,
            "reconfiguration_cost_s": self.reconfiguration_cost_s,
            "metadata": dict(self.metadata),
        }


@dataclass
class RuntimeResourceMetrics:
    """Online runtime signals used to trigger resource replanning."""

    step: int = 0
    queue_pressure: float = 0.0
    active_rollout_pressure: float = 0.0
    accepted_rollouts: int = 0
    rejected_rollouts: int = 0
    rollout_time_s: float = 0.0
    train_time_s: float = 0.0
    step_time_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def rollout_train_ratio(self) -> float:
        return self.rollout_time_s / max(self.train_time_s, 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "queue_pressure": self.queue_pressure,
            "active_rollout_pressure": self.active_rollout_pressure,
            "accepted_rollouts": self.accepted_rollouts,
            "rejected_rollouts": self.rejected_rollouts,
            "rollout_time_s": self.rollout_time_s,
            "train_time_s": self.train_time_s,
            "step_time_s": self.step_time_s,
            "rollout_train_ratio": self.rollout_train_ratio,
            "raw": dict(self.raw),
        }


@dataclass
class ElasticHybridSignal:
    """Versioned GRP command for the Elastic Hybrid Pool.

    The historical ``*_workers`` names are kept on the wire for compatibility,
    but every count is a *complete DP replica*, never an individual GPU/rank.
    ``max_workers`` is the instantaneous physical capacity, not a configured
    EHP limit.
    """

    step: int
    desired_workers: int
    current_workers: int
    max_workers: int
    action: str
    reason: str
    train_rollout_ratio: float
    rollout_pressure: float
    ttl_steps: int
    active_workers: int = 0
    joining_workers: int = 0
    replica_size_gpus: int = 1
    loan_instance_ids: list[str] = field(default_factory=list)
    current_loan_instance_ids: list[str] = field(default_factory=list)
    horizon_updates: int = 0
    predicted_savings_s: float = 0.0

    @property
    def desired_replicas(self) -> int:
        return self.desired_workers

    @property
    def current_replicas(self) -> int:
        return self.current_workers

    @property
    def capacity_replicas(self) -> int:
        return self.max_workers

    @property
    def should_adjust(self) -> bool:
        return (self.desired_workers != self.current_workers
                or set(self.loan_instance_ids) != set(self.current_loan_instance_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "desired_workers": self.desired_workers,
            "current_workers": self.current_workers,
            "max_workers": self.max_workers,
            "action": self.action,
            "reason": self.reason,
            "train_rollout_ratio": self.train_rollout_ratio,
            "rollout_pressure": self.rollout_pressure,
            "ttl_steps": self.ttl_steps,
            "expires_step": self.step + self.ttl_steps,
            "active_workers": self.active_workers,
            "joining_workers": self.joining_workers,
            "desired_replicas": self.desired_replicas,
            "current_replicas": self.current_replicas,
            "capacity_replicas": self.capacity_replicas,
            "replica_size_gpus": self.replica_size_gpus,
            "loan_instance_ids": list(self.loan_instance_ids),
            "current_loan_instance_ids": list(self.current_loan_instance_ids),
            "horizon_updates": self.horizon_updates,
            "predicted_savings_s": self.predicted_savings_s,
        }


@dataclass
class PlannerDecision:
    """Planner output for one invocation."""

    step: int
    should_reconfigure: bool
    reason: str
    current_plan: GlobalResourcePlan | None = None
    candidate_plan: GlobalResourcePlan | None = None
    optimization_result: OptimizationResult | None = None
    num_requests: int = 0
    trigger: str = ""
    runtime_metrics: RuntimeResourceMetrics | None = None
    elastic_hybrid_signal: ElasticHybridSignal | None = None
    # None preserves legacy hand-constructed decisions. Planner-produced
    # decisions explicitly distinguish full repartition from EHP adjustment.
    resource_should_reconfigure: bool | None = None
    _member_costs: _MemberPlanningCosts | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "should_reconfigure": self.should_reconfigure,
            "resource_should_reconfigure": self.resource_should_reconfigure,
            "elastic_should_adjust": bool(
                self.elastic_hybrid_signal is not None
                and self.elastic_hybrid_signal.should_adjust
            ),
            "reason": self.reason,
            "trigger": self.trigger,
            "num_requests": self.num_requests,
            "runtime_metrics": (
                self.runtime_metrics.to_dict() if self.runtime_metrics else None
            ),
            "current_plan": (
                self.current_plan.to_dict() if self.current_plan else None
            ),
            "candidate_plan": (
                self.candidate_plan.to_dict() if self.candidate_plan else None
            ),
            "elastic_hybrid_signal": (
                self.elastic_hybrid_signal.to_dict()
                if self.elastic_hybrid_signal
                else None
            ),
        }


@dataclass
class _MemberPlanningCosts:
    inputs: tuple
    calls_per_trace: float
    rollout: dict[tuple[int, ...], tuple[float, dict]] = field(default_factory=dict)
    current_inputs: tuple | None = None
    current_plan: GlobalResourcePlan | None = None


class GlobalResourcePlanner:
    """Periodic planner minimizing max(T_train, T_rollout).

    The evaluator is a CostModel-compatible facade.  Depending on config it may
    use the analytic model, Sailor for training, Vidur for rollout, or a hybrid
    with analytic fallback.
    """

    def __init__(
        self,
        evaluator: HybridSimulatorCostModel,
        n_total_gpus: int,
        plan_interval: int = 10,
        warmup_steps: int = 1,
        min_history_size: int = 8,
        min_gain_ratio: float = 0.05,
        reconfiguration_cost_s: float = 15.0,
        allowed_rollout_tp: list[int] | None = None,
        require_heterogeneous_rollout_tp: bool = False,
        allowed_train_tp: list[int] | None = None,
        allowed_train_ep: list[int] | None = None,
        allowed_train_pp: list[int] | None = None,
        train_comm_compute_ratio_threshold: float = float("inf"),
        train_ep_comm_compute_ratio_threshold: float = float("inf"),
        train_pipeline_bubble_ratio_threshold: float = 0.30,
        fixed_train_gpus: int = 0,
        initial_allocation_strategy: str = "grp",
        allocation_granularity_gpus: int = 1,
        min_train_gpus: int = 1,
        min_rollout_gpus: int = 1,
        micro_batch_sizes: list[int] | None = None,
        max_history_size: int = 4096,
        online_replanning: bool = True,
        replan_cooldown_steps: int = 1,
        queue_pressure_threshold: float = 0.75,
        active_rollout_pressure_threshold: float = 0.85,
        rejected_rollout_delta_threshold: int = 8,
        rollout_train_imbalance_threshold: float = 1.25,
        elastic_hybrid_planning_enabled: bool = False,
        elastic_hybrid_signal_ttl_steps: int = 20,
        verbose: bool = False,
    ):
        self.evaluator = evaluator
        self.n_total_gpus = int(n_total_gpus)
        self.plan_interval = max(1, int(plan_interval))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_history_size = max(1, int(min_history_size))
        self.min_gain_ratio = max(0.0, float(min_gain_ratio))
        self.reconfiguration_cost_s = max(0.0, float(reconfiguration_cost_s))
        self.allowed_rollout_tp = allowed_rollout_tp or [1, 2, 4, 8]
        self.require_heterogeneous_rollout_tp = bool(require_heterogeneous_rollout_tp)
        self.allowed_train_tp = allowed_train_tp
        self.allowed_train_ep = allowed_train_ep
        self.allowed_train_pp = allowed_train_pp
        self.fixed_train_gpus = int(fixed_train_gpus or 0)
        self.initial_allocation_strategy = str(initial_allocation_strategy or "grp")
        if self.initial_allocation_strategy not in {"grp", "configured"}:
            raise ValueError("initial_allocation_strategy must be 'grp' or 'configured'")
        self.allocation_granularity_gpus = max(1, int(allocation_granularity_gpus))
        self.min_train_gpus = max(1, int(min_train_gpus))
        self.min_rollout_gpus = max(1, int(min_rollout_gpus))
        self.micro_batch_sizes = micro_batch_sizes
        self.max_history_size = max(1, int(max_history_size))
        self.online_replanning = bool(online_replanning)
        self.replan_cooldown_steps = max(0, int(replan_cooldown_steps))
        self.queue_pressure_threshold = max(0.0, float(queue_pressure_threshold))
        self.active_rollout_pressure_threshold = max(
            0.0, float(active_rollout_pressure_threshold)
        )
        self.rejected_rollout_delta_threshold = max(
            0, int(rejected_rollout_delta_threshold)
        )
        self.rollout_train_imbalance_threshold = max(
            0.0, float(rollout_train_imbalance_threshold)
        )
        self.elastic_hybrid_planning_enabled = bool(
            elastic_hybrid_planning_enabled
        )
        self.elastic_hybrid_signal_ttl_steps = max(
            1, int(elastic_hybrid_signal_ttl_steps)
        )
        self.verbose = verbose
        self.runtime_capabilities: CoreRuntimeCapabilities | None = None

        self.optimizer = TwoLevelNestedOptimizer(
            cost_model=evaluator,
            allowed_rollout_tp=self.allowed_rollout_tp,
            require_heterogeneous_rollout_tp=self.require_heterogeneous_rollout_tp,
            train_comm_compute_ratio_threshold=train_comm_compute_ratio_threshold,
            train_ep_comm_compute_ratio_threshold=train_ep_comm_compute_ratio_threshold,
            train_pipeline_bubble_ratio_threshold=train_pipeline_bubble_ratio_threshold,
            verbose=verbose,
        )
        self._history: list[RequestInfo] = []
        self._training_lengths: list[int] = []
        self._calls_per_trace: float | None = None
        self._active_plan: GlobalResourcePlan | None = None
        self._last_decision: PlannerDecision | None = None
        self._runtime_metrics: list[RuntimeResourceMetrics] = []
        self._last_replan_step = -10**9
        self._last_rejected_rollouts = 0

    @classmethod
    def from_config(cls, config: AsyncRLConfig) -> "GlobalResourcePlanner":
        planner_cfg = config.global_resource_planner
        n_total = config.n_total_gpus or (config.train_gpus + config.rollout_gpus)
        return cls(
            evaluator=HybridSimulatorCostModel(config),
            n_total_gpus=n_total,
            plan_interval=planner_cfg.plan_interval,
            warmup_steps=planner_cfg.warmup_steps,
            min_history_size=planner_cfg.min_history_size,
            min_gain_ratio=planner_cfg.min_gain_ratio,
            reconfiguration_cost_s=planner_cfg.reconfiguration_cost_s,
            allowed_rollout_tp=planner_cfg.allowed_rollout_tp,
            require_heterogeneous_rollout_tp=planner_cfg.require_heterogeneous_rollout_tp,
            allowed_train_tp=planner_cfg.allowed_train_tp or None,
            allowed_train_ep=planner_cfg.allowed_train_ep or None,
            allowed_train_pp=planner_cfg.allowed_train_pp or None,
            train_comm_compute_ratio_threshold=planner_cfg.train_comm_compute_ratio_threshold,
            train_ep_comm_compute_ratio_threshold=planner_cfg.train_ep_comm_compute_ratio_threshold,
            train_pipeline_bubble_ratio_threshold=planner_cfg.train_pipeline_bubble_ratio_threshold,
            fixed_train_gpus=planner_cfg.fixed_train_gpus,
            initial_allocation_strategy=planner_cfg.initial_allocation_strategy,
            allocation_granularity_gpus=planner_cfg.allocation_granularity_gpus,
            min_train_gpus=planner_cfg.min_train_gpus,
            min_rollout_gpus=planner_cfg.min_rollout_gpus,
            micro_batch_sizes=planner_cfg.micro_batch_sizes or None,
            max_history_size=planner_cfg.max_history_size,
            online_replanning=planner_cfg.runtime_online_replanning,
            replan_cooldown_steps=planner_cfg.runtime_replan_cooldown_steps,
            queue_pressure_threshold=planner_cfg.runtime_queue_pressure_threshold,
            active_rollout_pressure_threshold=planner_cfg.runtime_active_rollout_pressure_threshold,
            rejected_rollout_delta_threshold=planner_cfg.runtime_rejected_rollout_delta_threshold,
            rollout_train_imbalance_threshold=planner_cfg.runtime_rollout_train_imbalance_threshold,
            elastic_hybrid_planning_enabled=planner_cfg.elastic_hybrid_planning_enabled,
            elastic_hybrid_signal_ttl_steps=planner_cfg.elastic_hybrid_signal_ttl_steps,
            verbose=planner_cfg.verbose,
        )

    @property
    def active_plan(self) -> GlobalResourcePlan | None:
        return self._active_plan

    @property
    def last_decision(self) -> PlannerDecision | None:
        return self._last_decision

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def latest_runtime_metrics(self) -> RuntimeResourceMetrics | None:
        return self._runtime_metrics[-1] if self._runtime_metrics else None

    def observe_training_batch(self, batch: list[Any]) -> int:
        """Keep only full-sequence CPU lengths, separately from generate history."""
        lengths = []
        for item in batch or []:
            if not isinstance(item, dict):
                continue
            input_ids = item.get("input_ids")
            if input_ids is not None:
                lengths.append(int(input_ids.shape[-1]))
            elif item.get("sequence_length", item.get("training_length", 0)):
                lengths.append(int(item.get("sequence_length", item.get("training_length"))))
        if lengths:
            self._training_lengths = lengths
        calls = [int(item["n_turns"]) for item in batch or []
                 if isinstance(item, dict) and item.get("n_turns", 0)]
        if calls:
            self._calls_per_trace = sum(calls) / len(calls)
        return len(lengths)

    def _training_workload_lengths(self, requests: list[RequestInfo], batch_size: int) -> list[int]:
        calibration = getattr(self.evaluator.config.profiling, "calibration", {}) or {}
        lengths = (self._training_lengths or calibration.get("training", {}).get("sequence_lengths")
                   or [request.training_length or request.total_length for request in requests] or [1024])
        return TrainingCostModel._representative_lengths(lengths, batch_size)

    def _rollout_calls_per_trace(self, config: AsyncRLConfig) -> float:
        calibration = getattr(config.profiling, "calibration", {}) or {}
        return max(1.0, float(self._calls_per_trace if self._calls_per_trace is not None
                              else calibration.get("calls_per_trace", 1.0)))

    def _complete_trace_rollout_seconds(
        self, config: AsyncRLConfig, tp_list: list[int], requests: list[RequestInfo],
        *, costs: _MemberPlanningCosts | None = None,
    ) -> float:
        single_calls, _ = self._layout_supply(tp_list, requests, costs)
        calls = costs.calls_per_trace if costs is not None else self._rollout_calls_per_trace(config)
        return single_calls * calls

    def _fixed_rollout_supply(
        self, tp_list: list[int], requests: list[RequestInfo],
    ) -> tuple[float, dict]:
        rollout = RolloutClusterConfig(tp_list=tp_list)
        if getattr(self.evaluator, "rollout_backend", "analytic") == "analytic":
            return self.optimizer.evaluate_fixed_rollout_config(rollout, requests)
        return self.evaluator.evaluate_rollout(rollout, requests)

    def _member_planning_costs(self, config, requests, previous=None) -> _MemberPlanningCosts:
        calibration = getattr(config.profiling, "calibration", {}) or {}
        calls = self._rollout_calls_per_trace(config)
        inputs = (
            id(self.evaluator),
            getattr(self.evaluator, "train_backend", "analytic"),
            getattr(self.evaluator, "rollout_backend", "analytic"),
            tuple((request.prompt_length, request.gen_length) for request in requests),
            calibration.get("rollout", {}), calibration.get("training", {}),
        )
        if previous is not None and previous.inputs == inputs and previous.calls_per_trace == calls:
            return previous
        return _MemberPlanningCosts(deepcopy(inputs), calls)

    def _layout_supply(self, tp_list, requests, costs):
        if costs is None:
            return self._fixed_rollout_supply(tp_list, requests)
        analytic = getattr(self.evaluator, "rollout_backend", "analytic") == "analytic"
        key = tuple(sorted(tp_list) if analytic else tp_list)
        if key not in costs.rollout:
            costs.rollout[key] = self._fixed_rollout_supply(tp_list, requests)
        return costs.rollout[key]

    def observe_batch(self, batch: list[Any]) -> int:
        self.observe_training_batch(batch)
        requests = self._extract_requests(batch)
        if not requests:
            return 0
        self._history.extend(requests)
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size :]
        return len(requests)

    def replace_history(self, history: list[Any]) -> int:
        """Replace the in-memory request history with a fresh profile window."""

        requests = self._extract_requests(history)
        if len(requests) > self.max_history_size:
            requests = requests[-self.max_history_size :]
        self._history = requests
        return len(requests)

    def extend_history(self, history: list[Any]) -> int:
        """Append a fresh batch of request history entries."""

        requests = self._extract_requests(history)
        if not requests:
            return 0
        self._history.extend(requests)
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size :]
        return len(requests)

    def observe_runtime(
        self,
        *,
        step: int,
        dispatcher_metrics: dict[str, Any] | None = None,
        step_stats: dict[str, Any] | None = None,
    ) -> RuntimeResourceMetrics:
        """Record online runtime pressure signals for dynamic replanning."""

        dispatcher_metrics = dict(dispatcher_metrics or {})
        step_stats = dict(step_stats or {})

        queue_items = (
            int(dispatcher_metrics.get("pending_inputs", 0) or 0)
            + int(dispatcher_metrics.get("runner_input_queue", 0) or 0)
            + int(dispatcher_metrics.get("runner_output_queue", 0) or 0)
        )
        queue_cap = max(
            1,
            int(dispatcher_metrics.get("runner_max_queue_size", 0) or 0)
            + int(dispatcher_metrics.get("staleness_pending_limit", 0) or 0),
        )
        running = int(dispatcher_metrics.get("staleness_running", 0) or 0)
        max_concurrent = max(1, int(step_stats.get("max_concurrent_rollouts", 0) or 0))
        if max_concurrent <= 1:
            max_concurrent = max(
                1,
                int(dispatcher_metrics.get("staleness_running", 0) or 0)
                + int(dispatcher_metrics.get("staleness_capacity", 0) or 0),
            )

        metrics = RuntimeResourceMetrics(
            step=int(step),
            queue_pressure=queue_items / queue_cap,
            active_rollout_pressure=running / max_concurrent,
            accepted_rollouts=int(dispatcher_metrics.get("staleness_accepted", 0) or 0),
            rejected_rollouts=int(dispatcher_metrics.get("staleness_rejected", 0) or 0),
            rollout_time_s=float(step_stats.get("rollout_time", 0.0) or 0.0),
            train_time_s=float(step_stats.get("train_time", 0.0) or 0.0),
            step_time_s=float(step_stats.get("step_time", 0.0) or 0.0),
            raw={
                "dispatcher": dispatcher_metrics,
                "step_stats": step_stats,
            },
        )
        self._runtime_metrics.append(metrics)
        if len(self._runtime_metrics) > self.max_history_size:
            self._runtime_metrics = self._runtime_metrics[-self.max_history_size :]
        return metrics

    def plan_if_needed(
        self,
        step: int,
        config: AsyncRLConfig,
        batch: list[Any] | None = None,
        runtime_metrics: RuntimeResourceMetrics | None = None,
    ) -> PlannerDecision:
        if batch is not None:
            self.observe_batch(batch)
        if runtime_metrics is None:
            runtime_metrics = self.latest_runtime_metrics

        if step < self.warmup_steps:
            return self._remember(
                PlannerDecision(
                    step,
                    False,
                    "warmup",
                    num_requests=len(self._history),
                    trigger="warmup",
                    runtime_metrics=runtime_metrics,
                    resource_should_reconfigure=False,
                )
            )

        trigger = self._planning_trigger(step, runtime_metrics)
        if trigger == "cooldown":
            return self._remember(
                PlannerDecision(
                    step,
                    False,
                    "cooldown",
                    current_plan=self._active_plan,
                    num_requests=len(self._history),
                    trigger=trigger,
                    runtime_metrics=runtime_metrics,
                    resource_should_reconfigure=False,
                )
            )
        if trigger == "interval_skip":
            return self._remember(
                PlannerDecision(
                    step,
                    False,
                    "interval_skip",
                    current_plan=self._active_plan,
                    num_requests=len(self._history),
                    trigger=trigger,
                    runtime_metrics=runtime_metrics,
                    resource_should_reconfigure=False,
                )
            )
        if len(self._history) < self.min_history_size:
            return self._remember(
                PlannerDecision(
                    step,
                    False,
                    "insufficient_history",
                    current_plan=self._active_plan,
                    num_requests=len(self._history),
                    trigger=trigger,
                    runtime_metrics=runtime_metrics,
                    resource_should_reconfigure=False,
                )
            )

        workload = representative_global_batch(self._history, int(config.batch_size))
        forced_rollout_tp = self._forced_rollout_tp_list(config)
        force_runtime_reconfigure = os.environ.get("GRP_FORCE_RUNTIME_RECONFIGURE", "0") == "1"
        explicit_repartition = (
            forced_rollout_tp is not None
            or bool(os.environ.get("GRP_FORCE_TRAIN_GPUS", "").strip())
            or bool(getattr(config.global_resource_planner, "runtime_forced_train_gpus", 0))
            or force_runtime_reconfigure
        )
        if (self.runtime_capabilities is not None and self.elastic_hybrid_planning_enabled
                and not explicit_repartition):
            costs = self._member_planning_costs(config, workload)
            current = self._build_current_plan(config, workload, costs=costs)
            signal = self._build_elastic_hybrid_signal(step=step, config=config,
                current=current, candidate=current, runtime_metrics=runtime_metrics,
                workload=workload, costs=costs)
            adjusting = bool(signal is not None and signal.should_adjust)
            decision = PlannerDecision(step=step, should_reconfigure=adjusting,
                reason="elastic_hybrid_adjustment" if adjusting else "same_plan",
                current_plan=current, candidate_plan=current, num_requests=len(self._history),
                trigger=trigger, runtime_metrics=runtime_metrics,
                elastic_hybrid_signal=signal, resource_should_reconfigure=False,
                _member_costs=costs)
            self._active_plan = current
            if adjusting:
                self._last_replan_step = int(step)
            return self._remember(decision)
        current = self._build_current_plan(config, workload)
        forced_train = self._forced_train_config(
            config,
            current,
            rollout_gpus=(
                sum(forced_rollout_tp)
                if forced_rollout_tp is not None
                else int(config.rollout_gpus or 0)
            ),
        )
        forced_candidate = forced_rollout_tp is not None or forced_train is not None
        result = self.optimizer.optimize(
            n_total_gpus=self.n_total_gpus,
            requests=workload,
            B_global=config.batch_size,
            allowed_train_tp=self.allowed_train_tp,
            allowed_train_ep=self.allowed_train_ep,
            allowed_train_pp=self.allowed_train_pp,
            micro_batch_sizes=self.micro_batch_sizes,
            fixed_train_gpus=(
                0
                if self.initial_allocation_strategy == "grp"
                else self.fixed_train_gpus
            ),
            allocation_granularity_gpus=self.allocation_granularity_gpus,
            min_train_gpus=self.min_train_gpus,
            min_rollout_gpus=self.min_rollout_gpus,
            rollout_node_tp_pattern=list(
                config.global_resource_planner.rollout_node_tp_pattern or []
            ),
            train_cp_size=max(1, int(config.train_cp_size or 1)),
            dp_batch_group_size=max(1, int(config.n_samples or 1)),
            training_sequence_lengths=self._training_workload_lengths(workload, config.batch_size),
            rollout_calls_per_trace=self._rollout_calls_per_trace(config),
        )
        candidate = self._plan_from_result(result, config)
        if candidate is None and not forced_candidate:
            self._mark_rejected_trigger_observed(trigger, runtime_metrics)
            return self._remember(
                PlannerDecision(
                    step,
                    False,
                    "no_feasible_candidate",
                    current_plan=current,
                    optimization_result=result,
                    num_requests=len(self._history),
                    trigger=trigger,
                    runtime_metrics=runtime_metrics,
                    resource_should_reconfigure=False,
                )
            )

        if forced_rollout_tp is not None:
            forced_rollout = RolloutClusterConfig(tp_list=forced_rollout_tp)
            forced_rollout_time, forced_rollout_details = self._fixed_rollout_supply(
                forced_rollout_tp, workload,
            )
            forced_rollout_time *= self._rollout_calls_per_trace(config)
            forced_train_time = current.t_train
            forced_train_details = current.metadata.get("training", {})
            if forced_train is not None:
                sequence_lengths = self._training_workload_lengths(workload, config.batch_size)
                forced_train_time, forced_train_details = self.evaluator.evaluate_training(
                    forced_train,
                    config.batch_size,
                    sequence_lengths,
                )
            candidate = GlobalResourcePlan(
                train_config=forced_train or current.train_config,
                rollout_config=forced_rollout,
                t_train=forced_train_time,
                t_rollout=forced_rollout_time,
                t_global=max(forced_train_time, forced_rollout_time),
                n_total_gpus=self.n_total_gpus,
                max_concurrent_rollouts=config.max_concurrent_rollouts,
                metadata={
                    "source": (
                        "forced_runtime_train_and_rollout"
                        if forced_train is not None
                        else "forced_runtime_rollout_tp"
                    ),
                    "training": forced_train_details,
                    "rollout": forced_rollout_details,
                },
            )
        elif forced_train is not None:
            sequence_lengths = self._training_workload_lengths(workload, config.batch_size)
            forced_train_time, forced_train_details = self.evaluator.evaluate_training(
                forced_train,
                config.batch_size,
                sequence_lengths,
            )
            candidate = GlobalResourcePlan(
                train_config=forced_train,
                rollout_config=current.rollout_config,
                t_train=forced_train_time,
                t_rollout=current.t_rollout,
                t_global=max(forced_train_time, current.t_rollout),
                n_total_gpus=self.n_total_gpus,
                max_concurrent_rollouts=config.max_concurrent_rollouts,
                metadata={
                    "source": "forced_runtime_train_gpus",
                    "training": forced_train_details,
                },
            )

        # Keep the unconstrained capacity recommendation as EHP demand, while
        # selecting Core/rollout layouts that the live backend can execute.
        # Startup has no live capabilities and retains the full 4/8/... search.
        elastic_candidate = candidate
        capabilities = self.runtime_capabilities
        rejected = capabilities.incompatibility(candidate) if capabilities is not None else ""
        if rejected:
            unconstrained = candidate.to_dict()
            if not forced_candidate:
                feasible = self.optimizer.optimize(
                    n_total_gpus=self.n_total_gpus, requests=workload,
                    B_global=config.batch_size,
                    allowed_train_tp=(self.allowed_train_tp if capabilities.rebuild_tp_dp else [capabilities.tp]),
                    allowed_train_ep=[capabilities.ep], allowed_train_pp=[capabilities.pp],
                    micro_batch_sizes=self.micro_batch_sizes,
                    fixed_train_gpus=capabilities.world_size,
                    allocation_granularity_gpus=self.allocation_granularity_gpus,
                    min_train_gpus=self.min_train_gpus, min_rollout_gpus=self.min_rollout_gpus,
                    rollout_node_tp_pattern=list(config.global_resource_planner.rollout_node_tp_pattern or []),
                    train_cp_size=capabilities.cp,
                    dp_batch_group_size=max(1, int(config.n_samples or 1)),
                    training_sequence_lengths=self._training_workload_lengths(workload, config.batch_size),
                    rollout_calls_per_trace=self._rollout_calls_per_trace(config))
                candidate = self._plan_from_result(feasible, config) or current
                result = feasible
            else:
                # Forced test commands also cannot authorize inventing ranks.
                candidate = current
            candidate.metadata["runtime_candidate_filter"] = {
                "reason": rejected, "capabilities": capabilities.to_dict(),
                "unconstrained_candidate": unconstrained,
                "elastic_demand_preserved": True}
        if capabilities is not None:
            candidate.metadata["runtime_core_capabilities"] = capabilities.to_dict()
        candidate.metadata["workload"] = {
            "global_batch_size": int(config.batch_size), "history_size": len(self._history),
            "representative_requests": len(workload), "selection": "deterministic_length_quantiles"}
        raw_gain = current.t_global - candidate.t_global
        net_gain = raw_gain - self.reconfiguration_cost_s
        gain_ratio = raw_gain / max(current.t_global, 1e-6)
        candidate.expected_gain_s = net_gain
        candidate.expected_gain_ratio = gain_ratio
        candidate.reconfiguration_cost_s = self.reconfiguration_cost_s

        same_plan = self._same_plan(current, candidate)
        elastic_signal = self._build_elastic_hybrid_signal(
            step=step,
            config=config,
            current=current,
            candidate=elastic_candidate,
            runtime_metrics=runtime_metrics,
        )
        elastic_adjust = bool(
            elastic_signal is not None and elastic_signal.should_adjust
        )
        resource_should = (
            (not same_plan or force_runtime_reconfigure)
            and not (rejected and forced_candidate)
            and (
                forced_candidate
                or force_runtime_reconfigure
                or (net_gain > 0.0 and gain_ratio >= self.min_gain_ratio)
            )
        )
        should = resource_should or elastic_adjust
        reason = (
            "elastic_hybrid_adjustment"
            if elastic_adjust
            else
            "forced_reconfigure"
            if should and forced_candidate
            else "forced_runtime_reconfigure"
            if should and force_runtime_reconfigure
            else "reconfigure"
            if should
            else "same_plan"
            if same_plan
            else "gain_below_threshold"
        )

        decision = PlannerDecision(
            step=step,
            should_reconfigure=should,
            reason=reason,
            current_plan=current,
            candidate_plan=candidate,
            optimization_result=result,
            num_requests=len(self._history),
            trigger=trigger,
            runtime_metrics=runtime_metrics,
            elastic_hybrid_signal=elastic_signal,
            resource_should_reconfigure=resource_should,
        )
        if resource_should:
            self._active_plan = candidate
        else:
            # An EHP join/release does not apply the rejected candidate's
            # training topology or rollout TP layout.
            self._active_plan = current
        if should:
            self._last_replan_step = int(step)
        self._mark_rejected_trigger_observed(trigger, runtime_metrics)
        return self._remember(decision)

    def recheck_elastic_signal(
        self, step: int, config: AsyncRLConfig,
        runtime_metrics: RuntimeResourceMetrics | None = None,
        *, decision: PlannerDecision | None = None,
    ) -> ElasticHybridSignal | None:
        """Recheck member costs and readiness once, then release the decision cache."""
        if not self.elastic_hybrid_planning_enabled:
            return None
        workload = representative_global_batch(self._history, int(config.batch_size))
        previous = decision._member_costs if decision is not None else None
        costs = self._member_planning_costs(config, workload, previous)
        current = self._build_current_plan(config, workload, costs=costs)
        if decision is not None:
            decision.current_plan = current
            decision._member_costs = None
        if not workload:
            return None
        return self._build_elastic_hybrid_signal(step=step, config=config,
            current=current, candidate=current, runtime_metrics=runtime_metrics,
            workload=workload, costs=costs)

    def _build_elastic_hybrid_signal(
        self, *, step: int, config: AsyncRLConfig, current: GlobalResourcePlan,
        candidate: GlobalResourcePlan, runtime_metrics: RuntimeResourceMetrics | None,
        workload: list[RequestInfo] | None = None, costs: _MemberPlanningCosts | None = None,
    ) -> ElasticHybridSignal | None:
        if not self.elastic_hybrid_planning_enabled:
            return None
        cfg = config.global_resource_planner
        width = (max(1, int(config.train_tp_size)) * max(1, int(config.train_ep_size))
                 * max(1, int(config.train_pp_size)) * max(1, int(config.train_cp_size)))
        configured_width = int(getattr(cfg, "elastic_hybrid_replica_size_gpus", 0) or width)
        if configured_width != width:
            raise ValueError(f"elastic_hybrid_replica_size_gpus must equal one complete DP Replica ({width})")
        raw = runtime_metrics.raw if runtime_metrics is not None else {}
        stats = raw.get("step_stats", {})
        dispatcher = raw.get("dispatcher", {})
        active = max(0, int(stats.get("active_hybrid_replicas", stats.get("active_hybrid_workers", 0)) or 0))
        joining = max(0, int(stats.get("joining_hybrid_replicas", stats.get("joining_hybrid_workers", 0)) or 0),
                      int(stats.get("pending_hybrid_replica_joins", stats.get("pending_hybrid_joins", 0)) or 0))
        instances = list(config.heterogeneous_rollout.instances)
        by_id = {str(instance.instance_id): instance for instance in instances}
        current_ids = tuple(sorted(str(identity) for identity in stats.get("loan_instance_ids", [])))
        if set(current_ids) == by_id.keys():
            # Legacy all-lent layouts keep their ready inventory. The next
            # step restores supply if a full batch cannot be collected.
            return None
        minimum = max(1, int(getattr(cfg, "elastic_hybrid_min_rollout_gpus", 0) or 0))
        from RL_Framework.infra.elastic.rollout_lending import can_expand_loan, loan_replica_capacity
        choices, capacities = [()], {(): 0}
        for count in range(1, len(instances) + 1):
            for subset in itertools.combinations(instances, count):
                replicas = loan_replica_capacity(instances, subset, width, minimum)
                # A removable whole service adds no training capacity and only
                # reduces rollout supply. Keep current loans separately below.
                if replicas and all(loan_replica_capacity(instances,
                        subset[:index] + subset[index + 1:], width, minimum) < replicas
                        for index in range(len(subset))):
                    identities = tuple(sorted(str(instance.instance_id) for instance in subset))
                    choices.append(identities)
                    capacities[identities] = replicas
        if not current_ids and active + joining:
            # Older callers report complete-replica counts. Resolve only a real
            # physical candidate; never invent a partially borrowed service.
            current_ids = next((choice for choice in choices
                                if capacities[choice] == active + joining), ())
        if current_ids not in choices:
            choices.append(current_ids)
            capacities[current_ids] = loan_replica_capacity(instances,
                [by_id[key] for key in current_ids], width, 0)
        current_workers = capacities[current_ids]
        calibration = dict(getattr(config.profiling, "calibration", {}) or {})
        lifecycle = calibration.get("lifecycle", {})
        horizon = min(max(1, int(calibration.get("horizon_updates", 8))),
                      max(0, int(stats.get("remaining_updates", max(0, config.total_steps - step - 1)))))
        error = float(calibration.get("error_margin_s", 0.0))
        remaining_load = max(0.0, float(stats.get("joining_remaining_s",
            0.0 if stats.get("activation_ready", False) else
            float(lifecycle.get("load_s", 0.0)) - float(stats.get("joining_elapsed_s", 0.0)))))
        if workload is None:
            workload = representative_global_batch(self._history, int(config.batch_size))
        if not workload:
            workload = [RequestInfo(1, 1)] * int(config.batch_size)
        if costs is None:
            costs = _MemberPlanningCosts((), self._rollout_calls_per_trace(config))
        lengths = self._training_workload_lengths(workload, config.batch_size)
        groups_per_batch = int(config.batch_size) // int(config.n_samples)
        age = max(0, int(getattr(config, "max_head_offpolicyness", 1)))
        revision = int(dispatcher.get("optimizer_revision", step + 1))
        reservations = list(dispatcher.get("group_reservations", []))
        publish = float(lifecycle.get("publish_s", 0.0))
        hybrid_publish = float(lifecycle.get("hybrid_publish_s", publish))
        publish_inflight = float(lifecycle.get("publish_inflight_progress_s", 0.0))
        hybrid_publish_inflight = float(lifecycle.get("hybrid_publish_inflight_progress_s", 0.0))
        tail = float(calibration.get("group_tool_s", 0.0)) + float(calibration.get("group_tail_s", 0.0))
        calls_per_trace = costs.calls_per_trace

        def supply(tp_list):
            return self._complete_trace_rollout_seconds(config, tp_list, workload, costs=costs)

        @cache
        def training(replicas):
            return self.evaluator.evaluate_elastic_training(
                current.train_config, config.batch_size, config.n_samples, replicas, lengths)[0]

        def borrow_frontend(ids):
            # These are current single generate calls, not active traces or
            # their remaining tool rounds. No token progress is assumed.
            counts = stats.get("generate_reservations_by_instance", {})
            drain = 0.0
            for identity in ids:
                count = max(0, int(counts.get(identity, 0)))
                if count:
                    calls = representative_global_batch(workload, count)
                    drain = max(drain, self.evaluator.evaluate_rollout_instance(
                        int(by_id[identity].tp), calls)[0])
            # Targets drain in parallel, then the executor stops them in
            # sequence. Execution cleanup may release an old snapshot, so a
            # new loan includes its necessary Core capture once.
            return (drain + len(ids) * float(lifecycle.get("borrow_stop_s", 0.0))
                    + float(lifecycle.get("borrow_snapshot_s", 0.0))
                    + float(lifecycle.get("borrow_coordination_s", 0.0)))

        forecasts = []
        for ids in choices:
            replicas = capacities[ids]
            surviving = [instance.tp for instance in instances if str(instance.instance_id) not in ids]
            if not surviving:
                continue
            rollout = supply(surviving)
            rollout_call = rollout / calls_per_trace
            train = training(replicas)
            changing = ids != current_ids
            expanding = bool(current_ids and can_expand_loan(instances, current_ids, ids, width))
            retains_current = not changing or expanding
            initial = float(lifecycle.get("return_frontend_s", 0.0)) if current_ids and not retains_current else 0.0
            new_ids = tuple(key for key in ids if not retains_current or key not in current_ids)
            first_active = min(active, replicas) if retains_current else 0
            before_train = training(first_active)
            initial += borrow_frontend(new_ids) if new_ids else 0.0
            new_load = initial + float(lifecycle.get("load_s", 0.0)) if new_ids else 0.0
            activation_times = [training(count) for count in range(first_active + 1, replicas + 1)]
            retained_pending = max(0, current_workers - first_active) if retains_current else 0
            ready_replicas = min(retained_pending, int(stats.get(
                "activation_ready_replicas", int(bool(stats.get("activation_ready", False))))))
            load = new_load if new_ids else remaining_load if retains_current and joining else 0.0
            activation_ready_times = None
            if expanding and retained_pending:
                activation_ready_times = ([0.0] * ready_replicas
                    + [remaining_load] * (retained_pending - ready_replicas)
                    + [new_load] * (replicas - current_workers))
            # During return/replace only services outside both loans can run;
            # restored services cannot receive work before this publication.
            unchanged = [instance.tp for instance in instances
                         if str(instance.instance_id) not in set(ids) | set(current_ids)]
            initial_rollout = (supply(unchanged)
                               if changing and unchanged else float("inf") if changing else None)
            seconds = bounded_update_time(train_s=train, core_train_s=before_train,
                rollout_batch_s=rollout, groups_per_batch=groups_per_batch,
                horizon=horizon, max_age=age, revision=revision, reservations=reservations,
                publish_s=publish, initial_s=initial, background_ready_s=load,
                hybrid_publish_s=hybrid_publish, active_hybrid_replicas=first_active,
                rollout_call_s=rollout_call, publish_inflight_progress_s=publish_inflight,
                hybrid_publish_inflight_progress_s=hybrid_publish_inflight,
                group_tail_s=tail, group_size=config.n_samples,
                max_concurrent_rollouts=config.max_concurrent_rollouts,
                pending_activation=bool(activation_times), activation_train_s=activation_times,
                activation_ready_times_s=activation_ready_times,
                activation_s=float(lifecycle.get("activate_s", 0.0)),
                activation_ready_replicas=ready_replicas,
                initial_rollout_batch_s=initial_rollout) if horizon else 0.0
            steady_publish = hybrid_publish if replicas else publish
            steady_inflight = hybrid_publish_inflight if replicas else publish_inflight
            steady_overlap = max(0.0, min(steady_publish, steady_inflight, rollout_call, rollout))
            forecasts.append({"loan_instance_ids": list(ids), "replicas": replicas,
                "seconds": seconds, "steady_update_s":
                    max(train, rollout + tail - steady_overlap) + steady_publish})
        baseline = next(row for row in forecasts if tuple(row["loan_instance_ids"]) == current_ids)
        eligible = [row for row in forecasts if row is baseline or
            (not (set(row["loan_instance_ids"]) - set(current_ids))
             or row["steady_update_s"] < baseline["steady_update_s"])]
        best = min(eligible, key=lambda row: row["seconds"])
        savings = baseline["seconds"] - best["seconds"]
        if savings <= error:
            best = baseline
        desired_ids = list(best["loan_instance_ids"])
        desired = int(best["replicas"])
        changed = tuple(desired_ids) != current_ids
        action = ("join" if desired > current_workers else "release" if desired < current_workers
                  else "replace" if changed else "hold")
        capacity = max(row["replicas"] for row in forecasts)
        return ElasticHybridSignal(step=int(step), desired_workers=desired,
            current_workers=current_workers, max_workers=capacity, action=action,
            reason="calibrated_gain" if changed else "calibrated_hold",
            train_rollout_ratio=0.0, rollout_pressure=0.0,
            ttl_steps=int(self.elastic_hybrid_signal_ttl_steps), active_workers=active,
            joining_workers=joining, replica_size_gpus=width,
            loan_instance_ids=desired_ids, current_loan_instance_ids=list(current_ids),
            horizon_updates=horizon, predicted_savings_s=savings if changed else 0.0)

    def _mark_rejected_trigger_observed(
        self,
        trigger: str,
        runtime_metrics: RuntimeResourceMetrics | None,
    ) -> None:
        if trigger == "rejected_rollout_delta" and runtime_metrics is not None:
            self._last_rejected_rollouts = runtime_metrics.rejected_rollouts

    def _forced_rollout_tp_list(self, config: AsyncRLConfig) -> list[int] | None:
        raw = os.environ.get("GRP_FORCE_ROLLOUT_TP_LIST", "").strip()
        if raw:
            try:
                tp_list = [
                    int(item.strip())
                    for item in re.split(r"[,;:]", raw)
                    if item.strip()
                ]
            except ValueError as exc:
                raise ValueError(f"Invalid GRP_FORCE_ROLLOUT_TP_LIST={raw!r}") from exc
        else:
            tp_list = [
                int(value)
                for value in (config.global_resource_planner.runtime_forced_rollout_tp_list or [])
            ]
        if not tp_list:
            return None
        if any(tp <= 0 for tp in tp_list):
            raise ValueError(f"GRP_FORCE_ROLLOUT_TP_LIST must be positive: {tp_list}")
        forced_train_raw = os.environ.get("GRP_FORCE_TRAIN_GPUS", "").strip()
        configured_forced_train = int(
            config.global_resource_planner.runtime_forced_train_gpus or 0
        )
        rollout_gpus = int(config.rollout_gpus or sum(tp_list))
        if forced_train_raw:
            try:
                forced_train_gpus = int(forced_train_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GRP_FORCE_TRAIN_GPUS={forced_train_raw!r}"
                ) from exc
            rollout_gpus = int(self.n_total_gpus) - forced_train_gpus
        elif configured_forced_train > 0:
            rollout_gpus = int(self.n_total_gpus) - configured_forced_train
        if sum(tp_list) != rollout_gpus:
            raise ValueError(
                "GRP_FORCE_ROLLOUT_TP_LIST must sum to rollout_gpus="
                f"{rollout_gpus}, got {tp_list}"
            )
        return tp_list

    def _forced_train_config(
        self,
        config: AsyncRLConfig,
        current: GlobalResourcePlan,
        *,
        rollout_gpus: int,
    ) -> TrainParallelConfig | None:
        raw = os.environ.get("GRP_FORCE_TRAIN_GPUS", "").strip()
        if raw:
            try:
                train_gpus = int(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid GRP_FORCE_TRAIN_GPUS={raw!r}") from exc
        else:
            train_gpus = int(config.global_resource_planner.runtime_forced_train_gpus or 0)
            if train_gpus == 0:
                return None
        if train_gpus <= 0:
            raise ValueError(f"GRP_FORCE_TRAIN_GPUS must be positive: {train_gpus}")
        if train_gpus + rollout_gpus > self.n_total_gpus:
            raise ValueError(
                "GRP_FORCE_TRAIN_GPUS plus rollout_gpus must fit n_total_gpus="
                f"{self.n_total_gpus}, got train={train_gpus} rollout={rollout_gpus}"
            )
        tp = int(current.train_config.tp)
        ep = int(current.train_config.ep)
        pp = int(current.train_config.pp)
        cp = max(1, int(config.train_cp_size or 1))
        model_parallel = max(1, tp * ep * pp * cp)
        if train_gpus % model_parallel != 0:
            raise ValueError(
                "GRP_FORCE_TRAIN_GPUS must be divisible by train TP*EP*PP*CP="
                f"{model_parallel}, got {train_gpus}"
            )
        return TrainParallelConfig(
            tp=tp,
            ep=ep,
            pp=pp,
            cp=cp,
            dp=train_gpus // model_parallel,
            b_micro=int(current.train_config.b_micro),
            zero_level=int(current.train_config.zero_level),
        )

    def _planning_trigger(
        self,
        step: int,
        metrics: RuntimeResourceMetrics | None,
    ) -> str:
        # Startup planning is handled explicitly by PreflightPlanner. Once the
        # job is running, disabling online replanning must also disable the
        # periodic interval trigger; otherwise a nominally fixed control arm
        # can still receive runtime resource decisions every plan_interval.
        if not self.online_replanning:
            return "interval_skip"
        if step - self._last_replan_step < self.replan_cooldown_steps:
            return "cooldown"
        if step % self.plan_interval == 0:
            return "interval"
        if metrics is None:
            return "interval_skip"

        if metrics.queue_pressure >= self.queue_pressure_threshold:
            return "queue_pressure"
        if metrics.active_rollout_pressure >= self.active_rollout_pressure_threshold:
            return "active_rollout_pressure"

        rejected_delta = metrics.rejected_rollouts - self._last_rejected_rollouts
        if rejected_delta >= self.rejected_rollout_delta_threshold:
            return "rejected_rollout_delta"

        if (
            metrics.train_time_s > 0
            and metrics.rollout_train_ratio >= self.rollout_train_imbalance_threshold
        ):
            return "rollout_train_imbalance"

        return "interval_skip"

    def apply_plan_to_config(
        self,
        plan: GlobalResourcePlan,
        config: AsyncRLConfig,
        *,
        allocation_hosts: Sequence[str] | None = None,
    ) -> None:
        preserve_slots = (int(config.train_gpus) == int(plan.train_gpus)
                          and int(config.rollout_gpus) == int(plan.rollout_gpus))
        previous_instances = list(config.heterogeneous_rollout.instances)
        config.n_total_gpus = plan.n_total_gpus
        config.train_gpus = plan.train_gpus
        config.rollout_gpus = plan.rollout_gpus
        config.train_tp_size = plan.train_config.tp
        config.tp_size = plan.train_config.tp
        config.train_ep_size = plan.train_config.ep
        config.train_pp_size = plan.train_config.pp
        config.train_cp_size = plan.train_config.cp
        # Megatron expresses EP as a subgroup of its runtime DP dimension,
        # whereas the paper's planner factors N as TP*EP*PP*DP.
        config.train_dp_size = plan.train_config.ep * plan.train_config.dp
        config.micro_batch_size = plan.train_config.b_micro
        config.max_concurrent_rollouts = plan.max_concurrent_rollouts

        hetero = config.heterogeneous_rollout
        hetero.enabled = True
        hetero.total_gpus = plan.rollout_gpus
        if allocation_hosts is None:
            host_gpu_pools = self._rollout_host_gpu_pools(config, plan.rollout_gpus)
        else:
            _, rollout_slots = split_gpu_slots(
                allocation_hosts, config.hardware.gpus_per_node, plan.train_gpus,
            )
            host_gpu_pools = list(rollout_slots.items())
        if preserve_slots and previous_instances:
            observed: dict[str, set[int]] = {}
            for instance in previous_instances:
                observed.setdefault(instance.host or hetero.vllm_host, set()).update(
                    int(gpu) for gpu in instance.gpus)
            # On a shared node, observing rollout GPUs [1, 2, 3] does not
            # authorize taking Core GPU 0. Repartition only the current pool.
            if sum(len(gpus) for gpus in observed.values()) == int(plan.rollout_gpus):
                host_gpu_pools = [(host, sorted(gpus)) for host, gpus in observed.items()]
            else:
                preserve_slots = False
        if host_gpu_pools:
            hetero.available_gpus = sorted(
                {gpu for _, gpus in host_gpu_pools for gpu in gpus}
            )
        else:
            available = list(hetero.available_gpus)
            if len(available) < plan.rollout_gpus:
                available = list(
                    range(config.train_gpus, config.train_gpus + plan.rollout_gpus)
                )
                hetero.available_gpus = list(available)
            host_gpu_pools = [(hetero.vllm_host, available)]

        allocations, retained = {}, {}
        if preserve_slots:
            # Retain an unchanged physical service only when the remaining
            # requested degrees still fit. This keeps the leading TP1 service
            # running through [1,1,1] -> [1,2] without fragmenting a TP4 pool.
            sizes = {host: len(gpus) for host, gpus in host_gpu_pools}
            for previous in sorted(previous_instances, key=lambda item: (
                    sizes.get(item.host or hetero.vllm_host, 0), min(item.gpus, default=-1))):
                index = next((index for index, tp in enumerate(plan.rollout_tp_list)
                              if index not in allocations and int(tp) == int(previous.tp)), None)
                if index is None:
                    continue
                host = previous.host or hetero.vllm_host
                trial = [(name, [gpu for gpu in gpus if name != host or gpu not in previous.gpus])
                         for name, gpus in host_gpu_pools]
                check = [(name, list(gpus)) for name, gpus in trial]
                try:
                    for candidate, tp in sorted(enumerate(plan.rollout_tp_list),
                                                key=lambda item: int(item[1]), reverse=True):
                        if candidate not in allocations and candidate != index:
                            self._allocate_rollout_instance(check, int(tp))
                except ValueError:
                    continue
                allocations[index] = (host, list(previous.gpus))
                retained[index] = previous
                host_gpu_pools = trial
        for index, tp in sorted(enumerate(plan.rollout_tp_list),
                                key=lambda item: int(item[1]), reverse=True):
            if index not in allocations:
                allocations[index] = self._allocate_rollout_instance(host_gpu_pools, int(tp))

        used_ids = {instance.instance_id for instance in retained.values()}
        used_ports = {int(instance.port) for instance in retained.values() if instance.port}
        instances = []
        for index, tp in enumerate(plan.rollout_tp_list):
            host, gpus = allocations[index]
            if index in retained:
                instances.append(replace(retained[index], host=host, gpus=gpus))
                continue
            suffix = index
            while f"grp_tp{tp}_{suffix}" in used_ids:
                suffix += 1
            identity = f"grp_tp{tp}_{suffix}"
            port = int(hetero.vllm_base_port) + index
            while port in used_ports:
                port += 1
            if port > 65535:
                raise ValueError("The configured rollout port range is invalid")
            used_ids.add(identity)
            used_ports.add(port)
            instances.append(HeterogeneousInstanceConfig(instance_id=identity, tp=tp,
                gpus=gpus, host=host or hetero.vllm_host, port=port,
                description="managed_by_global_resource_planner"))
        hetero.instances = instances

    @staticmethod
    def _rollout_host_gpu_pools(
        config: AsyncRLConfig,
        rollout_gpus: int,
    ) -> list[tuple[str, list[int]]]:
        hetero = config.heterogeneous_rollout
        by_host: dict[str, set[int]] = {}
        for inst in hetero.instances:
            host = inst.host or hetero.vllm_host
            by_host.setdefault(host, set()).update(int(g) for g in inst.gpus)

        pools: list[tuple[str, list[int]]] = []
        for host, observed in by_host.items():
            if observed:
                capacity = max(max(observed) + 1, len(observed))
            else:
                capacity = 0
            if capacity > 0:
                pools.append((host, list(range(capacity))))

        if pools:
            return pools

        available = list(hetero.available_gpus)
        if available:
            return [(hetero.vllm_host, available)]

        return [(hetero.vllm_host, list(range(max(1, rollout_gpus))))]

    @staticmethod
    def _allocate_rollout_instance(
        pools: list[tuple[str, list[int]]],
        tp: int,
    ) -> tuple[str, list[int]]:
        for idx, (host, gpus) in enumerate(pools):
            if len(gpus) >= tp:
                allocated = gpus[:tp]
                pools[idx] = (host, gpus[tp:])
                return host, allocated
        remaining = {host: list(gpus) for host, gpus in pools}
        raise ValueError(
            f"rollout plan cannot allocate a TP={tp} instance from host pools: "
            f"{remaining}"
        )

    def _remember(self, decision: PlannerDecision) -> PlannerDecision:
        self._last_decision = decision
        return decision

    def _extract_requests(self, batch: list[Any]) -> list[RequestInfo]:
        requests: list[RequestInfo] = []
        for item in batch or []:
            if not isinstance(item, dict):
                continue
            prompt = item.get("input_len", item.get("prompt_length", 0))
            gen = item.get("output_len", item.get("gen_length", 0))
            try:
                prompt_i = int(prompt)
                gen_i = int(gen)
            except (TypeError, ValueError):
                continue
            if prompt_i > 0 or gen_i > 0:
                input_ids = item.get("input_ids")
                training_length = (int(input_ids.shape[-1]) if input_ids is not None else
                    int(item.get("sequence_length", item.get("training_length", 0)) or 0))
                # Older profiles recorded trace totals without the number of
                # generation calls. Keep their original length approximation.
                calls = max(1, int(item.get("n_turns", 1) or 1))
                if calls > 1:
                    # Whole-trace batches hold action tokens summed across turns;
                    # the rollout history remains in per-generate units. The
                    # midpoint follows the context growth of the complete trace.
                    positions = {int(event["turn"]): int(event["token_position"])
                                 for event in item.get("tool_returns", ()) or ()
                                 if isinstance(event, dict)
                                 and "turn" in event and "token_position" in event
                                 and 0 <= int(event["turn"]) < calls - 1}
                    if len(positions) == calls - 1:
                        prompt_i += sum(positions.values()) // calls
                    else:
                        growth = max(0, training_length - prompt_i,
                                     int(item.get("total_output_tokens", 0) or 0))
                        prompt_i += growth * (calls - 1) // (2 * calls)
                    gen_i = math.ceil(gen_i / calls)
                requests.append(
                    RequestInfo(
                        prompt_length=max(1, prompt_i),
                        gen_length=max(1, gen_i),
                        training_length=training_length,
                    )
                )
        return requests

    def _build_current_plan(
        self,
        config: AsyncRLConfig,
        requests: list[RequestInfo],
        *, costs: _MemberPlanningCosts | None = None,
    ) -> GlobalResourcePlan:
        requests = representative_global_batch(requests, int(config.batch_size))
        ep = max(1, int(config.train_ep_size or 1))
        train = TrainParallelConfig(
            tp=config.train_tp_size,
            ep=ep,
            pp=config.train_pp_size,
            dp=max(1, int(config.train_dp_size) // ep),
            cp=config.train_cp_size,
            b_micro=config.micro_batch_size,
        )
        rollout_tp = list(config.heterogeneous_rollout.tp_list)
        if not rollout_tp:
            n_instances = max(1, config.rollout_gpus // max(1, config.vllm_tp_size))
            rollout_tp = [max(1, config.vllm_tp_size)] * n_instances
        rollout = RolloutClusterConfig(tp_list=rollout_tp)
        lengths = self._training_workload_lengths(requests, config.batch_size)
        current_inputs = (train, tuple(rollout_tp), tuple(lengths), config.max_concurrent_rollouts)
        if costs is not None and costs.current_inputs == current_inputs:
            return costs.current_plan
        t_train, train_details = self.evaluator.evaluate_training(
            train, config.batch_size, lengths
        )
        t_rollout, rollout_details = self._layout_supply(rollout_tp, requests, costs)
        t_rollout *= costs.calls_per_trace if costs is not None else self._rollout_calls_per_trace(config)
        plan = GlobalResourcePlan(
            train_config=train,
            rollout_config=rollout,
            t_train=t_train,
            t_rollout=t_rollout,
            t_global=max(t_train, t_rollout),
            n_total_gpus=self.n_total_gpus,
            max_concurrent_rollouts=config.max_concurrent_rollouts,
            metadata={
                "source": "current",
                "training": train_details,
                "rollout": rollout_details,
            },
        )
        if costs is not None:
            costs.current_plan = plan
            costs.current_inputs = current_inputs
        return plan

    def _plan_from_result(
        self,
        result: OptimizationResult,
        config: AsyncRLConfig,
    ) -> GlobalResourcePlan | None:
        if result.train_config is None or result.rollout_config is None:
            return None
        return GlobalResourcePlan(
            train_config=result.train_config,
            rollout_config=result.rollout_config,
            t_train=result.t_train,
            t_rollout=result.t_rollout,
            t_global=result.t_global,
            n_total_gpus=self.n_total_gpus,
            max_concurrent_rollouts=config.max_concurrent_rollouts,
            metadata={
                "source": "optimizer",
                "n_configs_explored": result.n_configs_explored,
                "optimization_time_ms": result.optimization_time_ms,
                "evaluated": result.all_evaluated[-10:],
            },
        )

    def _same_plan(
        self,
        a: GlobalResourcePlan,
        b: GlobalResourcePlan,
    ) -> bool:
        return (
            a.train_config.tp == b.train_config.tp
            and a.train_config.ep == b.train_config.ep
            and a.train_config.pp == b.train_config.pp
            and a.train_config.dp == b.train_config.dp
            and a.train_config.cp == b.train_config.cp
            and a.train_config.b_micro == b.train_config.b_micro
            and sorted(a.rollout_tp_list) == sorted(b.rollout_tp_list)
        )

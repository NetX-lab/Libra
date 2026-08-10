"""Global Resource Planner for cross-stage RL post-training resources."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.model import (
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)
from RL_Framework.infra.cost_model.optimizer import (
    OptimizationResult,
    TwoLevelNestedOptimizer,
)
from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel

logger = logging.getLogger(__name__)


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
        return self.desired_workers != self.current_workers

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "should_reconfigure": self.should_reconfigure,
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
        allowed_train_pp: list[int] | None = None,
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
        elastic_hybrid_max_workers: int = 0,
        elastic_hybrid_borrow_train_rollout_ratio: float = 1.15,
        elastic_hybrid_release_train_rollout_ratio: float = 0.90,
        elastic_hybrid_max_rollout_pressure: float = 0.80,
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
        self.elastic_hybrid_max_workers = max(0, int(elastic_hybrid_max_workers))
        self.elastic_hybrid_borrow_train_rollout_ratio = max(
            0.0, float(elastic_hybrid_borrow_train_rollout_ratio)
        )
        self.elastic_hybrid_release_train_rollout_ratio = max(
            0.0, float(elastic_hybrid_release_train_rollout_ratio)
        )
        if (
            self.elastic_hybrid_release_train_rollout_ratio
            > self.elastic_hybrid_borrow_train_rollout_ratio
        ):
            raise ValueError(
                "elastic hybrid release ratio cannot exceed borrow ratio"
            )
        self.elastic_hybrid_max_rollout_pressure = max(
            0.0, float(elastic_hybrid_max_rollout_pressure)
        )
        self.elastic_hybrid_signal_ttl_steps = max(
            1, int(elastic_hybrid_signal_ttl_steps)
        )
        self.verbose = verbose

        self.optimizer = TwoLevelNestedOptimizer(
            cost_model=evaluator,
            allowed_rollout_tp=self.allowed_rollout_tp,
            require_heterogeneous_rollout_tp=self.require_heterogeneous_rollout_tp,
            verbose=verbose,
        )
        self._history: list[RequestInfo] = []
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
            allowed_train_pp=planner_cfg.allowed_train_pp or None,
            fixed_train_gpus=planner_cfg.fixed_train_gpus,
            initial_allocation_strategy=getattr(
                planner_cfg, "initial_allocation_strategy", "grp"
            ),
            allocation_granularity_gpus=getattr(
                planner_cfg, "allocation_granularity_gpus", 1
            ),
            min_train_gpus=getattr(planner_cfg, "min_train_gpus", 1),
            min_rollout_gpus=getattr(planner_cfg, "min_rollout_gpus", 1),
            micro_batch_sizes=planner_cfg.micro_batch_sizes or None,
            max_history_size=planner_cfg.max_history_size,
            online_replanning=getattr(planner_cfg, "runtime_online_replanning", True),
            replan_cooldown_steps=getattr(
                planner_cfg, "runtime_replan_cooldown_steps", 1
            ),
            queue_pressure_threshold=getattr(
                planner_cfg, "runtime_queue_pressure_threshold", 0.75
            ),
            active_rollout_pressure_threshold=getattr(
                planner_cfg, "runtime_active_rollout_pressure_threshold", 0.85
            ),
            rejected_rollout_delta_threshold=getattr(
                planner_cfg, "runtime_rejected_rollout_delta_threshold", 8
            ),
            rollout_train_imbalance_threshold=getattr(
                planner_cfg, "runtime_rollout_train_imbalance_threshold", 1.25
            ),
            elastic_hybrid_planning_enabled=getattr(
                planner_cfg, "elastic_hybrid_planning_enabled", False
            ),
            elastic_hybrid_max_workers=getattr(
                planner_cfg, "elastic_hybrid_max_workers", 0
            ),
            elastic_hybrid_borrow_train_rollout_ratio=getattr(
                planner_cfg, "elastic_hybrid_borrow_train_rollout_ratio", 1.15
            ),
            elastic_hybrid_release_train_rollout_ratio=getattr(
                planner_cfg, "elastic_hybrid_release_train_rollout_ratio", 0.90
            ),
            elastic_hybrid_max_rollout_pressure=getattr(
                planner_cfg, "elastic_hybrid_max_rollout_pressure", 0.80
            ),
            elastic_hybrid_signal_ttl_steps=getattr(
                planner_cfg, "elastic_hybrid_signal_ttl_steps", 20
            ),
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

    def observe_batch(self, batch: list[Any]) -> int:
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
                )
            )

        current = self._build_current_plan(config, self._history)
        result = self.optimizer.optimize(
            n_total_gpus=self.n_total_gpus,
            requests=list(self._history),
            B_global=config.batch_size,
            allowed_train_tp=self.allowed_train_tp,
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
                getattr(config.global_resource_planner, "rollout_node_tp_pattern", [])
                or []
            ),
            train_cp_size=max(1, int(getattr(config, "train_cp_size", 1) or 1)),
            dp_batch_group_size=max(1, int(getattr(config, "n_samples", 1) or 1)),
        )
        candidate = self._plan_from_result(result, config)
        if candidate is None:
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
                )
            )

        forced_rollout_tp = self._forced_rollout_tp_list(config)
        forced_train = self._forced_train_config(
            config,
            current,
            rollout_gpus=(
                sum(forced_rollout_tp)
                if forced_rollout_tp is not None
                else int(getattr(config, "rollout_gpus", 0) or 0)
            ),
        )
        forced_candidate = forced_rollout_tp is not None or forced_train is not None
        if forced_rollout_tp is not None:
            forced_rollout = RolloutClusterConfig(tp_list=forced_rollout_tp)
            forced_rollout_time, forced_rollout_details = self.evaluator.evaluate_rollout(
                forced_rollout,
                list(self._history),
            )
            forced_train_time = current.t_train
            forced_train_details = current.metadata.get("training", {})
            if forced_train is not None:
                avg_len = (
                    int(np.mean([r.total_length for r in self._history]))
                    if self._history
                    else 1024
                )
                forced_train_time, forced_train_details = self.evaluator.evaluate_training(
                    forced_train,
                    config.batch_size,
                    avg_len,
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
            avg_len = (
                int(np.mean([r.total_length for r in self._history]))
                if self._history
                else 1024
            )
            forced_train_time, forced_train_details = self.evaluator.evaluate_training(
                forced_train,
                config.batch_size,
                avg_len,
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
            candidate=candidate,
            runtime_metrics=runtime_metrics,
        )
        elastic_adjust = bool(
            elastic_signal is not None and elastic_signal.should_adjust
        )
        force_runtime_reconfigure = (
            os.environ.get("GRP_FORCE_RUNTIME_RECONFIGURE", "0") == "1"
        )
        resource_should = (
            (not same_plan or force_runtime_reconfigure)
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
        )
        if should:
            self._active_plan = candidate
            self._last_replan_step = int(step)
        elif self._active_plan is None:
            self._active_plan = current
        self._mark_rejected_trigger_observed(trigger, runtime_metrics)
        return self._remember(decision)

    def _build_elastic_hybrid_signal(
        self,
        *,
        step: int,
        config: AsyncRLConfig,
        current: GlobalResourcePlan,
        candidate: GlobalResourcePlan,
        runtime_metrics: RuntimeResourceMetrics | None,
    ) -> ElasticHybridSignal | None:
        if not self.elastic_hybrid_planning_enabled:
            return None

        planner_cfg = config.global_resource_planner
        replica_size = int(
            getattr(planner_cfg, "elastic_hybrid_replica_size_gpus", 0) or 0
        )
        topology_width = (
            max(1, int(getattr(config, "train_tp_size", 1) or 1))
            * max(1, int(getattr(config, "train_pp_size", 1) or 1))
            * max(1, int(getattr(config, "train_cp_size", 1) or 1))
        )
        if replica_size > 0 and replica_size != topology_width:
            raise ValueError(
                "elastic_hybrid_replica_size_gpus must equal one complete DP "
                f"Replica ({topology_width}); got {replica_size}"
            )
        if replica_size <= 0:
            replica_size = topology_width
        min_rollout_gpus = max(
            0,
            int(getattr(planner_cfg, "elastic_hybrid_min_rollout_gpus", 0) or 0),
        )
        capacity_replicas = max(
            0,
            (int(config.rollout_gpus) - min_rollout_gpus) // replica_size,
        )
        raw_step_stats = (
            runtime_metrics.raw.get("step_stats", {})
            if runtime_metrics is not None
            else {}
        )
        active_workers = max(
            0,
            int(
                raw_step_stats.get(
                    "active_hybrid_replicas",
                    raw_step_stats.get("active_hybrid_workers", 0),
                )
                or 0
            ),
        )
        joining_workers = max(
            int(
                raw_step_stats.get(
                    "joining_hybrid_replicas",
                    raw_step_stats.get("joining_hybrid_workers", 0),
                )
                or 0
            ),
            int(
                raw_step_stats.get(
                    "pending_hybrid_replica_joins",
                    raw_step_stats.get("pending_hybrid_joins", 0),
                )
                or 0
            ),
            0,
        )
        # A worker in ACTIVATING/JOINING already consumes rollout capacity.  It
        # must therefore count as allocated so a GRP release signal can cancel
        # an in-flight non-blocking join immediately.
        current_workers = active_workers + joining_workers

        train_time = (
            float(runtime_metrics.train_time_s)
            if runtime_metrics is not None and runtime_metrics.train_time_s > 0
            else float(candidate.t_train or current.t_train)
        )
        rollout_time = (
            float(runtime_metrics.rollout_time_s)
            if runtime_metrics is not None and runtime_metrics.rollout_time_s > 0
            else float(candidate.t_rollout or current.t_rollout)
        )
        train_rollout_ratio = train_time / max(rollout_time, 1e-6)
        rollout_pressure = (
            max(runtime_metrics.queue_pressure, runtime_metrics.active_rollout_pressure)
            if runtime_metrics is not None
            else 0.0
        )

        planned_extra = max(
            0,
            (int(candidate.train_gpus) - int(config.train_gpus)) // replica_size,
        )
        if capacity_replicas <= 0:
            desired = 0
            reason = "no_complete_replica_capacity"
        elif rollout_pressure >= self.elastic_hybrid_max_rollout_pressure:
            desired = 0
            reason = "protect_rollout_capacity"
        elif planned_extra > 0:
            desired = min(capacity_replicas, planned_extra)
            reason = "candidate_requests_more_training_capacity"
        elif train_rollout_ratio >= self.elastic_hybrid_borrow_train_rollout_ratio:
            desired = capacity_replicas
            reason = "training_bottleneck"
        elif train_rollout_ratio <= self.elastic_hybrid_release_train_rollout_ratio:
            desired = 0
            reason = "rollout_bottleneck"
        else:
            desired = current_workers
            reason = "hysteresis_hold"

        action = (
            "join" if desired > current_workers
            else "release" if desired < current_workers
            else "hold"
        )
        return ElasticHybridSignal(
            step=int(step),
            desired_workers=int(desired),
            current_workers=int(current_workers),
            max_workers=int(capacity_replicas),
            action=action,
            reason=reason,
            train_rollout_ratio=float(train_rollout_ratio),
            rollout_pressure=float(rollout_pressure),
            ttl_steps=int(self.elastic_hybrid_signal_ttl_steps),
            active_workers=int(active_workers),
            joining_workers=int(joining_workers),
            replica_size_gpus=int(replica_size),
        )

    def _mark_rejected_trigger_observed(
        self,
        trigger: str,
        runtime_metrics: RuntimeResourceMetrics | None,
    ) -> None:
        if trigger == "rejected_rollout_delta" and runtime_metrics is not None:
            self._last_rejected_rollouts = runtime_metrics.rejected_rollouts

    def _forced_rollout_tp_list(self, config: AsyncRLConfig) -> list[int] | None:
        raw = os.environ.get("GRP_FORCE_ROLLOUT_TP_LIST", "").strip()
        if not raw:
            return None
        try:
            tp_list = [
                int(item.strip())
                for item in re.split(r"[,;:]", raw)
                if item.strip()
            ]
        except ValueError as exc:
            raise ValueError(f"Invalid GRP_FORCE_ROLLOUT_TP_LIST={raw!r}") from exc
        if not tp_list:
            return None
        if any(tp <= 0 for tp in tp_list):
            raise ValueError(f"GRP_FORCE_ROLLOUT_TP_LIST must be positive: {tp_list}")
        forced_train_raw = os.environ.get("GRP_FORCE_TRAIN_GPUS", "").strip()
        rollout_gpus = int(getattr(config, "rollout_gpus", 0) or sum(tp_list))
        if forced_train_raw:
            try:
                forced_train_gpus = int(forced_train_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GRP_FORCE_TRAIN_GPUS={forced_train_raw!r}"
                ) from exc
            rollout_gpus = int(self.n_total_gpus) - forced_train_gpus
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
        if not raw:
            return None
        try:
            train_gpus = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid GRP_FORCE_TRAIN_GPUS={raw!r}") from exc
        if train_gpus <= 0:
            raise ValueError(f"GRP_FORCE_TRAIN_GPUS must be positive: {train_gpus}")
        if train_gpus + rollout_gpus > self.n_total_gpus:
            raise ValueError(
                "GRP_FORCE_TRAIN_GPUS plus rollout_gpus must fit n_total_gpus="
                f"{self.n_total_gpus}, got train={train_gpus} rollout={rollout_gpus}"
            )
        tp = int(current.train_config.tp)
        pp = int(current.train_config.pp)
        cp = max(1, int(getattr(config, "train_cp_size", 1) or 1))
        model_parallel = max(1, tp * pp * cp)
        if train_gpus % model_parallel != 0:
            raise ValueError(
                "GRP_FORCE_TRAIN_GPUS must be divisible by train TP*PP*CP="
                f"{model_parallel}, got {train_gpus}"
            )
        return TrainParallelConfig(
            tp=tp,
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
        if step - self._last_replan_step < self.replan_cooldown_steps:
            return "cooldown"
        if step % self.plan_interval == 0:
            return "interval"
        if not self.online_replanning or metrics is None:
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
    ) -> None:
        config.n_total_gpus = plan.n_total_gpus
        config.train_gpus = plan.train_gpus
        config.rollout_gpus = plan.rollout_gpus
        config.train_tp_size = plan.train_config.tp
        config.tp_size = plan.train_config.tp
        config.train_pp_size = plan.train_config.pp
        config.train_cp_size = plan.train_config.cp
        config.train_dp_size = plan.train_config.dp
        config.micro_batch_size = plan.train_config.b_micro
        config.max_concurrent_rollouts = plan.max_concurrent_rollouts

        hetero = config.heterogeneous_rollout
        hetero.enabled = True
        hetero.total_gpus = plan.rollout_gpus
        host_gpu_pools = self._rollout_host_gpu_pools(config, plan.rollout_gpus)
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

        instances = []
        for idx, tp in enumerate(plan.rollout_tp_list):
            host, gpus = self._allocate_rollout_instance(host_gpu_pools, tp)
            instances.append(
                HeterogeneousInstanceConfig(
                    instance_id=f"grp_tp{tp}_{idx}",
                    tp=tp,
                    gpus=gpus,
                    host=host or hetero.vllm_host,
                    port=int(hetero.vllm_base_port) + idx,
                    description="managed_by_global_resource_planner",
                )
            )
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
        host = pools[0][0] if pools else ""
        return host, list(range(tp))

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
                requests.append(
                    RequestInfo(
                        prompt_length=max(1, prompt_i),
                        gen_length=max(1, gen_i),
                    )
                )
        return requests

    def _build_current_plan(
        self,
        config: AsyncRLConfig,
        requests: list[RequestInfo],
    ) -> GlobalResourcePlan:
        train = TrainParallelConfig(
            tp=config.train_tp_size,
            pp=config.train_pp_size,
            dp=config.train_dp_size,
            cp=config.train_cp_size,
            b_micro=config.micro_batch_size,
        )
        rollout_tp = list(config.heterogeneous_rollout.tp_list)
        if not rollout_tp:
            n_instances = max(1, config.rollout_gpus // max(1, config.vllm_tp_size))
            rollout_tp = [max(1, config.vllm_tp_size)] * n_instances
        rollout = RolloutClusterConfig(tp_list=rollout_tp)
        avg_len = int(np.mean([r.total_length for r in requests])) if requests else 1024
        t_train, train_details = self.evaluator.evaluate_training(
            train, config.batch_size, avg_len
        )
        t_rollout, rollout_details = self.evaluator.evaluate_rollout(rollout, requests)
        return GlobalResourcePlan(
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

    def _plan_from_result(
        self,
        result: OptimizationResult,
        config: AsyncRLConfig,
    ) -> GlobalResourcePlan | None:
        if result.train_config is None or result.rollout_config is None:
            return None
        max_concurrent = max(
            config.batch_size,
            result.rollout_config.n_instances * config.batch_size,
        )
        return GlobalResourcePlan(
            train_config=result.train_config,
            rollout_config=result.rollout_config,
            t_train=result.t_train,
            t_rollout=result.t_rollout,
            t_global=result.t_global,
            n_total_gpus=self.n_total_gpus,
            max_concurrent_rollouts=max_concurrent,
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
            and a.train_config.pp == b.train_config.pp
            and a.train_config.dp == b.train_config.dp
            and a.train_config.cp == b.train_config.cp
            and a.train_config.b_micro == b.train_config.b_micro
            and sorted(a.rollout_tp_list) == sorted(b.rollout_tp_list)
        )

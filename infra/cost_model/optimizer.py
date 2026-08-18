"""Support code for Optimizer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import (
    CostModel,
    CostModelResult,
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Optimization result implementation."""

    train_config: TrainParallelConfig | None = None
    rollout_config: RolloutClusterConfig | None = None


    t_train: float = float("inf")
    t_rollout: float = float("inf")
    t_global: float = float("inf")


    n_configs_explored: int = 0
    n_configs_pruned_oom: int = 0
    n_configs_pruned_early_stop: int = 0
    optimization_time_ms: float = 0.0
    all_evaluated: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Summary."""
        lines = [
            "=" * 60,
            "Two-level nested optimizer result",
            "=" * 60,
        ]
        if self.train_config is not None:
            tc = self.train_config
            lines.append(
                f"Training:  TP={tc.tp}, PP={tc.pp}, DP={tc.dp}, "
                f"b_micro={tc.b_micro}  ({tc.n_gpus} GPUs)"
            )
        if self.rollout_config is not None:
            rc = self.rollout_config
            lines.append(
                f"Rollout:   {rc.tp_list}  ({rc.n_gpus} GPUs, "
                f"{rc.n_instances} instances)"
            )
        lines.extend([
            f"T_train:   {self.t_train:.4f}s",
            f"T_rollout: {self.t_rollout:.4f}s",
            f"T_global:  {self.t_global:.4f}s  (bottleneck: "
            f"{'train' if self.t_train >= self.t_rollout else 'rollout'})",
            "-" * 60,
            f"Search statistics: explored={self.n_configs_explored}, "
            f"OOM_pruned={self.n_configs_pruned_oom}, "
            f"early_stop_pruned={self.n_configs_pruned_early_stop}",
            f"Optimization time: {self.optimization_time_ms:.2f}ms",
            "=" * 60,
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def generate_training_configs(
    n_total_gpus: int,
    max_tp: int = 8,
    max_pp: int = 8,
    allowed_tp: list[int] | None = None,
    allowed_pp: list[int] | None = None,
    micro_batch_sizes: list[int] | None = None,
) -> list[TrainParallelConfig]:
    """Generate training configs."""
    if allowed_tp is None:
        allowed_tp = [2 ** i for i in range(int(np.log2(max_tp)) + 1)]
        allowed_tp = [t for t in allowed_tp if t <= n_total_gpus]
    if allowed_pp is None:
        allowed_pp = [2 ** i for i in range(int(np.log2(max_pp)) + 1)]
        allowed_pp = [p for p in allowed_pp if p <= n_total_gpus]
    if micro_batch_sizes is None:
        micro_batch_sizes = [1, 2, 4, 8]

    configs = []
    for tp in allowed_tp:
        for pp in allowed_pp:
            if tp * pp > n_total_gpus:
                continue

            remaining = n_total_gpus // (tp * pp)
            if remaining < 1:
                continue

            for dp in range(1, remaining + 1):
                if tp * pp * dp > n_total_gpus:
                    break
                for b_micro in micro_batch_sizes:
                    configs.append(TrainParallelConfig(
                        tp=tp, pp=pp, dp=dp, b_micro=b_micro,
                    ))
    return configs


def generate_rollout_configs(
    n_gpus: int,
    allowed_tp_sizes: list[int] | None = None,
    max_instances: int = 32,
) -> list[RolloutClusterConfig]:
    """Generate rollout configs."""
    if n_gpus <= 0:
        return []
    if allowed_tp_sizes is None:
        allowed_tp_sizes = [1, 2, 4, 8]

    allowed_tp_sizes = sorted([t for t in allowed_tp_sizes if t <= n_gpus], reverse=True)
    if not allowed_tp_sizes:
        return []

    results: list[RolloutClusterConfig] = []

    def _partition(remaining: int, max_val: int, current: list[int]):
        """Partition."""
        if remaining == 0:
            results.append(RolloutClusterConfig(tp_list=list(current)))
            return
        if len(current) >= max_instances:
            return
        for tp in allowed_tp_sizes:
            if tp > remaining or tp > max_val:
                continue
            current.append(tp)
            _partition(remaining - tp, tp, current)
            current.pop()

    _partition(n_gpus, max(allowed_tp_sizes), [])


    if len(results) > 500:
        logger.warning(
            f"Rollout configuration space is too large ({len(results)}); truncating to 500"
        )
        results = results[:500]

    return results


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TwoLevelNestedOptimizer:
    """Two level nested optimizer implementation."""

    def __init__(
        self,
        cost_model: CostModel,
        allowed_rollout_tp: list[int] | None = None,
        max_rollout_instances: int = 32,
        require_heterogeneous_rollout_tp: bool = False,
        verbose: bool = False,
    ):
        self.cost_model = cost_model
        self.allowed_rollout_tp = allowed_rollout_tp or [1, 2, 4, 8]
        self.max_rollout_instances = max_rollout_instances
        self.require_heterogeneous_rollout_tp = require_heterogeneous_rollout_tp
        self.verbose = verbose

    def optimize(
        self,
        n_total_gpus: int,
        requests: list[RequestInfo],
        B_global: int = 32,
        allowed_train_tp: list[int] | None = None,
        allowed_train_pp: list[int] | None = None,
        micro_batch_sizes: list[int] | None = None,
        fixed_train_gpus: int = 0,
    ) -> OptimizationResult:
        """Optimize."""
        start_time = time.perf_counter()


        if requests:
            L_avg = int(np.mean([r.total_length for r in requests]))
        else:
            L_avg = 1024


        result = OptimizationResult()
        min_global_cost = float("inf")
        best_train_config = None
        best_rollout_config = None
        best_t_train = float("inf")
        best_t_rollout = float("inf")


        train_configs = generate_training_configs(
            n_total_gpus=n_total_gpus,
            allowed_tp=allowed_train_tp,
            allowed_pp=allowed_train_pp,
            micro_batch_sizes=micro_batch_sizes,
        )
        if fixed_train_gpus > 0:
            train_configs = [
                tc for tc in train_configs
                if tc.n_gpus == fixed_train_gpus
            ]

        if self.verbose:
            logger.info(f"Outer search space: {len(train_configs)} training configurations")

        for tc in train_configs:
            result.n_configs_explored += 1


            if self.cost_model.check_train_oom(tc, B_global, L_avg):
                result.n_configs_pruned_oom += 1
                continue


            t_train, _ = self.cost_model.evaluate_training(tc, B_global, L_avg)


            if t_train >= min_global_cost:
                result.n_configs_pruned_early_stop += 1
                continue


            n_rollout_gpus = n_total_gpus - tc.n_gpus
            if n_rollout_gpus <= 0:
                continue

            rollout_configs = generate_rollout_configs(
                n_gpus=n_rollout_gpus,
                allowed_tp_sizes=self.allowed_rollout_tp,
                max_instances=self.max_rollout_instances,
            )
            if self.require_heterogeneous_rollout_tp:
                rollout_configs = [
                    rc for rc in rollout_configs
                    if len(set(rc.tp_list)) > 1
                ]

            min_t_rollout = float("inf")
            best_rc_for_tc = None

            for rc in rollout_configs:
                t_rollout, _ = self.cost_model.evaluate_rollout(rc, requests)

                if t_rollout < min_t_rollout:
                    min_t_rollout = t_rollout
                    best_rc_for_tc = rc

            if best_rc_for_tc is None:
                continue


            current_global_cost = max(t_train, min_t_rollout)

            if self.verbose:
                result.all_evaluated.append({
                    "train": {"tp": tc.tp, "pp": tc.pp, "dp": tc.dp, "b_micro": tc.b_micro},
                    "rollout": best_rc_for_tc.tp_list if best_rc_for_tc else [],
                    "t_train": t_train,
                    "t_rollout": min_t_rollout,
                    "t_global": current_global_cost,
                })

            if current_global_cost < min_global_cost:
                min_global_cost = current_global_cost
                best_train_config = tc
                best_rollout_config = best_rc_for_tc
                best_t_train = t_train
                best_t_rollout = min_t_rollout


        result.train_config = best_train_config
        result.rollout_config = best_rollout_config
        result.t_train = best_t_train
        result.t_rollout = best_t_rollout
        result.t_global = min_global_cost
        result.optimization_time_ms = (time.perf_counter() - start_time) * 1000

        if self.verbose:
            logger.info(result.summary())

        return result

    def optimize_rollout_only(
        self,
        n_rollout_gpus: int,
        requests: list[RequestInfo],
    ) -> tuple[RolloutClusterConfig | None, float, dict]:
        """Optimize rollout only."""
        rollout_configs = generate_rollout_configs(
            n_gpus=n_rollout_gpus,
            allowed_tp_sizes=self.allowed_rollout_tp,
            max_instances=self.max_rollout_instances,
        )
        if self.require_heterogeneous_rollout_tp:
            rollout_configs = [
                rc for rc in rollout_configs
                if len(set(rc.tp_list)) > 1
            ]

        best_config = None
        best_makespan = float("inf")
        best_details = {}

        for rc in rollout_configs:
            makespan, details = self.cost_model.evaluate_rollout(rc, requests)
            if makespan < best_makespan:
                best_makespan = makespan
                best_config = rc
                best_details = details

        return best_config, best_makespan, best_details

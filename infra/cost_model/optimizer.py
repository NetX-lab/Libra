"""Support code for Optimizer."""

from __future__ import annotations

import logging
import math
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
                f"Training:  TP={tc.tp}, EP={tc.ep}, PP={tc.pp}, DP={tc.dp}, "
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
    allowed_ep: list[int] | None = None,
    is_moe: bool = False,
    num_experts: int = 1,
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
    if allowed_ep is None:
        allowed_ep = [1]
        if is_moe:
            allowed_ep = [
                value
                for value in (1, 2, 4, 8)
                if value <= n_total_gpus and num_experts % value == 0
            ]

    configs = []
    for tp in allowed_tp:
        for ep in allowed_ep:
            if not is_moe and ep != 1:
                continue
            if num_experts % ep:
                continue
            for pp in allowed_pp:
                if tp * ep * pp > n_total_gpus:
                    continue

                remaining = n_total_gpus // (tp * ep * pp)
                if remaining < 1:
                    continue

                for dp in range(1, remaining + 1):
                    if tp * ep * pp * dp > n_total_gpus:
                        break
                    for b_micro in micro_batch_sizes:
                        configs.append(TrainParallelConfig(
                            tp=tp, ep=ep, pp=pp, dp=dp, b_micro=b_micro,
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
    """Paper-style hierarchical training search plus rollout dynamic program."""

    def __init__(
        self,
        cost_model: CostModel,
        allowed_rollout_tp: list[int] | None = None,
        max_rollout_instances: int = 32,
        require_heterogeneous_rollout_tp: bool = False,
        train_comm_compute_ratio_threshold: float = float("inf"),
        train_ep_comm_compute_ratio_threshold: float = float("inf"),
        train_pipeline_bubble_ratio_threshold: float = 0.30,
        verbose: bool = False,
    ):
        self.cost_model = cost_model
        self.allowed_rollout_tp = sorted(set(allowed_rollout_tp or [1, 2, 4, 8]))
        self.max_rollout_instances = max_rollout_instances
        self.require_heterogeneous_rollout_tp = require_heterogeneous_rollout_tp
        self.train_comm_compute_ratio_threshold = float(train_comm_compute_ratio_threshold)
        self.train_ep_comm_compute_ratio_threshold = float(
            train_ep_comm_compute_ratio_threshold
        )
        self.train_pipeline_bubble_ratio_threshold = float(
            train_pipeline_bubble_ratio_threshold
        )
        self.verbose = verbose

    def _model_arch(self) -> Any:
        if hasattr(self.cost_model, "ma"):
            return self.cost_model.ma
        if hasattr(self.cost_model, "config"):
            return self.cost_model.config.model_arch
        if hasattr(self.cost_model, "analytic"):
            return self.cost_model.analytic.ma
        return None

    def _training_model(self) -> Any:
        if hasattr(self.cost_model, "training_model"):
            return self.cost_model.training_model
        analytic = getattr(self.cost_model, "analytic", None)
        return getattr(analytic, "training_model", None)

    def _pruning_metrics(
        self,
        config: TrainParallelConfig,
        B_global: int,
        sequence_length: int,
    ) -> dict[str, float]:
        model = self._training_model()
        n_micro = max(
            1,
            B_global // max(1, config.ep * config.dp * config.b_micro),
        )
        bubble_ratio = (config.pp - 1) / max(config.pp + n_micro - 1, 1)
        if model is None:
            return {
                "tp_comm_compute_ratio": 0.0,
                "ep_comm_compute_ratio": 0.0,
                "pipeline_bubble_ratio": bubble_ratio,
            }
        compute = (
            model.compute_pure_attn_time(sequence_length, config.b_micro, config.tp)
            + model.compute_pure_mlp_time(sequence_length, config.b_micro, config.tp)
        )
        tp_comm = model.compute_tp_comm(config.tp, config.b_micro, sequence_length)
        ep_comm = model.compute_ep_comm(config, sequence_length)
        return {
            "tp_comm_compute_ratio": tp_comm / max(compute, 1e-12),
            "ep_comm_compute_ratio": ep_comm / max(compute, 1e-12),
            "pipeline_bubble_ratio": bubble_ratio,
        }

    def decision_tree_search(
        self,
        n_train_gpus: int,
        *,
        B_global: int,
        sequence_lengths: list[int],
        allowed_train_tp: list[int] | None = None,
        allowed_train_ep: list[int] | None = None,
        allowed_train_pp: list[int] | None = None,
        micro_batch_sizes: list[int] | None = None,
        train_cp_size: int = 1,
        dp_batch_group_size: int = 1,
        result: OptimizationResult | None = None,
    ) -> list[TrainParallelConfig]:
        """Enumerate the TP->EP->PP->DP tree and prune invalid branches."""
        arch = self._model_arch()
        is_moe = bool(getattr(arch, "is_moe", False))
        num_experts = max(1, int(getattr(arch, "num_experts", 1)))
        cp = max(1, int(train_cp_size))
        max_tp = min(8, n_train_gpus)
        tp_values = sorted(set(allowed_train_tp or [1, 2, 4, 8]))
        tp_values = [tp for tp in tp_values if tp <= max_tp]
        if allowed_train_ep:
            ep_values = sorted(set(int(ep) for ep in allowed_train_ep))
        elif is_moe:
            ep_values = [
                ep for ep in (1, 2, 4, 8) if ep <= n_train_gpus and num_experts % ep == 0
            ]
        else:
            ep_values = [1]
        batch_sizes = sorted(set(micro_batch_sizes or [1, 2, 4, 8]))
        max_length = max(sequence_lengths or [1024])
        configs: list[TrainParallelConfig] = []

        for tp in tp_values:
            if n_train_gpus % (tp * cp):
                continue
            for ep in ep_values:
                if (not is_moe and ep != 1) or num_experts % ep:
                    continue
                base = tp * ep * cp
                if base > n_train_gpus or n_train_gpus % base:
                    continue
                if allowed_train_pp:
                    pp_values = sorted(set(int(pp) for pp in allowed_train_pp))
                else:
                    pp_values = list(range(1, n_train_gpus // base + 1))
                for pp in pp_values:
                    denominator = base * pp
                    if denominator > n_train_gpus or n_train_gpus % denominator:
                        continue
                    dp = n_train_gpus // denominator
                    for b_micro in batch_sizes:
                        candidate = TrainParallelConfig(
                            tp=tp,
                            ep=ep,
                            pp=pp,
                            dp=dp,
                            cp=cp,
                            b_micro=b_micro,
                        )
                        runtime_dp = ep * dp
                        if B_global % max(1, runtime_dp * b_micro):
                            continue
                        if (B_global // runtime_dp) % max(
                            1, int(dp_batch_group_size)
                        ):
                            continue
                        if result is not None:
                            result.n_configs_explored += 1
                        if self.cost_model.check_train_oom(
                            candidate, B_global, max_length
                        ):
                            if result is not None:
                                result.n_configs_pruned_oom += 1
                            continue
                        metrics = self._pruning_metrics(
                            candidate, B_global, max_length
                        )
                        if (
                            metrics["tp_comm_compute_ratio"]
                            > self.train_comm_compute_ratio_threshold
                            or metrics["ep_comm_compute_ratio"]
                            > self.train_ep_comm_compute_ratio_threshold
                            or metrics["pipeline_bubble_ratio"]
                            > self.train_pipeline_bubble_ratio_threshold
                        ):
                            if result is not None:
                                result.n_configs_pruned_early_stop += 1
                            continue
                        configs.append(candidate)
        return configs

    def _instance_cost(
        self,
        tp: int,
        requests: list[RequestInfo],
        *,
        num_requests: int | None = None,
        total_prompt_tokens: int = 0,
        total_gen_tokens: int = 0,
        max_seq: int = 0,
    ) -> tuple[float, dict]:
        aggregate = getattr(self.cost_model, "evaluate_rollout_segment", None)
        if (
            aggregate is not None
            and num_requests is not None
        ):
            return aggregate(
                tp,
                num_requests=num_requests,
                total_prompt_tokens=total_prompt_tokens,
                total_gen_tokens=total_gen_tokens,
                max_seq=max_seq,
            )
        evaluator = getattr(self.cost_model, "evaluate_rollout_instance", None)
        if evaluator is not None:
            return evaluator(tp, requests)
        return self.cost_model.evaluate_rollout(
            RolloutClusterConfig(tp_list=[tp]), requests
        )

    def rollout_dp(
        self,
        n_rollout_gpus: int,
        requests: list[RequestInfo],
    ) -> tuple[RolloutClusterConfig | None, float, dict]:
        """Solve Algorithm 1's sorted contiguous rollout partition exactly."""
        if n_rollout_gpus <= 0:
            return None, float("inf"), {"reason": "no_rollout_gpus"}
        sorted_requests = sorted(requests, key=lambda request: request.total_length)
        length_count = len(sorted_requests)
        tp_values = [tp for tp in self.allowed_rollout_tp if tp <= n_rollout_gpus]
        if not tp_values:
            return None, float("inf"), {"reason": "no_feasible_tp"}

        # Each dp[g][i] entry is a small map keyed by the TP-set bitmask. The
        # mask is only needed for the optional heterogeneous-cluster constraint;
        # without that constraint this reduces to the paper's scalar dp[g][i].
        dp: list[list[dict[tuple[int, int], float]]] = [
            [dict() for _ in range(length_count + 1)]
            for _ in range(n_rollout_gpus + 1)
        ]
        back: dict[
            tuple[int, int, int, int], tuple[int, int, int, int, int, int]
        ] = {}
        dp[0][0][(0, 0)] = 0.0
        segment_cache: dict[tuple[int, int, int], tuple[float, dict]] = {}
        prompt_prefix = [0]
        generation_prefix = [0]
        for request in sorted_requests:
            prompt_prefix.append(prompt_prefix[-1] + request.prompt_length)
            generation_prefix.append(generation_prefix[-1] + request.gen_length)

        for gpus in range(1, n_rollout_gpus + 1):
            for served in range(length_count + 1):
                for tp_index, tp in enumerate(tp_values):
                    if tp > gpus:
                        continue
                    for previous_served in range(served + 1):
                        segment_key = (tp, previous_served, served)
                        if segment_key not in segment_cache:
                            use_aggregate = (
                                hasattr(self.cost_model, "evaluate_rollout_segment")
                            )
                            segment_cache[segment_key] = self._instance_cost(
                                tp,
                                (
                                    []
                                    if use_aggregate
                                    else sorted_requests[previous_served:served]
                                ),
                                num_requests=served - previous_served,
                                total_prompt_tokens=(
                                    prompt_prefix[served] - prompt_prefix[previous_served]
                                ),
                                total_gen_tokens=(
                                    generation_prefix[served]
                                    - generation_prefix[previous_served]
                                ),
                                max_seq=(
                                    sorted_requests[served - 1].total_length
                                    if served > previous_served
                                    else 0
                                ),
                            )
                        instance_time = segment_cache[segment_key][0]
                        if not math.isfinite(instance_time):
                            continue
                        for (previous_mask, previous_count), previous_time in dp[gpus - tp][
                            previous_served
                        ].items():
                            instance_count = previous_count + 1
                            if instance_count > self.max_rollout_instances:
                                continue
                            mask = previous_mask | (1 << tp_index)
                            candidate = max(previous_time, instance_time)
                            state = (mask, instance_count)
                            old = dp[gpus][served].get(state, float("inf"))
                            if candidate < old:
                                dp[gpus][served][state] = candidate
                                back[(gpus, served, mask, instance_count)] = (
                                    gpus - tp,
                                    previous_served,
                                    previous_mask,
                                    previous_count,
                                    tp,
                                    previous_served,
                                )

        terminal = dp[n_rollout_gpus][length_count]
        states = [
            state
            for state in terminal
            for mask, _ in [state]
            if not self.require_heterogeneous_rollout_tp or mask.bit_count() > 1
        ]
        if not states:
            return None, float("inf"), {"reason": "no_feasible_partition"}
        best_mask, best_count = min(
            states, key=lambda state: (terminal[state], state[1], state[0])
        )
        makespan = terminal[(best_mask, best_count)]
        cursor = (n_rollout_gpus, length_count, best_mask, best_count)
        recovered: list[tuple[int, int, int]] = []
        while cursor[0] > 0:
            (
                previous_gpus,
                previous_served,
                previous_mask,
                previous_count,
                tp,
                start,
            ) = back[cursor]
            recovered.append((tp, start, cursor[1]))
            cursor = (
                previous_gpus,
                previous_served,
                previous_mask,
                previous_count,
            )
        recovered.reverse()
        config = RolloutClusterConfig(
            tp_list=sorted((tp for tp, _, _ in recovered), reverse=True)
        )
        return config, makespan, {
            "algorithm": "rollout_dp",
            "sorted_lengths": [request.total_length for request in sorted_requests],
            "segments": [
                {
                    "tp": tp,
                    "start": start,
                    "end": end,
                    "num_requests": end - start,
                    "time": segment_cache[(tp, start, end)][0],
                }
                for tp, start, end in recovered
            ],
            "states": sum(len(masks) for row in dp for masks in row),
        }

    def evaluate_fixed_rollout_config(
        self,
        config: RolloutClusterConfig,
        requests: list[RequestInfo],
    ) -> tuple[float, dict]:
        """Optimally assign contiguous length ranges to a fixed TP multiset."""
        tp_list = sorted(int(tp) for tp in config.tp_list)
        if not tp_list:
            return float("inf"), {"reason": "empty_rollout_config"}
        ordered = sorted(requests, key=lambda request: request.total_length)
        request_count = len(ordered)
        prompt_prefix = [0]
        generation_prefix = [0]
        for request in ordered:
            prompt_prefix.append(prompt_prefix[-1] + request.prompt_length)
            generation_prefix.append(generation_prefix[-1] + request.gen_length)
        dp = [[float("inf")] * (request_count + 1) for _ in range(len(tp_list) + 1)]
        back: dict[tuple[int, int], int] = {}
        cache: dict[tuple[int, int, int], float] = {}
        dp[0][0] = 0.0
        for instance_count, tp in enumerate(tp_list, start=1):
            for served in range(request_count + 1):
                for previous_served in range(served + 1):
                    key = (tp, previous_served, served)
                    if key not in cache:
                        use_aggregate = hasattr(
                            self.cost_model, "evaluate_rollout_segment"
                        )
                        cache[key] = self._instance_cost(
                            tp,
                            (
                                []
                                if use_aggregate
                                else ordered[previous_served:served]
                            ),
                            num_requests=served - previous_served,
                            total_prompt_tokens=(
                                prompt_prefix[served] - prompt_prefix[previous_served]
                            ),
                            total_gen_tokens=(
                                generation_prefix[served]
                                - generation_prefix[previous_served]
                            ),
                            max_seq=(
                                ordered[served - 1].total_length
                                if served > previous_served
                                else 0
                            ),
                        )[0]
                    candidate = max(
                        dp[instance_count - 1][previous_served], cache[key]
                    )
                    if candidate < dp[instance_count][served]:
                        dp[instance_count][served] = candidate
                        back[(instance_count, served)] = previous_served
        makespan = dp[len(tp_list)][request_count]
        if not math.isfinite(makespan):
            return makespan, {"reason": "no_feasible_fixed_assignment"}
        segments = []
        served = request_count
        for instance_count in range(len(tp_list), 0, -1):
            previous_served = back[(instance_count, served)]
            tp = tp_list[instance_count - 1]
            segments.append(
                {
                    "tp": tp,
                    "start": previous_served,
                    "end": served,
                    "num_requests": served - previous_served,
                    "time": cache[(tp, previous_served, served)],
                }
            )
            served = previous_served
        segments.reverse()
        return makespan, {
            "algorithm": "fixed_rollout_dp",
            "sorted_lengths": [request.total_length for request in ordered],
            "segments": segments,
        }

    def optimize(
        self,
        n_total_gpus: int,
        requests: list[RequestInfo],
        B_global: int = 32,
        allowed_train_tp: list[int] | None = None,
        allowed_train_ep: list[int] | None = None,
        allowed_train_pp: list[int] | None = None,
        micro_batch_sizes: list[int] | None = None,
        fixed_train_gpus: int = 0,
        allocation_granularity_gpus: int = 1,
        min_train_gpus: int = 1,
        min_rollout_gpus: int = 1,
        rollout_node_tp_pattern: list[int] | None = None,
        train_cp_size: int = 1,
        dp_batch_group_size: int = 1,
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        lengths = [request.total_length for request in requests] or [1024]
        result = OptimizationResult()
        granularity = max(1, int(allocation_granularity_gpus))
        train_budgets = range(
            max(1, int(min_train_gpus)),
            n_total_gpus - max(1, int(min_rollout_gpus)) + 1,
        )
        if fixed_train_gpus > 0:
            train_budgets = [int(fixed_train_gpus)]

        rollout_memo: dict[int, tuple[RolloutClusterConfig | None, float, dict]] = {}
        for n_train in train_budgets:
            if n_train % granularity:
                continue
            candidates = self.decision_tree_search(
                n_train,
                B_global=B_global,
                sequence_lengths=lengths,
                allowed_train_tp=allowed_train_tp,
                allowed_train_ep=allowed_train_ep,
                allowed_train_pp=allowed_train_pp,
                micro_batch_sizes=micro_batch_sizes,
                train_cp_size=train_cp_size,
                dp_batch_group_size=dp_batch_group_size,
                result=result,
            )
            if not candidates:
                continue
            best_train = None
            best_train_time = float("inf")
            best_train_details: dict[str, Any] = {}
            for candidate in candidates:
                train_time, train_details = self.cost_model.evaluate_training(
                    candidate, B_global, lengths
                )
                if train_time < best_train_time:
                    best_train = candidate
                    best_train_time = train_time
                    best_train_details = train_details
            if best_train is None or best_train_time >= result.t_global:
                result.n_configs_pruned_early_stop += 1
                continue

            n_rollout = n_total_gpus - n_train
            pattern = [int(tp) for tp in (rollout_node_tp_pattern or [])]
            if pattern:
                pattern_gpus = sum(pattern)
                if not pattern_gpus or n_rollout % pattern_gpus:
                    continue
                rollout_config = RolloutClusterConfig(
                    tp_list=pattern * (n_rollout // pattern_gpus)
                )
                rollout_time, rollout_details = self.evaluate_fixed_rollout_config(
                    rollout_config, requests
                )
            else:
                if n_rollout not in rollout_memo:
                    rollout_memo[n_rollout] = self.rollout_dp(n_rollout, requests)
                rollout_config, rollout_time, rollout_details = rollout_memo[n_rollout]
            if (
                rollout_config is not None
                and getattr(self.cost_model, "rollout_backend", "analytic")
                != "analytic"
            ):
                rollout_time, external_details = self.cost_model.evaluate_rollout(
                    rollout_config, requests
                )
                rollout_details = {
                    **rollout_details,
                    "external_validation": external_details,
                }
            if rollout_config is None or not math.isfinite(rollout_time):
                continue
            global_time = max(best_train_time, rollout_time)
            if self.verbose:
                result.all_evaluated.append({
                    "train": {
                        "tp": best_train.tp,
                        "ep": best_train.ep,
                        "pp": best_train.pp,
                        "cp": best_train.cp,
                        "dp": best_train.dp,
                        "b_micro": best_train.b_micro,
                    },
                    "rollout": rollout_config.tp_list,
                    "t_train": best_train_time,
                    "t_rollout": rollout_time,
                    "t_global": global_time,
                    "training_details": best_train_details,
                    "rollout_details": rollout_details,
                })
            if global_time < result.t_global:
                result.train_config = best_train
                result.rollout_config = rollout_config
                result.t_train = best_train_time
                result.t_rollout = rollout_time
                result.t_global = global_time

        result.optimization_time_ms = (time.perf_counter() - start_time) * 1000
        if self.verbose:
            logger.info(result.summary())
        return result

    def optimize_rollout_only(
        self,
        n_rollout_gpus: int,
        requests: list[RequestInfo],
    ) -> tuple[RolloutClusterConfig | None, float, dict]:
        return self.rollout_dp(n_rollout_gpus, requests)

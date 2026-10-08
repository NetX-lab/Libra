"""Paper-faithful GRP planning: a training decision tree and rollout DP.

The legacy optimizer enumerates whole training configurations and rollout
partitions.  This module is intentionally separate so the experiment can
compare that baseline with the algorithm described in resource_planner.tex.
Every retained or pruned tree node and every DP summary is exported in the
result, making the execution path auditable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import inf
from typing import Any

import numpy as np

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.model import (
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)
from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel


@dataclass
class PaperPlan:
    train: TrainParallelConfig
    rollout: RolloutClusterConfig
    t_train: float
    t_rollout: float
    n_total_gpus: int
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def train_gpus(self) -> int:
        return self.train.n_gpus

    @property
    def rollout_gpus(self) -> int:
        return self.rollout.n_gpus

    @property
    def t_global(self) -> float:
        return max(self.t_train, self.t_rollout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "paper_tree_dp_v1",
            "n_total_gpus": self.n_total_gpus,
            "train": {**_train_dict(self.train), "n_gpus": self.train_gpus},
            "rollout": {
                "tp_list": list(self.rollout.tp_list),
                "n_gpus": self.rollout_gpus,
                "n_instances": self.rollout.n_instances,
            },
            "t_train": self.t_train,
            "t_rollout": self.t_rollout,
            "t_global": self.t_global,
            "trace": self.trace,
        }


def _train_dict(config: TrainParallelConfig) -> dict[str, int]:
    return {
        "tp": int(config.tp), "pp": int(config.pp), "cp": int(config.cp),
        "dp": int(config.dp), "b_micro": int(config.b_micro),
    }


class PaperHierarchicalPlanner:
    """Algorithm 1 from the paper, for a dense-model NPU execution path."""

    def __init__(
        self,
        config: AsyncRLConfig,
        n_total_gpus: int,
        *,
        tp_comm_ratio_limit: float = 0.50,
        pp_bubble_ratio_limit: float = 0.30,
        max_dp_requests: int = 64,
    ):
        self.config = config
        self.n_total_gpus = int(n_total_gpus)
        if self.n_total_gpus < 2:
            raise ValueError("at least two NPUs are required")
        self.evaluator = HybridSimulatorCostModel(config)
        planner = config.global_resource_planner
        self.train_tp = self._allowed(planner.allowed_train_tp, [1, 2, 4, 8])
        self.train_pp = self._allowed(planner.allowed_train_pp, [])
        self.micro_batches = self._allowed(
            planner.micro_batch_sizes, [config.micro_batch_size]
        )
        self.rollout_tp = self._allowed(planner.allowed_rollout_tp, [1, 2, 4, 8])
        self.tp_comm_ratio_limit = float(tp_comm_ratio_limit)
        self.pp_bubble_ratio_limit = float(pp_bubble_ratio_limit)
        self.max_dp_requests = max(1, int(max_dp_requests))
        self.trace: dict[str, Any] = {
            "tree_nodes": [],
            "prune_counts": Counter(),
            "rollout_dp": {},
            "budget_evaluations": [],
        }

    @staticmethod
    def _allowed(values: list[int] | None, default: list[int]) -> list[int]:
        values = sorted({int(value) for value in values or [] if int(value) > 0})
        return values or default

    def plan(self, history: list[dict[str, Any]]) -> PaperPlan:
        requests = self._requests(history)
        if not requests:
            raise ValueError("paper planner requires non-empty length history")
        dp_requests = self._stratified_sample(requests)
        length = int(np.mean([request.total_length for request in requests]))
        rollout_cache: dict[int, tuple[RolloutClusterConfig, float]] = {}
        selected: PaperPlan | None = None

        # Root level: n_train from 1 to N-1, exactly as Algorithm 1.
        for n_train in range(1, self.n_total_gpus):
            leaves = self._decision_tree(n_train, length)
            if not leaves:
                continue
            train, t_train, details = min(leaves, key=lambda leaf: leaf[1])
            n_rollout = self.n_total_gpus - n_train
            if n_rollout not in rollout_cache:
                rollout_cache[n_rollout] = self._rollout_dp(n_rollout, dp_requests)
            rollout, t_rollout = rollout_cache[n_rollout]
            evaluation = {
                "n_train": n_train,
                "n_rollout": n_rollout,
                "surviving_train_leaves": len(leaves),
                "best_train": {**_train_dict(train), "n_gpus": train.n_gpus},
                "t_train": t_train,
                "training_details": details,
                "rollout_tp_list": list(rollout.tp_list),
                "t_rollout": t_rollout,
                "t_global": max(t_train, t_rollout),
            }
            self.trace["budget_evaluations"].append(evaluation)
            candidate = PaperPlan(train, rollout, t_train, t_rollout, self.n_total_gpus)
            if selected is None or candidate.t_global < selected.t_global:
                selected = candidate

        if selected is None:
            raise RuntimeError("paper decision tree found no feasible placement")
        self.trace["prune_counts"] = dict(self.trace["prune_counts"])
        self.trace["request_count"] = len(requests)
        self.trace["dp_request_count"] = len(dp_requests)
        self.trace["selected"] = {
            "train": {**_train_dict(selected.train), "n_gpus": selected.train_gpus},
            "rollout_tp_list": list(selected.rollout.tp_list),
            "t_global": selected.t_global,
        }
        selected.trace = self.trace
        return selected

    def _decision_tree(
        self, n_train: int, length: int
    ) -> list[tuple[TrainParallelConfig, float, dict[str, Any]]]:
        cp = max(1, int(self.config.train_cp_size))
        leaves: list[tuple[TrainParallelConfig, float, dict[str, Any]]] = []
        if n_train % cp:
            self._prune(n_train, "root", "cp_not_divisible")
            return leaves
        for tp in self.train_tp:
            if tp * cp > n_train or n_train % (tp * cp):
                self._prune(n_train, "tp", "tp_not_divisible", tp=tp)
                continue
            ratio = self._tp_comm_ratio(tp, length)
            if ratio > self.tp_comm_ratio_limit:
                self._prune(n_train, "tp", "tp_comm_ratio", tp=tp, ratio=ratio)
                continue
            self._node(n_train, "tp", tp=tp, ratio=ratio)
            # Current dense Qwen execution does not expose an EP coordinate.
            # Record that fact rather than pretending an unsupported MoE plan ran.
            if self.config.model_arch.is_moe:
                self._node(n_train, "ep", tp=tp, ep=1, note="EP fixed to 1 by runtime")
            max_pp = n_train // (tp * cp)
            pp_candidates = self.train_pp or list(range(1, max_pp + 1))
            for pp in pp_candidates:
                if pp > max_pp or n_train % (tp * pp * cp):
                    self._prune(n_train, "pp", "pp_not_divisible", tp=tp, pp=pp)
                    continue
                if self.config.model_arch.n_layers % pp:
                    self._prune(n_train, "pp", "layers_not_divisible", tp=tp, pp=pp)
                    continue
                dp = n_train // (tp * pp * cp)
                for micro in self.micro_batches:
                    if self.config.batch_size % (dp * micro):
                        self._prune(n_train, "dp", "global_batch_not_divisible", tp=tp, pp=pp, dp=dp, b_micro=micro)
                        continue
                    n_micro = self.config.batch_size // (dp * micro)
                    bubble = (pp - 1) / max(pp + n_micro - 1, 1)
                    if bubble > self.pp_bubble_ratio_limit:
                        self._prune(n_train, "pp", "pipeline_bubble", tp=tp, pp=pp, dp=dp, b_micro=micro, bubble=bubble)
                        continue
                    candidate = TrainParallelConfig(tp=tp, pp=pp, cp=cp, dp=dp, b_micro=micro)
                    if self.evaluator.check_train_oom(candidate, self.config.batch_size, length):
                        self._prune(n_train, "leaf", "memory_oom", **_train_dict(candidate))
                        continue
                    t_train, details = self.evaluator.evaluate_training(candidate, self.config.batch_size, length)
                    if not np.isfinite(t_train):
                        self._prune(n_train, "leaf", "unsupported_topology", **_train_dict(candidate))
                        continue
                    self._node(n_train, "leaf", **_train_dict(candidate), bubble=bubble, t_train=t_train)
                    leaves.append((candidate, t_train, details))
        return leaves

    def _rollout_dp(
        self, budget: int, requests: list[RequestInfo]
    ) -> tuple[RolloutClusterConfig, float]:
        # dp[g][i] = best makespan for g GPUs serving the first i sorted requests.
        if budget in self.trace["rollout_dp"]:
            item = self.trace["rollout_dp"][budget]
            return RolloutClusterConfig(list(item["tp_list"])), float(item["t_rollout"])
        ordered = sorted(requests, key=lambda request: request.total_length)
        n = len(ordered)
        table = [[(inf, []) for _ in range(n + 1)] for _ in range(budget + 1)]
        table[0][0] = (0.0, [])
        costs: dict[tuple[int, int, int], float] = {}
        transitions = 0
        for gpus in range(1, budget + 1):
            for done in range(n + 1):
                best_cost, best_layout = table[gpus][done]
                for tp in self.rollout_tp:
                    if tp > gpus:
                        continue
                    for count in range(done + 1):  # x=0 is the paper's unused-capacity transition.
                        previous, layout = table[gpus - tp][done - count]
                        if not np.isfinite(previous):
                            continue
                        transitions += 1
                        key = (tp, done - count, done)
                        if key not in costs:
                            segment = ordered[done - count:done]
                            costs[key] = self.evaluator.evaluate_rollout(
                                RolloutClusterConfig([tp]), segment
                            )[0]
                        makespan = max(previous, costs[key])
                        if makespan < best_cost:
                            best_cost, best_layout = makespan, layout + [tp]
                table[gpus][done] = (best_cost, best_layout)
        result_cost, result_layout = table[budget][n]
        self.trace["rollout_dp"][budget] = {
            "budget": budget, "request_count": n, "tp_list": result_layout,
            "t_rollout": result_cost, "transitions": transitions,
            "segment_evaluations": len(costs),
        }
        return RolloutClusterConfig(result_layout), result_cost

    def _tp_comm_ratio(self, tp: int, length: int) -> float:
        if tp == 1:
            return 0.0
        model = self.evaluator.analytic.training_model
        micro = min(self.micro_batches)
        config = TrainParallelConfig(tp=tp, pp=1, b_micro=micro)
        comm = model.compute_tp_comm(tp, micro, length)
        fwd = model.compute_stage_fwd_time(length, config)
        bwd = model.compute_stage_bwd_time(length, config)
        return comm / max(fwd + bwd - 2 * comm, 1e-12)

    @staticmethod
    def _requests(history: list[dict[str, Any]]) -> list[RequestInfo]:
        records = []
        for item in history:
            try:
                prompt = int(item.get("input_len", item.get("prompt_length", 0)))
                output = int(item.get("output_len", item.get("gen_length", 0)))
            except (AttributeError, TypeError, ValueError):
                continue
            if prompt > 0 or output > 0:
                records.append(RequestInfo(max(1, prompt), max(1, output)))
        return records

    def _stratified_sample(self, requests: list[RequestInfo]) -> list[RequestInfo]:
        if len(requests) <= self.max_dp_requests:
            return list(requests)
        ordered = sorted(requests, key=lambda request: request.total_length)
        positions = np.linspace(0, len(ordered) - 1, self.max_dp_requests, dtype=int)
        return [ordered[int(position)] for position in positions]

    def _node(self, budget: int, level: str, **data: Any) -> None:
        self.trace["tree_nodes"].append({"budget": budget, "level": level, "status": "kept", **data})

    def _prune(self, budget: int, level: str, reason: str, **data: Any) -> None:
        self.trace["prune_counts"][reason] += 1
        self.trace["tree_nodes"].append({"budget": budget, "level": level, "status": "pruned", "reason": reason, **data})


def apply_paper_plan(config: AsyncRLConfig, plan: PaperPlan) -> None:
    """Materialize a selected paper plan into the normal training config."""
    config.n_total_gpus = plan.n_total_gpus
    config.train_gpus = plan.train_gpus
    config.rollout_gpus = plan.rollout_gpus
    config.train_tp_size = config.tp_size = plan.train.tp
    config.train_pp_size = plan.train.pp
    config.train_cp_size = plan.train.cp
    config.train_dp_size = plan.train.dp
    config.micro_batch_size = plan.train.b_micro
    config.heterogeneous_rollout.total_gpus = plan.rollout_gpus
    config.global_resource_planner.initial_allocation_strategy = "paper_tree_dp"
    config.global_resource_planner.initial_allocation_applied = True

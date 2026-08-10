"""Startup preflight planning for Global Resource Planner."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlan,
    GlobalResourcePlanner,
    PlannerDecision,
)
from RL_Framework.infra.cost_model.startup_profile import summarize_length_profile


@dataclass
class PreflightPlannerResult:
    """Result produced by a startup preflight run."""

    decision: PlannerDecision
    applied: bool
    planned_config: AsyncRLConfig
    applied_plan: GlobalResourcePlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "decision": self.decision.to_dict(),
            "applied_plan": (
                self.applied_plan.to_dict() if self.applied_plan is not None else None
            ),
            "metadata": dict(self.metadata),
        }


class PreflightPlanner:
    """Run a one-shot planning pass before the trainer starts.

    The preflight pass uses the same planner and simulator adapters as runtime
    planning, but forces the interval/warmup gates open so startup can emit a
    concrete planned config.
    """

    def __init__(
        self,
        config: AsyncRLConfig,
        *,
        planner: GlobalResourcePlanner | None = None,
        apply_best_candidate: bool = True,
    ):
        self.config = copy.deepcopy(config)
        self.planner = planner or GlobalResourcePlanner.from_config(self.config)
        self.apply_best_candidate = apply_best_candidate

    def run(self, history: Iterable[dict[str, Any]]) -> PreflightPlannerResult:
        batch = list(history)
        original = {
            "plan_interval": self.planner.plan_interval,
            "warmup_steps": self.planner.warmup_steps,
            "min_history_size": self.planner.min_history_size,
            "reconfiguration_cost_s": self.planner.reconfiguration_cost_s,
            "min_gain_ratio": self.planner.min_gain_ratio,
            "fixed_train_gpus": self.planner.fixed_train_gpus,
        }
        try:
            self.planner.plan_interval = 1
            self.planner.warmup_steps = 0
            self.planner.min_history_size = max(1, min(len(batch), original["min_history_size"]))
            if self.apply_best_candidate:
                self.planner.reconfiguration_cost_s = 0.0
                self.planner.min_gain_ratio = 0.0
            if self.planner.initial_allocation_strategy == "grp":
                self.planner.fixed_train_gpus = 0
            decision = self.planner.plan_if_needed(
                step=0,
                config=self.config,
                batch=batch,
            )
        finally:
            self.planner.plan_interval = original["plan_interval"]
            self.planner.warmup_steps = original["warmup_steps"]
            self.planner.min_history_size = original["min_history_size"]
            self.planner.reconfiguration_cost_s = original["reconfiguration_cost_s"]
            self.planner.min_gain_ratio = original["min_gain_ratio"]
            self.planner.fixed_train_gpus = original["fixed_train_gpus"]

        applied_plan = self._select_plan(decision)
        applied = applied_plan is not None
        if applied_plan is not None:
            self.planner.apply_plan_to_config(applied_plan, self.config)
            # Persist the proof that startup allocation was decided by GRP.
            # The trainer checks this before constructing any training engine.
            self.config.global_resource_planner.initial_allocation_applied = True

        return PreflightPlannerResult(
            decision=decision,
            applied=applied,
            applied_plan=applied_plan,
            planned_config=self.config,
            metadata={
                "num_history_records": len(batch),
                "apply_best_candidate": self.apply_best_candidate,
                "initial_allocation_strategy": self.planner.initial_allocation_strategy,
                "length_profile": summarize_length_profile(batch),
            },
        )

    def _select_plan(self, decision: PlannerDecision) -> GlobalResourcePlan | None:
        candidate = decision.candidate_plan
        current = decision.current_plan
        if candidate is None:
            return None
        if self.planner.initial_allocation_strategy == "grp":
            # Startup has no reconfiguration cost and no already-running stage
            # to preserve.  The optimizer's feasible candidate is therefore
            # authoritative even when it happens to equal the YAML seed split.
            return candidate
        if decision.should_reconfigure:
            return candidate
        if not self.apply_best_candidate or current is None:
            return None
        if candidate.t_global <= current.t_global:
            return candidate
        return None


def load_history_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def synthetic_history(
    *,
    num_requests: int,
    input_len: int = 1024,
    output_len: int = 2048,
) -> list[dict[str, int]]:
    return [
        {
            "input_len": max(1, int(input_len)),
            "output_len": max(1, int(output_len)),
        }
        for _ in range(max(1, int(num_requests)))
    ]

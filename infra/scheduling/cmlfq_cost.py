"""Cost estimates shared by the distribution-based routing policy.

All public estimates are in seconds. Profile files use milliseconds explicitly.
Transfer measurements must include preparing/exporting the source cache and
the target import, not only the network copy.
"""

from __future__ import annotations

import json
import math
from typing import Any


class CMLFQRoutingCosts:
    def __init__(self, profile_path: str = "", rollout_model: Any = None):
        self.rollout_model = rollout_model
        self.decode_ms: dict[str, dict[str, float]] = {}
        self.migrations: list[dict] = []
        if profile_path:
            with open(profile_path, encoding="utf-8") as stream:
                data = json.load(stream)
            self.decode_ms = data.get("decode_ms", {})
            self.migrations = data.get("migrations", [])
            for bins in self.decode_ms.values():
                for value in bins.values():
                    self._validate_ms(value)
            for row in self.migrations:
                for key in ("source_tp", "target_tp", "seq_len"):
                    if int(row[key]) <= 0:
                        raise ValueError(f"{key} must be positive")
                for key in ("source_host", "target_host"):
                    if not row.get(key):
                        raise ValueError(f"migration profile requires {key}")
                self._validate_ms(row["recompute_ms"])
                if "transfer_ms" in row:
                    self._validate_ms(row["transfer_ms"])

    @staticmethod
    def _validate_ms(value: float) -> None:
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("latency must be finite and nonnegative")

    def decode_seconds(
        self, tp: int, bin_name: str, remaining: float, context: int,
    ) -> float:
        if self.rollout_model is not None and self.rollout_model.check_oom(math.ceil(context + remaining), tp):
            return math.inf
        value = self.decode_ms.get(str(tp), {}).get(bin_name)
        if value is not None:
            return float(value) / 1000.0
        if self.rollout_model is None:
            return math.inf
        # Midpoint integration is exact for the analytic model's affine
        # dependence on active context length. This is one trajectory's cost,
        # not the whole cluster makespan used by the planner.
        return remaining * self.rollout_model.compute_decode_step_time(
            math.ceil(context + remaining / 2), tp
        )

    def migration_seconds(
        self, source: dict, target: dict, context: int, allow_transfer: bool,
    ) -> tuple[str, float]:
        rows = [
            row for row in self.migrations
            if int(row["source_tp"]) == source["tp_degree"]
            and int(row["target_tp"]) == target["tp_degree"]
            and row["source_host"] == source["host"]
            and row["target_host"] == target["host"]
        ]
        if rows:
            row = min(rows, key=lambda item: abs(item["seq_len"] - context))
            recompute = float(row["recompute_ms"]) / 1000.0
            transfer = float(row.get("transfer_ms", math.inf)) / 1000.0
            if allow_transfer and transfer < recompute:
                return "transfer", transfer
            return "recompute", recompute
        if self.rollout_model is None:
            return "recompute", math.inf
        # No measured transfer cost => use the supported recompute path.
        return "recompute", self.prefill_seconds(target["tp_degree"], context)

    def prefill_seconds(self, tp: int, context: int) -> float:
        """Common cold-cache baseline; no cache residency is queried online."""
        if self.rollout_model is None:
            return 0.0
        model = self.rollout_model
        chunk = max(1, int(model.L_chunk))
        return max(0, context) / chunk * model.compute_prefill_step_time(1, chunk, tp)

    def completion_seconds(self, tp: int, bin_name: str, remaining: float,
                           context: int, other_requests: int = 0) -> float:
        """Estimate standalone completion, including one prefill, if the sequence fits."""
        if self.rollout_model is not None and self.rollout_model.check_oom(math.ceil(context + remaining), tp):
            return math.inf
        return self.decode_work_seconds(tp, bin_name, remaining, context, other_requests) + self.prefill_seconds(tp, context)

    def decode_work_seconds(self, tp: int, bin_name: str, remaining: float,
                            context: int, other_requests: int = 0) -> float:
        """Price decode for a representative cohort after admission checks."""
        count = max(0, int(other_requests)) + 1
        value = self.decode_ms.get(str(tp), {}).get(bin_name)
        if value is not None:
            return float(value) / 1000.0 * count
        if self.rollout_model is None:
            return math.inf
        elapsed = self.rollout_model.compute_instance_makespan_from_totals(
            num_requests=count,
            total_prompt_tokens=count * max(0, context),
            total_gen_tokens=count * max(0, math.ceil(remaining)),
            tp=tp,
        )
        return max(0.0, elapsed - count * self.prefill_seconds(tp, context))

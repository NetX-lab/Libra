"""Support code for Load balance."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

from RL_Framework.infra.scheduling.base import (
    DEFAULT_LOAD_METRIC,
    BaseScheduler,
    InstanceHandle,
    LoadBalanceStrategy,
    MetricsFeedbackConfig,
    SchedulingResult,
)

logger = logging.getLogger(__name__)


class LoadBalanceScheduler(BaseScheduler):
    """Load balance scheduler implementation."""

    def __init__(
        self,
        load_balance_strategy: str = "least_connections",
        max_queue_length: int = 100,
        weights: dict[int, float] | None = None,
        load_metric: str = DEFAULT_LOAD_METRIC,
        kv_capacity_tokens_by_tp: dict[int, int] | None = None,
        feedback: MetricsFeedbackConfig | None = None,
        prefix_affinity: bool = False,
        prefix_affinity_max_entries: int = 8192,
    ):
        super().__init__(
            name="LoadBalance",
            load_metric=load_metric,
            kv_capacity_tokens_by_tp=kv_capacity_tokens_by_tp,
            feedback=feedback,
            prefix_affinity=prefix_affinity,
            prefix_affinity_max_entries=prefix_affinity_max_entries,
        )

        if load_balance_strategy == "round_robin":
            self._strategy = LoadBalanceStrategy.ROUND_ROBIN
        elif load_balance_strategy == "weighted":
            self._strategy = LoadBalanceStrategy.WEIGHTED
        else:
            self._strategy = LoadBalanceStrategy.LEAST_CONNECTIONS

        self._max_queue_length = max_queue_length
        self._weights = weights or {}  # tp_degree -> weight
        self._rr_counter = 0

        logger.info(
            f"LoadBalanceScheduler initialized: strategy={self._strategy.value}, "
            f"max_queue_length={max_queue_length}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def schedule(
        self,
        input_tokens: int,
        prompt_id: str = "",
        n_samples: int = 1,
        epoch: int = -1,
        max_new_tokens: int = 0,
    ) -> SchedulingResult:
        """Schedule."""
        with self._lock:
            self._stats.total_requests += 1


            ready = self._selectable([h for h in self._instances if h.is_ready])
            candidates = [
                h for h in ready
                if (
                    self._max_queue_length <= 0
                    or h.active_requests < self._max_queue_length
                )
            ]

            if not candidates:

                candidates = ready

            if not candidates:
                self._stats.failed_routes += 1
                return SchedulingResult(
                    instance_index=-1,
                    tp_degree=0,
                    category="any",
                    is_fallback=False,
                    reason="no_available_instance",
                    prompt_id=prompt_id,
                )


            selected = self._affinity_pick(prompt_id, candidates)
            if selected is not None:
                self._stats.category_counts["any"] += 1
                self._stats.category_tp_counts["any"][selected.tp_degree] += 1
                reason = "prefix_affinity"
            else:
                selected = self._select(candidates)
                self._stats.category_counts["any"] += 1
                self._stats.category_tp_counts["any"][selected.tp_degree] += 1
                reason = ""
            self._stats.preferred_routes += 1
            reserved = self._record_route(selected, input_tokens, category="any", prompt_id=prompt_id, max_new_tokens=max_new_tokens)

            return SchedulingResult(
                instance_index=selected.index,
                reserved_tokens=reserved,
                tp_degree=selected.tp_degree,
                category="any",
                is_fallback=False,
                reason=reason,
                prompt_id=prompt_id,
            )

    def _select(self, candidates: list[InstanceHandle]) -> InstanceHandle:
        """Select."""
        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            idx = self._rr_counter % len(candidates)
            self._rr_counter += 1
            return candidates[idx]

        elif self._strategy == LoadBalanceStrategy.WEIGHTED:

            def weighted_load(h: InstanceHandle) -> float:
                w = self._weights.get(h.tp_degree, 1.0)
                return self.load_of(h) / max(w, 0.01)
            return min(candidates, key=weighted_load)

        else:
            return self._min_load_rotating(candidates)

    def on_request_done(
        self,
        instance_index: int,
        prompt_id: str = "",
        final_bucket: str = "",
        output_tokens: int = 0,
        reserved_tokens: int | None = None,
    ):
        """On request done."""
        with self._lock:
            self._complete_route(
                instance_index,
                output_tokens=output_tokens,
                category=final_bucket or "any",
                reserved_tokens=reserved_tokens,
            )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "LoadBalanceScheduler":
        """From config."""
        sched = hetero_config.scheduling
        return cls(
            load_balance_strategy=sched.load_balance_strategy,
            max_queue_length=sched.max_queue_length,
            load_metric=getattr(sched, "load_metric", DEFAULT_LOAD_METRIC),
            kv_capacity_tokens_by_tp=getattr(sched, "kv_capacity_tokens_by_tp", None),
            feedback=MetricsFeedbackConfig.from_scheduling(sched),
            prefix_affinity=bool(getattr(sched, "prefix_affinity", False)),
            prefix_affinity_max_entries=int(
                getattr(sched, "prefix_affinity_max_entries", 8192)
            ),
        )

"""Support code for Load balance."""

import logging
from collections import defaultdict
from typing import Any, Optional

from RL_Framework.infra.scheduling.base import (
    BaseScheduler,
    InstanceHandle,
    LoadBalanceStrategy,
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
    ):
        super().__init__(name="LoadBalance")

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
    ) -> SchedulingResult:
        """Schedule."""
        with self._lock:
            self._stats.total_requests += 1


            candidates = [
                h for h in self._instances
                if h.is_ready and (
                    self._max_queue_length <= 0
                    or h.active_requests < self._max_queue_length
                )
            ]

            if not candidates:

                candidates = [h for h in self._instances if h.is_ready]

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


            selected = self._select(candidates)
            self._stats.preferred_routes += 1
            self._stats.category_counts["any"] += 1
            self._stats.category_tp_counts["any"][selected.tp_degree] += 1
            selected.inc_active()

            return SchedulingResult(
                instance_index=selected.index,
                tp_degree=selected.tp_degree,
                category="any",
                is_fallback=False,
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
                return h.active_requests / max(w, 0.01)
            return min(candidates, key=weighted_load)

        else:

            return min(candidates, key=lambda h: h.active_requests)

    def on_request_done(
        self,
        instance_index: int,
        prompt_id: str = "",
        final_bucket: str = "",
        output_tokens: int = 0,
    ):
        """On request done."""
        handle = self.get_instance_handle(instance_index)
        if handle:
            with self._lock:
                handle.dec_active()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "LoadBalanceScheduler":
        """From config."""
        sched = hetero_config.scheduling
        return cls(
            load_balance_strategy=sched.load_balance_strategy,
            max_queue_length=sched.max_queue_length,
        )

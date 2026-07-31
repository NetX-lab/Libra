"""Support code for Length aware."""

import logging
from collections import defaultdict
from typing import Any

from RL_Framework.infra.scheduling.base import (
    BaseScheduler,
    InstanceHandle,
    LoadBalanceStrategy,
    RoutingRule,
    SchedulerStats,
    SchedulingResult,
)

logger = logging.getLogger(__name__)


class LengthAwareScheduler(BaseScheduler):
    """Length aware scheduler implementation."""

    DEFAULT_ROUTING_RULES = {
        "short": RoutingRule("short", [1, 2], [4, 8]),
        "medium": RoutingRule("medium", [2, 4], [1, 8]),
        "long": RoutingRule("long", [4, 8], [2, 1]),
        "extra_long": RoutingRule("extra_long", [8, 4], [2, 1]),
    }

    def __init__(
        self,
        routing_rules: dict[str, list[int]] | None = None,
        length_thresholds: dict[str, int] | None = None,
        load_balance_strategy: str = "least_connections",
        max_queue_length: int = 100,
        enable_fallback: bool = True,
    ):
        super().__init__(name="LengthAware")


        self._rules: dict[str, RoutingRule] = dict(self.DEFAULT_ROUTING_RULES)
        if routing_rules:
            for cat, tp_degrees in routing_rules.items():
                if cat in self._rules:
                    self._rules[cat].preferred_tp_degrees = list(tp_degrees)
                else:
                    self._rules[cat] = RoutingRule(cat, list(tp_degrees), [])


        self._thresholds = length_thresholds or {
            "short": 5000,
            "medium": 10000,
            "long": 15000,
        }


        if load_balance_strategy == "round_robin":
            self._strategy = LoadBalanceStrategy.ROUND_ROBIN
        else:
            self._strategy = LoadBalanceStrategy.LEAST_CONNECTIONS
        self._max_queue_length = max_queue_length
        self._enable_fallback = enable_fallback


        self._rr_index: dict[int, int] = defaultdict(int)


        self._perf_history: dict[tuple[str, int], list[float]] = defaultdict(list)

        logger.info(
            f"LengthAwareScheduler initialized: thresholds={self._thresholds}, "
            f"strategy={self._strategy.value}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def categorize(self, input_tokens: int) -> str:
        """Categorize."""
        short_th = self._thresholds.get("short", 5000)
        medium_th = self._thresholds.get("medium", 10000)
        long_th = self._thresholds.get("long", 15000)

        if input_tokens <= short_th:
            return "short"
        elif input_tokens <= medium_th:
            return "medium"
        elif input_tokens <= long_th:
            return "long"
        else:
            return "extra_long"

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
        category = self.categorize(input_tokens)
        rule = self._rules.get(category, self._rules.get("medium"))

        with self._lock:
            self._stats.total_requests += 1
            self._stats.category_counts[category] += 1


            selected = self._try_select(rule.preferred_tp_degrees)
            if selected is not None:
                self._stats.preferred_routes += 1
                self._stats.category_tp_counts[category][selected.tp_degree] += 1
                selected.inc_active()
                return SchedulingResult(
                    instance_index=selected.index,
                    tp_degree=selected.tp_degree,
                    category=category,
                    is_fallback=False,
                    prompt_id=prompt_id,
                )


            if self._enable_fallback and rule.fallback_tp_degrees:
                selected = self._try_select(rule.fallback_tp_degrees)
                if selected is not None:
                    self._stats.fallback_routes += 1
                    self._stats.category_tp_counts[category][selected.tp_degree] += 1
                    selected.inc_active()
                    return SchedulingResult(
                        instance_index=selected.index,
                        tp_degree=selected.tp_degree,
                        category=category,
                        is_fallback=True,
                        prompt_id=prompt_id,
                    )


            all_ready = [h for h in self._instances if h.is_ready]
            if all_ready:
                selected = min(all_ready, key=lambda h: h.active_requests)
                self._stats.fallback_routes += 1
                self._stats.category_tp_counts[category][selected.tp_degree] += 1
                selected.inc_active()
                return SchedulingResult(
                    instance_index=selected.index,
                    tp_degree=selected.tp_degree,
                    category=category,
                    is_fallback=True,
                    reason="global_fallback",
                    prompt_id=prompt_id,
                )


            self._stats.failed_routes += 1
            return SchedulingResult(
                instance_index=-1,
                tp_degree=0,
                category=category,
                is_fallback=False,
                reason="no_available_instance",
                prompt_id=prompt_id,
            )

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

    def _try_select(self, tp_preferences: list[int]) -> InstanceHandle | None:
        """Try select."""
        for tp_degree in tp_preferences:
            candidates = [
                h for h in self._instances_by_tp.get(tp_degree, [])
                if h.is_ready
            ]
            if not candidates:
                continue


            if self._max_queue_length > 0:
                candidates = [
                    h for h in candidates
                    if h.active_requests < self._max_queue_length
                ]
            if not candidates:
                continue


            if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
                idx = self._rr_index[tp_degree] % len(candidates)
                self._rr_index[tp_degree] += 1
                return candidates[idx]
            else:
                # least_connections
                return min(candidates, key=lambda h: h.active_requests)

        return None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def record_latency(self, category: str, tp_degree: int, latency: float):
        """Record latency."""
        key = (category, tp_degree)
        self._perf_history[key].append(latency)
        if len(self._perf_history[key]) > 200:
            self._perf_history[key] = self._perf_history[key][-200:]

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def reset_stats(self):
        super().reset_stats()
        self._perf_history.clear()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "LengthAwareScheduler":
        """From config."""
        sched = hetero_config.scheduling
        return cls(
            routing_rules=sched.routing_rules if isinstance(sched.routing_rules, dict) else {},
            length_thresholds=(
                sched.length_thresholds
                if isinstance(sched.length_thresholds, dict)
                else {}
            ),
            load_balance_strategy=sched.load_balance_strategy,
            max_queue_length=sched.max_queue_length,
            enable_fallback=sched.enable_fallback,
        )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

HeterogeneousScheduler = LengthAwareScheduler

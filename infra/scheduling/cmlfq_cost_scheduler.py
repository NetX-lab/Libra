"""Opt-in histogram/cost routing with transactional backend execution."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

from RL_Framework.infra.scheduling.base import SchedulingResult
from RL_Framework.infra.scheduling.cmlfq_cost import CMLFQRoutingCosts
from RL_Framework.infra.scheduling.cmlfq_scheduler import (
    CMLFQMigrationDecision, CMLFQScheduler,
)


@dataclass
class CostRoutingDecision(CMLFQMigrationDecision):
    source_instance_index: int = -1
    target_instance_index: int = -1
    execution_path: str = "stay"
    decode_seconds: float = 0.0
    migration_seconds: float = 0.0


class CMLFQCostScheduler(CMLFQScheduler):
    """Keep legacy C-MLFQ separate; commit moves only after backend success."""

    DEFAULT_BUCKETS = {
        "short": {"tp_degrees": [1], "max_tokens": 5000},
        "long": {"tp_degrees": [4], "max_tokens": 50000},
    }

    def __init__(self, *args, routing_costs: CMLFQRoutingCosts | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "C-MLFQ-Cost"
        self._lock = threading.RLock()
        tps = [cfg.get("tp_degrees", []) for cfg in self._buckets.values()]
        if any(len(tp) != 1 for tp in tps) or len({tp[0] for tp in tps}) != len(tps):
            raise ValueError("cmlfq_cost requires one distinct TP degree per bucket")
        if set(self._bucket_thresholds) != set(self._buckets):
            raise ValueError("bucket thresholds must match configured buckets")
        thresholds = list(self._bucket_thresholds.values())
        if any(value <= 0 for value in thresholds) or len(set(thresholds)) != len(thresholds):
            raise ValueError("bucket thresholds must be positive and distinct")
        self.routing_costs = routing_costs or CMLFQRoutingCosts()
        self._topology: dict[int, dict] = {}
        self._transfer_supported: Callable[[int, int], bool] = lambda src, dst: False
        self._input_lengths: dict[str, int] = {}
        self._pending_cost_routes: set[str] = set()
        self._reservations: dict[str, CostRoutingDecision] = {}

    def configure_runtime(self, topology: list[dict], transfer_supported=None):
        self._topology = {index: dict(cfg) for index, cfg in enumerate(topology)}
        if transfer_supported is not None:
            self._transfer_supported = transfer_supported

    def register_instance(self, index: int, instance_id: str, tp_degree: int):
        if self.bucket_for_tp(tp_degree) == "unmapped":
            raise ValueError(f"TP {tp_degree} is not mapped to a cmlfq_cost bucket")
        super().register_instance(index, instance_id, tp_degree)

    def schedule(self, input_tokens: int, prompt_id: str = "", n_samples: int = 1,
                 epoch: int = -1) -> SchedulingResult:
        if n_samples != 1:
            raise ValueError("cmlfq_cost tracks one trajectory per request")
        with self._lock:
            result = super().schedule(input_tokens, prompt_id, n_samples, epoch)
            if result.request_id:
                self._input_lengths[result.request_id] = max(0, input_tokens)
            return result

    def _route_to_bucket(self, bucket, input_tokens, prompt_id="", reason=""):
        """Initial placement honors readiness, capacity and fallback policy."""
        with self._lock:
            counts = self._get_global_active_counts()
            rule = self._bucket_rules[bucket]
            target = self._try_select(rule["preferred_tp_degrees"], counts)
            fallback = target is None
            if target is None and self._enable_fallback:
                target = self._try_select(rule["fallback_tp_degrees"], counts)
            if target is None:
                self._stats.failed_routes += 1
                return SchedulingResult(-1, 0, bucket, False, "no_available_instance")
            actual = self.bucket_for_tp(target.tp_degree)
            self._increment_active(target)
            self._stats.category_counts[actual] += 1
            self._stats.category_tp_counts[actual][target.tp_degree] += 1
            if fallback:
                self._stats.fallback_routes += 1
            else:
                self._stats.preferred_routes += 1
            return SchedulingResult(target.index, target.tp_degree, actual, fallback,
                                    reason, prompt_id=prompt_id)

    def on_tool_return(self, request_id: str, tool_result: Any,
                       generated_tokens: int = 0) -> CostRoutingDecision:
        with self._lock:
            state = self._request_states.get(request_id)
            if state is None:
                return CostRoutingDecision(False, "request_not_found", "unknown")
            if request_id in self._reservations:
                raise RuntimeError("tool return received during generation")
            if generated_tokens < state.generated_tokens:
                raise ValueError("tool token positions must be monotonic")
            self._tool_return_count += 1
            state.generated_tokens = generated_tokens
            state.collected_return_states.append(self._tool_registry.extract(tool_result))
            state.tool_return_token_positions.append(generated_tokens)
            self._pending_cost_routes.add(request_id)
            return self._choose(request_id)

    def _choose(self, request_id: str, context_tokens: int | None = None) -> CostRoutingDecision:
        state = self._request_states[request_id]
        source = self.get_instance_handle(state.current_instance_index)
        context = (self._input_lengths[request_id] + state.generated_tokens
                   if context_tokens is None else context_tokens)
        stay = CostRoutingDecision(
            False, "no_distribution_or_cost", state.current_bucket,
            target_bucket=state.current_bucket,
            source_instance_index=source.index, target_instance_index=source.index,
        )
        distribution = self.prefix_tree.residual_distribution(
            state.prompt_id, state.collected_return_states, self._bucket_thresholds,
        )
        if not distribution:
            if not source.is_ready:
                raise RuntimeError("current instance unavailable and no residual distribution")
            return stay
        global_counts = self._get_global_active_counts()
        choices = []
        for handle in self._instances:
            same = handle.index == source.index
            if not handle.is_ready:
                continue
            if not same and self._max_queue_length > 0 and self._active_count(
                handle, global_counts
            ) >= self._max_queue_length:
                continue
            decode = sum(
                probability * self.routing_costs.decode_seconds(
                    handle.tp_degree, name, remaining, context,
                ) for name, probability, remaining in distribution
            )
            path, movement = "stay", 0.0
            if not same:
                if source.index not in self._topology or handle.index not in self._topology:
                    continue
                path, movement = self.routing_costs.migration_seconds(
                    self._topology[source.index], self._topology[handle.index], context,
                    source.is_ready and self._transfer_supported(source.index, handle.index),
                )
            total = decode + movement
            if not math.isfinite(total) or total < 0:
                continue
            # Stable ties prefer staying, then the least-loaded instance.
            choices.append((total, not same, self._active_count(handle, global_counts),
                            handle.index, decode, movement, path))
        if not choices:
            if not source.is_ready:
                raise RuntimeError("no available instance with a finite routing cost")
            return stay
        _, _, _, index, decode, movement, path = min(choices)
        target = self.get_instance_handle(index)
        return CostRoutingDecision(
            index != source.index, "distribution_cost", state.current_bucket,
            target_bucket=self.bucket_for_tp(target.tp_degree),
            source_instance_index=source.index, target_instance_index=index,
            execution_path=path, decode_seconds=decode, migration_seconds=movement,
        )

    def execute_migration(self, request_id: str, decision: CMLFQMigrationDecision):
        """Queue intent; generation reserves/revalidates it with current load."""
        with self._lock:
            if request_id not in self._request_states:
                raise ValueError(f"Request {request_id} not found")
            if not decision.should_migrate:
                raise ValueError("Cannot execute a non-migration decision")
            self._pending_cost_routes.add(request_id)
            return self.get_request_route(request_id)

    def reserve_generation(self, request_id: str, context_tokens: int | None = None):
        with self._lock:
            if request_id in self._reservations:
                raise RuntimeError("concurrent generation for the same request")
            state = self._request_states[request_id]
            if request_id in self._pending_cost_routes:
                decision = self._choose(request_id, context_tokens)
            else:
                decision = CostRoutingDecision(
                    False, "resume", state.current_bucket, target_bucket=state.current_bucket,
                    source_instance_index=state.current_instance_index,
                    target_instance_index=state.current_instance_index,
                )
            target = self.get_instance_handle(decision.target_instance_index)
            if not target.is_ready:
                raise RuntimeError("selected instance is unavailable")
            if decision.should_migrate:
                self._increment_active(target)
            self._reservations[request_id] = decision
            return decision

    def commit_generation(self, request_id: str, decision: CostRoutingDecision):
        with self._lock:
            if self._reservations.get(request_id) is not decision:
                raise RuntimeError("generation reservation was cancelled or replaced")
            self._reservations.pop(request_id)
            state = self._request_states[request_id]
            if decision.should_migrate:
                self._decrement_active(self.get_instance_handle(state.current_instance_index))
                state.current_instance_index = decision.target_instance_index
                state.current_bucket = decision.target_bucket
                state.has_migrated = True
                self._migration_count += 1
                self._stats.migrated_routes += 1
            self._pending_cost_routes.discard(request_id)

    def rollback_generation(self, request_id: str):
        with self._lock:
            reservation = self._reservations.pop(request_id, None)
            if reservation is not None and reservation.should_migrate:
                self._decrement_active(self.get_instance_handle(reservation.target_instance_index))

    def cancel_request(self, request_id: str):
        with self._lock:
            self.rollback_generation(request_id)
            self._pending_cost_routes.discard(request_id)
            self._input_lengths.pop(request_id, None)
            super().cancel_request(request_id)

    def finish_request(self, request_id: str, total_output_tokens: int):
        with self._lock:
            self.rollback_generation(request_id)
            self._pending_cost_routes.discard(request_id)
            self._input_lengths.pop(request_id, None)
            super().finish_request(request_id, total_output_tokens)

    @classmethod
    def from_config(cls, hetero_config: Any):
        # Reuse legacy tree loading/persistence and extractor configuration.
        scheduler = super().from_config(hetero_config)
        scheduler.routing_costs = CMLFQRoutingCosts(
            getattr(hetero_config.scheduling, "cmlfq_cost_profile_path", "")
        )
        return scheduler

"""KV-aware routing layered on the official CMLFQ cost scheduler."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from RL_Framework.infra.scheduling.base import SchedulingResult
from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import (
    CMLFQCostScheduler,
    CostRoutingDecision,
)


@dataclass
class KVCostRoutingDecision(CostRoutingDecision):
    """Cost decision annotated with the KV snapshot used to route it."""

    kv_routing: dict[str, Any] = field(default_factory=dict)


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class CMLFQDynamicKVCostScheduler(CMLFQCostScheduler):
    """Filter by fresh KV headroom, then compare all candidates by cost."""

    POLICY = "cmlfq_cost_kv_dynamic"
    KV_LIMIT = 0.95
    KV_TTL_SECONDS = 5.0
    METRICS_POLL_INTERVAL_SECONDS = 1.0

    FEEDBACK_INTERVAL_SECONDS = 10.0
    FEEDBACK_EMA_ALPHA = 0.3
    FEEDBACK_DEADBAND = 0.05
    FEEDBACK_STEP_SCALE = 1000.0
    FEEDBACK_MAX_STEP = 250
    FEEDBACK_MIN_BOUNDARY = 2000
    FEEDBACK_MAX_BOUNDARY = 8000
    FEEDBACK_INITIAL_BOUNDARY = 4000
    FEEDBACK_KV_WEIGHT = 0.7
    FEEDBACK_QUEUE_WEIGHT = 0.3
    FEEDBACK_RUNNING_WEIGHT = 0.25
    FEEDBACK_QUEUE_SCALE = 4.0
    NONPREFERRED_PENALTY = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self._sorted_bucket_names) < 2:
            raise ValueError("cmlfq_cost_kv_dynamic requires at least two ordered buckets")
        self.name = "C-MLFQ-Cost-KV-Dynamic"
        self._initial_tie_counters: dict[str, int] = {}
        self._initial_route_records: dict[str, dict[str, Any]] = {}
        self._first_generation: set[str] = set()
        self._last_initial_routing: dict[str, Any] = {}
        self._dynamic_boundary = self.FEEDBACK_INITIAL_BOUNDARY
        self._last_boundary_epoch: tuple | None = None
        self._last_boundary_monotonic: float | None = None
        self._feedback_ema_difference: float | None = None
        self._last_feedback: dict[str, Any] = {"status": "uninitialized"}
        self._last_boundary_event: dict[str, Any] | None = None
        self._boundary_event_sequence = 0
        self._boundary_transition_count = 0
        self._boundary_ledger: list[dict[str, Any]] = []

    def schedule(self, input_tokens: int, prompt_id: str = "", n_samples: int = 1,
                 epoch: int = -1):
        with self._lock:
            self._last_initial_routing = {}
            result = super().schedule(input_tokens, prompt_id, n_samples, epoch)
            if result.request_id:
                record = dict(self._last_initial_routing)
                if result.instance_index >= 0:
                    record.setdefault(
                        "chosen_instance_id",
                        self.get_instance_handle(result.instance_index).instance_id,
                    )
                self._initial_route_records[result.request_id] = record
                self._first_generation.add(result.request_id)
            return result

    def _route_to_bucket(self, bucket, input_tokens, prompt_id="", reason=""):
        with self._lock:
            snapshot = self._capture_snapshot()
            self._update_boundary(snapshot)
            eligible, rejected = self._eligible(snapshot, source_index=None)
            preferred = [
                handle for handle in eligible
                if self.bucket_for_tp(handle.tp_degree) == bucket
            ]
            candidates = preferred or eligible
            if candidates:
                selected = self._least_kv_with_round_robin(candidates, snapshot, bucket)
                actual = self.bucket_for_tp(selected.tp_degree)
                fallback = not preferred
                self._increment_active(selected)
                self._stats.category_counts[actual] += 1
                self._stats.category_tp_counts[actual][selected.tp_degree] += 1
                if fallback:
                    self._stats.fallback_routes += 1
                else:
                    self._stats.preferred_routes += 1
                result = SchedulingResult(
                    selected.index, selected.tp_degree, actual, fallback,
                    "cmlfq_kv_initial_healthy_fallback" if fallback
                    else "cmlfq_kv_initial_healthy",
                    prompt_id=prompt_id,
                )
                meta = self._routing_metadata(
                    snapshot, eligible, rejected, selected.index,
                    preferred_bucket=bucket,
                    candidate_indices=[handle.index for handle in candidates],
                )
            else:
                fallback_reason = self._no_healthy_reason(rejected)
                result = super()._route_to_bucket(
                    bucket, input_tokens, prompt_id,
                    reason=f"{reason or 'cmlfq_initial_placement'}_kv_baseline_fallback",
                )
                meta = self._routing_metadata(
                    snapshot, eligible, rejected, result.instance_index,
                    preferred_bucket=bucket, fallback_reason=fallback_reason,
                )
            if result.instance_index >= 0:
                meta["chosen_instance_id"] = self.get_instance_handle(
                    result.instance_index
                ).instance_id
            self._last_initial_routing = meta
            return result

    def _capture_snapshot(self) -> dict[str, Any]:
        try:
            snapshots = self._metrics_feed.snapshots() if self._metrics_feed is not None else {}
        except Exception:
            snapshots = {}
        now = time.time()
        ready = [handle for handle in self._instances if handle.is_ready]
        instances = {}
        for handle in self._instances:
            raw = snapshots.get(handle.instance_id)
            info = {
                "kv": None, "age_s": None, "running": -1, "waiting": -1,
                "fresh": False, "reason": "missing_snapshot", "updated_at": None,
            }
            if raw is not None:
                try:
                    updated_at = float(raw.updated_at)
                    usage = float(raw.gpu_cache_usage)
                    age = now - updated_at
                    info["updated_at"] = updated_at if math.isfinite(updated_at) else None
                    info["age_s"] = age if math.isfinite(age) else None
                    try:
                        info["running"] = int(raw.running)
                        info["waiting"] = int(raw.waiting)
                    except (TypeError, ValueError, OverflowError):
                        pass
                    try:
                        capacity = int(raw.kv_capacity_tokens)
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        capacity = 0
                    if capacity > 0:
                        info["kv_capacity_tokens"] = capacity
                    if not math.isfinite(updated_at) or not math.isfinite(usage):
                        info["reason"] = "invalid_snapshot"
                    elif age < 0:
                        info["reason"] = "future_snapshot"
                    elif age > self.KV_TTL_SECONDS:
                        info["reason"] = "stale_snapshot"
                    elif usage < 0 or usage > 1:
                        info["reason"] = "invalid_kv"
                    else:
                        info.update(kv=usage, fresh=True, reason="")
                except (AttributeError, TypeError, ValueError, OverflowError):
                    info["reason"] = "invalid_snapshot"
            instances[handle.instance_id] = info
        complete = bool(ready) and all(
            instances[handle.instance_id]["fresh"] for handle in ready
        )
        epoch = None
        if complete:
            epoch = tuple(
                (handle.instance_id, instances[handle.instance_id]["updated_at"])
                for handle in sorted(ready, key=lambda item: item.index)
            )
        return {
            "instances": instances, "ready": ready,
            "complete": complete, "epoch": epoch,
        }

    def _eligible(self, snapshot, source_index):
        counts = self._get_global_active_counts()
        eligible, rejected = [], []
        for handle in self._instances:
            info = snapshot["instances"][handle.instance_id]
            reasons = []
            if not handle.is_ready:
                reasons.append("not_ready")
            elif source_index != handle.index and self._max_queue_length > 0 and (
                self._active_count(handle, counts) >= self._max_queue_length
            ):
                reasons.append("capacity")
            if handle.is_ready:
                if not info["fresh"]:
                    reasons.append(info["reason"] or "missing_snapshot")
                elif info["kv"] >= self.KV_LIMIT:
                    reasons.append("kv_pressure")
            if reasons:
                rejected.append({
                    "instance_id": handle.instance_id,
                    "reason": reasons[0], "reasons": reasons,
                })
            else:
                eligible.append(handle)
        return eligible, rejected

    @staticmethod
    def _no_healthy_reason(rejected):
        reasons = {reason for row in rejected for reason in row["reasons"]}
        if "capacity" in reasons:
            return "no_fresh_healthy_instance:capacity"
        if "stale_snapshot" in reasons:
            return "no_fresh_healthy_instance:stale_metrics"
        if "invalid_kv" in reasons or "invalid_snapshot" in reasons:
            return "no_fresh_healthy_instance:invalid_metrics"
        if "missing_snapshot" in reasons:
            return "no_fresh_healthy_instance:missing_metrics"
        if "future_snapshot" in reasons:
            return "no_fresh_healthy_instance:future_metrics"
        if "kv_pressure" in reasons:
            return "no_fresh_healthy_instance:kv_pressure"
        return "no_fresh_healthy_instance"

    def _least_kv_with_round_robin(self, candidates, snapshot, tie_key):
        lowest = min(snapshot["instances"][h.instance_id]["kv"] for h in candidates)
        tied = sorted(
            (h for h in candidates if snapshot["instances"][h.instance_id]["kv"] == lowest),
            key=lambda handle: handle.index,
        )
        counter = self._initial_tie_counters.get(tie_key, 0)
        self._initial_tie_counters[tie_key] = counter + 1
        return tied[counter % len(tied)]

    def _distribution(self, request_id):
        state = self._request_states[request_id]
        return self.prefix_tree.residual_distribution(
            state.prompt_id, state.collected_return_states, self._bucket_thresholds,
        )

    @staticmethod
    def _remaining_mean(distribution):
        if not distribution:
            return None
        try:
            mean = sum(float(probability) * float(remaining)
                       for _, probability, remaining in distribution)
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return None
        return mean if math.isfinite(mean) else None

    def _preferred_bucket(self, mean):
        if mean is None:
            return None
        return (self._sorted_bucket_names[0] if mean <= self._dynamic_boundary
                else self._sorted_bucket_names[-1])

    def _choose(self, request_id: str, context_tokens: int | None = None):
        with self._lock:
            snapshot = self._capture_snapshot()
            self._update_boundary(snapshot)
            return self._choose_with_snapshot(request_id, context_tokens, snapshot)

    def _choose_with_snapshot(self, request_id, context_tokens, snapshot):
        state = self._request_states[request_id]
        source = self.get_instance_handle(state.current_instance_index)
        context = (
            self._input_lengths[request_id] + state.generated_tokens
            if context_tokens is None else context_tokens
        )
        distribution = self._distribution(request_id)
        remaining_mean = self._remaining_mean(distribution)
        preferred = self._preferred_bucket(remaining_mean)
        eligible, rejected = self._eligible(snapshot, state.current_instance_index)
        fallback_reason = ""
        if not distribution:
            fallback_reason = "no_distribution"
        elif remaining_mean is None:
            fallback_reason = "invalid_distribution"
        elif not eligible:
            fallback_reason = self._no_healthy_reason(rejected)
        if fallback_reason:
            return self._baseline_fallback(
                request_id, context_tokens, snapshot, eligible, rejected,
                preferred, remaining_mean, fallback_reason,
            )

        candidate_costs = {}
        raw_rows = []
        for handle in eligible:
            same = handle.index == source.index
            row = {
                "instance_index": handle.index,
                "bucket": self.bucket_for_tp(handle.tp_degree),
                "decode_seconds": None, "migration_seconds": None,
                "raw_total": None, "penalty": None, "score": None,
                "execution_path": None, "unusable_reason": None,
            }
            try:
                decode = float(sum(
                    float(probability) * float(self.routing_costs.decode_seconds(
                        handle.tp_degree, name, remaining, context,
                    ))
                    for name, probability, remaining in distribution
                ))
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                decode = float("nan")
            if math.isfinite(decode):
                row["decode_seconds"] = decode
            path, movement = "stay", 0.0
            if not same:
                if source.index not in self._topology or handle.index not in self._topology:
                    row["unusable_reason"] = "missing_topology"
                    candidate_costs[handle.instance_id] = row
                    continue
                try:
                    path, movement = self.routing_costs.migration_seconds(
                        self._topology[source.index], self._topology[handle.index], context,
                        source.is_ready and self._transfer_supported(source.index, handle.index),
                    )
                    movement = float(movement)
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    movement = float("nan")
            row["execution_path"] = path
            if math.isfinite(movement):
                row["migration_seconds"] = movement
            if not math.isfinite(decode) or not math.isfinite(movement):
                row["unusable_reason"] = "non_finite_cost"
            else:
                total = decode + movement
                if not math.isfinite(total):
                    row["unusable_reason"] = "non_finite_cost"
                elif total < 0:
                    row.update(raw_total=total, unusable_reason="negative_cost")
                else:
                    row["raw_total"] = total
                    raw_rows.append((total, handle, decode, movement, path))
            candidate_costs[handle.instance_id] = row

        if not raw_rows:
            return self._baseline_fallback(
                request_id, context_tokens, snapshot, eligible, rejected,
                preferred, remaining_mean, "no_finite_healthy_cost",
                candidate_costs=candidate_costs,
            )

        minimum = min(row[0] for row in raw_rows)
        choices = []
        for total, handle, decode, movement, path in raw_rows:
            penalty = (
                self.NONPREFERRED_PENALTY * max(minimum, 0.001)
                if self.bucket_for_tp(handle.tp_degree) != preferred else 0.0
            )
            score = total + penalty
            diagnostic = candidate_costs[handle.instance_id]
            diagnostic.update(penalty=penalty, score=score)
            if math.isfinite(score):
                kv = snapshot["instances"][handle.instance_id]["kv"]
                choices.append((
                    score, handle.index != source.index, kv, handle.index,
                    decode, movement, path,
                ))
            else:
                diagnostic["score"] = None
                diagnostic["unusable_reason"] = "non_finite_score"

        if not choices:
            return self._baseline_fallback(
                request_id, context_tokens, snapshot, eligible, rejected,
                preferred, remaining_mean, "no_finite_healthy_cost",
                candidate_costs=candidate_costs,
            )

        _, _, _, index, decode, movement, path = min(choices)
        target = self.get_instance_handle(index)
        decision = KVCostRoutingDecision(
            should_migrate=index != source.index,
            reason="kv_distribution_cost",
            current_bucket=state.current_bucket,
            target_bucket=self.bucket_for_tp(target.tp_degree),
            mean_remaining_length=remaining_mean,
            source_instance_index=source.index,
            target_instance_index=index,
            execution_path=path,
            decode_seconds=decode,
            migration_seconds=movement,
        )
        decision.kv_routing = self._routing_metadata(
            snapshot, eligible, rejected, index, preferred, remaining_mean,
            candidate_indices=[handle.index for handle in eligible],
        )
        decision.kv_routing["candidate_costs"] = candidate_costs
        decision.kv_routing["initial_route"] = self._initial_route_records.get(
            request_id, {},
        )
        return decision

    def _baseline_fallback(self, request_id, context_tokens, snapshot, eligible,
                           rejected, preferred, remaining_mean, reason,
                           candidate_costs=None):
        decision = super()._choose(request_id, context_tokens)
        decision.reason = f"kv_baseline_fallback:{reason}"
        decision.kv_routing = self._routing_metadata(
            snapshot, eligible, rejected, decision.target_instance_index,
            preferred, remaining_mean, reason,
            candidate_indices=[handle.index for handle in eligible],
        )
        if candidate_costs is not None:
            decision.kv_routing["candidate_costs"] = candidate_costs
        decision.kv_routing["initial_route"] = self._initial_route_records.get(
            request_id, {},
        )
        return decision

    def _routing_metadata(self, snapshot, eligible, rejected, chosen_index,
                          preferred_bucket=None, remaining_mean=None,
                          fallback_reason="", candidate_indices=None):
        infos = snapshot["instances"]
        chosen = (
            self.get_instance_handle(chosen_index)
            if chosen_index is not None and chosen_index >= 0 else None
        )
        snapshots = {
            name: {
                key: info[key]
                for key in ("kv", "age_s", "running", "waiting", "fresh")
            }
            for name, info in infos.items()
        }
        for name, info in infos.items():
            if "kv_capacity_tokens" in info:
                snapshots[name]["kv_capacity_tokens"] = info["kv_capacity_tokens"]
        meta = {
            "policy": self.POLICY,
            "boundary": self._dynamic_boundary,
            "preferred_bucket": preferred_bucket,
            "remaining_mean": remaining_mean,
            "epoch": [list(item) for item in snapshot["epoch"]] if snapshot["epoch"] else None,
            "snapshots": snapshots,
            "eligible_instances": [handle.instance_id for handle in eligible],
            "rejected": [dict(item) for item in rejected],
            "candidate_instances": [
                self.get_instance_handle(index).instance_id
                for index in (candidate_indices or [])
            ],
            "fallback_reason": fallback_reason or None,
            "chosen_instance_id": chosen.instance_id if chosen else None,
            "chosenKV": infos[chosen.instance_id]["kv"] if chosen else None,
            "boundary_event": dict(self._last_boundary_event) if self._last_boundary_event else None,
            "boundary_event_sequence": self._boundary_event_sequence,
            "boundary_transition_count": self._boundary_transition_count,
            "boundary_ledger": [dict(event) for event in self._boundary_ledger],
            "feedback": dict(self._last_feedback),
        }
        return _json_safe(meta)

    def _dynamic_feedback(self, snapshot):
        if not snapshot["complete"]:
            reasons = [
                snapshot["instances"][handle.instance_id]["reason"]
                for handle in snapshot["ready"]
                if not snapshot["instances"][handle.instance_id]["fresh"]
            ]
            return None, reasons[0] if reasons else "incomplete_snapshot"

        groups = {
            bucket: [
                handle for handle in snapshot["ready"]
                if self.bucket_for_tp(handle.tp_degree) == bucket
            ]
            for bucket in self._sorted_bucket_names
        }
        if any(not group for group in groups.values()):
            return None, "missing_bucket"

        members = {}
        for bucket, group in groups.items():
            rows = []
            capacities_known = True
            for handle in group:
                info = snapshot["instances"][handle.instance_id]
                try:
                    kv = float(info["kv"])
                    running = float(info["running"])
                    waiting = float(info["waiting"])
                except (TypeError, ValueError, OverflowError):
                    return None, "invalid_queue"
                if not math.isfinite(kv) or kv < 0 or kv > 1:
                    return None, "invalid_feedback_metric"
                if (not math.isfinite(running) or not math.isfinite(waiting)
                        or running < 0 or waiting < 0):
                    return None, "invalid_queue"
                capacity = info.get("kv_capacity_tokens")
                if not isinstance(capacity, int) or capacity <= 0:
                    capacities_known = False
                rows.append((kv, running, waiting, capacity))
            if capacities_known:
                total_capacity = sum(row[3] for row in rows)
                bucket_kv = sum(row[0] * (row[3] / total_capacity) for row in rows)
                kv_source = "capacity_weighted"
            else:
                bucket_kv = sum(row[0] for row in rows) / len(rows)
                kv_source = "equal_replica"
            members[bucket] = (rows, bucket_kv, kv_source)

        short_bucket = self._sorted_bucket_names[0]
        estimates = []
        for bucket in self._sorted_bucket_names:
            tp = groups[bucket][0].tp_degree
            try:
                estimate = float(self.routing_costs.decode_seconds(
                    tp, short_bucket, 4000, 0,
                ))
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                estimate = float("nan")
            estimates.append(estimate)
        service_source, service_reason = "cost_service_proxy", None
        if all(math.isfinite(value) and value > 0 for value in estimates):
            base = min(estimates)
            ratios = [value / base for value in estimates]
            if not all(math.isfinite(value) and value > 0 for value in ratios):
                ratios = [1.0] * len(self._sorted_bucket_names)
                service_source, service_reason = "tag_count_proxy", "non_finite_service_ratio"
        else:
            ratios = [1.0] * len(self._sorted_bucket_names)
            service_source, service_reason = "tag_count_proxy", "unusable_service_estimate"

        pressure_by_bucket = {}
        for bucket, ratio in zip(self._sorted_bucket_names, ratios):
            rows, bucket_kv, kv_source = members[bucket]
            queue_terms = [
                (waiting + self.FEEDBACK_RUNNING_WEIGHT * running) * ratio
                for _, running, waiting, _ in rows
            ]
            if not all(math.isfinite(value) and value >= 0 for value in queue_terms):
                return None, "unusable_pressure"
            q = sum(queue_terms) / len(queue_terms)
            queue = q / (q + self.FEEDBACK_QUEUE_SCALE)
            pressure = (
                self.FEEDBACK_KV_WEIGHT * bucket_kv
                + self.FEEDBACK_QUEUE_WEIGHT * queue
            )
            if not all(math.isfinite(value) for value in (q, queue, pressure)):
                return None, "unusable_pressure"
            pressure_by_bucket[bucket] = {
                "kv": bucket_kv, "kv_source": kv_source,
                "queue": queue, "service_ratio": ratio, "pressure": pressure,
            }
        difference = (
            pressure_by_bucket[self._sorted_bucket_names[-1]]["pressure"]
            - pressure_by_bucket[short_bucket]["pressure"]
        )
        if not math.isfinite(difference):
            return None, "unusable_pressure"
        return {
            "epoch": [list(item) for item in snapshot["epoch"]],
            "bucket_pressure": pressure_by_bucket,
            "raw_difference": difference,
            "service_source": service_source,
            "service_fallback_reason": service_reason,
        }, None

    def _update_boundary(self, snapshot):
        feedback, reason = self._dynamic_feedback(snapshot)
        if feedback is None:
            self._last_feedback = {
                "status": "frozen", "reason": reason,
                "epoch": [list(item) for item in snapshot["epoch"]]
                if snapshot["epoch"] else None,
            }
            return
        epoch = snapshot["epoch"]
        difference = feedback["raw_difference"]
        if self._last_boundary_epoch is None:
            now = time.monotonic()
            self._last_boundary_epoch = epoch
            self._last_boundary_monotonic = now
            self._feedback_ema_difference = difference
            self._last_feedback = {
                **feedback, "status": "primed",
                "smoothed_difference": difference, "step": 0,
            }
            return
        if epoch == self._last_boundary_epoch:
            self._last_feedback = {
                **feedback, "status": "same_epoch",
                "smoothed_difference": self._feedback_ema_difference, "step": 0,
            }
            return

        now = time.monotonic()
        elapsed = now - self._last_boundary_monotonic
        smoothed = (
            (1.0 - self.FEEDBACK_EMA_ALPHA) * self._feedback_ema_difference
            + self.FEEDBACK_EMA_ALPHA * difference
        )
        if (not math.isfinite(now) or not math.isfinite(elapsed)
                or elapsed < self.FEEDBACK_INTERVAL_SECONDS):
            self._last_feedback = {
                **feedback, "status": "interval_wait",
                "smoothed_difference": smoothed, "step": 0,
            }
            return

        self._last_boundary_epoch = epoch
        self._last_boundary_monotonic = now
        self._feedback_ema_difference = smoothed
        if abs(smoothed) <= self.FEEDBACK_DEADBAND:
            self._last_feedback = {
                **feedback, "status": "deadband",
                "smoothed_difference": smoothed, "step": 0,
            }
            return

        step = min(
            self.FEEDBACK_MAX_STEP,
            round(self.FEEDBACK_STEP_SCALE * (abs(smoothed) - self.FEEDBACK_DEADBAND)),
        )
        target = min(
            self.FEEDBACK_MAX_BOUNDARY,
            max(self.FEEDBACK_MIN_BOUNDARY,
                self._dynamic_boundary + (step if smoothed > 0 else -step)),
        )
        self._last_feedback = {
            **feedback, "status": "evaluated",
            "smoothed_difference": smoothed, "step": step,
        }
        if target == self._dynamic_boundary:
            return

        origin = self._sorted_bucket_names[-1] if smoothed > 0 else self._sorted_bucket_names[0]
        previous = self._dynamic_boundary
        self._dynamic_boundary = target
        self._boundary_event_sequence += 1
        self._boundary_transition_count += 1
        event = {
            "sequence": self._boundary_event_sequence,
            "from": previous, "to": target,
            "reason": "pressure_differential",
            "origin_bucket": origin,
            "epoch": [list(item) for item in epoch],
            "pressure": {
                name: item["pressure"] for name, item in feedback["bucket_pressure"].items()
            },
            "raw_difference": difference,
            "smoothed_difference": smoothed,
            "service_source": feedback["service_source"],
            "wall_time": time.time(),
            "monotonic_time": now,
        }
        self._last_boundary_event = event
        self._boundary_ledger.append(event)

    def _is_healthy(self, handle, snapshot):
        info = snapshot["instances"][handle.instance_id]
        return handle.is_ready and info["fresh"] and info["kv"] < self.KV_LIMIT

    def reserve_generation(self, request_id: str, context_tokens: int | None = None):
        with self._lock:
            state = self._request_states[request_id]
            snapshot = self._capture_snapshot()
            self._update_boundary(snapshot)
            first_generation = request_id in self._first_generation
            pending = request_id in self._pending_cost_routes
            preserve_initial = (
                first_generation and not pending
                and self._is_healthy(
                    self.get_instance_handle(state.current_instance_index), snapshot,
                )
            )
            added_pending = False
            if first_generation and not pending and not preserve_initial:
                self._pending_cost_routes.add(request_id)
                added_pending = True
            try:
                decision = super().reserve_generation(request_id, context_tokens)
            except Exception:
                if added_pending:
                    self._pending_cost_routes.discard(request_id)
                raise
            if preserve_initial:
                decision.reason = "initial_route_preserved"
            if not hasattr(decision, "kv_routing"):
                eligible, rejected = self._eligible(snapshot, state.current_instance_index)
                decision.kv_routing = self._routing_metadata(
                    snapshot, eligible, rejected, decision.target_instance_index,
                    candidate_indices=[handle.index for handle in eligible],
                )
                decision.kv_routing["initial_route"] = self._initial_route_records.get(
                    request_id, {},
                )
            return decision

    def commit_generation(self, request_id: str, decision: CostRoutingDecision):
        with self._lock:
            super().commit_generation(request_id, decision)
            self._first_generation.discard(request_id)

    def cancel_request(self, request_id: str):
        with self._lock:
            super().cancel_request(request_id)
            self._first_generation.discard(request_id)
            self._initial_route_records.pop(request_id, None)

    def finish_request(self, request_id: str, total_output_tokens: int, *,
                       generated_tokens: int | None = None):
        with self._lock:
            super().finish_request(request_id, total_output_tokens, generated_tokens=generated_tokens)
            self._first_generation.discard(request_id)
            self._initial_route_records.pop(request_id, None)

"""Support code for Base."""

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


DEFAULT_LOAD_METRIC = "requests"
VALID_LOAD_METRICS = ("requests", "tokens", "kv_tokens")
DEFAULT_EXPECTED_OUTPUT_TOKENS = 512
OUTPUT_EMA_ALPHA = 0.2
DEFAULT_KV_ACTIVATION_RESERVE_BYTES = 4 * 1024**3


@dataclass
class MetricsFeedbackConfig:
    """Closed-loop /metrics feedback settings for schedulers.

    ``enabled`` defaults off so existing configs keep the pure open-loop
    behavior. When on, engine-layer poller snapshots drive (a) an additive
    bias correction of the local load estimate and (b) occupancy admission
    with hysteresis; preemption events penalize an instance for a TTL
    window instead of blacklisting it forever (the counter is cumulative).
    """

    enabled: bool = False
    admission_enter: float = 0.90
    admission_exit: float = 0.75
    preemption_penalty_ttl_s: float = 60.0
    poll_interval_s: float = 3.0
    request_timeout_s: float = 1.0
    staleness_ttl_s: float = 10.0
    bias_alpha: float = 0.3

    def __post_init__(self):
        if not 0 <= self.admission_exit < self.admission_enter <= 1:
            raise ValueError("metrics admission requires 0 <= exit < enter <= 1")
        if not 0 < self.bias_alpha <= 1:
            raise ValueError("metrics_bias_alpha must be in (0, 1]")
        for name in ("poll_interval_s", "request_timeout_s", "staleness_ttl_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_scheduling(cls, sched: Any) -> "MetricsFeedbackConfig":
        get = lambda name, default: getattr(sched, name, default) if sched is not None else default
        return cls(
            enabled=bool(get("enable_metrics_feedback", False)),
            admission_enter=float(get("metrics_admission_enter", 0.90)),
            admission_exit=float(get("metrics_admission_exit", 0.75)),
            preemption_penalty_ttl_s=float(get("metrics_preemption_penalty_ttl_s", 60.0)),
            poll_interval_s=float(get("metrics_poll_interval_s", 3.0)),
            request_timeout_s=float(get("metrics_request_timeout_s", 1.0)),
            staleness_ttl_s=float(get("metrics_staleness_ttl_s", 10.0)),
            bias_alpha=float(get("metrics_bias_alpha", 0.3)),
        )


def estimate_kv_capacity_tokens(
    *,
    tp_degree: int,
    mem_capacity_bytes: float,
    gpu_memory_utilization: float,
    weights_bytes: float,
    n_layers: int,
    n_kv_heads: int,
    n_heads: int,
    d_model: int,
    head_dim: int = 0,
    dtype_bytes: int = 2,
    activation_reserve_bytes: float = DEFAULT_KV_ACTIVATION_RESERVE_BYTES,
) -> int:
    """Estimate how many tokens of KV cache a TP bucket can hold.

    Per GPU: ``mem_capacity * gpu_memory_utilization`` is the vLLM budget,
    minus the sharded weights and a fixed activation/workspace reserve.
    KV cost per token per GPU is ``2 (K+V) * n_layers * kv_heads_per_gpu *
    head_dim * dtype_bytes``; GQA replicates KV heads when the TP degree
    exceeds the number of KV heads.
    """
    tp = max(1, int(tp_degree))
    dim = int(head_dim) or max(1, int(d_model) // max(1, int(n_heads)))
    kv_heads_per_gpu = max(1, -(-max(1, int(n_kv_heads)) // tp))
    kv_bytes_per_token = 2 * max(1, int(n_layers)) * kv_heads_per_gpu * dim * max(1, int(dtype_bytes))
    budget = (
        float(mem_capacity_bytes) * min(1.0, max(0.05, float(gpu_memory_utilization)))
        - float(weights_bytes) / tp
        - float(activation_reserve_bytes)
    )
    if kv_bytes_per_token <= 0 or budget <= 0:
        return 0
    return int(budget // kv_bytes_per_token)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """Routing rule implementation."""
    category: str                       # "short" / "medium" / "long" / "extra_long"
    preferred_tp_degrees: list[int]
    fallback_tp_degrees: list[int] = field(default_factory=list)


@dataclass
class SchedulingResult:
    """Scheduling result implementation."""
    instance_index: int
    tp_degree: int
    category: str
    is_fallback: bool
    reason: str = ""
    pending: bool = False
    prompt_id: str = ""
    request_id: str = ""
    reserved_tokens: int | None = None


@dataclass
class SchedulerStats:
    """Scheduler stats implementation."""
    total_requests: int = 0
    preferred_routes: int = 0
    fallback_routes: int = 0
    failed_routes: int = 0
    pending_routes: int = 0
    migrated_routes: int = 0
    category_counts: dict = field(default_factory=lambda: defaultdict(int))
    category_tp_counts: dict = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.preferred_routes + self.fallback_routes) / self.total_requests

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "preferred_routes": self.preferred_routes,
            "fallback_routes": self.fallback_routes,
            "failed_routes": self.failed_routes,
            "pending_routes": self.pending_routes,
            "migrated_routes": self.migrated_routes,
            "success_rate": self.success_rate,
            "category_counts": dict(self.category_counts),
            "category_tp_counts": {
                cat: dict(tp_counts)
                for cat, tp_counts in self.category_tp_counts.items()
            },
        }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class InstanceHandle:
    """Instance handle implementation."""
    index: int
    instance_id: str
    tp_degree: int
    is_ready: bool = True
    active_requests: int = 0
    active_tokens: int = 0
    kv_capacity_tokens: int = 0
    _token_estimates: deque = field(default_factory=deque)

    def inc_active(
        self,
        prompt_tokens: int = 0,
        expected_output_tokens: int = 0,
    ):
        """Account a routed request.

        ``active_requests`` counts in-flight requests. ``active_tokens``
        tracks the estimated in-flight token load (prompt + expected
        output). Each estimated request pushes its estimate onto a FIFO
        queue so the exact same amount is subtracted at completion.
        """
        self.active_requests += 1
        estimate = max(0, int(prompt_tokens)) + max(0, int(expected_output_tokens))
        self.active_tokens += estimate
        self._token_estimates.append(estimate)

    def dec_active(self, reserved_tokens: int | None = None) -> int:
        """Retire a completed request.

        Subtracts only what was added at ``inc_active`` time: the oldest
        pending token estimate, if any. Keeps the invariant
        ``active_tokens == sum(_token_estimates)`` so the counter returns
        exactly to zero once all in-flight requests drain. Returns the
        token estimate actually retired (0 when the queue was empty), so
        callers can mirror the exact delta into a shared cross-rank state.
        """
        if reserved_tokens is not None:
            # Completion order is not submission order. Retire this request's
            # reservation, not a still-running long request at the FIFO head.
            self._token_estimates.remove(reserved_tokens)
            self.active_tokens -= reserved_tokens
            self.active_requests = max(0, self.active_requests - 1)
            return reserved_tokens
        self.active_requests = max(0, self.active_requests - 1)
        retired = 0
        if self._token_estimates:
            estimate = self._token_estimates.popleft()
            retired = estimate
            self.active_tokens = max(0, self.active_tokens - estimate)
        return retired

    def load(self, metric: str = DEFAULT_LOAD_METRIC, kv_capacity_tokens: int = 0) -> float:
        """Current load under the requested metric.

        ``kv_tokens`` returns the occupancy ratio (0..1+) of estimated
        in-flight tokens against the bucket's KV capacity. Callers must
        NOT compare ratios against raw token counts (unit mismatch);
        schedulers filter uncapacitated instances out of ``kv_tokens``
        candidate sets via ``is_kv_capacitated``.
        """
        if metric == "kv_tokens":
            return self.active_tokens / max(1, int(kv_capacity_tokens))
        if metric == "tokens":
            return float(self.active_tokens)
        return float(self.active_requests)

    @property
    def is_kv_capacitated(self) -> bool:
        """Whether a KV capacity has been assigned to this handle."""
        return self.kv_capacity_tokens > 0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class BaseScheduler(ABC):
    """Base scheduler implementation."""

    def __init__(
        self,
        name: str = "base",
        load_metric: str = DEFAULT_LOAD_METRIC,
        kv_capacity_tokens_by_tp: dict[int, int] | None = None,
        feedback: MetricsFeedbackConfig | None = None,
        prefix_affinity: bool = False,
        prefix_affinity_max_entries: int = 8192,
    ):
        self.name = name
        if load_metric not in VALID_LOAD_METRICS:
            logger.warning(
                "[%s] unknown load_metric '%s', falling back to '%s'",
                name, load_metric, DEFAULT_LOAD_METRIC,
            )
            load_metric = DEFAULT_LOAD_METRIC
        self._load_metric = load_metric
        self._kv_capacity_by_tp: dict[int, int] = {
            int(tp): int(cap)
            for tp, cap in (kv_capacity_tokens_by_tp or {}).items()
            if int(cap) > 0
        }
        if load_metric == "kv_tokens" and not self._kv_capacity_by_tp:
            logger.warning(
                "[%s] load_metric='kv_tokens' without kv_capacity_tokens_by_tp; "
                "falling back to unnormalized token counts",
                name,
            )
        self._feedback = feedback or MetricsFeedbackConfig()
        self._metrics_feed: Any = None
        self._bias_ema: dict[str, float] = {}
        self._bias_seen_ts: dict[str, float] = {}
        self._admission_blocked: dict[str, bool] = {}
        self._shared_state: Any = None
        self._capacity_seen: dict[str, int] = {}
        # Prefix affinity: prompt_id -> instance_id, bounded LRU. RL
        # rollouts repeat prompts (n_samples, multi-turn replays, shared
        # system prompts); keeping them on one instance lets vLLM's prefix
        # cache serve the repeated prefill instead of recomputing it on a
        # cold instance.
        self._prefix_affinity_enabled = bool(prefix_affinity)
        self._prefix_affinity_max_entries = max(64, int(prefix_affinity_max_entries))
        self._affinity: OrderedDict[str, str] = OrderedDict()
        self._instances: list[InstanceHandle] = []
        self._instances_by_tp: dict[int, list[InstanceHandle]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stats = SchedulerStats()
        self._tie_rr_counter = 0
        # Per-category EMA of observed output lengths, used to estimate the
        # output-token share of in-flight requests at routing time.
        self._output_ema: dict[str, float] = defaultdict(
            lambda: float(DEFAULT_EXPECTED_OUTPUT_TOKENS)
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def kv_capacity_of(self, handle: InstanceHandle) -> int:
        """Configured KV capacity (tokens) for a handle's TP bucket."""
        if self._metrics_feed is not None and self._feedback.enabled:
            snapshot = self._metrics_feed.get(handle.instance_id)
            capacity = getattr(snapshot, "kv_capacity_tokens", -1)
            if capacity > 0:
                return int(capacity)
        return handle.kv_capacity_tokens or self._kv_capacity_by_tp.get(handle.tp_degree, 0)

    def attach_metrics_feed(self, feed: Any) -> None:
        """Attach an engine-layer metrics poller (``get(instance_id)``)."""
        self._metrics_feed = feed

    def attach_shared_state(self, state: Any) -> None:
        """Attach an engine-layer cross-rank load publisher/aggregator.

        The engine (which survives scheduler rebinding) owns the state;
        attaching only wires this scheduler's accounting publishes and
        aggregated ranking reads. Callers must attach with the scheduler's
        local counters at zero (fresh scheduler) — the engine resets the
        rank's published totals before attaching.
        """
        self._shared_state = state

    def load_of(self, handle: InstanceHandle) -> float:
        """Load of a handle under the configured metric.

        With a shared state attached the ranking value is the
        cluster-wide aggregate (all live ranks' in-flight accounting,
        ours included) instead of this rank's partial view; without fresh
        aggregate data it degrades to the local estimate. When /metrics
        feedback is enabled, an additive bias (EMA of observed - base)
        then anchors whichever base is used to the real occupancy.
        """
        capacity = self.kv_capacity_of(handle)
        if self._capacity_seen.get(handle.instance_id, capacity) != capacity:
            self._bias_ema.pop(handle.instance_id, None)
            self._bias_seen_ts.pop(handle.instance_id, None)
        self._capacity_seen[handle.instance_id] = capacity
        if self._load_metric == "kv_tokens" and capacity <= 0 and any(
            self.kv_capacity_of(h) > 0 for h in self._instances
        ):
            return float("inf")
        base = handle.load(
            self._load_metric,
            kv_capacity_tokens=capacity,
        )
        base = self._apply_shared_aggregate(handle, base)
        return self._apply_metrics_bias(handle, base)

    def _apply_shared_aggregate(
        self, handle: InstanceHandle, local: float
    ) -> float:
        if self._shared_state is None:
            return local
        try:
            totals = self._shared_state.totals()
        except Exception as exc:
            logger.warning(
                "[%s] shared aggregate read failed, using local view: %s",
                self.name, exc,
            )
            return local
        entry = totals.get(handle.instance_id) if totals else None
        if not entry:
            return local
        if self._load_metric == "requests":
            return float(entry.get("requests", 0))
        tokens = float(entry.get("tokens", 0))
        if self._load_metric == "kv_tokens":
            capacity = self.kv_capacity_of(handle)
            return tokens / max(1, capacity)
        return tokens

    def _observed_load(self, handle: InstanceHandle, snapshot: Any) -> float | None:
        """Map a metrics snapshot onto the current load metric's unit."""
        if self._load_metric == "kv_tokens":
            if self.kv_capacity_of(handle) <= 0:
                return None
            usage = float(getattr(snapshot, "gpu_cache_usage", -1.0))
            return usage if 0.0 <= usage <= 1.0 else None
        if self._load_metric == "tokens":
            tokens = float(getattr(snapshot, "kv_cache_tokens", -1.0))
            if tokens < 0:
                capacity = getattr(snapshot, "kv_capacity_tokens", -1)
                usage = getattr(snapshot, "gpu_cache_usage", -1)
                if capacity > 0 and 0 <= usage <= 1:
                    tokens = capacity * usage
            return tokens if tokens >= 0 else None
        running = int(getattr(snapshot, "running", -1))
        waiting = int(getattr(snapshot, "waiting", -1))
        return float(running + max(0, waiting)) if running >= 0 else None

    def _apply_metrics_bias(self, handle: InstanceHandle, local: float) -> float:
        if self._metrics_feed is None or not self._feedback.enabled:
            return local
        snapshot = self._metrics_feed.get(handle.instance_id)
        if snapshot is None:
            self._bias_ema.pop(handle.instance_id, None)
            self._bias_seen_ts.pop(handle.instance_id, None)
            return local
        observed = self._observed_load(handle, snapshot)
        if observed is None:
            return local
        key = handle.instance_id
        updated_at = float(getattr(snapshot, "updated_at", 0.0))
        if updated_at > self._bias_seen_ts.get(key, 0.0):
            self._bias_seen_ts[key] = updated_at
            alpha = self._feedback.bias_alpha
            previous = self._bias_ema.get(key, 0.0)
            self._bias_ema[key] = alpha * (observed - local) + (1.0 - alpha) * previous
        return max(0.0, local + self._bias_ema.get(key, 0.0))

    def _admission_ok(self, handle: InstanceHandle) -> bool:
        """Occupancy admission with hysteresis and preemption TTL.

        Blocking requires a FRESH snapshot; a stale or missing feed keeps
        the last decision sticky so a transient scrape failure cannot flap
        admission state.
        """
        if self._metrics_feed is None or not self._feedback.enabled:
            return True
        key = handle.instance_id
        blocked = self._admission_blocked.get(key, False)
        snapshot = self._metrics_feed.get(key)
        if snapshot is None:
            self._admission_blocked.pop(key, None)
            return True
        now = time.time()
        last_preemption_at = float(getattr(snapshot, "last_preemption_at", 0.0))
        if (
            last_preemption_at > 0.0
            and now - last_preemption_at < self._feedback.preemption_penalty_ttl_s
        ):
            blocked = True
        else:
            occupancy = float(getattr(snapshot, "gpu_cache_usage", -1.0))
            if 0.0 <= occupancy <= 1.0:
                if not blocked and occupancy > self._feedback.admission_enter:
                    blocked = True
                elif blocked and occupancy < self._feedback.admission_exit:
                    blocked = False
        self._admission_blocked[key] = blocked
        return not blocked

    def _apply_admission(self, handles: list[InstanceHandle]) -> list[InstanceHandle]:
        if self._metrics_feed is None or not self._feedback.enabled:
            return handles
        admitted = [h for h in handles if self._admission_ok(h)]
        if admitted:
            return admitted
        # A preferred TP subset being blocked is not a global overload.
        comparable = self._instances
        if self._load_metric == "kv_tokens" and any(self.kv_capacity_of(h) > 0 for h in comparable):
            comparable = [h for h in comparable if self.kv_capacity_of(h) > 0]
        if any(h.is_ready and self._admission_ok(h) for h in comparable):
            return []
        # All instances overloaded (or degraded): keep every candidate so
        # routing still works instead of hard-failing; the least-loaded
        # pick then applies graceful overload shedding.
        return handles

    def _selectable(
        self,
        handles: list[InstanceHandle],
        metric: str | None = None,
    ) -> list[InstanceHandle]:
        """Filter handles comparable under the metric.

        Under ``kv_tokens`` every compared value must be a ratio; an
        instance whose TP bucket has no configured capacity would
        otherwise contribute a raw token count into a ratio-sorted
        candidate set, starving the big buckets. If NO instance has a
        capacity we fall back to comparing raw token counts for all
        (consistent units), matching the legacy degradation.
        """
        if (metric or self._load_metric) != "kv_tokens":
            return self._apply_admission(list(handles))
        capacitated = [h for h in handles if self.kv_capacity_of(h) > 0]
        if any(self.kv_capacity_of(h) > 0 for h in self._instances):
            if len(capacitated) < len(handles):
                missing = [
                    h.instance_id
                    for h in handles
                    if self.kv_capacity_of(h) <= 0
                ]
                logger.warning(
                    "[%s] kv_tokens: excluding instances without KV capacity "
                    "from selection: %s",
                    self.name,
                    ", ".join(missing),
                )
            return self._apply_admission(capacitated)
        return self._apply_admission(list(handles))

    def _expected_output_tokens(
        self, category: str = "any", max_new_tokens: int = 0
    ) -> int:
        """Expected generation length for a category (EMA, floor at 1).

        A small request budget caps a large historical mean to avoid
        overestimating output. This does not fix underestimation of tails.
        """
        expected = max(1, int(self._output_ema[category]))
        if max_new_tokens > 0:
            expected = min(expected, max(1, int(max_new_tokens)))
        return expected

    def _update_output_ema(self, category: str, output_tokens: int) -> None:
        """Fold an observed completion length into the category EMA."""
        if output_tokens <= 0:
            return
        prev = self._output_ema[category or "any"]
        self._output_ema[category or "any"] = (
            OUTPUT_EMA_ALPHA * float(output_tokens)
            + (1.0 - OUTPUT_EMA_ALPHA) * prev
        )

    def _record_route(
        self,
        handle: InstanceHandle,
        input_tokens: int,
        category: str = "any",
        prompt_id: str = "",
        max_new_tokens: int = 0,
    ) -> int:
        """Account a routed request (callers must hold ``self._lock``)."""
        expected = self._expected_output_tokens(category, max_new_tokens)
        handle.inc_active(
            prompt_tokens=max(0, int(input_tokens)),
            expected_output_tokens=expected,
        )
        if prompt_id and self._prefix_affinity_enabled:
            self._affinity_record(prompt_id, handle)
        estimate = max(0, int(input_tokens)) + expected
        if self._shared_state is not None:
            try:
                self._shared_state.add(
                    handle.instance_id, delta_requests=1, delta_tokens=estimate
                )
            except Exception as exc:
                logger.warning(
                    "[%s] shared-state publish failed on route: %s",
                    self.name, exc,
                )
        return estimate

    def _complete_route(
        self,
        instance_index: int,
        output_tokens: int = 0,
        category: str = "any",
        reserved_tokens: int | None = None,
    ) -> None:
        """Retire a completed request and update the output EMA.

        Callers must hold ``self._lock``.
        """
        handle = self.get_instance_handle(instance_index)
        retired = handle.dec_active(reserved_tokens) if handle is not None else 0
        if self._shared_state is not None and handle is not None:
            try:
                self._shared_state.add(
                    handle.instance_id,
                    delta_requests=-1,
                    delta_tokens=-retired,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] shared-state publish failed on completion: %s",
                    self.name, exc,
                )
        self._update_output_ema(category, output_tokens)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def register_instance(self, index: int, instance_id: str, tp_degree: int):
        """Register instance."""
        handle = InstanceHandle(
            index=index,
            instance_id=instance_id,
            tp_degree=tp_degree,
        )
        self._instances.append(handle)
        self._instances_by_tp[tp_degree].append(handle)
        logger.info(
            f"[{self.name}] registered instance {instance_id}: index={index}, TP={tp_degree} "
            f"({len(self._instances)} instances total)"
        )

    def get_instance_handle(self, index: int) -> Optional[InstanceHandle]:
        """Get instance handle."""
        for h in self._instances:
            if h.index == index:
                return h
        return None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @abstractmethod
    def schedule(
        self,
        input_tokens: int,
        prompt_id: str = "",
        n_samples: int = 1,
        epoch: int = -1,
        max_new_tokens: int = 0,
    ) -> SchedulingResult:
        """Schedule."""
        ...

    @abstractmethod
    def on_request_done(
        self,
        instance_index: int,
        prompt_id: str = "",
        final_bucket: str = "",
        output_tokens: int = 0,
    ):
        """On request done."""
        ...

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        """On epoch start."""
        pass

    def on_epoch_end(self, epoch: int):
        """On epoch end."""
        pass

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _min_load_rotating(self, candidates: list[InstanceHandle]) -> InstanceHandle:
        """Least-load selection with rotation among equal-load endpoints.

        A plain ``min`` always picks the first registered endpoint when
        requests complete between scheduling calls, starving later
        endpoints under low-concurrency multi-turn workloads. Rotation
        position is stored after the selected handle so endpoints are not
        permanently skipped when requests enter/leave the candidate set.
        """
        scores = {id(h): self.load_of(h) for h in candidates}
        min_load = min(scores.values())
        least_loaded = {key for key, score in scores.items() if score == min_load}
        start = self._tie_rr_counter % len(self._instances)
        for offset in range(len(self._instances)):
            position = (start + offset) % len(self._instances)
            handle = self._instances[position]
            if id(handle) in least_loaded:
                self._tie_rr_counter = (position + 1) % len(self._instances)
                return handle
        # Candidates not present in _instances (stale handles): fall back
        # to a deterministic first-min pick.
        return min(candidates, key=lambda h: self.load_of(h))

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _affinity_pick(
        self,
        prompt_id: str,
        candidates: list[InstanceHandle],
    ) -> InstanceHandle | None:
        """Sticky-instance pick for ``prompt_id`` among valid candidates.

        Callers pass the post-filter candidate set (readiness, queue,
        capacity normalization, admission): affinity never overrides
        those safety filters, it only reorders preference among the
        survivors. Returns ``None`` when affinity is off, the prompt is
        unknown, or its sticky instance did not survive filtering.
        Callers must hold ``self._lock``.
        """
        if not self._prefix_affinity_enabled or not prompt_id:
            return None
        instance_id = self._affinity.get(prompt_id)
        if instance_id is None:
            return None
        self._affinity.move_to_end(prompt_id)
        for handle in candidates:
            if handle.instance_id == instance_id:
                # Global-overload fallback may have reintroduced blocked or
                # full endpoints. Affinity must not override that condition.
                if not self._admission_ok(handle):
                    return None
                limit = getattr(self, "_max_queue_length", 0)
                if limit > 0 and handle.active_requests >= limit:
                    return None
                if self._load_metric == "kv_tokens" and self.load_of(handle) >= 1:
                    return None
                return handle
        return None

    def _affinity_record(self, prompt_id: str, handle: InstanceHandle) -> None:
        """Bind prompt_id to the instance that just served it (LRU).

        Callers must hold ``self._lock``.
        """
        if not self._prefix_affinity_enabled or not prompt_id:
            return
        self._affinity[prompt_id] = handle.instance_id
        self._affinity.move_to_end(prompt_id)
        while len(self._affinity) > self._prefix_affinity_max_entries:
            self._affinity.popitem(last=False)

    def export_learned_state(self) -> dict:
        """Scheduler state that should survive a rebind/reconfigure.

        Weight-sync rebinding rebuilds schedulers every training step
        (sync_interval defaults to 1); without carrying this state the
        output-length EMA would reset to its prior every step and never
        learn. Subclasses with more learned state (e.g. LA-MLFQ history)
        should extend this dict.
        """
        return {
            "output_ema": {
                cat: float(v) for cat, v in self._output_ema.items()
            },
            "prefix_affinity": dict(self._affinity),
        }

    def import_learned_state(self, state: dict) -> None:
        """Restore state exported by ``export_learned_state``.

        Only categories with valid values are applied; unknown/shorter
        states are tolerated so a scheduler-type change at rebind time
        degrades gracefully instead of crashing.
        """
        if not isinstance(state, dict):
            return
        ema = state.get("output_ema")
        if isinstance(ema, dict):
            for cat, v in ema.items():
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    self._output_ema[str(cat)] = v
        affinity = state.get("prefix_affinity")
        if (
            self._prefix_affinity_enabled
            and isinstance(affinity, dict)
            and affinity
        ):
            for pid, iid in affinity.items():
                if isinstance(pid, str) and isinstance(iid, str):
                    self._affinity[pid] = iid
            while len(self._affinity) > self._prefix_affinity_max_entries:
                self._affinity.popitem(last=False)

    def get_stats(self) -> SchedulerStats:
        return self._stats

    def reset_stats(self):
        self._stats = SchedulerStats()

    def print_stats(self):
        """Print stats."""
        s = self._stats
        print("\n" + "=" * 60)
        print(f"{self.name} scheduler statistics")
        print("=" * 60)
        print(f"  total requests: {s.total_requests}")
        print(f"  preferred routes: {s.preferred_routes}")
        print(f"  fallback routes: {s.fallback_routes}")
        print(f"  failed: {s.failed_routes}")
        print(f"  pending: {s.pending_routes}")
        print(f"  migrated: {s.migrated_routes}")
        print(f"  success rate: {s.success_rate:.2%}")
        print(f"\n  category distribution:")
        for cat, cnt in sorted(s.category_counts.items()):
            print(f"    {cat}: {cnt}")
        print(f"\n  category-to-TP distribution:")
        for cat, tp_counts in sorted(s.category_tp_counts.items()):
            for tp, cnt in sorted(tp_counts.items()):
                print(f"    {cat} -> TP={tp}: {cnt}")
        print(f"\n  instance status:")
        for h in self._instances:
            print(
                f"    {h.instance_id}: TP={h.tp_degree}, "
                f"active={h.active_requests}, active_tokens={h.active_tokens}, "
                f"ready={h.is_ready}"
            )
        print("=" * 60)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "BaseScheduler":
        """From config."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_config(), "
            "use SchedulerFactory.create() or a subclass implementation of from_config()"
        )

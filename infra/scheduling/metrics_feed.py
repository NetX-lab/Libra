"""Background /metrics feed from vLLM instances for closed-loop scheduling.

The poller lives at the engine layer (it must survive scheduler rebinding
during weight sync) and stores the latest Prometheus snapshot per instance.
Schedulers read snapshots through the ``get(instance_id)`` interface and use
them for additive bias correction and admission control; any feed outage
degrades transparently back to the local open-loop estimate.
"""

from __future__ import annotations

import logging
import math
import re
import random
import threading
import time
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InstanceMetrics:
    """Latest observed metrics for one vLLM instance."""

    instance_id: str
    updated_at: float
    running: int = -1
    waiting: int = -1
    gpu_cache_usage: float = -1.0      # 0..1 occupancy of the GPU KV cache
    kv_cache_tokens: float = -1.0
    preemptions_total: float = -1.0    # cumulative counter, never resets
    last_preemption_at: float = 0.0
    consecutive_failures: int = 0
    kv_capacity_tokens: int = -1       # vLLM-profiled capacity (config_info)


_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "running": (
        "vllm:num_requests_running",
        "vllm_num_requests_running",
    ),
    "waiting": (
        "vllm:num_requests_waiting",
        "vllm_num_requests_waiting",
    ),
    "gpu_cache_usage": (
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
        "vllm_kv_cache_usage_perc",
    ),
    "kv_cache_tokens": (
        "vllm:kv_cache_tokens",
        "vllm_kv_cache_tokens",
        "vllm:gpu_cache_usage_tokens",
        "vllm_gpu_cache_usage_tokens",
    ),
    "preemptions_total": (
        "vllm:num_preemptions_total",
        "vllm_num_preemptions_total",
        "vllm:preemption_total",
        "vllm_preemption_total",
        "vllm:preemptions_total",
        "vllm_preemptions_total",
    ),
}

# cache_config_info carries vLLM's own profiled KV capacity (ground truth);
# the label value lives in the exposition line itself, so it is parsed
# separately from the numeric gauges.
_KV_SIZE_LABEL_KEYS = (
    'kv_cache_size_tokens="',
    'kv_cache_size_tokens=\'',
)


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Lenient Prometheus text exposition parsing (name -> last value).

    Metric names differ across vLLM versions and may carry ``{label}``
    suffixes; both are tolerated. Lines we cannot understand are skipped
    instead of raising so a partial scrape still yields the core gauges.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([\w:]+)(?:\{.*\})?\s+([^\s]+)', line)
        if match is None:
            continue
        name, raw_value = match.groups()
        try:
            value = float(raw_value)
            if not math.isfinite(value):
                continue
            if name in values:
                if "cache_usage_perc" in name:
                    value = max(values[name], value)
                else:
                    value += values[name]
            values[name] = value
        except ValueError:
            continue
    return values


def extract_kv_capacity_tokens(text: str) -> int:
    """Pull vLLM's own profiled KV capacity out of cache_config_info.

    The gauge line embeds ``kv_cache_size_tokens="<N>"`` as a label, which
    the numeric parser cannot see. -1 when the exposition does not carry
    it (older vLLM versions).
    """
    for line in text.splitlines():
        if line.lstrip().startswith("#") or "cache_config_info" not in line:
            continue
        for key in _KV_SIZE_LABEL_KEYS:
            start = line.find(key)
            if start < 0:
                continue
            start += len(key)
            end = line.find(key[-1], start)
            if end <= start:
                continue
            try:
                value = int(float(line[start:end]))
            except (ValueError, OverflowError):
                continue
            if value > 0:
                return value
    return -1


def extract_core_metrics(values: dict[str, float]) -> dict[str, float]:
    """Map raw exposition values onto the core scheduling metrics."""
    normalized = {key.replace(":", "_"): value for key, value in values.items()}
    extracted: dict[str, float] = {}
    for field, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias in values:
                extracted[field] = values[alias]
                break
            alias = alias.replace(":", "_")
            if alias in normalized:
                extracted[field] = normalized[alias]
                break
    return extracted


class VLLMMetricsPoller:
    """Polls ``/metrics`` of every registered instance on a staggered clock.

    Each endpoint gets a random initial phase so multiple training ranks /
    instances do not scrape in lockstep (which would synchronize admission
    flapping). Snapshots older than ``ttl_s`` are reported as stale.
    """

    def __init__(
        self,
        interval_s: float = 3.0,
        timeout_s: float = 1.0,
        ttl_s: float = 10.0,
    ):
        self.interval_s = max(0.2, float(interval_s))
        self.timeout_s = max(0.1, float(timeout_s))
        self.ttl_s = max(self.interval_s * 2.0, float(ttl_s))
        self._endpoints: dict[str, str] = {}
        self._next_due: dict[str, float] = {}
        self._snapshots: dict[str, InstanceMetrics] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failures: dict[str, int] = {}

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def set_endpoints(self, urls: dict[str, str]) -> None:
        """Replace the endpoint table; fresh snapshots keep their TTL."""
        now = time.time()
        with self._lock:
            for instance_id, old_url in self._endpoints.items():
                if urls.get(instance_id) != old_url:
                    self._snapshots.pop(instance_id, None)
                    self._next_due.pop(instance_id, None)
            self._endpoints = dict(urls)
            for instance_id in urls:
                if instance_id not in self._next_due:
                    self._next_due[instance_id] = now + random.uniform(
                        0.0, self.interval_s
                    )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="vllm-metrics-poller",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Started vLLM metrics poller: interval=%.1fs timeout=%.1fs ttl=%.1fs endpoints=%d",
            self.interval_s,
            self.timeout_s,
            self.ttl_s,
            len(self._endpoints),
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0, self.timeout_s * 2.0))
        self._thread = None

    def get(self, instance_id: str, ttl_s: float | None = None) -> InstanceMetrics | None:
        """Latest snapshot if still fresh, else ``None``."""
        ttl = self.ttl_s if ttl_s is None else ttl_s
        with self._lock:
            snapshot = self._snapshots.get(instance_id)
        if snapshot is None:
            return None
        if time.time() - snapshot.updated_at > ttl:
            return None
        return snapshot

    def snapshots(self) -> dict[str, InstanceMetrics]:
        with self._lock:
            return dict(self._snapshots)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                due = [
                    (instance_id, url)
                    for instance_id, url in self._endpoints.items()
                    if now >= self._next_due.get(instance_id, 0.0)
                ]
                for instance_id, _ in due:
                    self._next_due[instance_id] = now + self.interval_s
            for instance_id, url in due:
                if self._stop_event.is_set():
                    break
                self._poll_one(instance_id, url)
            self._stop_event.wait(0.2)

    def _poll_one(self, instance_id: str, url: str) -> None:
        with self._lock:
            registered = instance_id in self._endpoints
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read().decode("utf-8", errors="replace")
            core = extract_core_metrics(parse_prometheus_metrics(payload))
            if not core:
                raise ValueError("/metrics contains no recognized vLLM load gauges")
        except Exception as exc:
            self._record_failure(instance_id, exc)
            return

        now = time.time()
        with self._lock:
            if self._stop_event.is_set() or (registered and self._endpoints.get(instance_id) != url):
                return
            previous = self._snapshots.get(instance_id)
            last_preemption_at = 0.0
            preemptions_total = core.get("preemptions_total", -1.0)
            if (
                previous is not None
                and preemptions_total >= 0
                and previous.preemptions_total >= 0
                and preemptions_total > previous.preemptions_total
            ):
                # Preemption counters are cumulative; a preemption happened
                # between the two polls. Pin the event time to now (worst
                # case: up to one interval of uncertainty).
                last_preemption_at = now
            elif previous is not None and not (
                0 <= preemptions_total < previous.preemptions_total
            ):
                last_preemption_at = previous.last_preemption_at
            self._failures[instance_id] = 0
            self._snapshots[instance_id] = InstanceMetrics(
                instance_id=instance_id,
                updated_at=now,
                running=int(core.get("running", -1)),
                waiting=int(core.get("waiting", -1)),
                gpu_cache_usage=float(core.get("gpu_cache_usage", -1.0)),
                kv_cache_tokens=float(core.get("kv_cache_tokens", -1.0)),
                preemptions_total=preemptions_total,
                last_preemption_at=last_preemption_at,
                consecutive_failures=0,
                kv_capacity_tokens=extract_kv_capacity_tokens(payload),
            )

    def _record_failure(self, instance_id: str, exc: Exception) -> None:
        with self._lock:
            previous = self._snapshots.get(instance_id)
            failures = self._failures.get(instance_id, 0) + 1
            self._failures[instance_id] = failures
            if previous is not None:
                previous.consecutive_failures = failures
            if failures % 10 == 1:
                logger.warning(
                    "Metrics poll failed for %s (failures=%d): %s",
                    instance_id,
                    failures,
                    exc,
                )

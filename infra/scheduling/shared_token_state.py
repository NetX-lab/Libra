"""Cross-rank shared load publication for token-aware scheduling.

Generalizes the C-MLFQ ``SharedCMLFQLoadState`` pattern to every
scheduler: each training rank publishes its per-instance in-flight
accounting (request count + estimated token load) to a shared filesystem
using atomic renames plus a heartbeat TTL. Schedulers aggregate all live
ranks' files so instance ranking reflects cluster-wide load instead of a
single rank's partial view — without this, N independent least-connections
schedulers all see the same "empty" instance and herd traffic onto it.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class SharedTokenLoadState:
    """Publish per-rank token/request accounting; aggregate across ranks.

    Each process is the only writer of its own snapshot file. Aggregate
    reads sum every live file (heartbeat within TTL) so a crashed rank's
    counts disappear instead of poisoning the total. A short-lived cache
    (invalidated by our own writes) keeps per-schedule aggregate reads
    cheap while other ranks' updates appear within ``cache_ttl_s``.
    """

    def __init__(
        self,
        directory: str,
        ttl_s: float = 30.0,
        heartbeat_interval_s: float = 10.0,
        writer_id: str = "",
        cache_ttl_s: float = 1.0,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        self.writer_id = writer_id or f"rank_{rank}"
        if Path(self.writer_id).name != self.writer_id:
            raise ValueError("writer_id must be a file name, not a path")
        self._owner = uuid.uuid4().hex
        self.ttl_s = max(heartbeat_interval_s * 2.0, ttl_s)
        self.heartbeat_interval_s = max(1.0, heartbeat_interval_s)
        self.cache_ttl_s = max(0.0, float(cache_ttl_s))
        self._path = self.directory / f"{self.writer_id}.json"
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cache: dict[str, dict[str, int]] | None = None
        self._cache_at: float = 0.0
        self._publish()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"shared-token-load-{self.writer_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        atexit.register(self.close)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def add(
        self,
        instance_id: str,
        delta_requests: int = 0,
        delta_tokens: int = 0,
    ) -> None:
        """Apply a delta to this rank's published accounting."""
        with self._lock:
            if self._stop_event.is_set():
                return
            entry = self._counts.setdefault(
                instance_id, {"requests": 0, "tokens": 0}
            )
            entry["requests"] = max(0, entry["requests"] + int(delta_requests))
            entry["tokens"] = max(0, entry["tokens"] + int(delta_tokens))
            self._publish_locked()
            self._cache = None

    def reset(self) -> None:
        """Zero this rank's accounting (e.g. a fresh scheduler attached)."""
        with self._lock:
            if self._stop_event.is_set():
                return
            self._counts.clear()
            self._publish_locked()
            self._cache = None

    def totals(self) -> dict[str, dict[str, int]]:
        """Cluster-wide sums over live rank files (cached).

        Returns ``{instance_id: {"requests": int, "tokens": int}}``.
        """
        now = time.monotonic()
        with self._lock:
            if (
                self._cache is not None
                and self.cache_ttl_s > 0.0
                and now - self._cache_at < min(self.cache_ttl_s, self.ttl_s)
            ):
                return self._with_local(self._cache)
        wall = time.time()
        aggregated: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "tokens": 0}
        )
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError:
            paths = []
        for path in paths:
            if path == self._path:
                continue  # own contribution comes from live memory below
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if wall - float(payload.get("updated_at", 0.0)) > self.ttl_s:
                    continue
                counts = payload.get("counts", {})
                if not isinstance(counts, dict):
                    continue
                for instance_id, entry in counts.items():
                    if not isinstance(entry, dict):
                        continue
                    total = aggregated[str(instance_id)]
                    total["requests"] += max(0, int(entry.get("requests", 0)))
                    total["tokens"] += max(0, int(entry.get("tokens", 0)))
            except (OSError, ValueError, TypeError, OverflowError):
                continue
        result = {
            instance_id: dict(entry) for instance_id, entry in aggregated.items()
        }
        with self._lock:
            self._cache = result
            self._cache_at = time.monotonic()
            return self._with_local(result)

    def _with_local(self, peers):
        result = {iid: dict(entry) for iid, entry in peers.items()}
        for iid, entry in self._counts.items():
            total = result.setdefault(iid, {"requests": 0, "tokens": 0})
            for key in ("requests", "tokens"):
                total[key] += entry[key]
        return result

    def close(self) -> None:
        """Stop the heartbeat and remove our file so peers drop our counts."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._heartbeat_thread.join(timeout=self.heartbeat_interval_s + 1)
        with self._lock:
            self._counts.clear()
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                if payload.get("owner") == self._owner:
                    self._path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
            self._cache = None
        atexit.unregister(self.close)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_s):
            try:
                self._publish()
            except OSError as exc:
                logger.warning("Shared load heartbeat failed: %s", exc)

    def _publish(self) -> None:
        with self._lock:
            self._publish_locked()

    def _publish_locked(self) -> None:
        if self._stop_event.is_set():
            return
        payload = {
            "owner": self._owner,
            "writer_id": self.writer_id,
            "rank": int(os.environ.get("RANK", "0")),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "updated_at": time.time(),
            "counts": {
                iid: dict(entry) for iid, entry in self._counts.items()
            },
        }
        tmp_path = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._path)

"""Cross-process load snapshots for C-MLFQ schedulers."""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path


class SharedCMLFQLoadState:
    """Publish per-rank counters and aggregate them from a shared filesystem.

    Each process is the only writer of its own snapshot. Atomic rename keeps
    readers from observing partial JSON, while heartbeats let readers discard
    counters left behind by a failed training process.
    """

    def __init__(
        self,
        directory: str,
        ttl_s: float = 60.0,
        heartbeat_interval_s: float = 10.0,
        writer_id: str = "",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        self.writer_id = writer_id or f"rank_{rank}"
        self.ttl_s = max(heartbeat_interval_s * 2.0, ttl_s)
        self.heartbeat_interval_s = max(1.0, heartbeat_interval_s)
        self._path = self.directory / f"{self.writer_id}.json"
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._publish()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"cmlfq-heartbeat-{self.writer_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        atexit.register(self.close)

    def increment(self, instance_id: str) -> None:
        with self._lock:
            self._counts[instance_id] = self._counts.get(instance_id, 0) + 1
            self._publish_locked()

    def decrement(self, instance_id: str) -> None:
        with self._lock:
            self._counts[instance_id] = max(
                0, self._counts.get(instance_id, 0) - 1
            )
            self._publish_locked()

    def aggregate(self) -> dict[str, int]:
        now = time.time()
        totals: dict[str, int] = defaultdict(int)
        for path in self.directory.glob("rank_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                updated_at = float(payload.get("updated_at", 0.0))
                if now - updated_at > self.ttl_s:
                    continue
                for instance_id, count in payload.get("counts", {}).items():
                    totals[str(instance_id)] += max(0, int(count))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return dict(totals)

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        with self._lock:
            self._counts.clear()
            self._publish_locked()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_s):
            self._publish()

    def _publish(self) -> None:
        with self._lock:
            self._publish_locked()

    def _publish_locked(self) -> None:
        payload = {
            "writer_id": self.writer_id,
            "rank": int(os.environ.get("RANK", "0")),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "updated_at": time.time(),
            "counts": dict(self._counts),
        }
        tmp_path = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._path)

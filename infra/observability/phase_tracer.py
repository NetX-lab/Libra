"""Low-overhead per-rank phase tracing for long RL runs."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class PhaseTracer:
    """Write append-only JSONL spans that can be aggregated across ranks."""

    def __init__(self, path: str | os.PathLike[str], *, rank: int, world_size: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._starts: dict[tuple[int, str], tuple[int, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _write(self, record: dict[str, Any]) -> None:
        record = {
            "schema": "libra.phase_trace.v1",
            "rank": self.rank,
            "world_size": self.world_size,
            **record,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()

    def start(self, step: int, phase: str, details: dict[str, Any]) -> None:
        now = time.time_ns()
        key = (int(step), str(phase))
        with self._lock:
            self._starts[key] = (now, dict(details))
            self._write({"event": "phase_start", "step": int(step), "phase": str(phase), "ts_ns": now, "details": details})

    def end(self, step: int, phase: str, details: dict[str, Any]) -> None:
        now = time.time_ns()
        key = (int(step), str(phase))
        with self._lock:
            started = self._starts.pop(key, None)
            start_ns = started[0] if started else now
            merged = dict(started[1]) if started else {}
            merged.update(details)
            self._write({
                "event": "phase_span",
                "step": int(step),
                "phase": str(phase),
                "start_ns": start_ns,
                "end_ns": now,
                "duration_s": (now - start_ns) / 1.0e9,
                "details": merged,
            })

    def close(self) -> None:
        now = time.time_ns()
        with self._lock:
            for (step, phase), (start_ns, details) in list(self._starts.items()):
                self._write({
                    "event": "phase_unfinished",
                    "step": step,
                    "phase": phase,
                    "start_ns": start_ns,
                    "end_ns": now,
                    "duration_s": (now - start_ns) / 1.0e9,
                    "details": details,
                })
            self._starts.clear()

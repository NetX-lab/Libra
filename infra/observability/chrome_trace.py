"""Small, process-local Chrome Trace Event collector.

Each process writes its own trace file.  This is intentional: multiple trainer
and rollout processes must not append to the same JSON document concurrently.
Use ``scripts/merge_chrome_traces.py`` after a run to create one trace for
Chrome Tracing or Perfetto.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import socket
import threading
import time
from typing import Any, Iterator


class ChromeTraceCollector:
    """Write Chrome Trace Event JSON from one process.

    The collector is deliberately dependency-free and cheap when disabled.
    Event timestamps use wall-clock microseconds so traces from different
    processes can be merged, while span durations use a monotonic clock.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str] = "",
        *,
        enabled: bool = True,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        process_name: str = "rl_framework",
    ) -> None:
        self.enabled = bool(enabled and output_dir)
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.process_name = str(process_name)
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._closed = False
        self._has_events = False
        self._file = None
        self.path: Path | None = None

        if not self.enabled:
            return

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"trace.rank_{self.rank}.pid_{self.pid}.json"
        self._file = self.path.open("w", encoding="utf-8")
        self._file.write('{"traceEvents":[\n')
        self._write_metadata()

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        process_name: str = "rl_framework",
    ) -> "ChromeTraceCollector":
        """Create a collector from an AsyncRLConfig-like object."""
        output_dir = str(getattr(config, "chrome_trace_output_dir", "") or "")
        if not output_dir:
            log_dir = str(getattr(config, "log_dir", "") or "")
            output_dir = str(Path(log_dir or ".") / "chrome_trace")
        return cls(
            output_dir,
            enabled=bool(getattr(config, "enable_chrome_trace", False)),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            process_name=process_name,
        )

    @staticmethod
    def _json_safe(value: Any, *, max_string_length: int = 1024) -> Any:
        """Convert event arguments to bounded JSON-compatible values."""
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            if len(value) <= max_string_length:
                return value
            return value[:max_string_length] + "..."
        if isinstance(value, dict):
            return {
                str(key): ChromeTraceCollector._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [ChromeTraceCollector._json_safe(item) for item in value]
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return ChromeTraceCollector._json_safe(str(value))
        return value

    @staticmethod
    def _wall_timestamp_us() -> int:
        return time.time_ns() // 1_000

    def _write_event(self, event: dict[str, Any]) -> None:
        if not self.enabled or self._file is None:
            return
        event.setdefault("pid", self.pid)
        event.setdefault("tid", threading.get_ident())
        event = self._json_safe(event)
        with self._lock:
            if self._closed or self._file is None:
                return
            if self._has_events:
                self._file.write(",\n")
            self._file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            self._file.flush()
            self._has_events = True

    def _write_metadata(self) -> None:
        metadata = {
            "process_name": self.process_name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "host": socket.gethostname(),
            "pid": self.pid,
        }
        for name, value in metadata.items():
            self._write_event({
                "name": name,
                "cat": "metadata",
                "ph": "M",
                "ts": self._wall_timestamp_us(),
                "args": {"name": value} if name == "process_name" else {"value": value},
            })

    def instant(
        self,
        name: str,
        *,
        cat: str = "rl",
        args: dict[str, Any] | None = None,
        tid: int | str | None = None,
    ) -> None:
        """Record an instant event on the current thread."""
        event: dict[str, Any] = {
            "name": str(name),
            "cat": str(cat),
            "ph": "i",
            "s": "t",
            "ts": self._wall_timestamp_us(),
            "args": args or {},
        }
        if tid is not None:
            event["tid"] = tid
        self._write_event(event)

    def begin(
        self,
        name: str,
        *,
        cat: str = "rl",
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Start a span and return an opaque token for :meth:`end`."""
        if not self.enabled:
            return None
        return {
            "name": str(name),
            "cat": str(cat),
            "args": dict(args or {}),
            "wall_start_us": self._wall_timestamp_us(),
            "mono_start_ns": time.monotonic_ns(),
            "tid": threading.get_ident(),
        }

    def end(
        self,
        token: dict[str, Any] | None,
        *,
        args: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Finish a span.  Missing/disabled tokens are harmless."""
        if not token or not self.enabled:
            return
        event_args = dict(token.get("args") or {})
        if args:
            event_args.update(args)
        if error:
            event_args["error"] = error
        duration_us = max(
            0,
            (time.monotonic_ns() - int(token["mono_start_ns"])) // 1_000,
        )
        self._write_event({
            "name": token["name"],
            "cat": token["cat"],
            "ph": "X",
            "ts": token["wall_start_us"],
            "dur": duration_us,
            "tid": token["tid"],
            "args": event_args,
        })

    @contextmanager
    def span(
        self,
        name: str,
        *,
        cat: str = "rl",
        args: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record one complete duration event around a block."""
        token = self.begin(name, cat=cat, args=args)
        try:
            yield
        except Exception as exc:
            self.end(token, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.end(token)

    def close(self) -> None:
        """Close the JSON document; safe to call more than once."""
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._file is not None:
                self._file.write('\n],"displayTimeUnit":"ms"}\n')
                self._file.flush()
                self._file.close()
                self._file = None

    def __enter__(self) -> "ChromeTraceCollector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ChromeTraceCollector"]

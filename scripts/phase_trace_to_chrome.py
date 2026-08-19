#!/usr/bin/env python3
"""Convert Libra per-rank phase JSONL files to Chrome Trace JSON.

Example:
  python scripts/phase_trace_to_chrome.py \
    --input-glob 'logs/phase_trace/phase_trace_rank*.jsonl' \
    --output logs/phase_trace/chrome_trace.json

The output opens in Perfetto or Chrome's ``chrome://tracing`` and keeps the
same phase names across runs, making baseline comparisons straightforward.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def convert(paths: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("event") not in {"phase_span", "phase_unfinished"}:
                    continue
                rank = int(record.get("rank", 0))
                phase = str(record.get("phase", "unknown"))
                events.append(
                    {
                        "name": phase,
                        "cat": "libra.phase",
                        "ph": "X",
                        "ts": int(record["start_ns"]) / 1000.0,
                        "dur": max(
                            0.0,
                            (int(record["end_ns"]) - int(record["start_ns"]))
                            / 1000.0,
                        ),
                        "pid": "rank-%d" % rank,
                        "tid": phase,
                        "args": {
                            "step": record.get("step"),
                            **dict(record.get("details") or {}),
                        },
                    }
                )
    events.sort(key=lambda event: (event["ts"], str(event["pid"])))
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"no trace files matched: {args.input_glob}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"traceEvents": convert(paths)}, indent=2), encoding="utf-8")
    print(f"wrote {output} from {len(paths)} rank files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge process-local Chrome Trace JSON files into one trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def merge_trace_files(paths: Iterable[Path]) -> dict:
    events: list[dict] = []
    for path in sorted(paths):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        events.extend(payload.get("traceEvents", []))
    events.sort(key=lambda event: (event.get("ts", 0), event.get("pid", 0)))
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pattern",
        default="trace.rank_*.pid_*.json",
        help="Input filename pattern (default: %(default)s)",
    )
    args = parser.parse_args()

    paths = list(args.input_dir.glob(args.pattern))
    if not paths:
        parser.error(f"no trace files found in {args.input_dir}: {args.pattern}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merge_trace_files(paths), ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"merged {len(paths)} trace files -> {args.output}")


if __name__ == "__main__":
    main()

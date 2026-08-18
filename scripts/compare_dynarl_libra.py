#!/usr/bin/env python3
"""Summarize RLinf/DynaRL logs and Libra rollout refresh results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NUMBER_PATTERN = r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
METRIC_RE = re.compile(rf"([A-Za-z0-9_/.-]+)=({NUMBER_PATTERN})")
MIGRATION_RE = re.compile(
    rf"nccl_build_cost=(?P<build>{NUMBER_PATTERN})ms, "
    rf"transfer_cost=(?P<transfer>{NUMBER_PATTERN})ms, "
    rf"update_model_cost=(?P<update>{NUMBER_PATTERN})ms, "
    rf"transfer_bytes=(?P<bytes>{NUMBER_PATTERN})GB, "
    rf"bandwidth=(?P<bandwidth>{NUMBER_PATTERN}) GB/s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynarl-log", type=Path)
    parser.add_argument("--static-log", type=Path)
    parser.add_argument("--libra-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def parse_rlinf_log(path: Path) -> dict[str, Any]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    steps = []
    for line in text.splitlines():
        if "Global Step:" not in line or "step_time=" not in line:
            continue
        metrics = {name: float(value) for name, value in METRIC_RE.findall(line)}
        if metrics:
            steps.append(metrics)

    migrations = [
        {
            "nccl_build_ms": float(match.group("build")),
            "transfer_ms": float(match.group("transfer")),
            "update_model_ms": float(match.group("update")),
            "transfer_gb": float(match.group("bytes")),
            "bandwidth_gbps": float(match.group("bandwidth")),
        }
        for match in MIGRATION_RE.finditer(text)
    ]
    warm_steps = steps[1:] if len(steps) > 1 else steps
    return {
        "path": str(path),
        "num_steps": len(steps),
        "steps": steps,
        "warm_step_time_mean_s": _mean(
            [step["step_time"] for step in warm_steps if "step_time" in step]
        ),
        "warm_sync_time_mean_s": _mean(
            [
                step["sync_weights_time"]
                for step in warm_steps
                if "sync_weights_time" in step
            ]
        ),
        "migrations": migrations,
        "migration_transfer_mean_ms": _mean(
            [migration["transfer_ms"] for migration in migrations]
        ),
    }


def parse_libra_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds") or [payload]
    return {
        "path": str(path),
        "model": payload.get("model"),
        "outputs_stable": payload.get("outputs_stable"),
        "rounds": rounds,
        "cold_reload_s": rounds[0].get("send_seconds"),
        "steady_reload_s": rounds[-1].get("send_seconds"),
        "steady_payload_gib_per_second": rounds[-1].get(
            "payload_gib_per_second"
        ),
    }


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {}
    if args.dynarl_log:
        result["dynarl"] = parse_rlinf_log(args.dynarl_log)
    if args.static_log:
        result["rlinf_static"] = parse_rlinf_log(args.static_log)
    if args.libra_result:
        result["libra"] = parse_libra_result(args.libra_result)

    dynarl_step = result.get("dynarl", {}).get("warm_step_time_mean_s")
    static_step = result.get("rlinf_static", {}).get("warm_step_time_mean_s")
    if dynarl_step and static_step:
        result["pilot_dynarl_vs_static_step_speedup"] = static_step / dynarl_step

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

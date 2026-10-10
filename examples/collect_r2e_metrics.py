"""Aggregate completed R2E-Gym steps and compare one arm with Baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, .50),
        "p95": percentile(values, .95),
        "p99": percentile(values, .99),
    }


def collect(log_dir: Path, warmup_steps: int) -> dict:
    timings = {}
    for path in (log_dir / "step_timings").glob("step_*.json"):
        record = json.loads(path.read_text())
        timings[int(record["step"])] = float(record["step_latency_s"])
    completed = sorted(timings)
    measured = completed[warmup_steps:]
    if not measured:
        raise ValueError("No completed steady-state steps")
    records = []
    for step in measured:
        paths = list((log_dir / "rank_ready").glob(f"job_*/step_{step}/rank_*.json"))
        if not paths:
            raise ValueError(f"Missing rank-ready telemetry for step {step}")
        rows = [json.loads(path.read_text()) for path in paths]
        if len(rows) != int(rows[0]["world_size"]):
            raise ValueError(f"Incomplete rank telemetry for step {step}")
        sources = [row for row in rows if row.get("is_batch_source")]
        if not sources:
            raise ValueError(f"No source ranks for step {step}")
        records.extend(sources)
    tokens = sum(sum(int(n) for n in row["output_lengths"]) for row in records)
    trajectories = sum(int(row["batch_size"]) for row in records)
    elapsed = sum(timings[step] for step in measured)
    first = [float(n) for row in records for n in row.get("first_token_latencies_s", [])]
    request = [float(n) for row in records for n in row.get("request_e2e_latencies_s", [])]
    if not first or not request:
        raise ValueError("Missing request timing samples; refusing incomplete report")
    return {
        "completed_steps": completed,
        "measured_steps": measured,
        "warmup_steps": warmup_steps,
        "generated_tokens": tokens,
        "trajectories": trajectories,
        "measured_elapsed_s": elapsed,
        "generated_tokens_per_s": tokens / elapsed,
        "trajectories_per_s": trajectories / elapsed,
        "first_token_latency_s": distribution(first),
        "request_e2e_latency_s": distribution(request),
        "steady_step_latency_s": distribution([timings[step] for step in measured]),
    }


def compare(report: dict, baseline: dict) -> dict:
    def throughput(key: str) -> float:
        return (report[key] / baseline[key] - 1) * 100

    def latency(kind: str, stat: str) -> float:
        return (1 - report[kind][stat] / baseline[kind][stat]) * 100

    return {
        "generated_tokens_per_s_gain_pct": throughput("generated_tokens_per_s"),
        "trajectories_per_s_gain_pct": throughput("trajectories_per_s"),
        "latency_reduction_pct": {
            kind: {stat: latency(kind, stat) for stat in ("mean", "p50", "p95", "p99")}
            for kind in ("first_token_latency_s", "request_e2e_latency_s", "steady_step_latency_s")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect(args.log_dir, args.warmup_steps)
    if args.baseline:
        report["relative_to_baseline"] = compare(report, json.loads(args.baseline.read_text()))
    result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()

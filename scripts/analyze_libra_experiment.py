#!/usr/bin/env python3
"""Summarize Libra RL and systems-performance experiment logs.

The analyzer is intentionally dependency-light so it can run on login nodes.
It reads history JSONL files, runtime reconfiguration event logs, C-MLFQ tree
snapshots, and rank0 training logs, then emits a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any


STEP_RE = re.compile(r"^Step\s+(\d+)/(\d+)")
TIME_RE = re.compile(r"Time:\s+([0-9.]+)s\s+\(rollout=([0-9.]+)s,\s+train=([0-9.]+)s\)")
REWARD_RE = re.compile(r"Reward:\s+([-+0-9.eE]+)")
TRAJ_RE = re.compile(r"Trajectories:\s+(\d+)")


@dataclass
class RunSummary:
    name: str
    path: Path
    history_files: list[Path] = field(default_factory=list)
    history_records: list[dict[str, Any]] = field(default_factory=list)
    log_step_records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cmlfq_trees: list[Path] = field(default_factory=list)
    cmlfq_tree_stats: dict[str, Any] = field(default_factory=dict)
    log_counts: dict[str, int] = field(default_factory=dict)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _last_window(values: list[float], n: int = 10) -> list[float]:
    if len(values) < n * 2 and len(values) >= 2:
        return values[len(values) // 2 :]
    return values[-n:] if len(values) >= n else values


def _first_window(values: list[float], n: int = 10) -> list[float]:
    if len(values) < n * 2 and len(values) >= 2:
        return values[: len(values) // 2]
    return values[:n] if len(values) >= n else values


def _history_metric(records: list[dict[str, Any]], *path: str) -> list[Any]:
    out: list[Any] = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            out.append(value)
    return out


def _tokens_per_second(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        timing = record.get("timing", {})
        step_time = _safe_float(timing.get("step_total_time"))
        if step_time <= 0:
            continue
        seqs = record.get("sequences") or []
        total_tokens = 0
        for seq in seqs:
            total_tokens += _safe_int(seq.get("input_len")) + _safe_int(
                seq.get("output_len")
            )
        if total_tokens > 0:
            values.append(total_tokens / step_time)
    return values


def _trajectory_throughput(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        timing = record.get("timing", {})
        step_time = _safe_float(timing.get("step_total_time"))
        metrics = record.get("training_metrics", {})
        n_traj = _safe_int(metrics.get("n_trajectories"))
        if step_time > 0 and n_traj > 0:
            values.append(n_traj / step_time)
    return values


def _parse_rank0_logs(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for log_path in sorted(path.rglob("training_rank0_*.log")) + sorted(
        path.glob("*.out")
    ):
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                match = STEP_RE.match(line)
                if match:
                    if current:
                        records.append(current)
                    current = {
                        "step": int(match.group(1)),
                        "total_steps": int(match.group(2)),
                        "source": str(log_path),
                    }
                    continue
                if current is None:
                    continue
                if (match := TIME_RE.search(line)):
                    current["step_time"] = float(match.group(1))
                    current["rollout_time"] = float(match.group(2))
                    current["train_time"] = float(match.group(3))
                elif (match := REWARD_RE.search(line)):
                    current["reward_mean"] = float(match.group(1))
                elif (match := TRAJ_RE.search(line)):
                    current["n_trajectories"] = int(match.group(1))
        if current:
            records.append(current)
            current = None
    dedup: dict[int, dict[str, Any]] = {}
    for record in records:
        dedup[int(record["step"])] = record
    return [dedup[k] for k in sorted(dedup)]


def _derive_log_step_durations(records: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    prev_elapsed = 0.0
    for record in records:
        elapsed = _safe_float(record.get("step_time"))
        if elapsed <= 0:
            continue
        delta = elapsed - prev_elapsed
        if delta > 0:
            durations.append(delta)
            prev_elapsed = elapsed
    return durations


def _log_trajectory_throughput(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    prev_elapsed = 0.0
    for record in records:
        elapsed = _safe_float(record.get("step_time"))
        n_traj = _safe_int(record.get("n_trajectories"))
        if elapsed <= 0:
            continue
        delta = elapsed - prev_elapsed
        if delta > 0 and n_traj > 0:
            values.append(n_traj / delta)
            prev_elapsed = elapsed
    return values


def _count_log_patterns(path: Path) -> dict[str, int]:
    patterns = {
        "cmlfq_decisions": "[CMLFQ]",
        "global_resource_planner": "[GlobalResourcePlanner]",
        "runtime_elastic_executor": "[RuntimeElasticExecutor]",
        "elastic_gradient_domain": "elastic_gradient_domain",
        "cluster_swap_complete": "cluster_swap_complete",
        "planner_applied": "applied actions=",
        "prefix_tree_updates": "prefix-tree",
    }
    counts = {key: 0 for key in patterns}
    for log_path in sorted(path.rglob("*.log")) + sorted(path.glob("*.out")) + sorted(
        path.glob("*.err")
    ):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for key, pattern in patterns.items():
            counts[key] += text.count(pattern)
    return counts


def _load_tree_stats(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        return {}
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"latest_path": str(latest), "readable": False}
    stats = payload.get("stats") if isinstance(payload, dict) else None
    if not isinstance(stats, dict):
        stats = {}
        if isinstance(payload, dict):
            for key in ("n_prompts", "n_nodes", "n_edges", "n_trajectories"):
                if key in payload:
                    stats[key] = payload[key]
    stats["latest_path"] = str(latest)
    stats["readable"] = True
    return stats


def load_run(name: str, path: Path) -> RunSummary:
    run = RunSummary(name=name, path=path)
    run.history_files = sorted(path.rglob("*history*.jsonl"))
    for history_path in run.history_files:
        run.history_records.extend(_load_jsonl(history_path))
    for events_path in sorted(path.rglob("runtime_reconfiguration_events.jsonl")):
        run.events.extend(_load_jsonl(events_path))
    run.cmlfq_trees = [
        p for p in sorted(path.rglob("*cmlfq*tree*.json")) if p.is_file()
    ]
    job_match = re.search(r"job_(\d+)", path.name)
    if job_match:
        job_id = job_match.group(1)
        run.cmlfq_trees.extend(
            p
            for p in sorted(path.parent.glob(f"cmlfq_tree_{job_id}*.json"))
            if p.is_file()
        )
    run.cmlfq_trees = sorted(set(run.cmlfq_trees))
    run.cmlfq_tree_stats = _load_tree_stats(run.cmlfq_trees)
    run.log_step_records = _parse_rank0_logs(path)
    run.log_counts = _count_log_patterns(path)
    return run


def _reward_series(run: RunSummary) -> list[float]:
    rewards = [
        _safe_float(v)
        for v in _history_metric(
            run.history_records, "training_metrics", "reward_mean"
        )
    ]
    if rewards:
        return rewards
    return [_safe_float(r.get("reward_mean")) for r in run.log_step_records]


def _step_times(run: RunSummary) -> list[float]:
    values = [
        _safe_float(v)
        for v in _history_metric(run.history_records, "timing", "step_total_time")
    ]
    if values:
        return [v for v in values if v > 0]
    return _derive_log_step_durations(run.log_step_records)


def summarize_run(run: RunSummary) -> dict[str, Any]:
    rewards = _reward_series(run)
    step_times = _step_times(run)
    traj_tps = _trajectory_throughput(run.history_records)
    if not traj_tps:
        traj_tps = _log_trajectory_throughput(run.log_step_records)
    token_tps = _tokens_per_second(run.history_records)
    gen_means = [
        _safe_float(v)
        for v in _history_metric(
            run.history_records, "sequence_stats", "generation", "mean"
        )
    ]
    rollout_times = [
        _safe_float(v)
        for v in _history_metric(run.history_records, "timing", "rollout_time")
    ]
    train_times = [
        _safe_float(v)
        for v in _history_metric(run.history_records, "timing", "train_time")
    ]
    planner_decisions = [
        event for event in run.events
        if event.get("decision", {}).get("should_reconfigure")
        or event.get("runtime", {}).get("applied")
    ]
    planner_applied = max(
        len(planner_decisions),
        run.log_counts.get("planner_applied", 0),
        run.log_counts.get("cluster_swap_complete", 0),
    )
    return {
        "steps": len(rewards) or len(run.history_records) or len(run.log_step_records),
        "max_logged_step": max(
            [_safe_int(r.get("step")) for r in run.log_step_records] or [0]
        ),
        "reward_first": _mean(_first_window(rewards)),
        "reward_last": _mean(_last_window(rewards)),
        "reward_delta": _mean(_last_window(rewards)) - _mean(_first_window(rewards))
        if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "step_time_mean": _mean(step_times),
        "step_time_median": median(step_times) if step_times else 0.0,
        "traj_per_s_mean": _mean(traj_tps),
        "tokens_per_s_mean": _mean(token_tps),
        "rollout_time_mean": _mean([v for v in rollout_times if v > 0]),
        "train_time_mean": _mean([v for v in train_times if v > 0]),
        "gen_len_first": _mean(_first_window(gen_means)),
        "gen_len_last": _mean(_last_window(gen_means)),
        "runtime_events": len(run.events),
        "planner_applied_events": planner_applied,
        "cmlfq_trees": len(run.cmlfq_trees),
        "log_counts": run.log_counts,
        "cmlfq_tree_stats": run.cmlfq_tree_stats,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(runs: list[RunSummary]) -> str:
    summaries = {run.name: summarize_run(run) for run in runs}
    lines = [
        "# Libra Validation Report",
        "",
        "## Run Summary",
        "",
        "| Run | Steps | Reward first | Reward last | Reward delta | Step time mean | Traj/s | Tokens/s | Planner applied | C-MLFQ trees |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        s = summaries[run.name]
        lines.append(
            "| {name} | {steps} | {rf} | {rl} | {rd} | {st} | {tr} | {tok} | {pa} | {ct} |".format(
                name=run.name,
                steps=s["steps"],
                rf=_fmt(s["reward_first"]),
                rl=_fmt(s["reward_last"]),
                rd=_fmt(s["reward_delta"]),
                st=_fmt(s["step_time_mean"]),
                tr=_fmt(s["traj_per_s_mean"]),
                tok=_fmt(s["tokens_per_s_mean"]),
                pa=s["planner_applied_events"],
                ct=s["cmlfq_trees"],
            )
        )
    lines.extend(["", "## Capability Trend", ""])
    for run in runs:
        s = summaries[run.name]
        direction = "improved" if s["reward_delta"] > 0 else "flat/regressed"
        lines.append(
            f"- **{run.name}**: reward {direction}; first-window={s['reward_first']:.4f}, "
            f"last-window={s['reward_last']:.4f}, delta={s['reward_delta']:.4f}; "
            f"generation mean changed {s['gen_len_first']:.1f} -> {s['gen_len_last']:.1f} tokens."
        )
    if len(runs) >= 2:
        base = summaries[runs[0].name]
        lines.extend(["", "## Performance Deltas", ""])
        for run in runs[1:]:
            s = summaries[run.name]
            speedup = (
                base["step_time_mean"] / s["step_time_mean"]
                if s["step_time_mean"] > 0 else 0.0
            )
            traj_gain = (
                (s["traj_per_s_mean"] / base["traj_per_s_mean"] - 1.0) * 100.0
                if base["traj_per_s_mean"] > 0 else 0.0
            )
            token_gain = (
                (s["tokens_per_s_mean"] / base["tokens_per_s_mean"] - 1.0) * 100.0
                if base["tokens_per_s_mean"] > 0 else 0.0
            )
            lines.append(
                f"- **{run.name} vs {runs[0].name}**: step-time speedup={speedup:.3f}x, "
                f"trajectory throughput gain={traj_gain:.2f}%, token throughput gain={token_gain:.2f}%."
            )
    lines.extend(["", "## Design Signals", ""])
    for run in runs:
        s = summaries[run.name]
        counts = s["log_counts"]
        tree = s["cmlfq_tree_stats"]
        lines.append(
            f"- **{run.name}**: C-MLFQ log events={counts.get('cmlfq_decisions', 0)}, "
            f"prefix-tree updates={counts.get('prefix_tree_updates', 0)}, "
            f"GRP log events={counts.get('global_resource_planner', 0)}, "
            f"EHP/runtime events={counts.get('runtime_elastic_executor', 0)}, "
            f"elastic domain events={counts.get('elastic_gradient_domain', 0)}, "
            f"cluster swaps={counts.get('cluster_swap_complete', 0)}."
        )
        if tree:
            lines.append(f"  Prefix tree latest: `{tree.get('latest_path')}`; stats={tree}.")
    lines.extend(["", "## Inputs", ""])
    for run in runs:
        lines.append(f"- **{run.name}**: `{run.path}`")
        for history in run.history_files[-3:]:
            lines.append(f"  history: `{history}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Run name and log directory. Can be passed multiple times.",
    )
    parser.add_argument("--output", default="", help="Markdown output path.")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("at least one --run NAME=PATH is required")
    runs: list[RunSummary] = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"invalid --run value: {item!r}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"run path does not exist: {path}")
        runs.append(load_run(name, path))
    report = render_report(runs)
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(out)
    else:
        print(report)


if __name__ == "__main__":
    main()

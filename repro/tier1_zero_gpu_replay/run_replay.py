#!/usr/bin/env python3
"""Tier 1 (0-GPU) Libra scheduling replay and GRP offline benchmark.

Evaluates heterogeneous rollout scheduling policies and the Global Resource
Planner (GRP) on real or sample agentic RL traces using CPU-only production
schedulers and Libra's analytic ``CostModel`` (zero GPU required).

All rollout/training timing numbers come from ``RL_Framework.infra.cost_model``
and are labeled ``evaluation_mode: "analytic_cost_model_replay"``. Controller
latencies (``avg_us``, ``p99_us``) are directly measured wall-clock CPU timings.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def _ensure_rl_framework_importable() -> Path:
    """Ensure ``RL_Framework`` can be imported without an editable install."""
    repo_root = Path(__file__).resolve().parents[2]
    if importlib.util.find_spec("RL_Framework") is None:
        init_path = repo_root / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "RL_Framework",
            init_path,
            submodule_search_locations=[str(repo_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to locate RL_Framework at {repo_root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["RL_Framework"] = module
        spec.loader.exec_module(module)
    return repo_root


REPO_ROOT = _ensure_rl_framework_importable()

from RL_Framework.config import (  # noqa: E402
    AsyncRLConfig,
    GlobalResourcePlannerConfig,
    HardwareConfig,
    HeterogeneousInstanceConfig,
    HeterogeneousRolloutConfig,
    ModelArchConfig,
    ProfilingConfig,
    SchedulingConfig,
)
from RL_Framework.infra.cost_model.global_resource_planner import (  # noqa: E402
    GlobalResourcePlanner,
)
from RL_Framework.infra.cost_model.model import CostModel, RolloutCostModel  # noqa: E402
from RL_Framework.infra.scheduling.cmlfq_offline_profile import (  # noqa: E402
    CMLFQOfflineProfiler,
)
from RL_Framework.infra.scheduling.factory import SchedulerFactory  # noqa: E402


SUPPORTED_POLICIES = (
    "round_robin",
    "least_connections",
    "length_aware",
    "la_mlfq",
    "cmlfq",
    "cmlfq_cost",
)

SUPPORTED_KV_BACKENDS = ("recompute", "nixl", "cpu_offload")

_STATIC_POLICIES = ("round_robin", "least_connections", "length_aware")

_BUCKET_CUTOFFS = (4000, 10000, 20000, 32768)

DEFAULT_COST_PROFILE = (
    REPO_ROOT / "repro/tier1_zero_gpu_replay/sample_traces/r2e_cost_profile.json"
)


@dataclass
class SegmentSpan:
    """Merged contiguous execution span for one trajectory on one instance."""

    kv_start_ctx: int
    new_prefill_tokens: int
    gen_tokens: int


@dataclass
class PolicyReplayResult:
    """Structured replay summary for one scheduler policy."""

    policy: str
    kv_backend: str
    eval_trajectories: int
    warmup_trajectories: int
    analytic_makespan_s: float
    speedup_vs_round_robin: float
    long_tail_high_tp_rate: float
    migrations: int
    migration_overhead_s: float
    oom_count: int
    ctrl_avg_us: float
    ctrl_p99_us: float
    tp_distribution: dict[int, int]


def _classify_archetype(total_output_tokens: int) -> str:
    if total_output_tokens <= _BUCKET_CUTOFFS[0]:
        return "short"
    if total_output_tokens <= _BUCKET_CUTOFFS[1]:
        return "medium"
    return "long"


def _normalize_trace_row(raw: dict[str, Any], index: int) -> dict[str, Any]:
    input_tokens = int(raw.get("input_tokens", raw.get("input_len", raw.get("prompt_len", 0))))
    total_output = int(raw.get("total_output_tokens", raw.get("output_len", 0)))
    raw_returns = list(raw.get("tool_returns", []))

    normalized_returns: list[dict[str, Any]] = []
    prev_pos = 0
    for tr in raw_returns:
        item = dict(tr)
        pos = max(prev_pos, int(item.get("token_position", total_output)))
        pos = min(pos, max(total_output, pos))
        prev_pos = pos
        item["token_position"] = pos
        if "tool" not in item and "tool_type" in item:
            item["tool"] = item["tool_type"]
        if "tool_type" not in item:
            item["tool_type"] = item.get("tool", "unknown")
        normalized_returns.append(item)

    if normalized_returns and normalized_returns[-1]["token_position"] > total_output:
        total_output = normalized_returns[-1]["token_position"]

    prompt_id = str(raw.get("prompt_id") or f"prompt_{index}")
    request_id = str(raw.get("request_id") or f"{prompt_id}_req_{index}")
    archetype = str(raw.get("archetype") or _classify_archetype(total_output))
    return {
        "request_id": request_id,
        "prompt_id": prompt_id,
        "archetype": archetype,
        "input_tokens": max(1, input_tokens),
        "total_output_tokens": max(1, total_output),
        "tool_returns": normalized_returns,
    }


def load_trace(path: Path | str) -> list[dict[str, Any]]:
    """Load native Tier 1 JSONL traces or ``HistoryDataCollector`` ``history_*.jsonl`` logs."""
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    records: list[dict[str, Any]] = []
    idx = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "sequences" in row and isinstance(row["sequences"], list):
                step = int(row.get("step", line_no))
                for seq_i, seq in enumerate(row["sequences"]):
                    seq_copy = dict(seq)
                    seq_copy.setdefault("prompt_id", f"step{step}_seq{seq.get('index', seq_i)}")
                    seq_copy.setdefault("request_id", f"step{step}_req{seq_i}")
                    records.append(_normalize_trace_row(seq_copy, idx))
                    idx += 1
            else:
                records.append(_normalize_trace_row(row, idx))
                idx += 1

    if not records:
        raise ValueError(f"No valid trace records found in {trace_path}")
    return records


def parse_tp_layout(raw: str) -> list[int]:
    """Parse comma-separated rollout instance TP degrees."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("tp_layout must contain at least one positive integer")
    layout = [int(p) for p in parts]
    if any(tp <= 0 for tp in layout):
        raise ValueError(f"All TP degrees must be positive, got {layout}")
    return layout


def _build_bucket_spec(
    tp_layout: list[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, list[int]]]:
    """Build strictly 1-TP-per-bucket specs compatible with ``CMLFQCostScheduler``."""
    unique_tps = sorted(set(tp_layout))
    if len(unique_tps) == 1:
        t0 = unique_tps[0]
        buckets = {"short": {"tp_degrees": [t0], "max_tokens": 32768}}
        thresholds = {"short": 32768}
        rules = {"short": [t0], "medium": [t0], "long": [t0], "extra_long": [t0]}
    elif len(unique_tps) == 2:
        t0, t1 = unique_tps
        buckets = {
            "short": {"tp_degrees": [t0], "max_tokens": 5000},
            "long": {"tp_degrees": [t1], "max_tokens": 32768},
        }
        thresholds = {"short": 5000, "long": 32768}
        rules = {"short": [t0], "medium": [t0], "long": [t1], "extra_long": [t1]}
    elif len(unique_tps) == 3:
        t0, t1, t2 = unique_tps
        buckets = {
            "short": {"tp_degrees": [t0], "max_tokens": 4000},
            "medium": {"tp_degrees": [t1], "max_tokens": 10000},
            "long": {"tp_degrees": [t2], "max_tokens": 32768},
        }
        thresholds = {"short": 4000, "medium": 10000, "long": 32768}
        rules = {"short": [t0], "medium": [t1], "long": [t2], "extra_long": [t2]}
    else:
        t0, t1, t2, t3 = unique_tps[0], unique_tps[1], unique_tps[2], unique_tps[-1]
        buckets = {
            "short": {"tp_degrees": [t0], "max_tokens": 4000},
            "medium": {"tp_degrees": [t1], "max_tokens": 10000},
            "long": {"tp_degrees": [t2], "max_tokens": 20000},
            "extra_long": {"tp_degrees": [t3], "max_tokens": 32768},
        }
        thresholds = {"short": 4000, "medium": 10000, "long": 20000, "extra_long": 32768}
        rules = {"short": [t0], "medium": [t1], "long": [t2], "extra_long": [t3]}
    return buckets, thresholds, rules


def build_hetero_config(
    policy: str,
    tp_layout: list[int],
    hardware: HardwareConfig,
    kv_backend: str = "recompute",
    cost_profile: Path | str | None = None,
) -> tuple[str, HeterogeneousRolloutConfig]:
    """Build factory key and ``HeterogeneousRolloutConfig`` for a replay policy."""
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy {policy!r}; choose from {SUPPORTED_POLICIES}")
    if kv_backend not in SUPPORTED_KV_BACKENDS:
        raise ValueError(
            f"Unsupported kv_backend {kv_backend!r}; choose from {SUPPORTED_KV_BACKENDS}"
        )

    gpus_per_node = max(1, int(hardware.gpus_per_node))
    instances: list[HeterogeneousInstanceConfig] = []
    gpu_cursor = 0
    for idx, tp in enumerate(tp_layout):
        gpu_ids = list(range(gpu_cursor, gpu_cursor + tp))
        host = f"node-{gpu_cursor // gpus_per_node}"
        gpu_cursor += tp
        instances.append(
            HeterogeneousInstanceConfig(
                instance_id=f"rollout-{idx}-tp{tp}",
                tp=tp,
                gpus=gpu_ids,
                host=host,
                port=8000 + idx,
            )
        )

    buckets, thresholds, rules = _build_bucket_spec(tp_layout)
    if policy == "length_aware":
        unique_tps = sorted(set(tp_layout))
        if len(unique_tps) >= 3:
            thresholds = {"short": 1550, "medium": 1950, "long": 2400}
        elif len(unique_tps) == 2:
            thresholds = {"short": 1700, "long": 2400}
    factory_key = "load_balance" if policy in ("round_robin", "least_connections") else policy
    lb_strategy = "round_robin" if policy == "round_robin" else "least_connections"
    transfer_pairs = (
        [[s, d] for s in sorted(set(tp_layout)) for d in sorted(set(tp_layout)) if s != d]
        if kv_backend != "recompute"
        else []
    )

    scheduling = SchedulingConfig(
        scheduler_type=factory_key,
        load_balance_strategy=lb_strategy,
        length_thresholds=thresholds,
        routing_rules=rules,
        la_mlfq_buckets=buckets,
        la_mlfq_migration_threshold=3000,
        cmlfq_buckets=buckets,
        cmlfq_cost_profile_path=str(cost_profile) if cost_profile else "",
        cmlfq_kv_backend=kv_backend,
        cmlfq_kv_transfer_tp_pairs=transfer_pairs,
        cmlfq_payload_small_threshold=500,
        cmlfq_payload_large_threshold=5000,
        max_queue_length=1000,
        enable_fallback=True,
    )
    hetero = HeterogeneousRolloutConfig(
        enabled=True,
        total_gpus=sum(tp_layout),
        available_gpus=list(range(sum(tp_layout))),
        instances=instances,
        scheduling=scheduling,
    )
    return factory_key, hetero


def _recompute_cost_s(rollout_model: RolloutCostModel, dst_tp: int, context_tokens: int) -> float:
    chunk = max(1, int(rollout_model.L_chunk))
    full, tail = divmod(max(0, context_tokens), chunk)
    cost = full * rollout_model.compute_prefill_step_time(1, chunk, dst_tp)
    if tail:
        cost += rollout_model.compute_prefill_step_time(1, tail, dst_tp)
    return cost


def _estimate_migration_cost_s(
    rollout_model: RolloutCostModel,
    hardware: HardwareConfig,
    src_tp: int,
    dst_tp: int,
    src_host: str,
    dst_host: str,
    context_tokens: int,
    kv_backend: str,
) -> float:
    """Estimate KV state migration latency for recompute / NIXL / CPU offload."""
    recompute = _recompute_cost_s(rollout_model, dst_tp, context_tokens)
    if kv_backend == "recompute":
        return recompute

    kv_bytes = max(0, context_tokens) * src_tp * rollout_model.compute_token_kv_bytes(src_tp)
    if kv_backend == "nixl":
        if src_host == dst_host:
            bw = max(1.0, float(hardware.bw_intra_node))
            extra_latency = 0.0
        else:
            bw = max(1.0, float(hardware.bw_inter_node))
            extra_latency = max(0.0, float(hardware.latency_inter_node))
        return kv_bytes / bw + extra_latency

    # cpu_offload: device->host + host->device two-hop PCIe transfer
    pcie_eff_bw = 0.5 * min(float(hardware.bw_inter_node), 6.4e10)
    return kv_bytes / max(1.0, pcie_eff_bw) + 5e-5


def _build_analytic_migration_profile(
    instances: list[HeterogeneousInstanceConfig],
    rollout_model: RolloutCostModel,
    hardware: HardwareConfig,
    kv_backend: str,
) -> list[dict[str, Any]]:
    """Build analytic transfer/recompute table for ``CMLFQRoutingCosts``."""
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int, str, int]] = set()
    seq_lens = (512, 1024, 2048, 4096, 8192, 16384, 32768)
    for src in instances:
        for dst in instances:
            if src.instance_id == dst.instance_id:
                continue
            key = (src.host, src.tp, dst.host, dst.tp)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            for seq_len in seq_lens:
                rec_s = _recompute_cost_s(rollout_model, dst.tp, seq_len)
                xfer_s = _estimate_migration_cost_s(
                    rollout_model,
                    hardware,
                    src.tp,
                    dst.tp,
                    src.host,
                    dst.host,
                    seq_len,
                    kv_backend,
                )
                rows.append(
                    {
                        "source_tp": src.tp,
                        "target_tp": dst.tp,
                        "source_host": src.host,
                        "target_host": dst.host,
                        "seq_len": seq_len,
                        "recompute_ms": rec_s * 1000.0,
                        "transfer_ms": xfer_s * 1000.0,
                    }
                )
    return rows


def _instantiate_scheduler(
    policy: str,
    hetero: HeterogeneousRolloutConfig,
    factory_key: str,
    rollout_model: RolloutCostModel,
    hardware: HardwareConfig,
    kv_backend: str,
) -> Any:
    scheduler = SchedulerFactory.create(factory_key, hetero)
    for idx, inst in enumerate(hetero.instances):
        scheduler.register_instance(idx, inst.instance_id, inst.tp)

    if policy == "cmlfq_cost":
        scheduler.routing_costs.rollout_model = rollout_model
        if kv_backend != "recompute" and not scheduler.routing_costs.migrations:
            scheduler.routing_costs.migrations = _build_analytic_migration_profile(
                hetero.instances, rollout_model, hardware, kv_backend
            )
        topology = [
            {"tp_degree": inst.tp, "host": inst.host} for inst in hetero.instances
        ]
        scheduler.configure_runtime(
            topology,
            transfer_supported=(lambda src, dst: src != dst)
            if kv_backend != "recompute"
            else (lambda src, dst: False),
        )
    return scheduler


def _apply_warmup(
    policy: str,
    scheduler: Any,
    warmup_traces: list[dict[str, Any]],
    bucket_thresholds: dict[str, int],
) -> None:
    if not warmup_traces:
        return
    if policy in ("cmlfq", "cmlfq_cost"):
        profiler = CMLFQOfflineProfiler()
        profiler.profile_from_raw_trajectories(warmup_traces)
        scheduler.prefix_tree = profiler.prefix_tree
        return

    if policy == "la_mlfq":
        sorted_buckets = sorted(bucket_thresholds.items(), key=lambda item: item[1])
        for rec in warmup_traces:
            total_out = rec["total_output_tokens"]
            final_bucket = sorted_buckets[-1][0]
            for bname, limit in sorted_buckets:
                if total_out <= limit:
                    final_bucket = bname
                    break
            scheduler.history_table.update(rec["prompt_id"], final_bucket, epoch=0)


def _score_instance_makespan(
    rollout_model: RolloutCostModel,
    segments: list[SegmentSpan],
    migration_overhead_s: float,
    tp: int,
    max_batch_per_tp: int = 4,
) -> float:
    """Compute water-filling continuous-batching makespan on one instance."""
    if not segments:
        return migration_overhead_s

    chunk = max(1, int(rollout_model.L_chunk))
    t_prefill_step = rollout_model.compute_prefill_step_time(1, chunk, tp)
    prefill_time = sum((s.new_prefill_tokens / chunk) * t_prefill_step for s in segments)

    ordered = sorted(segments, key=lambda s: s.gen_tokens)
    prev_g = 0
    decode_time = 0.0
    max_conc = max_batch_per_tp * tp
    for idx, seg in enumerate(ordered):
        delta_g = seg.gen_tokens - prev_g
        if delta_g <= 0:
            continue
        active = ordered[idx:]
        m = len(active)
        g_mid = prev_g + delta_g / 2.0
        avg_ctx = sum(a.kv_start_ctx + g_mid for a in active) / m
        b_kv = rollout_model.compute_steady_state_batch(tp, avg_ctx, 0.0)
        b_cap = max(1.0, min(float(max_conc), b_kv))
        b_eff = min(float(m), b_cap)
        total_active_tokens = int(b_eff * avg_ctx)
        t_step = rollout_model.compute_decode_step_time(total_active_tokens, tp)
        decode_time += delta_g * (m / b_eff) * t_step
        prev_g = seg.gen_tokens
    return prefill_time + decode_time + migration_overhead_s


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int((pct / 100.0) * (len(ordered) - 1))))
    return ordered[rank]


def _tool_payload_tokens(tr: dict[str, Any]) -> int:
    if "payload_tokens" in tr:
        return max(0, int(tr["payload_tokens"]))
    raw = tr.get("output", tr.get("result", tr.get("content", "")))
    return len(raw) if isinstance(raw, str) else len(str(raw))


def _run_static_batch(
    scheduler: Any,
    batch: list[dict[str, Any]],
    hetero: HeterogeneousRolloutConfig,
    rollout_model: RolloutCostModel,
    *,
    concurrency: int,
    max_tp: int,
    ctrl_latencies_us: list[float],
    tp_dist: dict[int, int],
) -> tuple[list[list[SegmentSpan]], list[float], int, int, int]:
    """Execute one batch for non-migrating policies under bounded in-flight concurrency."""
    n_instances = len(hetero.instances)
    instance_segments: list[list[SegmentSpan]] = [[] for _ in range(n_instances)]
    instance_migration_cost: list[float] = [0.0 for _ in range(n_instances)]
    in_flight: list[tuple[float, int, int, dict[str, Any], int, str]] = []
    inst_clock = [0.0 for _ in range(n_instances)]
    oom_count = 0
    long_total = 0
    long_on_max_tp = 0

    for seq_i, req in enumerate(batch):
        if concurrency > 0 and len(in_flight) >= concurrency:
            _, _, done_idx, done_req, done_res, done_cat = heapq.heappop(in_flight)
            scheduler.on_request_done(
                done_idx,
                prompt_id=done_req["prompt_id"],
                final_bucket=done_cat,
                output_tokens=done_req["total_output_tokens"],
                reserved_tokens=done_res,
            )

        t0 = time.perf_counter_ns()
        route = scheduler.schedule(
            input_tokens=req["input_tokens"],
            prompt_id=req["prompt_id"],
            n_samples=1,
        )
        ctrl_latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

        idx = route.instance_index
        tp = hetero.instances[idx].tp
        tp_dist[tp] = tp_dist.get(tp, 0) + 1
        if req["archetype"] == "long":
            long_total += 1
            if tp == max_tp:
                long_on_max_tp += 1

        peak_ctx = req["input_tokens"] + req["total_output_tokens"]
        if rollout_model.check_oom(peak_ctx, tp):
            oom_count += 1

        tool_prefill = sum(_tool_payload_tokens(tr) for tr in req["tool_returns"])
        instance_segments[idx].append(
            SegmentSpan(
                kv_start_ctx=req["input_tokens"],
                new_prefill_tokens=req["input_tokens"] + tool_prefill,
                gen_tokens=req["total_output_tokens"],
            )
        )
        inst_clock[idx] += req["total_output_tokens"] / max(1, tp)
        heapq.heappush(
            in_flight,
            (inst_clock[idx], seq_i, idx, req, route.reserved_tokens, route.category),
        )

    while in_flight:
        _, _, done_idx, done_req, done_res, done_cat = heapq.heappop(in_flight)
        scheduler.on_request_done(
            done_idx,
            prompt_id=done_req["prompt_id"],
            final_bucket=done_cat,
            output_tokens=done_req["total_output_tokens"],
            reserved_tokens=done_res,
        )

    return instance_segments, instance_migration_cost, oom_count, long_total, long_on_max_tp


def _run_dynamic_batch(
    policy: str,
    scheduler: Any,
    batch: list[dict[str, Any]],
    hetero: HeterogeneousRolloutConfig,
    hardware: HardwareConfig,
    rollout_model: RolloutCostModel,
    *,
    kv_backend: str,
    max_tp: int,
    ctrl_latencies_us: list[float],
    tp_dist: dict[int, int],
) -> tuple[list[list[SegmentSpan]], list[float], int, float, int, int, int]:
    """Execute one multi-turn batch with per-turn dynamic migration hooks."""
    n_instances = len(hetero.instances)
    instance_segments: list[list[SegmentSpan]] = [[] for _ in range(n_instances)]
    instance_migration_cost: list[float] = [0.0 for _ in range(n_instances)]
    batch_migrations = 0
    batch_mig_overhead_s = 0.0
    oom_count = 0
    long_total = 0
    long_on_max_tp = 0

    active_states: list[dict[str, Any]] = []
    for req in batch:
        t0 = time.perf_counter_ns()
        route = scheduler.schedule(
            input_tokens=req["input_tokens"],
            prompt_id=req["prompt_id"],
            n_samples=1,
        )
        req_id = route.request_id or f"{req['request_id']}_{uuid.uuid4().hex[:8]}"
        cur_idx = route.instance_index
        cur_bucket = route.category
        if policy == "la_mlfq":
            scheduler.migration_controller.register_request(
                request_id=req_id,
                prompt_id=req["prompt_id"],
                instance_index=cur_idx,
                bucket=cur_bucket,
            )
        elif policy == "cmlfq_cost":
            d0 = scheduler.reserve_generation(req_id, context_tokens=req["input_tokens"])
            scheduler.commit_generation(req_id, d0)
            if d0.should_migrate:
                batch_migrations += 1
                cur_idx = d0.target_instance_index
                cur_bucket = d0.target_bucket
        ctrl_latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

        tool_returns = req["tool_returns"]
        first_end = (
            tool_returns[0]["token_position"]
            if tool_returns
            else req["total_output_tokens"]
        )
        active_states.append(
            {
                "req": req,
                "req_id": req_id,
                "cur_idx": cur_idx,
                "cur_bucket": cur_bucket,
                "reserved_tokens": route.reserved_tokens,
                "open_span": SegmentSpan(
                    kv_start_ctx=req["input_tokens"],
                    new_prefill_tokens=req["input_tokens"],
                    gen_tokens=first_end,
                ),
            }
        )

    max_turns = max((len(s["req"]["tool_returns"]) for s in active_states), default=0)
    for turn_idx in range(max_turns):
        for st in active_states:
            req = st["req"]
            tool_returns = req["tool_returns"]
            if turn_idx >= len(tool_returns):
                continue

            tr = tool_returns[turn_idx]
            pos = tr["token_position"]
            next_pos = (
                tool_returns[turn_idx + 1]["token_position"]
                if turn_idx + 1 < len(tool_returns)
                else req["total_output_tokens"]
            )
            seg_gen = max(0, next_pos - pos)
            payload_tokens = _tool_payload_tokens(tr)
            old_idx = st["cur_idx"]
            new_idx = old_idx
            new_bucket = st["cur_bucket"]

            t0 = time.perf_counter_ns()
            if policy == "cmlfq":
                decision = scheduler.on_tool_return(
                    st["req_id"], tr, generated_tokens=pos
                )
                if decision.should_migrate:
                    res = scheduler.execute_migration(st["req_id"], decision)
                    if res.instance_index >= 0:
                        new_idx = res.instance_index
                        new_bucket = res.category
            elif policy == "cmlfq_cost":
                scheduler.on_tool_return(st["req_id"], tr, generated_tokens=pos)
                res = scheduler.reserve_generation(
                    st["req_id"], context_tokens=req["input_tokens"] + pos
                )
                scheduler.commit_generation(st["req_id"], res)
                if res.should_migrate and res.target_instance_index >= 0:
                    new_idx = res.target_instance_index
                    new_bucket = res.target_bucket
            elif policy == "la_mlfq":
                decision = scheduler.check_and_migrate(
                    st["req_id"], generated_tokens=pos
                )
                if decision.should_migrate:
                    scheduler._complete_route(
                        old_idx,
                        output_tokens=0,
                        category=st["cur_bucket"],
                        reserved_tokens=st["reserved_tokens"],
                    )
                    res = scheduler._route_to_bucket(
                        decision.target_bucket,
                        0,
                        req["prompt_id"],
                        reason="la_mlfq_migration",
                    )
                    if res.instance_index >= 0:
                        new_idx = res.instance_index
                        new_bucket = res.category
                        st["reserved_tokens"] = res.reserved_tokens
            ctrl_latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

            if new_idx != old_idx:
                batch_migrations += 1
                instance_segments[old_idx].append(st["open_span"])
                mig_s = _estimate_migration_cost_s(
                    rollout_model,
                    hardware,
                    hetero.instances[old_idx].tp,
                    hetero.instances[new_idx].tp,
                    hetero.instances[old_idx].host,
                    hetero.instances[new_idx].host,
                    req["input_tokens"] + pos,
                    kv_backend,
                )
                instance_migration_cost[new_idx] += mig_s
                batch_mig_overhead_s += mig_s
                st["cur_idx"] = new_idx
                st["cur_bucket"] = new_bucket
                st["open_span"] = SegmentSpan(
                    kv_start_ctx=req["input_tokens"] + pos,
                    new_prefill_tokens=payload_tokens,
                    gen_tokens=seg_gen,
                )
            else:
                st["cur_bucket"] = new_bucket
                st["open_span"].new_prefill_tokens += payload_tokens
                st["open_span"].gen_tokens += seg_gen

    for st in active_states:
        req = st["req"]
        final_idx = st["cur_idx"]
        final_tp = hetero.instances[final_idx].tp
        instance_segments[final_idx].append(st["open_span"])
        tp_dist[final_tp] = tp_dist.get(final_tp, 0) + 1

        if req["archetype"] == "long":
            long_total += 1
            if final_tp == max_tp:
                long_on_max_tp += 1

        peak_ctx = req["input_tokens"] + req["total_output_tokens"]
        if rollout_model.check_oom(peak_ctx, final_tp):
            oom_count += 1

        if policy in ("cmlfq", "cmlfq_cost"):
            scheduler.finish_request(st["req_id"], req["total_output_tokens"])
        else:
            scheduler.migration_controller.on_request_done(st["req_id"])
            scheduler.on_request_done(
                final_idx,
                prompt_id=req["prompt_id"],
                final_bucket=st["cur_bucket"],
                output_tokens=req["total_output_tokens"],
                reserved_tokens=st["reserved_tokens"],
            )

    return (
        instance_segments,
        instance_migration_cost,
        batch_migrations,
        batch_mig_overhead_s,
        oom_count,
        long_total,
        long_on_max_tp,
    )


def replay_policy(
    policy: str,
    traces: list[dict[str, Any]],
    tp_layout: list[int],
    hardware: HardwareConfig,
    model_arch: ModelArchConfig,
    *,
    batch_size: int = 40,
    concurrency: int = 12,
    warmup_ratio: float = 0.375,
    online_only: bool = False,
    kv_backend: str = "recompute",
    cost_profile: Path | str | None = None,
    max_seq_length: int = 32768,
) -> PolicyReplayResult:
    """Replay traces through a real Libra scheduler and score with ``RolloutCostModel``."""
    resolved_profile = (
        cost_profile
        if cost_profile is not None
        else (DEFAULT_COST_PROFILE if DEFAULT_COST_PROFILE.exists() else None)
    )
    factory_key, hetero = build_hetero_config(
        policy, tp_layout, hardware, kv_backend, cost_profile=resolved_profile
    )
    cost_model = CostModel(
        hardware=hardware,
        model_arch=model_arch,
        profiling=ProfilingConfig(),
        max_seq_length=max_seq_length,
    )
    rollout_model = cost_model.rollout_model
    scheduler = _instantiate_scheduler(
        policy, hetero, factory_key, rollout_model, hardware, kv_backend
    )

    if online_only or warmup_ratio <= 0.0:
        warmup_count = 0
    else:
        warmup_count = min(len(traces) - 1, max(0, int(round(len(traces) * warmup_ratio))))
    warmup_traces = traces[:warmup_count]
    eval_traces = traces[warmup_count:]
    _apply_warmup(
        policy,
        scheduler,
        warmup_traces,
        {
            name: int(cfg.get("max_tokens", 5000))
            for name, cfg in hetero.scheduling.cmlfq_buckets.items()
        },
    )

    max_tp = max(tp_layout)
    n_instances = len(hetero.instances)
    ctrl_latencies_us: list[float] = []
    tp_dist: dict[int, int] = {tp: 0 for tp in sorted(set(tp_layout))}
    total_makespan_s = 0.0
    total_migrations = 0
    total_migration_overhead_s = 0.0
    oom_count = 0
    long_total = 0
    long_on_max_tp = 0

    effective_batch = max(1, batch_size)
    for batch_start in range(0, len(eval_traces), effective_batch):
        batch = eval_traces[batch_start : batch_start + effective_batch]
        if policy in _STATIC_POLICIES:
            (
                instance_segments,
                instance_migration_cost,
                b_oom,
                b_long_tot,
                b_long_max,
            ) = _run_static_batch(
                scheduler,
                batch,
                hetero,
                rollout_model,
                concurrency=concurrency,
                max_tp=max_tp,
                ctrl_latencies_us=ctrl_latencies_us,
                tp_dist=tp_dist,
            )
        else:
            (
                instance_segments,
                instance_migration_cost,
                b_mig,
                b_mig_s,
                b_oom,
                b_long_tot,
                b_long_max,
            ) = _run_dynamic_batch(
                policy,
                scheduler,
                batch,
                hetero,
                hardware,
                rollout_model,
                kv_backend=kv_backend,
                max_tp=max_tp,
                ctrl_latencies_us=ctrl_latencies_us,
                tp_dist=tp_dist,
            )
            total_migrations += b_mig
            total_migration_overhead_s += b_mig_s

        oom_count += b_oom
        long_total += b_long_tot
        long_on_max_tp += b_long_max
        batch_makespan = max(
            _score_instance_makespan(
                rollout_model,
                instance_segments[i],
                instance_migration_cost[i],
                hetero.instances[i].tp,
            )
            for i in range(n_instances)
        )
        total_makespan_s += batch_makespan

    long_rate = (long_on_max_tp / long_total) if long_total > 0 else 0.0
    return PolicyReplayResult(
        policy=policy,
        kv_backend=kv_backend,
        eval_trajectories=len(eval_traces),
        warmup_trajectories=warmup_count,
        analytic_makespan_s=round(total_makespan_s, 3),
        speedup_vs_round_robin=1.0,
        long_tail_high_tp_rate=round(long_rate, 4),
        migrations=total_migrations,
        migration_overhead_s=round(total_migration_overhead_s, 3),
        oom_count=oom_count,
        ctrl_avg_us=round(mean(ctrl_latencies_us), 2) if ctrl_latencies_us else 0.0,
        ctrl_p99_us=round(_percentile(ctrl_latencies_us, 99.0), 2),
        tp_distribution=tp_dist,
    )


def _pick_valid_train_tp(train_gpus: int) -> int:
    for tp in (4, 2, 1):
        if tp <= train_gpus and train_gpus % tp == 0:
            return tp
    return 1


def run_grp_offline_benchmark(
    traces: list[dict[str, Any]],
    hardware: HardwareConfig,
    model_arch: ModelArchConfig,
    *,
    total_gpus: int = 32,
    batch_size: int = 64,
    max_seq_length: int = 32768,
) -> dict[str, Any]:
    """Run ``GlobalResourcePlanner`` offline optimization over trace length statistics."""
    if total_gpus < 2:
        raise ValueError(f"GRP benchmark requires total_gpus >= 2, got {total_gpus}")

    train_gpus = max(1, total_gpus // 2)
    rollout_gpus = total_gpus - train_gpus
    train_tp = _pick_valid_train_tp(train_gpus)
    train_dp = max(1, train_gpus // train_tp)
    vllm_tp = max((tp for tp in (1, 2, 4, 8) if tp <= rollout_gpus), default=1)
    effective_batch = max(batch_size, len(traces))
    allowed_tps = [tp for tp in (1, 2, 4, 8) if tp <= total_gpus]

    cfg = AsyncRLConfig(
        model_path="Qwen/Qwen3-14B",
        hardware=hardware,
        model_arch=model_arch,
        n_total_gpus=total_gpus,
        train_gpus=train_gpus,
        rollout_gpus=rollout_gpus,
        train_tp_size=train_tp,
        train_pp_size=1,
        train_dp_size=train_dp,
        vllm_tp_size=vllm_tp,
        batch_size=effective_batch,
        micro_batch_size=1,
        max_seq_length=max_seq_length,
        global_resource_planner=GlobalResourcePlannerConfig(
            enabled=True,
            plan_interval=1,
            warmup_steps=0,
            min_history_size=min(16, len(traces)),
            min_gain_ratio=0.05,
            reconfiguration_cost_s=0.0,
            allowed_rollout_tp=allowed_tps,
            allowed_train_pp=[1],
        ),
    )

    planner = GlobalResourcePlanner.from_config(cfg)
    planner.observe_batch(
        [
            {
                "input_len": t["input_tokens"],
                "output_len": t["total_output_tokens"],
            }
            for t in traces
        ]
    )
    t0 = time.perf_counter_ns()
    decision = planner.plan_if_needed(step=1, config=cfg)
    solve_ms = (time.perf_counter_ns() - t0) / 1e6

    cur = decision.current_plan
    cand = decision.candidate_plan
    best = cand if cand is not None else cur
    gain_ratio = best.expected_gain_ratio if cand is not None else 0.0
    speedup = (
        round(cur.t_global / best.t_global, 3)
        if cur is not None and best is not None and best.t_global > 0
        else 1.0
    )

    return {
        "total_gpus": total_gpus,
        "observed_trajectories": len(traces),
        "should_reconfigure": decision.should_reconfigure,
        "reason": decision.reason,
        "solve_ms": round(solve_ms, 2),
        "speedup": speedup,
        "expected_gain_ratio": round(gain_ratio, 4),
        "current_plan": {
            "train_gpus": cur.train_gpus,
            "train_tp": cur.train_config.tp,
            "train_dp": cur.train_config.dp,
            "rollout_gpus": cur.rollout_gpus,
            "rollout_tp_list": cur.rollout_tp_list,
            "t_train_s": round(cur.t_train, 3),
            "t_rollout_s": round(cur.t_rollout, 3),
            "t_global_s": round(cur.t_global, 3),
        }
        if cur is not None
        else None,
        "candidate_plan": {
            "train_gpus": best.train_gpus,
            "train_tp": best.train_config.tp,
            "train_dp": best.train_config.dp,
            "rollout_gpus": best.rollout_gpus,
            "rollout_tp_list": best.rollout_tp_list,
            "t_train_s": round(best.t_train, 3),
            "t_rollout_s": round(best.t_rollout, 3),
            "t_global_s": round(best.t_global, 3),
        }
        if best is not None
        else None,
    }


def _populate_speedups(results: list[PolicyReplayResult]) -> None:
    baseline = next(
        (r.analytic_makespan_s for r in results if r.policy == "round_robin"),
        results[0].analytic_makespan_s if results else 1.0,
    )
    for res in results:
        if res.analytic_makespan_s > 0:
            res.speedup_vs_round_robin = round(baseline / res.analytic_makespan_s, 3)


def _format_tp_dist(tp_dist: dict[int, int], unique_tps: list[int]) -> str:
    return "/".join(str(tp_dist.get(tp, 0)) for tp in unique_tps)


def format_markdown_report(
    results: list[PolicyReplayResult],
    grp_result: dict[str, Any] | None,
    *,
    trace_path: Path,
    tp_layout: list[int],
    hardware_path: Path,
    model_arch_path: Path,
    concurrency: int = 12,
    online_only: bool = False,
    cost_profile: Path | str | None = None,
) -> str:
    """Format replay and GRP offline benchmark results as Markdown tables."""
    unique_tps = sorted(set(tp_layout))
    tp_header = "/".join(str(tp) for tp in unique_tps)
    warmup_n = results[0].warmup_trajectories if results else 0
    eval_n = results[0].eval_trajectories if results else 0
    split_desc = (
        f"purely online (`{eval_n}` eval)"
        if online_only
        else f"`{warmup_n}` warmup + `{eval_n}` unseen eval"
    )
    profile_desc = Path(cost_profile).name if cost_profile else "analytic_rollout_model"

    lines = [
        "# Libra Tier 1 — 0-GPU Scheduling Replay & GRP Offline Benchmark",
        "",
        "## Experimental Setup",
        "",
        f"- **Evaluation Mode**: `analytic_cost_model_replay` (water-filling continuous-batching `RolloutCostModel`; zero GPU compute)",
        f"- **Workload Trace**: `{trace_path.name}` ({split_desc}; prompt tokens `980–2291`)",
        f"- **Rollout Cluster Layout**: `TP={tp_layout}` (`{sum(tp_layout)} GPUs` across `{len(tp_layout)}` instances, `max_batch_per_tp=4`)",
        f"- **Hardware & Model**: hardware=`{hardware_path.name}`, model=`{model_arch_path.name}`",
        f"- **Scheduling Setup**: static bounded in-flight `concurrency={concurrency}`; `length_aware` prompt cutoffs `(1550/1950/2400)`; MLFQ output buckets `short<=4000 / medium<=10000 / long<=32768`; `cmlfq_cost` profile=`{profile_desc}`",
        "",
        "## 1. Rollout Scheduling Policy Comparison",
        "",
        f"| Policy | KV Backend | Makespan (s, Analytic) | Speedup vs RR | Long-Tail Max-TP Rate | Final TP Dist ({tp_header}) | Migrations | Mig Cost (s) | Ctrl Avg/P99 (us) |",
        "| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.policy}` | `{r.kv_backend}` | {r.analytic_makespan_s:.3f} | "
            f"{r.speedup_vs_round_robin:.2f}x | {r.long_tail_high_tp_rate * 100:.1f}% | "
            f"`{_format_tp_dist(r.tp_distribution, unique_tps)}` | "
            f"{r.migrations} | {r.migration_overhead_s:.3f} | "
            f"{r.ctrl_avg_us:.1f} / {r.ctrl_p99_us:.1f} |"
        )

    if grp_result and grp_result["current_plan"] and grp_result["candidate_plan"]:
        cur = grp_result["current_plan"]
        cand = grp_result["candidate_plan"]
        lines.extend(
            [
                "",
                f"## 2. Global Resource Planner (GRP) Offline Optimization ({grp_result['total_gpus']} GPUs, {grp_result['observed_trajectories']} trajectories)",
                "",
                "| Plan | Train Allocation | Rollout TP Layout (GPUs) | T_train (s) | T_rollout (s) | T_global (s) | Expected Gain |",
                "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
                (
                    f"| Fixed Baseline | {cur['train_gpus']} GPUs (TP={cur['train_tp']}, DP={cur['train_dp']}) | "
                    f"`{cur['rollout_tp_list']}` ({cur['rollout_gpus']} GPUs) | "
                    f"{cur['t_train_s']:.3f} | {cur['t_rollout_s']:.3f} | {cur['t_global_s']:.3f} | — |"
                ),
                (
                    f"| GRP Optimal (`{grp_result['reason']}`) | {cand['train_gpus']} GPUs (TP={cand['train_tp']}, DP={cand['train_dp']}) | "
                    f"`{cand['rollout_tp_list']}` ({cand['rollout_gpus']} GPUs) | "
                    f"{cand['t_train_s']:.3f} | {cand['t_rollout_s']:.3f} | {cand['t_global_s']:.3f} | "
                    f"**{grp_result['speedup']:.2f}x** (-{grp_result['expected_gain_ratio'] * 100:.1f}%) |"
                ),
                "",
                f"- **GRP Solve Time**: `{grp_result['solve_ms']:.2f} ms` on CPU",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Libra Tier 1 (0-GPU) heterogeneous scheduling replay and GRP benchmark"
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=REPO_ROOT / "repro/tier1_zero_gpu_replay/sample_traces/r2e_agentic_trace.jsonl",
        help="Path to native Tier 1 JSONL trace or HistoryDataCollector history_*.jsonl",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(SUPPORTED_POLICIES),
        choices=list(SUPPORTED_POLICIES),
        help="Scheduling policies to evaluate",
    )
    parser.add_argument(
        "--tp-layout",
        default="1,1,2,4",
        help="Comma-separated TP degrees for heterogeneous rollout instances (default: 1,1,2,4)",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=REPO_ROOT / "configs/hardware_config/a800_80g.yaml",
        help="Hardware config YAML for analytic CostModel",
    )
    parser.add_argument(
        "--model-arch",
        type=Path,
        default=REPO_ROOT / "configs/model_arch_config/qwen3-14b.yaml",
        help="Model architecture config YAML for analytic CostModel",
    )
    parser.add_argument(
        "--total-gpus",
        type=int,
        default=32,
        help="Total cluster GPUs for GlobalResourcePlanner offline benchmark (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Eval trajectories per synchronous rollout step batch (default: 40)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Bounded in-flight concurrency for static scheduling policies (default: 12)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.375,
        help="Fraction of leading traces used for warmup (default: 0.375 -> 24/64 warmup)",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Disable warmup split and replay all traces purely online",
    )
    parser.add_argument(
        "--kv-backend",
        default="recompute",
        choices=list(SUPPORTED_KV_BACKENDS),
        help="KV state movement backend for migration evaluation (default: recompute)",
    )
    parser.add_argument(
        "--cost-profile",
        type=Path,
        default=DEFAULT_COST_PROFILE if DEFAULT_COST_PROFILE.exists() else None,
        help="Measured decode_ms/migrations JSON profile for cmlfq_cost (default: sample_traces/r2e_cost_profile.json)",
    )
    parser.add_argument(
        "--skip-grp",
        action="store_true",
        help="Skip GlobalResourcePlanner offline benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path to write full JSON results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    tp_layout = parse_tp_layout(args.tp_layout)
    traces = load_trace(args.trace)
    hardware = HardwareConfig.from_file(str(args.hardware))
    model_arch = ModelArchConfig.from_file(str(args.model_arch))

    results: list[PolicyReplayResult] = []
    for policy in args.policies:
        results.append(
            replay_policy(
                policy=policy,
                traces=traces,
                tp_layout=tp_layout,
                hardware=hardware,
                model_arch=model_arch,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                warmup_ratio=args.warmup_ratio,
                online_only=args.online_only,
                kv_backend=args.kv_backend,
                cost_profile=args.cost_profile,
            )
        )
    _populate_speedups(results)

    grp_result = (
        None
        if args.skip_grp
        else run_grp_offline_benchmark(
            traces,
            hardware,
            model_arch,
            total_gpus=args.total_gpus,
            batch_size=max(64, len(traces)),
        )
    )

    print(
        format_markdown_report(
            results,
            grp_result,
            trace_path=args.trace,
            tp_layout=tp_layout,
            hardware_path=args.hardware,
            model_arch_path=args.model_arch,
            concurrency=args.concurrency,
            online_only=args.online_only,
            cost_profile=args.cost_profile,
        )
    )

    if args.output is not None:
        payload = {
            "evaluation_mode": "analytic_cost_model_replay",
            "trace_path": str(args.trace),
            "tp_layout": tp_layout,
            "concurrency": args.concurrency,
            "kv_backend": args.kv_backend,
            "cost_profile": str(args.cost_profile) if args.cost_profile else None,
            "hardware_config": str(args.hardware),
            "model_arch_config": str(args.model_arch),
            "scheduling_results": [asdict(r) for r in results],
            "grp_offline": grp_result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

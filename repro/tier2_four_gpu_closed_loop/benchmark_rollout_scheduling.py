#!/usr/bin/env python3
"""Controlled multi-turn heterogeneous rollout scheduling benchmark (Tier 2).

Runs Phase 1 cold micro-calibration across distinct tensor-parallel tiers to fit
analytical `CostModel` scaling factors and migration recompute costs, followed by
Phase 2 counterbalanced rounds comparing 6 Libra scheduling policies with live
vLLM Automatic Prefix Caching (`--enable-prefix-caching`) and session-isolated
prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _ensure_rl_framework_importable() -> Path:
    """Resolve the repository root and make `RL_Framework.*` importable."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if repo_root.name == "RL_Framework":
        parent = str(repo_root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    if importlib.util.find_spec("RL_Framework") is None:
        spec = importlib.util.spec_from_file_location(
            "RL_Framework",
            repo_root / "__init__.py",
            submodule_search_locations=[str(repo_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to locate RL_Framework at {repo_root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["RL_Framework"] = module
        spec.loader.exec_module(module)
    return repo_root


REPO_ROOT = _ensure_rl_framework_importable()

from RL_Framework.config import AsyncRLConfig  # noqa: E402
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine  # noqa: E402
from RL_Framework.infra.cost_model.model import CostModel, RequestInfo  # noqa: E402
from RL_Framework.infra.scheduling.cmlfq_cost import CMLFQRoutingCosts  # noqa: E402
from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import CMLFQCostScheduler  # noqa: E402

POLICIES_FORWARD: tuple[str, ...] = (
    "round_robin",
    "least_connections",
    "length_aware",
    "la_mlfq",
    "cmlfq",
    "cmlfq_cost",
)

DEFAULT_HW_CFG = REPO_ROOT / "configs/hardware_config/h20_3e_141g.yaml"
DEFAULT_ARCH_CFG = REPO_ROOT / "configs/model_arch_config/qwen3-14b.yaml"


def parse_topology(
    topology_spec: str,
    gpus: list[int] | None = None,
    host: str = "127.0.0.1",
    base_port: int = 19001,
) -> list[dict[str, Any]]:
    """Parse a comma-separated TP layout into heterogeneous instance configs."""
    parts = [p.strip() for p in topology_spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("topology specification must not be empty")
    tp_degrees = [int(p) for p in parts]
    if any(tp <= 0 for tp in tp_degrees):
        raise ValueError(f"TP degrees must be positive: {tp_degrees}")
    if len(tp_degrees) < 2 or len(set(tp_degrees)) != 2:
        raise ValueError(
            "cmlfq_cost requires >= 2 instances spanning exactly 2 distinct TP tiers, "
            f"got {tp_degrees}"
        )

    total_gpus = sum(tp_degrees)
    gpu_list = list(range(total_gpus)) if gpus is None else list(gpus)
    if len(gpu_list) != total_gpus:
        raise ValueError(
            f"topology {tp_degrees} requires {total_gpus} GPUs, got {len(gpu_list)} ({gpu_list})"
        )

    instances: list[dict[str, Any]] = []
    tp_counts: Counter[int] = Counter()
    cursor = 0
    for idx, tp in enumerate(tp_degrees):
        assigned = gpu_list[cursor : cursor + tp]
        cursor += tp
        suffix = tp_counts[tp]
        tp_counts[tp] += 1
        instances.append({
            "instance_id": f"tp{tp}_{suffix}",
            "tp": tp,
            "gpus": assigned,
            "host": host,
            "port": base_port + idx,
        })
    return instances


def make_buckets_and_thresholds(
    instances: list[dict[str, Any]],
    *,
    max_model_len: int = 24576,
    short_threshold: int = 5000,
    medium_threshold: int = 12000,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Build unified Libra two-bucket routing definitions and length thresholds."""
    tps = sorted({int(inst["tp"]) for inst in instances})
    if len(tps) != 2:
        raise ValueError(f"Expected 2 distinct TP tiers, got {tps}")
    min_tp, max_tp = tps[0], tps[1]
    buckets = {
        "short": {"tp_degrees": [min_tp], "max_tokens": short_threshold},
        "long": {"tp_degrees": [max_tp], "max_tokens": max_model_len},
    }
    length_thresholds = {
        "short": short_threshold,
        "medium": medium_threshold,
        "long": max_model_len,
    }
    return buckets, length_thresholds


def _make_padded_text(tokenizer, target_tokens: int, seed_text: str, filler_unit: str) -> str:
    """Build exact-token-budget text using the configured tokenizer."""
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler_unit, add_special_tokens=False)
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _chat_prompt(tokenizer, system_msg: str, user_msg: str) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def _fit_chat_prompt_exact(
    tokenizer,
    system_msg: str,
    user_head: str,
    filler_unit: str,
    target_tokens: int = 1024,
) -> tuple[str, int]:
    """Pad user filler so apply_chat_template hits ~target_tokens across sessions."""
    _, base_len = _chat_prompt(tokenizer, system_msg, user_head)
    filler_ids = tokenizer.encode(filler_unit, add_special_tokens=False)
    filler_n = max(16, target_tokens - base_len)
    pad_text = ""
    for _ in range(3):
        rep = (filler_n // max(1, len(filler_ids))) + 2
        pad_text = tokenizer.decode((filler_ids * rep)[: max(0, filler_n)], skip_special_tokens=True)
        full_text, full_len = _chat_prompt(tokenizer, system_msg, user_head + pad_text)
        diff = target_tokens - full_len
        if diff == 0:
            return full_text, full_len
        filler_n = max(0, filler_n + diff)
    return _chat_prompt(tokenizer, system_msg, user_head + pad_text)


def _interp_pf(pf_map: dict[int, float], ctx: int) -> float:
    """Lookup or piecewise-linearly interpolate measured prefill latency (ms)."""
    if ctx in pf_map:
        return pf_map[ctx]
    keys = sorted(pf_map)
    if ctx <= keys[0]:
        return pf_map[keys[0]] * (ctx / max(1, keys[0]))
    if ctx >= keys[-1]:
        return pf_map[keys[-1]] * (ctx / max(1, keys[-1]))
    for lo, hi in zip(keys[:-1], keys[1:]):
        if lo <= ctx <= hi:
            alpha = (ctx - lo) / (hi - lo)
            return pf_map[lo] * (1.0 - alpha) + pf_map[hi] * alpha
    return pf_map[keys[-1]]


def fit_rollout_calibration(
    prefill_ms: dict[str, dict[int, float]],
    raw_decode_ms: dict[str, dict[str, dict[str, float]]],
    prefill_specs: list[int],
    decode_specs: list[tuple[str, int, int]],
    hw_cfg_path: Path,
    arch_cfg_path: Path,
    model_path: str,
) -> dict[str, dict[str, float]]:
    """Fit analytical CostModel prefill/decode scale factors against live cold measurements."""
    uncalib_raw = AsyncRLConfig(model_path=model_path, tokenizer_path=model_path).to_dict()
    uncalib_raw["hardware"] = yaml.safe_load(hw_cfg_path.read_text(encoding="utf-8"))
    uncalib_raw["model_arch"] = yaml.safe_load(arch_cfg_path.read_text(encoding="utf-8"))
    uncalib_cfg = AsyncRLConfig.from_dict(uncalib_raw)
    rm_uncalib = CostModel(
        hardware=uncalib_cfg.hardware,
        model_arch=uncalib_cfg.model_arch,
        profiling=uncalib_cfg.profiling,
    ).rollout_model
    chunk = max(1, int(rm_uncalib.L_chunk))

    rollout_calibration: dict[str, dict[str, float]] = {}
    for tp_key in sorted(prefill_ms, key=lambda k: int(k)):
        tp = int(tp_key)
        pf_ratios = []
        for ctx in prefill_specs:
            uncalib_pf_ms = (ctx / chunk) * rm_uncalib.compute_prefill_step_time(1, chunk, tp) * 1000.0
            pf_ratios.append(prefill_ms[tp_key][ctx] / max(1e-3, uncalib_pf_ms))
        pf_scale = max(0.2, min(5.0, statistics.median(pf_ratios)))

        dec_ratios = []
        for label, ctx, gen in decode_specs:
            meas_dec_ms = max(
                1.0,
                raw_decode_ms[tp_key][label]["b1_total_ms"] - _interp_pf(prefill_ms[tp_key], ctx),
            )
            uncalib_pf_s = (ctx / chunk) * rm_uncalib.compute_prefill_step_time(1, chunk, tp)
            uncalib_dec_ms = max(
                1.0,
                (rm_uncalib.compute_instance_makespan([RequestInfo(ctx, gen)], tp) - uncalib_pf_s)
                * 1000.0,
            )
            dec_ratios.append(meas_dec_ms / uncalib_dec_ms)
        dec_scale = max(0.2, min(5.0, statistics.median(dec_ratios)))
        rollout_calibration[tp_key] = {
            "prefill_scale": round(pf_scale, 4),
            "decode_scale": round(dec_scale, 4),
        }
    return rollout_calibration


def _make_bare_engine(
    policy: str,
    instances: list[dict[str, Any]],
    *,
    model_path: str,
    hw_cfg_path: Path,
    arch_cfg_path: Path,
    out_dir: Path,
    max_model_len: int = 24576,
    gpu_memory_utilization: float = 0.85,
    short_threshold: int = 5000,
    medium_threshold: int = 12000,
    cost_profile_path: str = "",
    rollout_calibration: dict[str, dict[str, float]] | None = None,
) -> HeterogeneousRolloutEngine:
    """Instantiate a client-only HeterogeneousRolloutEngine pointing at running vLLM servers."""
    buckets, length_thresholds = make_buckets_and_thresholds(
        instances,
        max_model_len=max_model_len,
        short_threshold=short_threshold,
        medium_threshold=medium_threshold,
    )
    tps = sorted({int(inst["tp"]) for inst in instances})
    min_tp, max_tp = tps[0], tps[1]

    raw = AsyncRLConfig(
        model_path=model_path,
        tokenizer_path=model_path,
        sync_path=str(out_dir / "_unused_sync"),
        log_dir=str(out_dir / "_unused_logs"),
    ).to_dict()
    raw["enable_history_collection"] = False
    raw["hardware"] = yaml.safe_load(hw_cfg_path.read_text(encoding="utf-8"))
    raw["model_arch"] = yaml.safe_load(arch_cfg_path.read_text(encoding="utf-8"))
    if rollout_calibration and policy == "cmlfq_cost":
        raw.setdefault("profiling", {})["calibration"] = {"rollout": rollout_calibration}

    hr = raw["heterogeneous_rollout"]
    hr["enabled"] = True
    hr["total_gpus"] = sum(int(inst["tp"]) for inst in instances)
    hr["max_model_len"] = max_model_len
    hr["gpu_memory_utilization"] = gpu_memory_utilization
    hr["instances"] = [dict(inst) for inst in instances]

    sc = hr["scheduling"]
    sc["max_queue_length"] = 64
    sc["enable_fallback"] = True
    if policy in ("round_robin", "least_connections"):
        sc["scheduler_type"] = "load_balance"
        sc["load_balance_strategy"] = policy
    elif policy == "length_aware":
        sc["scheduler_type"] = "length_aware"
        sc["load_balance_strategy"] = "least_connections"
        sc["routing_rules"] = {"short": [min_tp], "medium": [max_tp], "long": [max_tp]}
        sc["length_thresholds"] = length_thresholds
    elif policy == "la_mlfq":
        sc["scheduler_type"] = "la_mlfq"
        sc["load_balance_strategy"] = "least_connections"
        sc["la_mlfq_buckets"] = buckets
        sc["length_thresholds"] = length_thresholds
        sc["la_mlfq_history_ttl"] = 10
    elif policy == "cmlfq":
        sc["scheduler_type"] = "cmlfq"
        sc["cmlfq_buckets"] = buckets
        sc["cmlfq_tree_path"] = ""
        sc["cmlfq_rebuild_interval"] = 32
        sc["cmlfq_kv_backend"] = "recompute"
    elif policy == "cmlfq_cost":
        sc["scheduler_type"] = "cmlfq_cost"
        sc["cmlfq_buckets"] = buckets
        sc["cmlfq_tree_path"] = ""
        sc["cmlfq_rebuild_interval"] = 32
        sc["cmlfq_kv_backend"] = "recompute"
        sc["cmlfq_cost_profile_path"] = cost_profile_path
    else:
        raise ValueError(f"Unknown policy: {policy}")

    full_cfg = AsyncRLConfig.from_dict(raw)
    engine = HeterogeneousRolloutEngine.from_config(full_cfg)
    if isinstance(engine.scheduler, CMLFQCostScheduler):
        cm = CostModel(
            hardware=full_cfg.hardware,
            model_arch=full_cfg.model_arch,
            profiling=full_cfg.profiling,
        )
        engine.scheduler.routing_costs = CMLFQRoutingCosts(
            profile_path=cost_profile_path,
            rollout_model=cm.rollout_model,
        )
    return engine


async def run_phase1_calibration(
    tokenizer,
    instances: list[dict[str, Any]],
    *,
    model_path: str,
    hw_cfg_path: Path,
    arch_cfg_path: Path,
    out_dir: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    short_threshold: int,
    medium_threshold: int,
) -> dict[str, Any]:
    """Run live cold micro-calibration on the two distinct TP tiers."""
    print("=== PHASE 1: Live Cold Micro-Calibration (TP_short vs TP_long) ===", flush=True)
    engine = _make_bare_engine(
        "round_robin",
        instances,
        model_path=model_path,
        hw_cfg_path=hw_cfg_path,
        arch_cfg_path=arch_cfg_path,
        out_dir=out_dir,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        short_threshold=short_threshold,
        medium_threshold=medium_threshold,
    )
    try:
        warm_prompt = _make_padded_text(
            tokenizer,
            128,
            "[calib_warmup_nonce_00] Solve 2x + 3 = 11 briefly. ",
            "Verify each arithmetic step cleanly. ",
        )
        await asyncio.gather(*(
            sub.generate(
                prompt=warm_prompt,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                n=1,
                ignore_eos=True,
            )
            for sub in engine.engines
        ))

        tps = sorted({int(inst["tp"]) for inst in instances})
        min_tp, max_tp = tps[0], tps[1]
        probe_indices = [
            next(i for i, inst in enumerate(instances) if int(inst["tp"]) == min_tp),
            next(i for i, inst in enumerate(instances) if int(inst["tp"]) == max_tp),
        ]
        prefill_specs = [c for c in (2048, 4096, 8192, 16384) if c < max_model_len]
        if not prefill_specs:
            prefill_specs = [max(256, max_model_len // 2)]

        prefill_ms: dict[str, dict[int, float]] = {}
        for idx in probe_indices:
            tp_key = str(engine.instance_configs[idx]["tp_degree"])
            sub = engine.engines[idx]
            prefill_ms[tp_key] = {}
            for ctx in prefill_specs:
                samples = []
                for trial in range(3):
                    # Unique token-0 prefix guarantees cold prefill even with vLLM APC enabled.
                    prompt = _make_padded_text(
                        tokenizer,
                        ctx,
                        f"[calib_pf_tp{tp_key}_L{ctx}_t{trial}] Analyze distributed KV state across {ctx} tokens:\n",
                        f"Block trace tp{tp_key}_L{ctx}_t{trial}: entry status ok, prefix alignment verified, worker nominal.\n",
                    )
                    t0 = time.perf_counter()
                    await sub.generate(
                        prompt=prompt,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                        n=1,
                        ignore_eos=True,
                    )
                    samples.append((time.perf_counter() - t0) * 1000.0)
                prefill_ms[tp_key][ctx] = statistics.median(samples)
                print(
                    f"  [Calib Prefill] tp={tp_key} ctx={ctx:5d} median={prefill_ms[tp_key][ctx]:7.1f}ms",
                    flush=True,
                )

        decode_specs = [
            (lbl, ctx, gen)
            for lbl, ctx, gen in (("ctx2048_g1024", 2048, 1024), ("ctx6144_g1024", 6144, 1024))
            if ctx + gen <= max_model_len
        ]
        if not decode_specs:
            ctx_fallback = max(256, max_model_len // 2)
            gen_fallback = max(64, min(512, max_model_len - ctx_fallback - 16))
            decode_specs = [(f"ctx{ctx_fallback}_g{gen_fallback}", ctx_fallback, gen_fallback)]

        raw_decode_ms: dict[str, dict[str, dict[str, float]]] = {}
        for idx in probe_indices:
            tp_key = str(engine.instance_configs[idx]["tp_degree"])
            sub = engine.engines[idx]
            raw_decode_ms[tp_key] = {}
            for label, ctx, gen in decode_specs:
                prompt_b1 = _make_padded_text(
                    tokenizer,
                    ctx,
                    f"[calib_dec_tp{tp_key}_{label}_b1] Derive a complete formal proof for variant {label}:\n",
                    f"Constraint {label}_b1: invariant A_i <= B_i + 1 preserved over state transitions.\n",
                )
                t0 = time.perf_counter()
                await sub.generate(
                    prompt=prompt_b1,
                    max_new_tokens=gen,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    ignore_eos=True,
                )
                b1_total_ms = (time.perf_counter() - t0) * 1000.0

                b8_prompts = [
                    _make_padded_text(
                        tokenizer,
                        ctx,
                        f"[calib_dec_tp{tp_key}_{label}_b8_s{k}] Derive proof variant {label} sample {k}:\n",
                        f"Constraint {label}_b8_s{k}: invariant A_i <= B_i + 1 preserved over state transitions.\n",
                    )
                    for k in range(8)
                ]
                t0 = time.perf_counter()
                await asyncio.gather(*(
                    sub.generate(
                        prompt=p_k,
                        max_new_tokens=gen,
                        temperature=0.0,
                        top_p=1.0,
                        n=1,
                        ignore_eos=True,
                    )
                    for p_k in b8_prompts
                ))
                b8_wall_ms = (time.perf_counter() - t0) * 1000.0
                raw_decode_ms[tp_key][label] = {
                    "b1_total_ms": round(b1_total_ms, 2),
                    "b8_wall_ms": round(b8_wall_ms, 2),
                }
                print(
                    f"  [Calib Decode] tp={tp_key} {label:13s}: b1={b1_total_ms:.0f}ms, b8_wall={b8_wall_ms:.0f}ms",
                    flush=True,
                )

        rollout_calibration = fit_rollout_calibration(
            prefill_ms,
            raw_decode_ms,
            prefill_specs,
            decode_specs,
            hw_cfg_path,
            arch_cfg_path,
            model_path,
        )
        print(f"  [Calib Fit] rollout_calibration={rollout_calibration}", flush=True)

        migrations = []
        host_min = instances[probe_indices[0]]["host"]
        host_max = instances[probe_indices[1]]["host"]
        for src_tp, dst_tp, src_host, dst_host in (
            (min_tp, max_tp, host_min, host_max),
            (max_tp, min_tp, host_max, host_min),
        ):
            for ctx in prefill_specs:
                migrations.append({
                    "source_tp": src_tp,
                    "target_tp": dst_tp,
                    "source_host": src_host,
                    "target_host": dst_host,
                    "seq_len": ctx,
                    "recompute_ms": round(prefill_ms[str(dst_tp)][ctx], 2),
                })

        profile = {
            "decode_ms": {},
            "migrations": migrations,
            "rollout_calibration": rollout_calibration,
            "raw_measurements": {
                "prefill_ms": prefill_ms,
                "decode": raw_decode_ms,
            },
        }
        profile_path = out_dir / "h20_rollout_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        return {
            "profile_path": str(profile_path),
            "profile": profile,
            "rollout_calibration": rollout_calibration,
        }
    finally:
        await engine.close()


def build_workload(
    tokenizer,
    session_tag: str,
    *,
    is_warmup: bool,
    max_model_len: int = 24576,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build session-isolated SWE multi-turn workload (zero cross-run APC leakage)."""
    sys_short = (
        f"[session={session_tag}] You are a meticulous repository QA & math verification assistant. "
        "Provide a clear step-by-step derivation, sanity check boundary values, and end with \\boxed{} (~220 words)."
    )
    sys_agent = (
        f"[session={session_tag}] You are an analytical software verification & distributed systems engineer. "
        "Always write thorough step-by-step reasoning, state invariants, and complete verification details."
    )

    def make_item(kind: str, tag: str, idx: int) -> dict[str, Any]:
        pid = f"{kind}_{tag}_{idx:03d}"
        filler = (
            f"def verify_{pid}(shard_id, seq_len, active_reqs): "
            f"assert shard_id >= 0 and seq_len <= {max_model_len}; return active_reqs * 2 + shard_id\n"
        )
        if kind == "short":
            a = 5 + (idx % 11)
            b = 9 + (idx % 17)
            x = 3 + (idx % 13)
            rhs = a * x + b
            head = (
                f"[{pid}] Repository Verification Task:\n"
                f"Solve for shard parameter $x$ where ${a}x + {b} = {rhs}$, verify uniqueness "
                f"and overflow safety against `max_seq={max_model_len}`, explain each arithmetic check clearly "
                "in ~200 words, and finish with \\boxed{}.\nSnapshot context:\n"
            )
            p0, n0 = _fit_chat_prompt_exact(tokenizer, sys_short, head, filler, target_tokens=1024)
            return {
                "prompt_id": pid,
                "kind": "short",
                "prompt0": p0,
                "input_tokens0": n0,
                "turns": [{"max_new_tokens": 256}],
            }
        if kind == "quick_fix":
            m = 7 + idx
            head = (
                f"[{pid}] Repository Patch Triage:\n"
                f"Inspect off-by-one boundary in ring buffer size {m}. Draft an initial hypothesis in 3 concise "
                "bullet points before running the test suite.\nSnapshot context:\n"
            )
            p0, n0 = _fit_chat_prompt_exact(tokenizer, sys_agent, head, filler, target_tokens=1024)
            fb1 = _make_padded_text(
                tokenizer,
                1000,
                f"### test_runner ({pid})\nstatus=failure\nMinor IndexError in ring_buffer_pop at offset {m}. "
                "Summarize the exact 1-line fix, state why surrounding callers stay valid, and provide a short unified diff.\n"
                "--- pytest traceback ---\n",
                f"FAILED tests/test_ring_{m % 5}.py::test_boundary_{idx} - IndexError: pop index {m} out of range\n",
            )
            return {
                "prompt_id": pid,
                "kind": "quick_fix",
                "prompt0": p0,
                "input_tokens0": n0,
                "turns": [
                    {"max_new_tokens": 96},
                    {
                        "tool_type": "test_runner",
                        "status": "failure",
                        "payload_tokens": 1000,
                        "feedback": fb1,
                        "max_new_tokens": 192,
                    },
                ],
            }
        if kind == "deep_fail":
            m = 19 + idx
            head = (
                f"[{pid}] Deep Distributed Consensus Bug:\n"
                f"Inspect multi-shard state replication across {m} replicas for epoch fencepost & lock ordering bugs. "
                "List initial diagnostic hypotheses before inspecting failure logs.\nSnapshot context:\n"
            )
            p0, n0 = _fit_chat_prompt_exact(tokenizer, sys_agent, head, filler, target_tokens=1024)
            turns: list[dict[str, Any]] = [{"max_new_tokens": 96}]
            for t in range(1, 11):
                fb = _make_padded_text(
                    tokenizer,
                    1000,
                    f"### test_runner ({pid} phase {t}/10)\nstatus=failure\nDeep AssertionError in replica-{m} state machine "
                    f"(sub-case {t}/10)! Write a comprehensive mathematical invariant proof, full transition table, "
                    "edge-case analysis, and complete corrected Python implementation.\n--- state dump ---\n",
                    f"E AssertionError: epoch={t} replica={m} expected barrier commit <= snapshot_seq\n",
                )
                turns.append({
                    "tool_type": "test_runner",
                    "status": "failure",
                    "payload_tokens": 1000,
                    "feedback": fb,
                    "max_new_tokens": 640,
                })
            return {
                "prompt_id": pid,
                "kind": "deep_fail",
                "prompt0": p0,
                "input_tokens0": n0,
                "turns": turns,
            }

        m = 31 + idx
        head = (
            f"[{pid}] Deep Async Pipeline Tail-Latency Bug:\n"
            f"Diagnose worst-case tail latency stall in a {m}-worker asynchronous pipeline with preemption and rollback. "
            "State initial hypotheses before inspecting profiler logs.\nSnapshot context:\n"
        )
        p0, n0 = _fit_chat_prompt_exact(tokenizer, sys_agent, head, filler, target_tokens=1024)
        turns = [{"max_new_tokens": 96}]
        for t in range(1, 11):
            fb = _make_padded_text(
                tokenizer,
                1000,
                f"### test_runner ({pid} stress {t}/10)\nstatus=timeout\nPerformance regression timed out on {m}-worker "
                f"stress harness (phase {t}/10)! Derive the complete asymptotic complexity proof, amortization "
                "equations, step-by-step execution trace, and full async Python rewrite.\n--- profiler trace ---\n",
                f"TIMEOUT worker={m} phase={t} queue_wait_ms=14820 stalling_barrier=commit_lock\n",
            )
            turns.append({
                "tool_type": "test_runner",
                "status": "timeout",
                "payload_tokens": 1000,
                "feedback": fb,
                "max_new_tokens": 640,
            })
        return {
            "prompt_id": pid,
            "kind": "deep_timeout",
            "prompt0": p0,
            "input_tokens0": n0,
            "turns": turns,
        }

    if is_warmup:
        counts = {"short": 6, "quick_fix": 4, "deep_fail": 2, "deep_timeout": 1}
        tag = "w"
        rng_seed = seed
    else:
        counts = {"short": 48, "quick_fix": 10, "deep_fail": 1, "deep_timeout": 1}
        tag = "e"
        rng_seed = seed + 1

    items: list[dict[str, Any]] = []
    for kind in ("short", "quick_fix", "deep_fail", "deep_timeout"):
        for idx in range(counts[kind]):
            items.append(make_item(kind, tag, idx))
    random.Random(rng_seed).shuffle(items)
    return items


async def run_one_episode(
    engine: HeterogeneousRolloutEngine,
    tokenizer,
    item: dict[str, Any],
    *,
    epoch: int,
) -> dict[str, Any]:
    t_ep0 = time.perf_counter()
    req_id = ""
    if hasattr(engine.scheduler, "get_request_route"):
        req_id = await engine.begin_cmlfq_request_when_available(
            item["prompt_id"], item["input_tokens0"], epoch=epoch
        )

    cur_text = item["prompt0"]
    cur_tokens = item["input_tokens0"]
    pure_generated = 0
    total_output_tokens = 0
    calls = []
    try:
        for turn_idx, turn_spec in enumerate(item["turns"]):
            if turn_idx > 0:
                fb = turn_spec["feedback"]
                event = {
                    "tool_type": turn_spec["tool_type"],
                    "status": turn_spec["status"],
                    "payload_tokens": turn_spec["payload_tokens"],
                    "token_position": total_output_tokens,
                    "generated_token_position": pure_generated,
                    "turn": turn_idx - 1,
                }
                if req_id:
                    engine.record_cmlfq_tool_return(req_id, event, total_output_tokens)
                cur_text += "\n" + fb + "\n"
                cur_tokens = len(tokenizer.encode(cur_text, add_special_tokens=False))

            kwargs: dict[str, Any] = {
                "prompt": cur_text,
                "max_new_tokens": turn_spec["max_new_tokens"],
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
                "prompt_id": item["prompt_id"],
                "input_tokens": cur_tokens,
                "epoch": epoch,
            }
            if req_id:
                kwargs["request_id"] = req_id

            t_g0 = time.perf_counter()
            resp = await engine.generate(**kwargs)
            t_g1 = time.perf_counter()
            out_text = resp.get("text", "")
            out_toks = len(resp.get("tokens", [])) or len(
                tokenizer.encode(out_text, add_special_tokens=False)
            )
            pure_generated += out_toks
            total_output_tokens += out_toks + (
                turn_spec.get("payload_tokens", 0) if turn_idx > 0 else 0
            )
            cur_text += out_text
            info = resp.get("_schedule_info", {})
            calls.append({
                "turn": turn_idx,
                "tp": info.get("tp_degree"),
                "instance_id": info.get("instance_id"),
                "reason": info.get("reason", ""),
                "path": info.get("execution_path", ""),
                "out_tokens": out_toks,
                "dur_s": t_g1 - t_g0,
            })

        if req_id:
            engine.finish_cmlfq_request(
                req_id, total_output_tokens, generated_tokens=pure_generated
            )
    except BaseException:
        if req_id:
            engine.cancel_cmlfq_request(req_id)
        raise

    return {
        "prompt_id": item["prompt_id"],
        "kind": item["kind"],
        "dur_s": time.perf_counter() - t_ep0,
        "pure_generated_tokens": pure_generated,
        "calls": calls,
    }


async def eval_policy(
    policy: str,
    round_idx: int,
    run_slot: int,
    tokenizer,
    instances: list[dict[str, Any]],
    *,
    model_path: str,
    hw_cfg_path: Path,
    arch_cfg_path: Path,
    out_dir: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    short_threshold: int,
    medium_threshold: int,
    cost_profile_path: str,
    rollout_calibration: dict[str, dict[str, float]],
    seed: int,
) -> dict[str, Any]:
    w_tag = f"W{run_slot:02d}"
    e_tag = f"E{run_slot:02d}"
    warmup_items = build_workload(
        tokenizer, w_tag, is_warmup=True, max_model_len=max_model_len, seed=seed
    )
    eval_items = build_workload(
        tokenizer, e_tag, is_warmup=False, max_model_len=max_model_len, seed=seed
    )
    engine = _make_bare_engine(
        policy,
        instances,
        model_path=model_path,
        hw_cfg_path=hw_cfg_path,
        arch_cfg_path=arch_cfg_path,
        out_dir=out_dir,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        short_threshold=short_threshold,
        medium_threshold=medium_threshold,
        cost_profile_path=cost_profile_path,
        rollout_calibration=rollout_calibration,
    )
    try:
        async def _run_batch(items: list[dict[str, Any]], epoch: int) -> list[dict[str, Any]]:
            sem = asyncio.Semaphore(len(items))

            async def _bounded(it: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    return await run_one_episode(engine, tokenizer, it, epoch=epoch)

            return await asyncio.gather(*(_bounded(it) for it in items))

        engine.notify_epoch_start(0)
        await _run_batch(warmup_items, 0)
        engine.notify_epoch_end(0)

        engine.notify_epoch_start(1)
        t0 = time.perf_counter()
        ep_results = await _run_batch(eval_items, 1)
        makespan_s = time.perf_counter() - t0
        engine.notify_epoch_end(1)

        durs = sorted(ep["dur_s"] for ep in ep_results)
        p50 = durs[len(durs) // 2]
        p90 = durs[min(len(durs) - 1, int(0.9 * len(durs)))]
        total_gen_tokens = sum(ep["pure_generated_tokens"] for ep in ep_results)
        tp_calls: Counter[str] = Counter()
        tp_tokens: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        paths: Counter[str] = Counter()
        for ep in ep_results:
            for c in ep["calls"]:
                tp_key = str(c["tp"])
                tp_calls[tp_key] += 1
                tp_tokens[tp_key] += c["out_tokens"]
                reasons[c["reason"]] += 1
                if c["path"]:
                    paths[c["path"]] += 1

        stats = engine.scheduler.get_stats().to_dict()
        migrations = stats.get("migrated_routes", 0)
        by_kind = {
            k: round(statistics.mean(ep["dur_s"] for ep in ep_results if ep["kind"] == k), 3)
            for k in ("short", "quick_fix", "deep_fail", "deep_timeout")
        }
        by_kind_gen = {
            k: round(
                statistics.mean(
                    ep["pure_generated_tokens"] for ep in ep_results if ep["kind"] == k
                ),
                1,
            )
            for k in ("short", "quick_fix", "deep_fail", "deep_timeout")
        }
        print(
            f"[R{round_idx} {policy:18s}] makespan={makespan_s:6.2f}s | "
            f"tok/s={total_gen_tokens / makespan_s:7.1f} | p50={p50:5.2f}s p90={p90:5.2f}s | "
            f"tp_calls={dict(sorted(tp_calls.items()))} | mig={migrations} | "
            f"gen_tok={total_gen_tokens} | gen_by_kind={by_kind_gen}",
            flush=True,
        )
        return {
            "round": round_idx,
            "session_tags": {"warmup": w_tag, "eval": e_tag},
            "policy": policy,
            "makespan_s": makespan_s,
            "total_generated_tokens": total_gen_tokens,
            "tokens_per_second": total_gen_tokens / makespan_s,
            "trajectories_per_second": len(ep_results) / makespan_s,
            "latency_p50_s": p50,
            "latency_p90_s": p90,
            "latency_max_s": durs[-1],
            "tp_calls": dict(sorted(tp_calls.items())),
            "tp_tokens": dict(sorted(tp_tokens.items())),
            "migrations": migrations,
            "reasons": dict(reasons),
            "paths": dict(paths),
            "by_kind": by_kind,
            "by_kind_gen_tokens": by_kind_gen,
        }
    finally:
        await engine.close()


def _resolve_policies(raw_policies: str) -> list[str]:
    if raw_policies.strip().lower() == "all":
        return list(POLICIES_FORWARD)
    selected = [p.strip() for p in raw_policies.split(",") if p.strip()]
    unknown = [p for p in selected if p not in POLICIES_FORWARD]
    if unknown:
        raise ValueError(f"Unknown policies {unknown}; valid choices: {list(POLICIES_FORWARD)}")
    if not selected:
        raise ValueError("No policies selected")
    return selected


def run_dry_run(
    args: argparse.Namespace,
    instances: list[dict[str, Any]],
    selected_policies: list[str],
) -> dict[str, Any]:
    """Instantiate all selected schedulers offline and verify CostModel wiring."""
    buckets, length_thresholds = make_buckets_and_thresholds(
        instances,
        max_model_len=args.max_model_len,
        short_threshold=args.short_threshold,
        medium_threshold=args.medium_threshold,
    )
    tps = sorted({int(inst["tp"]) for inst in instances})
    min_tp, max_tp = tps[0], tps[1]
    dummy_cal = {
        str(min_tp): {"prefill_scale": 0.67, "decode_scale": 1.46},
        str(max_tp): {"prefill_scale": 0.78, "decode_scale": 2.65},
    }
    hw_cfg_path = Path(args.hardware_config)
    arch_cfg_path = Path(args.model_arch_config)

    with tempfile.TemporaryDirectory(prefix="libra_rollout_dryrun_") as tmpdir:
        tmp_path = Path(tmpdir)
        profile_path = tmp_path / "dummy_profile.json"
        profile_path.write_text(
            json.dumps({
                "decode_ms": {},
                "migrations": [
                    {
                        "source_tp": min_tp,
                        "target_tp": max_tp,
                        "source_host": instances[0]["host"],
                        "target_host": instances[-1]["host"],
                        "seq_len": 2048,
                        "recompute_ms": 130.0,
                    },
                    {
                        "source_tp": max_tp,
                        "target_tp": min_tp,
                        "source_host": instances[-1]["host"],
                        "target_host": instances[0]["host"],
                        "seq_len": 2048,
                        "recompute_ms": 430.0,
                    },
                ],
            }),
            encoding="utf-8",
        )
        schedulers: dict[str, str] = {}
        for policy in selected_policies:
            eng = _make_bare_engine(
                policy,
                instances,
                model_path=args.model,
                hw_cfg_path=hw_cfg_path,
                arch_cfg_path=arch_cfg_path,
                out_dir=tmp_path,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                short_threshold=args.short_threshold,
                medium_threshold=args.medium_threshold,
                cost_profile_path=str(profile_path),
                rollout_calibration=dummy_cal,
            )
            schedulers[policy] = type(eng.scheduler).__name__
            if policy == "cmlfq_cost":
                rc = eng.scheduler.routing_costs
                assert rc.decode_ms == {}, f"Expected empty decode_ms, got {rc.decode_ms}"
                assert rc.rollout_model is not None
                assert rc.rollout_model.calibration == dummy_cal
                assert len(rc.migrations) == 2

    payload = {
        "dry_run": True,
        "model": args.model,
        "instances": instances,
        "buckets": buckets,
        "length_thresholds": length_thresholds,
        "policies": schedulers,
        "rounds": args.rounds,
    }
    print(json.dumps(payload, indent=2))
    return payload


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    gpus = (
        [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        if args.gpus.strip()
        else None
    )
    instances = parse_topology(args.topology, gpus=gpus, host=args.host, base_port=args.base_port)
    selected_policies = _resolve_policies(args.policies)

    if args.dry_run:
        return run_dry_run(args, instances, selected_policies)

    from transformers import AutoTokenizer

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hw_cfg_path = Path(args.hardware_config)
    arch_cfg_path = Path(args.model_arch_config)
    buckets, length_thresholds = make_buckets_and_thresholds(
        instances,
        max_model_len=args.max_model_len,
        short_threshold=args.short_threshold,
        medium_threshold=args.medium_threshold,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    calib = await run_phase1_calibration(
        tokenizer,
        instances,
        model_path=args.model,
        hw_cfg_path=hw_cfg_path,
        arch_cfg_path=arch_cfg_path,
        out_dir=out_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        short_threshold=args.short_threshold,
        medium_threshold=args.medium_threshold,
    )

    print(
        f"=== PHASE 2: {len(selected_policies)}-Policy Comparison "
        f"(warmup=13, eval=60, full-batch, vLLM APC ON, short={args.short_threshold}) ===",
        flush=True,
    )

    runs = []
    slot = 0
    for round_idx in range(1, args.rounds + 1):
        order = selected_policies if round_idx % 2 == 1 else list(reversed(selected_policies))
        print(f"--- Round {round_idx}: {order} ---", flush=True)
        for policy in order:
            slot += 1
            runs.append(
                await eval_policy(
                    policy,
                    round_idx,
                    slot,
                    tokenizer,
                    instances,
                    model_path=args.model,
                    hw_cfg_path=hw_cfg_path,
                    arch_cfg_path=arch_cfg_path,
                    out_dir=out_dir,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    short_threshold=args.short_threshold,
                    medium_threshold=args.medium_threshold,
                    cost_profile_path=calib["profile_path"],
                    rollout_calibration=calib["rollout_calibration"],
                    seed=args.seed,
                )
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for policy in selected_policies:
        rows = [r for r in runs if r["policy"] == policy]
        mean_mk = statistics.mean(r["makespan_s"] for r in rows)
        mean_tps = statistics.mean(r["tokens_per_second"] for r in rows)
        mean_p50 = statistics.mean(r["latency_p50_s"] for r in rows)
        mean_p90 = statistics.mean(r["latency_p90_s"] for r in rows)
        mean_mig = statistics.mean(r["migrations"] for r in rows)
        aggregate[policy] = {
            "rounds": len(rows),
            "mean_makespan_s": mean_mk,
            "mean_tokens_per_second": mean_tps,
            "mean_latency_p50_s": mean_p50,
            "mean_latency_p90_s": mean_p90,
            "mean_migrations": mean_mig,
            "round_makespans_s": [r["makespan_s"] for r in rows],
            "total_generated_tokens": [r["total_generated_tokens"] for r in rows],
            "tp_calls_per_round": [r["tp_calls"] for r in rows],
            "by_kind_mean_s": {
                k: round(statistics.mean(r["by_kind"][k] for r in rows), 3)
                for k in ("short", "quick_fix", "deep_fail", "deep_timeout")
            },
            "by_kind_gen_tokens": {
                k: round(statistics.mean(r["by_kind_gen_tokens"][k] for r in rows), 1)
                for k in ("short", "quick_fix", "deep_fail", "deep_timeout")
            },
        }

    if "round_robin" in aggregate:
        rr = aggregate["round_robin"]
        for stats_d in aggregate.values():
            stats_d["speedup_vs_round_robin"] = rr["mean_makespan_s"] / stats_d["mean_makespan_s"]
            stats_d["throughput_gain_vs_rr_pct"] = (
                stats_d["mean_tokens_per_second"] / rr["mean_tokens_per_second"] - 1.0
            ) * 100.0
            stats_d["makespan_reduction_vs_rr_pct"] = (
                1.0 - stats_d["mean_makespan_s"] / rr["mean_makespan_s"]
            ) * 100.0
    if "least_connections" in aggregate:
        lc = aggregate["least_connections"]
        for stats_d in aggregate.values():
            stats_d["speedup_vs_least_connections"] = (
                lc["mean_makespan_s"] / stats_d["mean_makespan_s"]
            )
            stats_d["throughput_gain_vs_lc_pct"] = (
                stats_d["mean_tokens_per_second"] / lc["mean_tokens_per_second"] - 1.0
            ) * 100.0
    if "length_aware" in aggregate:
        la = aggregate["length_aware"]
        for stats_d in aggregate.values():
            stats_d["throughput_gain_vs_la_pct"] = (
                stats_d["mean_tokens_per_second"] / la["mean_tokens_per_second"] - 1.0
            ) * 100.0
    if "cmlfq" in aggregate:
        cm = aggregate["cmlfq"]
        for stats_d in aggregate.values():
            stats_d["throughput_gain_vs_cmlfq_pct"] = (
                stats_d["mean_tokens_per_second"] / cm["mean_tokens_per_second"] - 1.0
            ) * 100.0

    summary = {
        "topology": args.topology,
        "instances": instances,
        "model": args.model,
        "thresholds": {"buckets": buckets, "length_thresholds": length_thresholds},
        "calibration": calib["profile"],
        "workload": {"warmup": 13, "eval": 60, "concurrency": {"warmup": 13, "eval": 60}},
        "runs": runs,
        "aggregate": aggregate,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== AGGREGATE SUMMARY ===", flush=True)
    for policy in selected_policies:
        s = aggregate[policy]
        vs_rr = (
            f" | vs_RR={s['speedup_vs_round_robin']:.2f}x "
            f"(tok/s {s['throughput_gain_vs_rr_pct']:+5.1f}%, mk {s['makespan_reduction_vs_rr_pct']:+5.1f}%)"
            if "speedup_vs_round_robin" in s
            else ""
        )
        print(
            f"{policy:18s}: makespan={s['mean_makespan_s']:6.2f}s | "
            f"tok/s={s['mean_tokens_per_second']:7.1f} | p90={s['mean_latency_p90_s']:5.2f}s{vs_rr}",
            flush=True,
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier 2 controlled multi-turn heterogeneous rollout scheduling benchmark."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="Model ID or local path")
    parser.add_argument(
        "--topology",
        default="1,1,1,1,4",
        help="Comma-separated TP degrees across instances (default: 1,1,1,1,4)",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Comma-separated GPU IDs (default: 0..sum(topology)-1)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--base-port", type=int, default=19001, help="Starting vLLM server port")
    parser.add_argument(
        "--hardware-config",
        default=str(DEFAULT_HW_CFG),
        help="Hardware config YAML for analytical CostModel",
    )
    parser.add_argument(
        "--model-arch-config",
        default=str(DEFAULT_ARCH_CFG),
        help="Model architecture config YAML for analytical CostModel",
    )
    parser.add_argument(
        "--output-dir",
        default="./logs/tier2_rollout_scheduling",
        help="Directory for calibration profile and results.json",
    )
    parser.add_argument("--max-model-len", type=int, default=24576, help="Max sequence length")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85, help="vLLM GPU memory utilization"
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=5000,
        help="Unified token threshold separating short vs long bucket (default: 5000)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=int,
        default=12000,
        help="Medium length threshold for length_aware/la_mlfq (default: 12000)",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Counterbalanced rounds (default: 2)")
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma-separated policy names or 'all'",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random workload shuffle seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify topology, thresholds, and scheduler initialization without GPUs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run_benchmark(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

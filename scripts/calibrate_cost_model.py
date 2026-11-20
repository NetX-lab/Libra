#!/usr/bin/env python3
"""Measure, fit, and validate Qwen3-14B cost-model calibration on GPU clusters."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def _load_length_profile(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _step_lengths(
    rows: list[dict[str, Any]],
    step: int,
    fallback: list[int],
    limit: int = 24,
) -> list[int]:
    step_rows = [int(r["sequence_length"]) for r in rows if r.get("step") == step and r.get("sequence_length")]
    if len(step_rows) >= limit:
        return step_rows[:limit]
    if step_rows:
        return step_rows
    return list(fallback[:limit])


def _rollout_requests_from_profile(
    rows: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    step: int | None = None,
    fallback: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    pool = [r for r in rows if (step is None or r.get("step") == step) and r.get("input_len") and r.get("output_len")]
    if len(pool) >= count:
        if step is not None:
            chosen = pool[:count]
        else:
            rng = random.Random(seed)
            chosen = rng.sample(pool, count)
        return [(int(r["input_len"]), max(1, int(r["output_len"]))) for r in chosen]
    return list(fallback[:count])


def cmd_measure_rollout(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    profile_rows = _load_length_profile(args.length_profile)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=42,
    )

    def run_once(requests: list[tuple[int, int]], seed_base: int) -> dict[str, Any]:
        rng = random.Random(seed_base)
        prompts = [
            TokensPrompt(prompt_token_ids=[rng.randint(100, 100000) for _ in range(p_len)])
            for p_len, _ in requests
        ]
        params = [
            SamplingParams(temperature=0.0, max_tokens=g_len, min_tokens=g_len, ignore_eos=True)
            for _, g_len in requests
        ]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        actual_gen = [len(out.outputs[0].token_ids) for out in outputs]
        return {"elapsed_s": elapsed, "actual_gen": actual_gen}

    # Warmup
    run_once([(256, 16), (256, 16)], seed_base=1)
    run_once([(512, 32)], seed_base=2)

    prefill_cases = [
        ("prefill_b1_512", [(512, 1)], 3),
        ("prefill_b1_1024", [(1024, 1)], 3),
        ("prefill_b1_2048", [(2048, 1)], 3),
        ("prefill_b1_4096", [(4096, 1)], 3),
        ("prefill_b1_6144", [(6144, 1)], 3),
        ("prefill_b4_1024", [(1024, 1)] * 4, 3),
    ]
    prefill_results = []
    prefill_lookup: dict[tuple[int, int], float] = {}
    for idx, (name, reqs, repeats) in enumerate(prefill_cases):
        runs = [run_once(reqs, seed_base=1000 + idx * 10 + r) for r in range(repeats)]
        times = [r["elapsed_s"] for r in runs]
        med = statistics.median(times)
        prefill_lookup[(len(reqs), reqs[0][0])] = med
        prefill_results.append({
            "name": name,
            "requests": reqs,
            "repeats_s": times,
            "median_s": med,
        })
        print(f"[Rollout TP={args.tp}] {name}: median={med:.4f}s (runs={[round(x, 4) for x in times]})", flush=True)

    decode_cases = [
        ("decode_b1_p1024_g128", [(1024, 128)], 3),
        ("decode_b1_p1024_g256", [(1024, 256)], 3),
        ("decode_b1_p1024_g512", [(1024, 512)], 3),
        ("decode_b1_p1024_g1024", [(1024, 1024)], 3),
        ("decode_b1_p4096_g512", [(4096, 512)], 3),
        ("decode_b4_p1024_g128", [(1024, 128)] * 4, 3),
        ("decode_b4_p1024_g256", [(1024, 256)] * 4, 3),
        ("decode_b4_p1024_g512", [(1024, 512)] * 4, 3),
    ]
    decode_results = []
    for idx, (name, reqs, repeats) in enumerate(decode_cases):
        runs = [run_once(reqs, seed_base=2000 + idx * 10 + r) for r in range(repeats)]
        times = [r["elapsed_s"] for r in runs]
        med = statistics.median(times)
        pre_s = prefill_lookup[(len(reqs), reqs[0][0])]
        pure_decode_s = max(0.0, med - pre_s)
        decode_results.append({
            "name": name,
            "requests": reqs,
            "repeats_s": times,
            "total_median_s": med,
            "matched_prefill_s": pre_s,
            "pure_decode_s": pure_decode_s,
            "actual_gen": runs[-1]["actual_gen"],
        })
        print(
            f"[Rollout TP={args.tp}] {name}: total={med:.4f}s - prefill={pre_s:.4f}s = decode={pure_decode_s:.4f}s",
            flush=True,
        )

    cohort_cases = [
        (
            "r2e_short_8",
            _rollout_requests_from_profile(
                profile_rows, 8, seed=101,
                fallback=[(2150, 256)] * 6 + [(2150, 512)] * 2,
            ),
            2,
        ),
        (
            "r2e_mixed_16",
            _rollout_requests_from_profile(
                profile_rows, 16, seed=202,
                fallback=[(2150, 192)] * 10 + [(2150, 384)] * 4 + [(2150, 512)] * 2,
            ),
            2,
        ),
        (
            "skewed_tail_8",
            [(2048, g) for g in (128, 128, 128, 128, 256, 256, 512, 2048)],
            2,
        ),
        (
            "long_tail_4",
            [(2048, g) for g in (256, 256, 512, 2048)],
            2,
        ),
        (
            "r2e_step1_24",
            _rollout_requests_from_profile(
                profile_rows, 24, seed=303, step=1,
                fallback=[(2150, 320)] * 16 + [(2150, 512)] * 8,
            ),
            2,
        ),
    ]
    cohort_results = []
    for idx, (name, reqs, repeats) in enumerate(cohort_cases):
        runs = [run_once(reqs, seed_base=3000 + idx * 10 + r) for r in range(repeats)]
        times = [r["elapsed_s"] for r in runs]
        med = statistics.median(times)
        cohort_results.append({
            "name": name,
            "requests": reqs,
            "repeats_s": times,
            "median_s": med,
            "actual_gen": runs[-1]["actual_gen"],
        })
        print(f"[Rollout TP={args.tp}] cohort {name} (n={len(reqs)}): median={med:.4f}s", flush=True)

    payload = {
        "tp": args.tp,
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prefill": prefill_results,
        "decode": decode_results,
        "cohorts": cohort_results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[Rollout TP={args.tp}] wrote {out}", flush=True)


def cmd_measure_training(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world = args.tp * args.dp
    if world_size != expected_world:
        raise ValueError(f"WORLD_SIZE={world_size} must equal tp*dp={expected_world}")

    engine = MegatronCoreTrainEngine(
        model_path=args.model,
        learning_rate=1.0e-6,
        kl_coef=0.001,
        clip_epsilon=0.2,
        micro_batch_size=args.micro_batch_size,
        recompute_micro_batch_size=args.recompute_micro_batch_size,
        train_tp_size=args.tp,
        train_pp_size=1,
        train_cp_size=1,
        train_ep_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        use_distributed_optimizer=True,
        grad_reduce_in_fp32=False,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=0.0,
        streaming_export=True,
        use_transformer_engine=False,
        use_cpu_initialization=False,
        recompute_num_layers=1,
        grpo_objective="decoupled",
    )
    engine.initialize(max_seq_length=args.max_seq_length, initialize_optimizer=True)

    opt_timer = {"last_s": 0.0}
    orig_step = engine.optimizer.step

    def timed_step(*s_args, **s_kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = orig_step(*s_args, **s_kwargs)
        torch.cuda.synchronize()
        opt_timer["last_s"] = time.perf_counter() - t0
        return res

    engine.optimizer.step = timed_step

    dp_rank = engine.get_data_parallel_rank()
    dp_world = engine.get_data_parallel_world_size()
    assert dp_world == args.dp

    def make_trajectories(lengths: list[int], seed: int) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        trajs = []
        for idx, length in enumerate(lengths):
            ids = torch.tensor([[rng.randint(100, 50000) for _ in range(length)]], dtype=torch.long)
            loss_mask = torch.ones((1, length), dtype=torch.float32)
            loss_mask[:, 0] = 0.0
            adv = 1.0 if (idx % 2 == 0) else -1.0
            trajs.append({
                "input_ids": ids,
                "loss_mask": loss_mask,
                "logprobs": torch.zeros((1, length), dtype=torch.float32),
                "behavior_logprobs": torch.zeros((1, length), dtype=torch.float32),
                "rewards": torch.tensor([adv], dtype=torch.float32),
                "advantages": torch.tensor([adv], dtype=torch.float32),
            })
        return trajs

    def shard_lengths(lengths: list[int]) -> list[int]:
        base, rem = divmod(len(lengths), dp_world)
        offset = dp_rank * base + min(dp_rank, rem)
        count = base + (1 if dp_rank < rem else 0)
        return lengths[offset : offset + count]

    def run_step(name: str, lengths: list[int], seed: int, role: str) -> dict[str, Any]:
        local_lengths = shard_lengths(lengths)
        local_trajs = make_trajectories(local_lengths, seed=seed + dp_rank * 997)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        engine.recompute_logprobs(local_trajs)
        torch.cuda.synchronize()
        dist.barrier()
        recompute_s = time.perf_counter() - t0

        for t in local_trajs:
            t["behavior_logprobs"] = t["logprobs"].clone()

        opt_timer["last_s"] = 0.0
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()
        engine.grpo_update(local_trajs, ppo_epochs=1)
        torch.cuda.synchronize()
        dist.barrier()
        grpo_s = time.perf_counter() - t1

        opt_tensor = torch.tensor([opt_timer["last_s"]], dtype=torch.float64, device=torch.cuda.current_device())
        dist.all_reduce(opt_tensor, op=dist.ReduceOp.MAX)
        opt_s = float(opt_tensor.item())
        fwd_bwd_comm_s = max(0.0, grpo_s - opt_s)
        total_s = recompute_s + grpo_s
        if rank == 0:
            print(
                f"[Train TP={args.tp} DP={args.dp}] {name} ({role}, B={len(lengths)}): "
                f"recompute={recompute_s:.4f}s, fwd_bwd_comm={fwd_bwd_comm_s:.4f}s, "
                f"opt={opt_s:.4f}s, grpo={grpo_s:.4f}s, total={total_s:.4f}s",
                flush=True,
            )
        return {
            "name": name,
            "role": role,
            "B_global": len(lengths),
            "lengths": lengths,
            "recompute_logprobs_s": recompute_s,
            "fwd_bwd_comm_s": fwd_bwd_comm_s,
            "optimizer_s": opt_s,
            "grpo_update_s": grpo_s,
            "total_train_s": total_s,
        }

    profile_rows = _load_length_profile(args.length_profile)
    # Warmup
    run_step("warmup", [2048] * (4 * args.dp), seed=10, role="warmup")

    if args.dp == 1:
        step0 = _step_lengths(profile_rows, 0, [2500] * 24)
        step1 = _step_lengths(profile_rows, 1, [2700] * 24)
        step2 = _step_lengths(profile_rows, 2, [2560] * 24)
        step3 = _step_lengths(profile_rows, 3, [2680] * 24)
        cases = [
            ("cal_uniform_2k_b8", [2048] * 8, "calibration"),
            ("cal_uniform_4k_b8", [4096] * 8, "calibration"),
            ("cal_r2e_step2_b16", step2[:16], "calibration"),
            ("cal_r2e_step0_b24", step0, "calibration"),
            ("val_mixed_b16", [2048] * 10 + [4096] * 4 + [6144] * 2, "validation"),
            ("val_r2e_step1_b24", step1, "validation"),
            ("val_r2e_step3_b24", step3, "validation"),
        ]
    else:
        step1 = _step_lengths(profile_rows, 1, [2700] * 24)
        step3 = _step_lengths(profile_rows, 3, [2680] * 24)
        cases = [
            ("val_dp2_uniform_b16", [3072] * 16, "validation"),
            ("val_dp2_r2e_step1_b24", step1, "validation"),
            ("val_dp2_skewed_b16", [4096] * 8 + [2048] * 8, "validation"),
            ("val_dp2_r2e_step3_b24", step3, "validation"),
        ]

    records = []
    for idx, (name, lengths, role) in enumerate(cases):
        records.append(run_step(name, lengths, seed=100 + idx * 17, role=role))

    if rank == 0:
        payload = {
            "tp": args.tp,
            "dp": args.dp,
            "model": args.model,
            "max_seq_length": args.max_seq_length,
            "records": records,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[Train TP={args.tp} DP={args.dp}] wrote {out}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _ratio_ols(preds: list[float], actuals: list[float]) -> float:
    denom = sum(p * p for p in preds)
    if denom <= 0:
        return 1.0
    return max(1e-6, sum(p * a for p, a in zip(preds, actuals)) / denom)


def _error_metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    abs_errs = [abs(p - a) for p, a in pairs]
    rel_errs = [abs(p - a) / max(a, 1e-9) * 100.0 for p, a in pairs]
    return {
        "n": len(pairs),
        "mae_s": statistics.mean(abs_errs) if abs_errs else 0.0,
        "max_abs_s": max(abs_errs) if abs_errs else 0.0,
        "mape_pct": statistics.mean(rel_errs) if rel_errs else 0.0,
        "max_ape_pct": max(rel_errs) if rel_errs else 0.0,
    }


def _evaluate_training_unsharded_anchor(cost_model, config, B_global: int, L: list[int]) -> tuple[float, dict]:
    """Reproduce PR #21 pre-fix evaluate_training where anchor was not DP-sharded."""
    seconds, details = cost_model.training_model.compute_iteration_time(config, B_global, L)
    cal = cost_model.calibration.get("training", {})
    compute = seconds * float(cal.get("compute_scale", 1.0))
    anchor = 0.0
    if cost_model.recompute_logprobs:
        lengths = cost_model.training_model._representative_lengths(L, B_global)
        ashape = replace(config, dp=1, b_micro=cost_model.recompute_micro_batch_size)
        anchor = sum(
            cost_model.training_model.compute_stage_fwd_time(
                max(lengths[s : s + ashape.b_micro]), ashape
            )
            for s in range(0, B_global, ashape.b_micro)
        )
        anchor *= float(cal.get("anchor_scale", 1.0))
    optimizer = float(cal.get("optimizer_s", 0.0))
    return compute + anchor + optimizer, {"compute_s": compute, "anchor_s": anchor, "optimizer_s": optimizer}


def _legacy_steady_state_makespan(rollout_model, requests, tp: int) -> float:
    """Reproduce pre-PR-#21 steady-state makespan formula for baseline comparison."""
    if not requests:
        return 0.0
    total_prompt = sum(r.prompt_length for r in requests)
    total_gen = sum(r.gen_length for r in requests)
    avg_prompt = total_prompt / len(requests)
    avg_gen = total_gen / len(requests)
    b_bar = rollout_model.compute_steady_state_batch(tp, avg_prompt, avg_gen)
    t_pre_step = rollout_model.compute_prefill_step_time(B_prefill=1, L_chunk=rollout_model.L_chunk, tp=tp)
    t_pre = (total_prompt / max(rollout_model.L_chunk, 1)) * t_pre_step
    avg_active = int(b_bar * (avg_prompt + avg_gen / 2.0))
    t_dec_step = rollout_model.compute_decode_step_time(avg_active, tp)
    t_dec = (total_gen / max(b_bar, 1.0)) * t_dec_step
    return t_pre + t_dec


def _fit_calibration(
    hw,
    ma,
    prof_base,
    rollout_files: dict[int, dict[str, Any]],
    train_dp1: dict[str, Any],
    hybrid_gradient_s: float,
) -> dict[str, Any]:
    from RL_Framework.config import ProfilingConfig
    from RL_Framework.infra.cost_model.model import CostModel, RequestInfo, TrainParallelConfig

    raw_prof = replace(prof_base, calibration={})
    raw_cm = CostModel(hardware=hw, model_arch=ma, profiling=raw_prof, max_seq_length=8192, recompute_logprobs=True)

    rollout_cal = {}
    for tp, data in sorted(rollout_files.items()):
        pre_preds, pre_acts = [], []
        for item in data["prefill"]:
            reqs = [RequestInfo(p, 0) for p, _ in item["requests"]]
            pre_preds.append(raw_cm.rollout_model.compute_instance_makespan(reqs, tp))
            pre_acts.append(item["median_s"])
        pre_scale = _ratio_ols(pre_preds, pre_acts)

        dec_preds, dec_acts = [], []
        for item in data["decode"]:
            reqs = [RequestInfo(p, g) for p, g in item["requests"]]
            # Isolate pure decode time from raw model (prefill prompt_only subtracted)
            full_raw = raw_cm.rollout_model.compute_instance_makespan(reqs, tp)
            pre_only = raw_cm.rollout_model.compute_instance_makespan(
                [RequestInfo(p, 0) for p, _ in item["requests"]], tp
            )
            dec_preds.append(max(1e-9, full_raw - pre_only))
            dec_acts.append(item["pure_decode_s"])
        dec_scale = _ratio_ols(dec_preds, dec_acts)
        rollout_cal[str(tp)] = {
            "prefill_scale": round(pre_scale, 6),
            "decode_scale": round(dec_scale, 6),
        }

    cal_records = [r for r in train_dp1["records"] if r["role"] == "calibration"]
    opt_s = statistics.median(r["optimizer_s"] for r in cal_records)

    cfg_dp1 = TrainParallelConfig(tp=train_dp1["tp"], dp=1, b_micro=1)
    anc_preds, anc_acts = [], []
    comp_preds, comp_acts = [], []
    for r in cal_records:
        _, det = raw_cm.evaluate_training(cfg_dp1, r["B_global"], r["lengths"])
        anc_preds.append(det["anchor_s"])
        anc_acts.append(r["recompute_logprobs_s"])
        comp_preds.append(det["compute_s"])
        comp_acts.append(r["fwd_bwd_comm_s"])

    anchor_scale = _ratio_ols(anc_preds, anc_acts)
    compute_scale = _ratio_ols(comp_preds, comp_acts)
    return {
        "rollout": rollout_cal,
        "training": {
            "compute_scale": round(compute_scale, 6),
            "anchor_scale": round(anchor_scale, 6),
            "optimizer_s": round(opt_s, 6),
            "hybrid_gradient_s": round(hybrid_gradient_s, 6),
        },
    }


def _load_e2e_runs(e2e_dirs: list[str]) -> list[dict[str, Any]]:
    runs = []
    for run_dir in e2e_dirs:
        p = Path(run_dir)
        lp_rows = _load_length_profile(str(p / "length-profile.jsonl"))
        # Read phase traces for exact phase durations
        phases_by_step: dict[int, dict[str, float]] = {}
        phase_file = p / "phases" / "phase_trace_rank0.jsonl"
        if phase_file.exists():
            for line in phase_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev.get("event") == "phase_span":
                    phases_by_step.setdefault(int(ev["step"]), {})[ev["phase"]] = float(ev["duration_s"])
        # Detect active hybrids from train.log
        hybrids_by_step: dict[int, int] = {}
        train_log = p / "train.log"
        if train_log.exists():
            for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("QS_HYBRID "):
                    parts = line.split()
                    step_val = int(parts[1].split("=")[1])
                    active_json = line.split("active=", 1)[1]
                    hybrids_by_step[step_val] = len(json.loads(active_json))
        steps = []
        for s in sorted(phases_by_step):
            seqs = [int(r["sequence_length"]) for r in lp_rows if r.get("step") == s and r.get("sequence_length")]
            ph = phases_by_step[s]
            if seqs and "recompute_logprobs" in ph and "grpo_update" in ph:
                steps.append({
                    "step": s,
                    "hybrids": hybrids_by_step.get(s, 0),
                    "lengths": seqs,
                    "recompute_logprobs_s": ph["recompute_logprobs"],
                    "grpo_update_s": ph["grpo_update"],
                    "total_train_s": ph["recompute_logprobs"] + ph["grpo_update"],
                })
        runs.append({"run": p.name, "steps": steps})
    return runs


def cmd_fit_and_validate(args: argparse.Namespace) -> None:
    from RL_Framework.config import HardwareConfig, ModelArchConfig, ProfilingConfig, load_config
    from RL_Framework.infra.cost_model.model import CostModel, RequestInfo, TrainParallelConfig

    hw_h20 = HardwareConfig.from_file(args.h20_hw)
    hw_a100 = HardwareConfig.from_file(args.a100_hw)
    ma = ModelArchConfig.from_file(args.model_arch)
    prof_base = ProfilingConfig()

    rollout_files = {}
    for spec in args.rollout_json:
        data = json.loads(Path(spec).read_text(encoding="utf-8"))
        rollout_files[int(data["tp"])] = data

    train_dp1 = json.loads(Path(args.train_dp1_json).read_text(encoding="utf-8"))
    train_dp2 = (
        json.loads(Path(args.train_dp2_json).read_text(encoding="utf-8"))
        if args.train_dp2_json
        else None
    )
    e2e_runs = _load_e2e_runs(args.e2e_dir or [])

    # Preliminary fit without hybrid_gradient_s so we can infer hybrid_gradient_s from real e2e Hybrid step
    cal_h20 = _fit_calibration(hw_h20, ma, prof_base, rollout_files, train_dp1, hybrid_gradient_s=0.0)

    # Infer real hybrid_gradient_s from any active Hybrid step in e2e runs
    hybrid_grads = []
    temp_cm = CostModel(
        hardware=hw_h20, model_arch=ma,
        profiling=replace(prof_base, calibration=cal_h20),
        max_seq_length=8192, recompute_logprobs=True,
    )
    core_cfg = TrainParallelConfig(tp=4, dp=1, b_micro=1)
    for run in e2e_runs:
        for st in run["steps"]:
            if st["hybrids"] > 0:
                pred_elastic, det = temp_cm.evaluate_elastic_training(
                    core_cfg, len(st["lengths"]), 4, st["hybrids"], st["lengths"]
                )
                inferred = max(0.0, st["total_train_s"] - pred_elastic)
                hybrid_grads.append(inferred)
    hybrid_grad_s = statistics.median(hybrid_grads) if hybrid_grads else 0.0
    cal_h20["training"]["hybrid_gradient_s"] = round(hybrid_grad_s, 6)
    cal_a100 = _fit_calibration(hw_a100, ma, prof_base, rollout_files, train_dp1, hybrid_gradient_s=hybrid_grad_s)

    # Also load old A100 calibration from preset if provided
    old_a100_cal = {}
    if args.old_preset:
        old_cfg = load_config(args.old_preset)
        old_a100_cal = dict(getattr(old_cfg.profiling, "calibration", {}) or {})

    cm_raw_h20 = CostModel(hardware=hw_h20, model_arch=ma, profiling=replace(prof_base, calibration={}), max_seq_length=8192, recompute_logprobs=True)
    cm_cal_h20 = CostModel(hardware=hw_h20, model_arch=ma, profiling=replace(prof_base, calibration=cal_h20), max_seq_length=8192, recompute_logprobs=True)
    cm_old_a100 = CostModel(hardware=hw_a100, model_arch=ma, profiling=replace(prof_base, calibration=old_a100_cal), max_seq_length=8192, recompute_logprobs=True)

    # 1. Validate Rollout Held-out Cohorts
    rollout_val_rows = []
    pairs_legacy, pairs_uncal, pairs_old_cal, pairs_h20_cal = [], [], [], []
    for tp, data in sorted(rollout_files.items()):
        for item in data["cohorts"]:
            reqs = [RequestInfo(p, g) for p, g in item["requests"]]
            actual = item["median_s"]
            pred_legacy = _legacy_steady_state_makespan(cm_raw_h20.rollout_model, reqs, tp)
            pred_uncal = cm_raw_h20.rollout_model.compute_instance_makespan(reqs, tp)
            pred_old = cm_old_a100.rollout_model.compute_instance_makespan(reqs, tp) if old_a100_cal else float("nan")
            pred_new = cm_cal_h20.rollout_model.compute_instance_makespan(reqs, tp)
            pairs_legacy.append((pred_legacy, actual))
            pairs_uncal.append((pred_uncal, actual))
            if old_a100_cal:
                pairs_old_cal.append((pred_old, actual))
            pairs_h20_cal.append((pred_new, actual))
            rollout_val_rows.append({
                "tp": tp, "name": item["name"], "n_reqs": len(reqs),
                "actual_s": actual,
                "legacy_main_s": pred_legacy,
                "uncalibrated_h20_s": pred_uncal,
                "old_a100_cal_s": pred_old,
                "calibrated_h20_s": pred_new,
                "ape_pct": abs(pred_new - actual) / actual * 100.0,
            })

    # 2. Validate Training Held-out (DP=1 and DP=2, contrasting with vs without anchor DP fix)
    train_val_rows = []
    tr_uncal_pairs, tr_old_cal_pairs, tr_nofix_pairs, tr_fixed_pairs = [], [], [], []
    dp2_nofix_pairs, dp2_fixed_pairs = [], []
    for dataset in [d for d in (train_dp1, train_dp2) if d is not None]:
        tp, dp = dataset["tp"], dataset["dp"]
        cfg = TrainParallelConfig(tp=tp, dp=dp, b_micro=1)
        for r in dataset["records"]:
            if r["role"] != "validation":
                continue
            actual = r["total_train_s"]
            pred_uncal, _ = cm_raw_h20.evaluate_training(cfg, r["B_global"], r["lengths"])
            pred_old_cal, _ = cm_old_a100.evaluate_training(cfg, r["B_global"], r["lengths"]) if old_a100_cal else (float("nan"), {})
            pred_nofix, det_nofix = _evaluate_training_unsharded_anchor(cm_cal_h20, cfg, r["B_global"], r["lengths"])
            pred_fixed, det_fixed = cm_cal_h20.evaluate_training(cfg, r["B_global"], r["lengths"])

            tr_uncal_pairs.append((pred_uncal, actual))
            if old_a100_cal:
                tr_old_cal_pairs.append((pred_old_cal, actual))
            tr_nofix_pairs.append((pred_nofix, actual))
            tr_fixed_pairs.append((pred_fixed, actual))
            if dp > 1:
                dp2_nofix_pairs.append((pred_nofix, actual))
                dp2_fixed_pairs.append((pred_fixed, actual))

            train_val_rows.append({
                "tp": tp, "dp": dp, "name": r["name"], "B": r["B_global"],
                "actual_anchor_s": r["recompute_logprobs_s"],
                "actual_total_s": actual,
                "uncalibrated_h20_s": pred_uncal,
                "old_a100_cal_s": pred_old_cal,
                "nofix_h20_s": pred_nofix,
                "nofix_anchor_s": det_nofix["anchor_s"],
                "fixed_h20_s": pred_fixed,
                "fixed_anchor_s": det_fixed["anchor_s"],
                "fixed_ape_pct": abs(pred_fixed - actual) / actual * 100.0,
                "nofix_ape_pct": abs(pred_nofix - actual) / actual * 100.0,
            })

    # 3. Validate Real E2E Runs (e2e-run-02 & e2e-run-03)
    e2e_val_rows = []
    e2e_nofix_pairs, e2e_fixed_pairs = [], []
    for run in e2e_runs:
        for st in run["steps"]:
            B = len(st["lengths"])
            actual = st["total_train_s"]
            if st["hybrids"] == 0:
                pred_fixed, det_fixed = cm_cal_h20.evaluate_training(core_cfg, B, st["lengths"])
                pred_nofix, det_nofix = _evaluate_training_unsharded_anchor(cm_cal_h20, core_cfg, B, st["lengths"])
            else:
                # Core + hybrids
                pred_fixed, det_fixed = cm_cal_h20.evaluate_elastic_training(core_cfg, B, 4, st["hybrids"], st["lengths"])
                # Compare with what unsharded anchor on total DP=1+hybrids would have overpredicted
                dp_equiv = TrainParallelConfig(tp=4, dp=1 + st["hybrids"], b_micro=1)
                pred_nofix, det_nofix = _evaluate_training_unsharded_anchor(cm_cal_h20, dp_equiv, B, st["lengths"])
                pred_nofix += float(cal_h20["training"]["hybrid_gradient_s"])
            e2e_fixed_pairs.append((pred_fixed, actual))
            e2e_nofix_pairs.append((pred_nofix, actual))
            e2e_val_rows.append({
                "run": run["run"], "step": st["step"], "hybrids": st["hybrids"],
                "actual_anchor_s": st["recompute_logprobs_s"],
                "actual_grpo_s": st["grpo_update_s"],
                "actual_total_s": actual,
                "pred_anchor_s": det_fixed["anchor_s"],
                "pred_total_s": pred_fixed,
                "pred_nofix_total_s": pred_nofix,
                "fixed_ape_pct": abs(pred_fixed - actual) / actual * 100.0,
            })

    report = {
        "calibrated_h20_3e": cal_h20,
        "calibrated_a100_equivalent": cal_a100,
        "rollout_summary": {
            "legacy_main": _error_metrics(pairs_legacy),
            "uncalibrated_h20": _error_metrics(pairs_uncal),
            "old_a100_cal": _error_metrics(pairs_old_cal) if pairs_old_cal else None,
            "calibrated_h20": _error_metrics(pairs_h20_cal),
        },
        "training_summary": {
            "all_uncalibrated_h20": _error_metrics(tr_uncal_pairs),
            "all_old_a100_cal": _error_metrics(tr_old_cal_pairs) if tr_old_cal_pairs else None,
            "all_h20_nofix": _error_metrics(tr_nofix_pairs),
            "all_h20_fixed": _error_metrics(tr_fixed_pairs),
            "dp2_h20_nofix": _error_metrics(dp2_nofix_pairs),
            "dp2_h20_fixed": _error_metrics(dp2_fixed_pairs),
        },
        "e2e_summary": {
            "h20_nofix": _error_metrics(e2e_nofix_pairs),
            "h20_fixed": _error_metrics(e2e_fixed_pairs),
        },
        "rollout_validation_rows": rollout_val_rows,
        "training_validation_rows": train_val_rows,
        "e2e_validation_rows": e2e_val_rows,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== Fitted profiling.calibration (H20-3e 141G) ===")
    print(json.dumps(cal_h20, indent=2))
    print("\n=== Rollout Held-out Summary ===")
    print(json.dumps(report["rollout_summary"], indent=2))
    print("\n=== Training Held-out Summary ===")
    print(json.dumps(report["training_summary"], indent=2))
    print("\n=== E2E Run-02/03 Summary ===")
    print(json.dumps(report["e2e_summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ro = sub.add_parser("measure-rollout")
    p_ro.add_argument("--model", required=True)
    p_ro.add_argument("--tp", type=int, required=True)
    p_ro.add_argument("--output", required=True)
    p_ro.add_argument("--length-profile", default="")
    p_ro.add_argument("--max-model-len", type=int, default=8192)
    p_ro.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p_ro.add_argument("--max-num-seqs", type=int, default=48)

    p_tr = sub.add_parser("measure-training")
    p_tr.add_argument("--model", required=True)
    p_tr.add_argument("--tp", type=int, default=4)
    p_tr.add_argument("--dp", type=int, default=1)
    p_tr.add_argument("--output", required=True)
    p_tr.add_argument("--length-profile", default="")
    p_tr.add_argument("--max-seq-length", type=int, default=8192)
    p_tr.add_argument("--micro-batch-size", type=int, default=1)
    p_tr.add_argument("--recompute-micro-batch-size", type=int, default=1)

    p_fit = sub.add_parser("fit-and-validate")
    p_fit.add_argument("--h20-hw", required=True)
    p_fit.add_argument("--a100-hw", required=True)
    p_fit.add_argument("--model-arch", required=True)
    p_fit.add_argument("--rollout-json", action="append", required=True)
    p_fit.add_argument("--train-dp1-json", required=True)
    p_fit.add_argument("--train-dp2-json", default="")
    p_fit.add_argument("--e2e-dir", action="append", default=[])
    p_fit.add_argument("--old-preset", default="")
    p_fit.add_argument("--output", default="")

    args = parser.parse_args()
    if args.cmd == "measure-rollout":
        cmd_measure_rollout(args)
    elif args.cmd == "measure-training":
        cmd_measure_training(args)
    else:
        cmd_fit_and_validate(args)


if __name__ == "__main__":
    main()

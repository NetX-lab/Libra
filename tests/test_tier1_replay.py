"""Unit and CLI tests for Tier 1 (0-GPU) replay and GRP offline benchmark."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repro.tier1_zero_gpu_replay.run_replay import (
    REPO_ROOT,
    SUPPORTED_POLICIES,
    load_trace,
    replay_policy,
    run_grp_offline_benchmark,
)
from RL_Framework.config import HardwareConfig, ModelArchConfig


SAMPLE_TRACE = REPO_ROOT / "repro/tier1_zero_gpu_replay/sample_traces/r2e_agentic_trace.jsonl"
HARDWARE_CFG = REPO_ROOT / "configs/hardware_config/a800_80g.yaml"
MODEL_ARCH_CFG = REPO_ROOT / "configs/model_arch_config/qwen3-14b.yaml"


def test_load_native_and_history_step_traces(tmp_path: Path):
    native = load_trace(SAMPLE_TRACE)
    assert len(native) == 64
    assert {r["archetype"] for r in native} == {"short", "medium", "long"}

    history_file = tmp_path / "history_sample.jsonl"
    step_record = {
        "step": 3,
        "sequences": [
            {
                "index": 0,
                "prompt_id": "hist_p0",
                "input_len": 1200,
                "output_len": 15000,
                "tool_returns": [
                    {
                        "tool_type": "bash",
                        "status": "failure",
                        "payload_tokens": 6000,
                        "token_position": 900,
                    }
                ],
            },
            {
                "index": 1,
                "prompt_id": "hist_p1",
                "input_len": 1100,
                "output_len": 1400,
            },
        ],
    }
    history_file.write_text(json.dumps(step_record) + "\n", encoding="utf-8")

    parsed = load_trace(history_file)
    assert len(parsed) == 2
    assert parsed[0]["prompt_id"] == "hist_p0"
    assert parsed[0]["archetype"] == "long"
    assert parsed[0]["tool_returns"][0]["token_position"] == 900
    assert parsed[1]["archetype"] == "short"
    assert parsed[1]["tool_returns"] == []


def test_replay_all_policies_and_strict_ranking():
    traces = load_trace(SAMPLE_TRACE)
    hw = HardwareConfig.from_file(str(HARDWARE_CFG))
    arch = ModelArchConfig.from_file(str(MODEL_ARCH_CFG))
    tp_layout = [1, 1, 2, 4]

    results = {
        policy: replay_policy(policy, traces, tp_layout, hw, arch)
        for policy in SUPPORTED_POLICIES
    }

    for policy_name, res in results.items():
        assert res.eval_trajectories == 40, policy_name
        assert res.warmup_trajectories == 24, policy_name
        assert res.analytic_makespan_s > 0.0, policy_name
        assert res.oom_count == 0, policy_name

    rr = results["round_robin"]
    lc = results["least_connections"]
    la = results["length_aware"]
    la_mlfq = results["la_mlfq"]
    cmlfq = results["cmlfq"]
    cmlfq_cost = results["cmlfq_cost"]

    assert (
        cmlfq_cost.analytic_makespan_s
        < cmlfq.analytic_makespan_s
        < la_mlfq.analytic_makespan_s
        < la.analytic_makespan_s
        < lc.analytic_makespan_s
        < rr.analytic_makespan_s
    )
    assert cmlfq.long_tail_high_tp_rate == 1.0
    assert cmlfq_cost.long_tail_high_tp_rate == 1.0
    assert rr.long_tail_high_tp_rate < 1.0
    assert cmlfq.tp_distribution[2] > 0
    assert la_mlfq.tp_distribution[2] == 0
    assert cmlfq.migrations > 0
    assert cmlfq_cost.migrations > 0


def test_kv_backends_and_online_only_mode():
    traces = load_trace(SAMPLE_TRACE)
    hw = HardwareConfig.from_file(str(HARDWARE_CFG))
    arch = ModelArchConfig.from_file(str(MODEL_ARCH_CFG))
    tp_layout = [1, 1, 2, 4]

    recompute = replay_policy("cmlfq_cost", traces, tp_layout, hw, arch, kv_backend="recompute")
    cpu_offload = replay_policy(
        "cmlfq_cost", traces, tp_layout, hw, arch, kv_backend="cpu_offload"
    )
    nixl = replay_policy("cmlfq_cost", traces, tp_layout, hw, arch, kv_backend="nixl")
    assert nixl.migration_overhead_s <= cpu_offload.migration_overhead_s <= recompute.migration_overhead_s

    online_rr = replay_policy(
        "round_robin", traces, tp_layout, hw, arch, batch_size=16, online_only=True
    )
    online_cmlfq = replay_policy(
        "cmlfq", traces, tp_layout, hw, arch, batch_size=16, online_only=True
    )
    assert online_cmlfq.warmup_trajectories == 0
    assert online_cmlfq.eval_trajectories == 64
    assert online_cmlfq.migrations > 0
    assert online_cmlfq.analytic_makespan_s < online_rr.analytic_makespan_s


def test_grp_offline_benchmark():
    traces = load_trace(SAMPLE_TRACE)
    hw = HardwareConfig.from_file(str(HARDWARE_CFG))
    arch = ModelArchConfig.from_file(str(MODEL_ARCH_CFG))

    grp = run_grp_offline_benchmark(traces, hw, arch, total_gpus=32, batch_size=64)
    assert grp["current_plan"] is not None
    assert grp["candidate_plan"] is not None
    assert grp["candidate_plan"]["t_global_s"] <= grp["current_plan"]["t_global_s"]
    assert grp["expected_gain_ratio"] > 0.0
    assert grp["speedup"] >= 1.0


@pytest.mark.parametrize("kv_backend", ["recompute"])
def test_tier1_cli_invocation_emits_json(tmp_path: Path, kv_backend: str):
    out_json = tmp_path / "tier1_out.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "repro/tier1_zero_gpu_replay/run_replay.py"),
            "--kv-backend",
            kv_backend,
            "--output",
            str(out_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "analytic_cost_model_replay" in proc.stdout

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["evaluation_mode"] == "analytic_cost_model_replay"
    assert [r["policy"] for r in payload["scheduling_results"]] == list(SUPPORTED_POLICIES)
    assert payload["grp_offline"]["candidate_plan"] is not None

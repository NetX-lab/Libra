#!/usr/bin/env python3
"""Slurm-free Tier 2 launcher for 4-GPU heterogeneous and 1-GPU Colab closed-loop RL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
import yaml


def _build_pythonpath(repo_root: Path, existing: str | None) -> str:
    """Build an order-preserving, deduplicated PYTHONPATH for subprocesses."""
    entries: list[str] = []
    if repo_root.name == "RL_Framework":
        entries.append(str(repo_root.parent))
    entries.append(str(repo_root))
    if existing:
        for item in existing.split(os.pathsep):
            if item:
                entries.append(item)
    deduped = list(dict.fromkeys(entries))
    return os.pathsep.join(deduped)


def _ensure_rl_framework_importable() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
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
from RL_Framework.scripts.summarize_scheduler_rl import summarize  # noqa: E402


DEFAULT_DATASET = (
    REPO_ROOT / "repro/tier2_four_gpu_closed_loop/sample_data/dapo_math_mini.jsonl"
)

FOUR_GPU_POLICIES = ("round_robin", "least_connections", "cmlfq", "cmlfq_cost")
COLAB_POLICIES = ("round_robin", "least_connections")

COLAB_TIER_PRESETS: dict[str, dict[str, Any]] = {
    "t4": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_arch_yaml": "qwen2.5-3b.yaml",
        "vllm_dtype": "half",
        "fsdp_dtype": "float32",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 1024,
        "max_new_tokens": 256,
        "batch_size": 4,
        "max_num_seqs": 8,
        "sync_interval": 2,
    },
    "l4": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_arch_yaml": "qwen2.5-3b.yaml",
        "vllm_dtype": "bfloat16",
        "fsdp_dtype": "bfloat16",
        "gpu_memory_utilization": 0.35,
        "max_model_len": 1536,
        "max_new_tokens": 384,
        "batch_size": 4,
        "max_num_seqs": 8,
        "sync_interval": 2,
    },
    "a100-40g": {
        "model": "Qwen/Qwen3-1.7B",
        "model_arch_yaml": "qwen2.5-3b.yaml",
        "vllm_dtype": "bfloat16",
        "fsdp_dtype": "bfloat16",
        "gpu_memory_utilization": 0.38,
        "max_model_len": 2048,
        "max_new_tokens": 512,
        "batch_size": 8,
        "max_num_seqs": 16,
        "sync_interval": 2,
    },
    "a100-80g": {
        "model": "Qwen/Qwen3-4B",
        "model_arch_yaml": "qwen3-4b.yaml",
        "vllm_dtype": "bfloat16",
        "fsdp_dtype": "bfloat16",
        "gpu_memory_utilization": 0.38,
        "max_model_len": 2048,
        "max_new_tokens": 512,
        "batch_size": 8,
        "max_num_seqs": 16,
        "sync_interval": 2,
    },
}


def detect_colab_tier() -> str:
    """Detect GPU tier from CUDA compute capability and VRAM (defaults to t4 on CPU/T4)."""
    if not torch.cuda.is_available():
        return "t4"
    props = torch.cuda.get_device_properties(0)
    total_mem_gib = props.total_memory / (1024**3)
    if props.major < 8:
        return "t4"
    if total_mem_gib >= 70.0:
        return "a100-80g"
    if total_mem_gib >= 36.0:
        return "a100-40g"
    if total_mem_gib >= 20.0:
        return "l4"
    return "t4"


def parse_gpu_list(raw: str | None, *, mode: str) -> list[int]:
    """Parse GPU indices from --gpus or CUDA_VISIBLE_DEVICES."""
    default_str = "0,1,2,3" if mode == "4gpu" else "0"
    source = (
        raw
        if raw is not None
        else (os.environ.get("CUDA_VISIBLE_DEVICES") or default_str)
    )
    gpus = [int(item.strip()) for item in source.split(",") if item.strip()]
    min_gpus = 4 if mode == "4gpu" else 1
    if len(gpus) < min_gpus:
        raise ValueError(
            f"Mode {mode!r} requires at least {min_gpus} GPU(s), got {gpus}"
        )
    return gpus


def resolve_policies(policy: str, mode: str) -> list[str]:
    """Resolve CLI --policy string into an ordered list of policies to run."""
    if mode == "4gpu":
        if policy == "auto":
            return ["cmlfq_cost"]
        if policy == "all":
            return ["round_robin", "least_connections", "cmlfq_cost"]
        if policy not in FOUR_GPU_POLICIES:
            raise ValueError(
                f"Unsupported policy {policy!r} for 4gpu; choose from {FOUR_GPU_POLICIES} or 'all'"
            )
        return [policy]

    if policy == "auto":
        return ["least_connections"]
    if policy == "all":
        return ["round_robin", "least_connections"]
    if policy not in COLAB_POLICIES:
        raise ValueError(
            f"Unsupported policy {policy!r} for colab-1gpu; choose from {COLAB_POLICIES} or 'all'"
        )
    return [policy]


def _scheduling_block(
    policy: str,
    run_dir: Path,
    kv_backend: str,
    mode: str,
    max_model_len: int,
) -> dict[str, Any]:
    if mode == "colab-1gpu":
        return {
            "scheduler_type": "load_balance",
            "load_balance_strategy": (
                "round_robin" if policy == "round_robin" else "least_connections"
            ),
            "max_queue_length": 8,
            "enable_fallback": True,
        }

    is_causal = policy in ("cmlfq", "cmlfq_cost")
    return {
        "scheduler_type": policy if is_causal else "load_balance",
        "load_balance_strategy": (
            "round_robin" if policy == "round_robin" else "least_connections"
        ),
        "max_queue_length": 16,
        "enable_fallback": True,
        "cmlfq_kv_backend": kv_backend if is_causal else "recompute",
        "cmlfq_tree_path": str(run_dir / "tree.json"),
        "cmlfq_rebuild_interval": 8,
        "cmlfq_kv_transfer_tp_pairs": (
            [[1, 2], [2, 1]] if (is_causal and kv_backend == "cpu_offload") else []
        ),
        "cmlfq_buckets": {
            "short": {"tp_degrees": [1], "max_tokens": 512},
            "long": {"tp_degrees": [2], "max_tokens": max(2048, max_model_len)},
        },
    }


def _vllm_supervisor_command(
    python_bin: str,
    *,
    instance_id: str,
    control_dir: Path,
    health_url: str,
    initial_model: str,
    port: int,
    tp_size: int,
    vllm_dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    seed: int,
    kv_storage_dir: Path | None,
) -> list[str]:
    vllm_cmd = [
        python_bin,
        str(REPO_ROOT / "scripts/vllm_hot_reload_api_server.py"),
        "--model",
        "__MODEL_PATH__",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
        "--dtype",
        vllm_dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        "--enforce-eager",
        "--disable-log-requests",
        "--no-enable-prefix-caching",
        "--seed",
        str(seed),
        "--worker-extension-cls",
        "RL_Framework.vllm_hot_reload.InplaceReloadWorkerExtension",
    ]
    if kv_storage_dir is not None:
        vllm_cmd.extend(
            [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "LibraCPUOffloadConnector",
                        "kv_role": "kv_both",
                        "kv_connector_module_path": "RL_Framework.infra.scheduling.vllm_cpu_offload",
                        "kv_connector_extra_config": {
                            "storage_path": str(kv_storage_dir)
                        },
                    }
                ),
            ]
        )
    return [
        python_bin,
        str(REPO_ROOT / "scripts/restartable_vllm_server.py"),
        "--instance-id",
        instance_id,
        "--control-dir",
        str(control_dir),
        "--health-url",
        health_url,
        "--initial-model",
        initial_model,
        "--reload-method",
        "inplace",
        "--",
        *vllm_cmd,
    ]


def build_run_plan(
    *,
    mode: str,
    policy: str,
    gpus: list[int],
    output_dir: Path,
    dataset_path: Path,
    model_path: str | None = None,
    colab_tier: str = "auto",
    steps: int | None = None,
    batch_size: int | None = None,
    n_samples: int = 2,
    sync_interval: int | None = None,
    seed: int = 42,
    base_port: int = 18920,
    kv_backend: str = "recompute",
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_new_tokens: int | None = None,
    max_turns: int = 2,
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    """Build a complete Slurm-free launch plan for one trial run."""
    run_dir = (output_dir / f"{policy}_{seed}").resolve()
    control_dir = run_dir / "control"
    resolved_tier: str | None = None

    if mode == "colab-1gpu":
        resolved_tier = detect_colab_tier() if colab_tier == "auto" else colab_tier
        preset = COLAB_TIER_PRESETS[resolved_tier]
        resolved_model = model_path or preset["model"]
        arch_yaml = preset["model_arch_yaml"]
        vllm_dtype = preset["vllm_dtype"]
        fsdp_dtype = preset["fsdp_dtype"]
        gpu_util = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else preset["gpu_memory_utilization"]
        )
        resolved_max_len = max_model_len or preset["max_model_len"]
        resolved_max_new = max_new_tokens or preset["max_new_tokens"]
        resolved_batch = batch_size or preset["batch_size"]
        resolved_steps = steps or 4
        resolved_sync = sync_interval or preset["sync_interval"]
        max_num_seqs = preset["max_num_seqs"]
        max_concurrent = resolved_batch * n_samples
        train_gpu = gpus[0]
        rollout_layout = [("tp1", 1, [gpus[0]], base_port)]
        rollout_gpus = 1
        kv_storage_dir = None
    else:
        resolved_model = model_path or "Qwen/Qwen3-4B"
        arch_yaml = "qwen3-4b.yaml"
        vllm_dtype = "bfloat16"
        fsdp_dtype = "bfloat16"
        gpu_util = gpu_memory_utilization if gpu_memory_utilization is not None else 0.65
        resolved_max_len = max_model_len or 2048
        resolved_max_new = max_new_tokens or 512
        resolved_batch = batch_size or 8
        resolved_steps = steps or 8
        resolved_sync = sync_interval or 4
        max_num_seqs = 16
        max_concurrent = max(16, resolved_batch * n_samples)
        train_gpu = gpus[0]
        rollout_layout = [
            ("tp1", 1, [gpus[1]], base_port),
            ("tp2", 2, [gpus[2], gpus[3]], base_port + 1),
        ]
        rollout_gpus = 3
        kv_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else run_dir
        kv_storage_dir = (
            kv_base / f"libra-tier2-{os.getpid()}-{policy}-s{seed}"
            if (kv_backend == "cpu_offload" and policy in ("cmlfq", "cmlfq_cost"))
            else None
        )

    instances = [
        {
            "instance_id": inst_id,
            "tp": tp,
            "gpus": inst_gpus,
            "host": "127.0.0.1",
            "port": port,
        }
        for inst_id, tp, inst_gpus, port in rollout_layout
    ]

    config_dict: dict[str, Any] = {
        "hardware_config": str(
            (REPO_ROOT / "configs/hardware_config/a800_80g.yaml").resolve()
        ),
        "model_arch_config": str(
            (REPO_ROOT / "configs/model_arch_config" / arch_yaml).resolve()
        ),
        "model_path": resolved_model,
        "tokenizer_path": resolved_model,
        "train_backend": "fsdp",
        "train_gpus": 1,
        "rollout_gpus": rollout_gpus,
        "tp_size": 1,
        "train_tp_size": 1,
        "train_pp_size": 1,
        "train_dp_size": 1,
        "batch_size": resolved_batch,
        "micro_batch_size": 1,
        "total_steps": resolved_steps,
        "n_samples": n_samples,
        "max_concurrent_rollouts": max_concurrent,
        "max_head_offpolicyness": 4,
        "queue_size": 32,
        "max_new_tokens": resolved_max_new,
        "max_seq_length": resolved_max_len,
        "temperature": 0.8,
        "top_p": 0.95,
        "learning_rate": 1.0e-6,
        "ppo_epochs": 1,
        "seed": seed,
        "recompute_logprobs": True,
        "sync_interval": resolved_sync,
        "rollout_sync_drain_lead_steps": 1,
        "weight_sync_mode": "disk",
        "sync_path": str(run_dir / "weights"),
        "rollout_weight_sync_mode": "restart",
        "rollout_weight_reload_method": "inplace",
        "rollout_weight_reload_strategy": "parallel",
        "rollout_weight_sync_control_dir": str(control_dir),
        "require_rollout_weight_sync": True,
        "eval_interval": 0,
        "save_interval": 0,
        "log_dir": str(run_dir),
        "enable_history_collection": True,
        "history_output_dir": str(run_dir / "history"),
        "history_save_raw_lengths": True,
        "history_flush_interval": 1,
        "history_experiment_name": f"tier2_{mode}_{policy}",
        "heterogeneous_rollout": {
            "enabled": True,
            "total_gpus": rollout_gpus,
            "vllm_host": "127.0.0.1",
            "vllm_base_port": base_port,
            "max_model_len": resolved_max_len,
            "gpu_memory_utilization": gpu_util,
            "instances": instances,
            "scheduling": _scheduling_block(
                policy, run_dir, kv_backend, mode, resolved_max_len
            ),
        },
    }

    env_overrides: dict[str, str] = {
        "PYTHONPATH": _build_pythonpath(REPO_ROOT, os.environ.get("PYTHONPATH")),
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "FSDP_GRADIENT_CHECKPOINTING": "1",
        "FSDP_DTYPE": fsdp_dtype,
        "FSDP_PROGRESS_DIR": str(run_dir / "progress"),
        "BENCH_RUN_DIR": str(run_dir),
        "DAPO_MATH_PATH": str(dataset_path.resolve()),
        "DAPO_MAX_TURNS": str(max_turns),
        "DAPO_MAX_PROMPT_TOKENS": "384" if resolved_max_len <= 1024 else "512",
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }
    if kv_storage_dir is not None:
        env_overrides["LIBRA_KV_STORAGE_PATH"] = str(kv_storage_dir)
        env_overrides["VLLM_ATTENTION_BACKEND"] = os.environ.get(
            "VLLM_ATTENTION_BACKEND", "FLASH_ATTN"
        )

    config_path = run_dir / "config.yaml"
    rollout_specs = []
    for inst_id, tp, inst_gpus, port in rollout_layout:
        health_url = f"http://127.0.0.1:{port}/health"
        rollout_specs.append(
            {
                "instance_id": inst_id,
                "tp": tp,
                "gpus": inst_gpus,
                "cuda_visible_devices": ",".join(str(g) for g in inst_gpus),
                "port": port,
                "health_url": health_url,
                "log_path": str(run_dir / f"vllm_{inst_id}.log"),
                "command": _vllm_supervisor_command(
                    python_bin,
                    instance_id=inst_id,
                    control_dir=control_dir,
                    health_url=health_url,
                    initial_model=resolved_model,
                    port=port,
                    tp_size=tp,
                    vllm_dtype=vllm_dtype,
                    max_model_len=resolved_max_len,
                    gpu_memory_utilization=gpu_util,
                    max_num_seqs=max_num_seqs,
                    seed=seed,
                    kv_storage_dir=kv_storage_dir,
                ),
            }
        )

    train_spec = {
        "gpus": [train_gpu],
        "cuda_visible_devices": str(train_gpu),
        "log_path": str(run_dir / "train.log"),
        "env_overrides": {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        },
        "command": [
            python_bin,
            str(REPO_ROOT / "scripts/benchmark_scheduler_train.py"),
            "--config",
            str(config_path),
        ],
    }

    return {
        "mode": mode,
        "policy": policy,
        "seed": seed,
        "colab_tier": resolved_tier,
        "model": resolved_model,
        "vllm_dtype": vllm_dtype,
        "fsdp_dtype": fsdp_dtype,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "config": config_dict,
        "env_overrides": env_overrides,
        "rollout_specs": rollout_specs,
        "train_spec": train_spec,
        "kv_storage_dir": str(kv_storage_dir) if kv_storage_dir is not None else None,
    }


def materialize_run_plan(plan: dict[str, Any]) -> None:
    """Write config.yaml + launch_plan.json and validate with AsyncRLConfig.from_yaml."""
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["config"]["rollout_weight_sync_control_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    if plan["kv_storage_dir"]:
        Path(plan["kv_storage_dir"]).mkdir(parents=True, exist_ok=True)

    config_path = Path(plan["config_path"])
    config_path.write_text(
        yaml.safe_dump(plan["config"], sort_keys=False), encoding="utf-8"
    )
    AsyncRLConfig.from_yaml(str(config_path))

    plan_file = run_dir / "launch_plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "mode": plan["mode"],
                "policy": plan["policy"],
                "seed": plan["seed"],
                "colab_tier": plan["colab_tier"],
                "model": plan["model"],
                "vllm_dtype": plan["vllm_dtype"],
                "fsdp_dtype": plan["fsdp_dtype"],
                "config_path": plan["config_path"],
                "env_overrides": plan["env_overrides"],
                "rollout_specs": plan["rollout_specs"],
                "train_spec": plan["train_spec"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _stop_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        except ProcessLookupError:
            pass


def _wait_for_health(
    processes: list[subprocess.Popen[Any]],
    health_urls: list[str],
    timeout_s: float = 600.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    for url in health_urls:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Rollout server exited early with code {proc.returncode}; check vLLM logs."
                    )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Rollout server health check timed out: {url}")
            time.sleep(2)


def execute_run_plan(plan: dict[str, Any], timeout_s: float = 2400.0) -> None:
    """Launch supervised vLLM rollout servers and FSDP trainer, cleaning up on exit."""
    materialize_run_plan(plan)
    env = dict(os.environ, **plan["env_overrides"])
    processes: list[subprocess.Popen[Any]] = []
    log_handles = []

    try:
        health_urls = []
        for spec in plan["rollout_specs"]:
            log_file = Path(spec["log_path"]).open("w", encoding="utf-8")
            log_handles.append(log_file)
            proc = subprocess.Popen(
                spec["command"],
                env=dict(env, CUDA_VISIBLE_DEVICES=spec["cuda_visible_devices"]),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proc)
            health_urls.append(spec["health_url"])

        print(f"[Tier 2] Waiting for rollout health checks: {health_urls}")
        _wait_for_health(processes, health_urls, timeout_s=600.0)
        print("[Tier 2] All rollout servers ready. Starting FSDP trainer...")

        train_spec = plan["train_spec"]
        train_log = Path(train_spec["log_path"]).open("w", encoding="utf-8")
        log_handles.append(train_log)
        train_env = dict(
            env,
            **train_spec.get("env_overrides", {}),
            CUDA_VISIBLE_DEVICES=train_spec["cuda_visible_devices"],
        )
        for key in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "LOCAL_WORLD_SIZE",
        ):
            train_env.pop(key, None)
        train_proc = subprocess.Popen(
            train_spec["command"],
            env=train_env,
            stdout=train_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(train_proc)

        deadline = time.monotonic() + timeout_s
        while train_proc.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Training exceeded {timeout_s}s timeout")
            time.sleep(5)

        if train_proc.returncode != 0:
            raise RuntimeError(
                f"Trainer exited with code {train_proc.returncode}; inspect {train_spec['log_path']}"
            )
    finally:
        for proc in processes:
            _stop_process(proc)
        for handle in log_handles:
            handle.close()
        if plan["kv_storage_dir"]:
            shutil.rmtree(plan["kv_storage_dir"], ignore_errors=True)


def _print_dry_run_summary(plans: list[dict[str, Any]]) -> None:
    print("# Libra Tier 2 — Dry-Run Launch Plan (Slurm-Free)")
    for plan in plans:
        print(
            f"\n## Policy: `{plan['policy']}` | Mode: `{plan['mode']}`"
            + (f" (`{plan['colab_tier']}`)" if plan["colab_tier"] else "")
        )
        print(f"- Model: `{plan['model']}` (`vllm={plan['vllm_dtype']}`, `fsdp={plan['fsdp_dtype']}`)")
        print(f"- Config: `{plan['config_path']}` (validated via `AsyncRLConfig.from_yaml`)")
        for spec in plan["rollout_specs"]:
            print(
                f"- Rollout `{spec['instance_id']}` (GPU={spec['cuda_visible_devices']}, TP={spec['tp']}): "
                f"`{' '.join(spec['command'])}`"
            )
        train_spec = plan["train_spec"]
        print(
            f"- Trainer (GPU={train_spec['cuda_visible_devices']}): "
            f"`{' '.join(train_spec['command'])}`"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tier 2 Slurm-free launcher for 4-GPU closed loop and 1-GPU Google Colab"
    )
    parser.add_argument(
        "--mode",
        choices=("4gpu", "colab-1gpu"),
        default="4gpu",
        help="4gpu: 1 Train + 3 heterogeneous Rollout; colab-1gpu: single co-located GPU",
    )
    parser.add_argument(
        "--policy",
        default="auto",
        help="Scheduling policy (4gpu: round_robin/least_connections/cmlfq/cmlfq_cost/all; colab-1gpu: least_connections/round_robin/all)",
    )
    parser.add_argument(
        "--colab-tier",
        choices=("auto", *COLAB_TIER_PRESETS.keys()),
        default="auto",
        help="GPU preset tier when --mode colab-1gpu (default: auto-detect VRAM/SM)",
    )
    parser.add_argument("--model", default=None, help="Override HuggingFace/local model path")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="DAPO-Math JSONL or Parquet dataset path",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU IDs (defaults to CUDA_VISIBLE_DEVICES or 0,1,2,3 / 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs/tier2_closed_loop",
        help="Directory for run configs, checkpoints, history, and summary.json",
    )
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Prompts per RL step")
    parser.add_argument("--n-samples", type=int, default=2, help="Rollouts per prompt")
    parser.add_argument("--sync-interval", type=int, default=None, help="Weight reload interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base-port", type=int, default=18920, help="Starting vLLM port")
    parser.add_argument(
        "--kv-backend",
        choices=("recompute", "cpu_offload"),
        default="recompute",
        help="KV movement backend for cmlfq/cmlfq_cost (default: recompute)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=2, help="Max rollout turns per episode")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize and validate configs/commands without launching GPUs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    gpus = parse_gpu_list(args.gpus, mode=args.mode)
    policies = resolve_policies(args.policy, args.mode)

    plans = [
        build_run_plan(
            mode=args.mode,
            policy=pol,
            gpus=gpus,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            model_path=args.model,
            colab_tier=args.colab_tier,
            steps=args.steps,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            sync_interval=args.sync_interval,
            seed=args.seed,
            base_port=args.base_port,
            kv_backend=args.kv_backend,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            max_turns=args.max_turns,
        )
        for pol in policies
    ]

    if args.dry_run:
        for plan in plans:
            materialize_run_plan(plan)
        _print_dry_run_summary(plans)
        return 0

    for plan in plans:
        print(f"\n[Tier 2] Starting run: policy={plan['policy']} mode={plan['mode']} dir={plan['run_dir']}")
        execute_run_plan(plan)

    try:
        summary = summarize(args.output_dir.resolve())
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print("\n=== Tier 2 Summary ===")
        print(json.dumps(summary, indent=2))
    except ValueError as exc:
        print(f"[Tier 2] Note: summary skipped ({exc})")

    history_glob = next(args.output_dir.glob("*/history/*.jsonl"), None)
    if history_glob is not None:
        print(
            "\nReplay collected rollout trace on CPU (Tier 1):\n"
            f"  python repro/tier1_zero_gpu_replay/run_replay.py --trace {history_glob} --tp-layout 1,2"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch a four-update Qwen3-14B R2E verification on eight existing QS Pods."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def configure(preset, model, run_dir, hosts):
    from RL_Framework.config import HeterogeneousInstanceConfig, load_config
    from RL_Framework.scripts.run_r2e_qwen3_14b_16gpu import bind_placement

    if len(hosts) != 8 or len({ipaddress.IPv4Address(host) for host in hosts}) != 8:
        raise ValueError("Expected eight distinct QS Pod IPv4 addresses")
    config = load_config(str(preset))
    bind_placement(config, hosts[:4], model, run_dir)
    config.hardware.gpus_per_node = config.gpus_per_node_override = 8
    config.rollout_gpus_per_node = 8
    config.rollout_gpus, config.n_total_gpus = 60, 64
    config.total_steps, config.batch_size = 4, 24
    config.max_concurrent_rollouts = 48
    config.max_new_tokens, config.r2e_max_turns = 512, 2
    config.wandb_run_name = "r2e-qwen3-14b-qs-h20"
    rollout = config.heterogeneous_rollout
    rollout.total_gpus, rollout.available_gpus = 60, list(range(8))
    rollout.scheduling.scheduler_type = "cmlfq_cost_kv_dynamic"
    slots = [(hosts[0], list(range(4, 8)))]
    for index, host in enumerate(hosts[1:]):
        width = 2 if index == 0 else 4
        slots.extend((host, list(range(first, first + width))) for first in range(0, 8, width))
    rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f"rollout-{i}", host=host, tp=len(gpus), gpus=gpus,
                                    port=rollout.vllm_base_port + i)
        for i, (host, gpus) in enumerate(slots)
    ]
    config.vllm_num_instances = len(rollout.instances)
    config.vllm_endpoints = ",".join(f"{item.host}:{item.port}" for item in rollout.instances)
    planner = config.global_resource_planner
    planner.startup_profile_enabled = False
    planner.runtime_async_planning = False
    planner.elastic_hybrid_join_timeout_s = 720
    return config


def verification_loan(config):
    return next(item.instance_id for item in config.heterogeneous_rollout.instances
                if item.host != config.master_addr and item.tp == 4)


def train(config_path, tasks):
    from unittest.mock import patch
    import torch.distributed as dist
    from RL_Framework.config import load_config
    from RL_Framework.infra.cost_model.global_resource_planner import ElasticHybridSignal, PlannerDecision
    from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
    from examples.r2e_gym_async_rl import main as run_training

    class VerificationTrainer(AsyncRLTrainer):
        def setup(self, workflow):
            super().setup(workflow)
            self.control = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
            self._sync_weights_nccl(0)

        def _run_global_resource_planner(self, step, batch, stats):
            desired = int(step < 2)
            executor = self.runtime_elastic_executor
            state = executor.hybrid_runtime_state()
            if step == 1:
                # Cold model/optimizer loading may outlast two short Core updates.
                # Wait for the worker receipt before this boundary activates it.
                deadline = time.monotonic() + 660
                print("QS_HYBRID_AWAITING_ACTIVATION", flush=True)
                while not (state["activation_ready"] or state["active_hybrid_replicas"]):
                    if not state["pending_hybrid_replica_joins"]:
                        raise RuntimeError("QS Hybrid join stopped before acknowledging its snapshot")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("QS Hybrid worker did not acknowledge its initial snapshot")
                    time.sleep(2)
                    state = executor.hybrid_runtime_state()
            current = int(state["active_hybrid_replicas"])
            loan = verification_loan(self.config)
            signal = ElasticHybridSignal(
                step=step, desired_workers=desired, current_workers=current, max_workers=1,
                action="join" if desired and not current else "hold" if desired else "release",
                reason="qs_verification", train_rollout_ratio=0, rollout_pressure=0, ttl_steps=3,
                active_workers=current, replica_size_gpus=4, loan_instance_ids=[loan] if desired else [],
                current_loan_instance_ids=state.get("loan_instance_ids", []),
            )
            planner = self.global_resource_planner
            plan = planner._build_current_plan(self.config, planner._history)
            decision = PlannerDecision(step=step, should_reconfigure=True, reason=signal.reason,
                                       current_plan=plan, candidate_plan=plan,
                                       resource_should_reconfigure=False, elastic_hybrid_signal=signal)
            # Fix the planner decision for this test while exercising normal execution and rank coordination.
            with patch.object(self.global_resource_planner, "recheck_elastic_signal", return_value=signal):
                self._apply_global_resource_planner_decision(step, decision, stats)

        def _prepare_elastic_training_step(self, step):
            active = super()._prepare_elastic_training_step(step)
            if self.is_main_process:
                print(f"QS_HYBRID step={step} active={json.dumps(active)}", flush=True)
            return active

        def _sync_weights_nccl(self, *args, **kwargs):
            receipt = super()._sync_weights_nccl(*args, **kwargs)
            if self.is_main_process and receipt:
                print(f"QS_WEIGHT_PUBLICATION {json.dumps(receipt, sort_keys=True)}", flush=True)
            return receipt

    config = load_config(str(config_path))
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"QS_EXPECTED_LOAN {verification_loan(config)}", flush=True)
    try:
        run_training(config=config, data_path=str(tasks),
                     trainer_factory=VerificationTrainer, workflow_env={})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def verify(logs):
    target = re.search(r"^QS_EXPECTED_LOAN (\S+)", logs, re.MULTILINE)
    if not target:
        raise RuntimeError("Missing the expected QS loan owner")
    config_rollout_id = target.group(1)
    receipts = [json.loads(match) for match in re.findall(r"QS_WEIGHT_PUBLICATION (\{.*\})", logs)]
    if not receipts or receipts[0]["optimizer_revision"] != 0 or receipts[0]["nccl_world_size"] != 61:
        raise RuntimeError("Missing initial NCCL publication to all 60 rollout ranks")
    first = next(row for row in receipts[0]["acknowledgements"] if row["instance_id"] == config_rollout_id)
    if not any(r["optimizer_revision"] > 0 and config_rollout_id not in r["target_ids"] for r in receipts):
        raise RuntimeError("No weight publication excluded the borrowed GPUs")
    if not any(row["instance_id"] == config_rollout_id and
               row["server_instance_id"] != first["server_instance_id"]
               for receipt in receipts[1:] for row in receipt["acknowledgements"]):
        raise RuntimeError("The returned rollout never acknowledged a fresh publication")
    if not any(json.loads(match) for match in re.findall(r"QS_HYBRID step=\d+ active=(\[.*\])", logs)):
        raise RuntimeError("No optimizer update used a Hybrid worker")


def _exec(pod, *args):
    return ["qs", "exec", pod, "-c", "pytorch", "--", *map(str, args)]


def _probe(pod, source, model, tasks, python, run_dir):
    code = ("import json,pathlib,sys,torch; import megatron.bridge; "
            "from RL_Framework.scripts import vllm_hot_reload_api_server; "
            "model=pathlib.Path(sys.argv[1]); "
            "weights=set(json.loads((model/'model.safetensors.index.json').read_text())['weight_map'].values()); "
            "assert all((model/name).stat().st_size>0 for name in weights|{'tokenizer.json'}); "
            "assert torch.cuda.device_count()==8; torch.empty(1,device='cuda')")
    shell = ('for p in "$1/pyproject.toml" "$2/model.safetensors.index.json" "$3" "$5/shared-marker"; '
             'do test -r "$p" || { printf "Missing QS input: %s\n" "$p" >&2; exit 9; }; done; '
             'PYTHONPATH="$5/imports:$1" "$4" -c "$6" "$2" || exit; '
             'hostname -i > "$7.tmp" && mv "$7.tmp" "$7"')
    log = run_dir / f"preflight-{pod}.log"
    receipt = log.with_suffix(".ready")
    pid = start_remote(pod, log, "sh", "-c", shell, "_", source, model, tasks, python,
                       run_dir, code, receipt)
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        if check_files(pod, [receipt]):
            output = subprocess.check_output(_exec(pod, "cat", receipt), text=True, timeout=40)
            return str(ipaddress.IPv4Address(output.strip()))
        if not running(pod, pid) and not check_files(pod, [receipt]):
            output = subprocess.check_output(_exec(pod, "tail", "-30", log), text=True, timeout=40)
            raise RuntimeError(output)
        time.sleep(5)
    raise TimeoutError(f"QS preflight timed out on {pod}; see {log}")


def supervise(args):
    # All runtime paths live on the same QS shared filesystem.
    root = args.run_dir
    for path in (root, args.source, args.model, args.tasks, args.python):
        if not path.is_absolute():
            raise ValueError("QS paths must be absolute")
    script = args.source / "scripts" / Path(__file__).name
    legacy_script = args.source / "scripts/run_r2e_qwen3_14b_16gpu.py"
    preset = args.source / "configs/r2e_qwen3_14b_16gpu.yaml"
    driver = args.pods[0]
    if len(set(args.pods)) != 8:
        raise ValueError("Expected eight distinct QS Pods, master first")
    subprocess.run(_exec(driver, "sh", "-c",
                        'umask 077; mkdir "$1" && mkdir "$1/imports" && ln -s "$2" "$1/imports/RL_Framework" && touch "$1/shared-marker"',
                        "_", root, args.source), check=True, timeout=40)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_probe, pod, args.source, args.model, args.tasks, args.python, root)
                   for pod in args.pods]
        failures = []
        for pod, future in zip(args.pods, futures):
            if (exc := future.exception()) is not None:
                failures.append((pod, str(exc.output).strip() if isinstance(exc, subprocess.CalledProcessError)
                                 else str(exc)))
        if failures:
            raise RuntimeError(f"QS preflight failed: {failures}")
        hosts = [future.result() for future in futures]

    env = {
        "PYTHONPATH": f"{root}/imports:{args.source}",
        "LIBRA_QS_ROLLOUT_DIR": f"{root}/qs-rollout-control",
        "LIBRA_HYBRID_RUN_ID": str(root),
        "JOB_ID": root.name,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SERVER_MODULE": "RL_Framework.scripts.vllm_hot_reload_api_server",
        "NCCL_CUMEM_ENABLE": "0",
        "OMP_NUM_THREADS": "1",
    }
    remote = ["env", *(f"{key}={val}" for key, val in env.items()), args.python]
    config = root / "planned.yaml"
    subprocess.run(_exec(driver, *remote, script, "--prepare", "--run-dir", root,
                        "--model", args.model, "--preset", preset, "--hosts", *hosts),
                   check=True, timeout=60)
    archive = args.local_logs
    archive.mkdir(parents=True, exist_ok=True)
    controllers = []
    try:
        for pod, host in zip(args.pods, hosts):
            argv = ["env", f"LIBRA_QS_NODE={host}", "LIBRA_QS_CONTROLLER=1", *remote,
                    legacy_script, "--node", host, "--config", config, "--run-dir", root]
            pid = start_remote(pod, root / f"controller-{host}.log", *argv)
            controllers.append((pod, host, pid))
        ready = [root / f"node-{host}.ready.json" for host in hosts]
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if check_files(driver, ready):
                break
            # Check the process on its owning Pod. A QS exec stream may end
            # while its remote command is still running.
            for pod, host, pid in controllers:
                if not running(pod, pid):
                    raise RuntimeError(f"QS controller for {host} stopped before becoming ready")
            time.sleep(5)
        else:
            raise TimeoutError("QS rollout controllers did not become ready")
        # Ready files are JSON arrays. Flatten them into the trainer's JSONL registry.
        flatten = ('import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); '
                   'p.write_text("".join(json.dumps(row)+"\\n" for f in sys.argv[2:] '
                   'for row in json.loads(pathlib.Path(f).read_text())))')
        subprocess.run(_exec(driver, *remote, "-c", flatten, root / "rollout-pids.jsonl", *ready),
                       check=True, timeout=40)
        trainer = ["env", f"LIBRA_QS_NODE={hosts[0]}", "CUDA_VISIBLE_DEVICES=0,1,2,3",
                   *remote, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
                   script, "--train", "--config", config, "--tasks", args.tasks]
        receipt = root / "train.exit"
        wrapper = 'cd "$1" || exit; exit_file="$2"; shift 2; "$@"; rc=$?; printf "%s\n" "$rc" > "$exit_file"'
        train_pid = start_remote(driver, root / "train.log", "sh", "-c", wrapper, "_",
                                 root, receipt, *trainer)
        print(f"QS trainer started. Shared log: {root / 'train.log'}", flush=True)
        while not check_files(driver, [receipt]):
            if not running(driver, train_pid):
                if not check_files(driver, [receipt]):
                    raise RuntimeError("QS trainer stopped without an exit receipt")
                break
            time.sleep(10)
        result = subprocess.check_output(_exec(driver, "cat", receipt), text=True, timeout=30)
        logs = subprocess.check_output(_exec(driver, "cat", root / "train.log"), text=True, timeout=120)
        (archive / "train.log").write_text(logs)
        if int(result.strip()) != 0:
            raise RuntimeError(f"QS trainer failed. See {archive / 'train.log'}")
        verify(logs)
    finally:
        subprocess.run(_exec(driver, "touch", root / "controllers.stop"), timeout=40)
        for pod, host, _ in controllers:
            result = subprocess.run(_exec(pod, "cat", root / f"controller-{host}.log"),
                                    capture_output=True, text=True, timeout=40)
            (archive / f"{pod}.log").write_text(result.stdout + result.stderr)


def start_remote(pod, log, *argv):
    shell = ('log="$1"; shift; nohup setsid "$@" > "$log" 2>&1 < /dev/null & '
             'pid=$!; sleep 1; printf "QS_PID %s\n" "$pid"')
    output = subprocess.check_output(_exec(pod, "sh", "-c", shell, "_", log, *argv),
                                     text=True, timeout=40)
    match = re.search(r"^QS_PID (\d+)", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"QS did not return a PID for {pod}")
    return int(match.group(1))


def check_files(pod, paths):
    # The QS CLI does not always propagate its remote exit code.
    shell = 'for p do test -s "$p" || exit; done; printf "QS_FILES_READY\n"'
    output = subprocess.check_output(_exec(pod, "sh", "-c", shell, "_", *paths),
                                     text=True, timeout=40)
    return "QS_FILES_READY" in output


def running(pod, pid):
    shell = 'stat=$(ps -o stat= -p "$1"); case "$stat" in ""|Z*) exit;; esac; printf "QS_ALIVE\n"'
    output = subprocess.check_output(_exec(pod, "sh", "-c", shell, "_", pid),
                                     text=True, timeout=40)
    return "QS_ALIVE" in output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("launch", "prepare", "train"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--local-logs", type=Path)
    parser.add_argument("--pods", nargs=8)
    parser.add_argument("--hosts", nargs=8)
    parser.add_argument("--preset", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if not all((args.preset, args.model, args.run_dir, args.hosts)):
            parser.error("--prepare requires preset, model, run path and eight host IPs")
        configure(args.preset, args.model, args.run_dir, args.hosts).to_yaml(str(args.run_dir / "planned.yaml"))
    elif args.train:
        if not all((args.config, args.tasks)):
            parser.error("--train requires config and tasks")
        train(args.config, args.tasks)
    else:
        needed = (args.run_dir, args.source, args.model, args.tasks, args.python, args.local_logs, args.pods)
        if not all(needed):
            parser.error("--launch requires run, source, model, tasks, python, log paths and eight Pods")
        supervise(args)


if __name__ == "__main__":
    main()

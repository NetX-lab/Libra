"""Launch the native R2E-Gym trainer on one Core and three rollout nodes."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


LAUNCH_IDENTITY = ("run_id", "generation", "launch_owner")


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bind_placement(config, nodes, model, run_dir):
    from RL_Framework.config import HeterogeneousInstanceConfig

    if (config.train_gpus, config.rollout_gpus, config.n_total_gpus,
            config.train_tp_size, config.train_dp_size) != (4, 12, 16, 4, 1):
        raise ValueError("Expected a four-GPU TP4/DP1 Core and twelve rollout GPUs")
    config.model_path = config.tokenizer_path = str(model)
    config.num_nodes = 1
    config.master_addr = config.rollout_nccl_host = nodes[0]
    config.train_gpus_per_node = config.rollout_gpus_per_node = 4
    config.log_dir = str(run_dir / "logs")
    config.sync_path = str(run_dir / "weights")
    config.rollout_weight_sync_control_dir = str(run_dir / "control")
    config.phase_trace_dir = str(run_dir / "phases")
    config.history_output_dir = str(run_dir / "history")

    rollout = config.heterogeneous_rollout
    rollout.vllm_host = nodes[1]
    rollout.available_gpus = list(range(4))
    rollout.instances = []
    for host, tp in zip(nodes[1:], (4, 2, 1)):
        for first in range(0, 4, tp):
            index = len(rollout.instances)
            rollout.instances.append(HeterogeneousInstanceConfig(
                instance_id=f"rollout-{index}", host=host, tp=tp,
                gpus=list(range(first, first + tp)), port=rollout.vllm_base_port + index))
    config.vllm_num_instances = len(rollout.instances)
    config.vllm_endpoints = ",".join(f"{item.host}:{item.port}" for item in rollout.instances)
    rollout.scheduling.cmlfq_tree_path = str(run_dir / "cmlfq/tree.json")
    rollout.scheduling.cmlfq_shared_load_dir = str(run_dir / "cmlfq/load")

    planner = config.global_resource_planner
    planner.gradient_server_host = "0.0.0.0"
    planner.gradient_server_public_host = nodes[0]
    planner.hybrid_worker_python = sys.executable
    planner.hybrid_worker_remote_control_enabled = True
    planner.hybrid_worker_task_dir = str(run_dir / "hybrid")
    planner.hybrid_worker_remote_control_dir = planner.hybrid_worker_task_dir
    planner.runtime_rollout_log_dir = str(run_dir / "rollouts")
    planner.runtime_rollout_manifest_path = str(run_dir / "rollout-layout.json")
    planner.runtime_rollout_pid_registry_path = str(run_dir / "rollout-pids.jsonl")
    planner.runtime_length_profile_jsonl = str(run_dir / "length-profile.jsonl")
    planner.simulator_cache_dir = str(run_dir / "simulator-cache")


def slurm_step(host, *args, cpus=4):
    return ["srun", "--overlap", "--nodes=1", "--ntasks=1", f"--nodelist={host}",
            "--gres=gpu:4", f"--cpus-per-task={cpus}", "--cpu-bind=none", *map(str, args)]


def train_command(host, config_path, tasks):
    return slurm_step(host, sys.executable, "-B", "-m", "torch.distributed.run",
                      "--standalone", "--nproc-per-node=4", "--module", "RL_Framework.examples.r2e_gym_1938329",
                      "--config", config_path, "--tasks", tasks, cpus=24)


def launch_host():
    return os.environ.get("LIBRA_QS_NODE") or os.environ.get("SLURMD_NODENAME") or socket.gethostname()


def same_host(left, right):
    for host in (left, right):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            continue
        return left == right
    return left.split(".", 1)[0] == right.split(".", 1)[0]


class NodeWorkers:
    """Run and acknowledge only the Hybrid processes this rollout node owns."""

    def __init__(self, task_dir, host, gpus):
        self.task_dir, self.host, self.gpus = task_dir, host, gpus
        self.workers = {}
        task_dir.mkdir(parents=True, exist_ok=True)

    def launch(self, worker, request):
        gpus = request["gpus"]
        if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= self.gpus:
            raise ValueError("Hybrid request must use distinct GPUs from this rollout node")
        if worker in self.workers:
            raise RuntimeError("This node already owns a process for the Hybrid worker")
        devices = ",".join(map(str, gpus))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices, "PLANNED_CUDA_VISIBLE_DEVICES": devices}
        log_path = self.task_dir / f"{worker}.controller.log"
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(["bash", "-c", request["command"]], env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.workers[worker] = (proc, request, log_path)
        return "started", {"pid": proc.pid, **{key: request[key] for key in LAUNCH_IDENTITY if key in request}}

    def stop(self, worker, request):
        from RL_Framework.infra.elastic.rollout_process import stop_owned_session

        pending = self.task_dir / f".launch_{worker}.json"
        started = self.task_dir / f".launch_{worker}.started"
        proc, launch, _ = self.workers.get(worker, (None, None, None))
        queued = read_json(pending) if pending.exists() else None
        owned = launch or queued
        if owned is not None and any(request.get(key) != owned[key]
                                     for key in ("host", *LAUNCH_IDENTITY) if key in owned):
            raise RuntimeError("Stop request does not identify the owned Hybrid launch")
        if proc is not None:
            if int(request.get("pid", -1)) > 0 and int(request["pid"]) != proc.pid:
                raise RuntimeError("Stop request PID differs from the owned Hybrid process")
            stop_owned_session(proc)
            self.workers.pop(worker)
        elif started.exists():
            raise RuntimeError("Hybrid launch is recorded but its process is no longer owned")
        if queued is not None and (launch is None or queued == launch):
            pending.unlink()
        started.unlink(missing_ok=True)
        return "done", {"pid": proc.pid if proc is not None else -1,
                        "exited": True, "finished_at": time.time(),
                        **{key: owned[key] for key in LAUNCH_IDENTITY if owned and key in owned}}

    def poll(self):
        # Process stops first so a cancelled launch can be withdrawn without starting it.
        for action, handle in (("stop", self.stop), ("launch", self.launch)):
            for pending in self.task_dir.glob(f".{action}_*.json"):
                try:
                    request = read_json(pending)
                    worker = request["worker_id"]
                    if not same_host(request["host"], self.host) or pending.name != f".{action}_{worker}.json":
                        continue
                    running = pending.with_suffix(".running")
                    pending.replace(running)
                except FileNotFoundError:
                    continue
                reply = {"worker_id": worker, "host": self.host}
                if action == "stop":
                    reply["request_id"] = request["request_id"]
                try:
                    result, fields = handle(worker, request)
                    reply.update(fields)
                except (OSError, RuntimeError, ValueError) as exc:
                    result = "error"
                    reply["error"] = str(exc)
                write_json(pending.with_suffix(f".{result}"), reply)
                running.unlink()
        for worker, (proc, request, log_path) in self.workers.items():
            code = proc.poll()
            error = self.task_dir / f".launch_{worker}.error"
            if code is not None and not error.exists():
                write_json(error, {"worker_id": worker, "host": self.host,
                                   "error": f"Hybrid process exited with status {code}; see {log_path}",
                                   **{key: request[key] for key in LAUNCH_IDENTITY if key in request}})

    def close(self):
        from RL_Framework.infra.elastic.rollout_process import stop_owned_session

        for proc, _, _ in self.workers.values():
            stop_owned_session(proc)
        self.workers.clear()


def run_node(config_path, run_dir, host):
    from RL_Framework.config import load_config
    from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor

    if not same_host(launch_host(), host):
        raise RuntimeError(f"Controller requested for {host} started on {launch_host()}")
    config = load_config(str(config_path))
    executor = RuntimeElasticExecutor(config=config, planner=None)
    services = [meta for meta in executor._desired_rollout_processes(None) if same_host(meta.host, host)]
    worker_dir = Path(config.global_resource_planner.hybrid_worker_remote_control_dir)
    controller = NodeWorkers(worker_dir, host, {gpu for meta in services for gpu in meta.gpus})
    rollout_controller = None
    if root := os.getenv("LIBRA_QS_ROLLOUT_DIR"):
        from RL_Framework.infra.elastic.qs_rollout_process import Controller
        rollout_controller = Controller(root, host, controller.gpus)
    try:
        started = []
        for meta in services:
            # The trainer adopts the PID from this node. Make it the supervisor PID.
            meta.command = "exec " + meta.command
            service = executor._start_one_rollout_process(meta)
            if rollout_controller is not None:
                rollout_controller.register(executor._processes[meta.instance_id],
                                            meta.instance_id, meta.port, meta.gpus)
            started.append(vars(service))
            if rollout_controller is None:
                executor._wait_rollout_process_start_ready(service)
        if rollout_controller is not None:
            for service in executor._process_meta.values():
                executor._wait_rollout_process_start_ready(service)
        write_json(run_dir / f"node-{host}.ready.json", started)
        while not (run_dir / "controllers.stop").exists():
            if rollout_controller is not None:
                rollout_controller.poll()
            controller.poll()
            time.sleep(0.2)
    finally:
        try:
            controller.close()
        finally:
            if rollout_controller is not None:
                try:
                    rollout_controller.close()
                finally:
                    executor._processes.clear()
                    executor._process_meta.clear()
            executor.close()


def supervise(args):
    run_dir = args.run_dir.resolve()
    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    tasks = args.tasks.resolve(strict=True)
    preset = (args.config or source / "configs/r2e_qwen3_14b_16gpu.yaml").resolve(strict=True)
    run_dir.mkdir(mode=0o700, parents=True)
    os.chdir(run_dir)
    imports = run_dir / "imports"
    imports.mkdir()
    (imports / "RL_Framework").symlink_to(source, target_is_directory=True)
    source_paths = [str(imports), str(source)]
    sys.path[:0] = source_paths
    if os.environ.get("PYTHONPATH"):
        source_paths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(source_paths)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", WANDB_MODE="disabled",
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_USE_V1="1", NCCL_CUMEM_ENABLE="0",
                      VLLM_SERVER_MODULE="RL_Framework.scripts.vllm_hot_reload_api_server",
                      TOKENIZERS_PARALLELISM="false", MCORE_MOE_GROUPED_GEMM="0", OMP_NUM_THREADS="1",
                      SLURM_EXPORT_ENV="ALL", LIBRA_HYBRID_RUN_ID=str(run_dir))
    for name in ("SLURM_GPUS", "SLURM_GPUS_PER_TASK", "SLURM_GPUS_PER_NODE"):
        os.environ.pop(name, None)
    os.environ.setdefault("VLLM_RPC_BASE_PATH", os.environ.get("TMPDIR", "/tmp"))

    from RL_Framework.config import load_config
    from RL_Framework.infra.elastic.rollout_process import allocation_nodes, stop_owned_session

    nodes = allocation_nodes(os.environ)
    if len(nodes) != 4 or len(set(nodes)) != 4:
        raise RuntimeError("This launch requires four distinct Slurm nodes")
    config = load_config(str(preset))
    bind_placement(config, nodes, model, run_dir)
    config_path = run_dir / "planned.yaml"
    config.to_yaml(str(config_path))
    controllers = []
    training = None
    try:
        for host in nodes[1:]:
            argv = slurm_step(host, sys.executable, "-B", Path(__file__).resolve(),
                              "--node", host, "--config", config_path, "--run-dir", run_dir)
            with (run_dir / f"node-{host}.log").open("x", encoding="utf-8") as log:
                controllers.append(subprocess.Popen(argv, cwd=run_dir, stdout=log,
                                                    stderr=subprocess.STDOUT, start_new_session=True))
        ready = [run_dir / f"node-{host}.ready.json" for host in nodes[1:]]
        deadline = time.monotonic() + config.global_resource_planner.vllm_ready_timeout_s * 4
        while not all(path.exists() for path in ready):
            if any(proc.poll() is not None for proc in controllers) or time.monotonic() >= deadline:
                raise RuntimeError(f"Rollout nodes did not become ready; see node logs in {run_dir}")
            time.sleep(1)
        pid_registry = Path(config.global_resource_planner.runtime_rollout_pid_registry_path)
        pid_registry.write_text("".join(json.dumps(service) + "\n" for path in ready
                                        for service in read_json(path)), encoding="utf-8")
        with (run_dir / "train.log").open("x", encoding="utf-8") as log:
            training = subprocess.Popen(train_command(nodes[0], config_path, tasks), cwd=run_dir,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while training.poll() is None:
                if any(proc.poll() is not None for proc in controllers):
                    raise RuntimeError(f"A rollout controller exited during training; see node logs in {run_dir}")
                time.sleep(1)
        if training.returncode < 0:
            return 128 - training.returncode
        return training.returncode
    finally:
        try:
            if training is not None and training.poll() is None:
                stop_owned_session(training)
        finally:
            (run_dir / "controllers.stop").touch()
            errors = []
            for host, proc in zip(nodes[1:], controllers):
                try:
                    status = proc.wait(timeout=60)
                    if status:
                        errors.append(f"{host} exited with status {status}")
                except subprocess.TimeoutExpired:
                    try:
                        stop_owned_session(proc)
                    except RuntimeError as exc:
                        errors.append(f"{host} cleanup failed ({exc})")
                    else:
                        errors.append(f"{host} did not exit before the cleanup timeout")
            if errors:
                raise RuntimeError(f"Rollout controller cleanup failed ({', '.join(errors)}); see node logs in {run_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Shared Libra checkout")
    parser.add_argument("--model", type=Path, help="Local Qwen3-14B weights and tokenizer")
    parser.add_argument("--tasks", type=Path, help="Compact R2E-Gym task index JSONL")
    parser.add_argument("--run-dir", type=Path, required=True, help="New writable shared run directory")
    parser.add_argument("--config", type=Path, help="Override the shipped 16-GPU preset")
    parser.add_argument("--node", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.node is None and not (args.source and args.model and args.tasks):
        parser.error("--source, --model and --tasks are required")

    def terminate(signum, _frame):
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.node:
            return run_node(args.config, args.run_dir, args.node)
        return supervise(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

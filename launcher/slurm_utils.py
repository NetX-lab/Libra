"""Support code for Slurm utils."""

import os
import subprocess
from dataclasses import dataclass


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def get_slurm_nodes() -> list[str]:
    """Get slurm nodes."""
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        import socket
        return [socket.gethostname()]

    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True, text=True, check=True,
        )
        nodes = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
        return nodes if nodes else [os.environ.get("HOSTNAME", "localhost")]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [os.environ.get("HOSTNAME", "localhost")]


def get_slurm_info() -> dict:
    """Get slurm info."""
    return {
        "job_id": os.environ.get("SLURM_JOBID", ""),
        "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "nodelist": os.environ.get("SLURM_NODELIST", ""),
        "num_nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "num_tasks": int(os.environ.get("SLURM_NTASKS", "1")),
        "gpus": os.environ.get("SLURM_GPUS", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
    }


def is_slurm_env() -> bool:
    """Is slurm env."""
    return bool(os.environ.get("SLURM_NODELIST", ""))


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class NodeGPUAllocation:
    """Node g p u allocation implementation."""
    node: str
    train_gpu_ids: list[int]
    rollout_gpu_ids: list[int]
    gpus_per_node: int


def compute_gpu_allocation(
    nodes: list[str],
    gpus_per_node: int = 4,
    train_gpus_per_node: int = 2,
    rollout_gpus_per_node: int = 2,
) -> list[NodeGPUAllocation]:
    """Compute gpu allocation."""
    total_per_node = train_gpus_per_node + rollout_gpus_per_node
    if total_per_node > gpus_per_node:
        raise ValueError(
            f"Per-node GPU allocation overflow: train({train_gpus_per_node}) + "
            f"rollout({rollout_gpus_per_node}) > gpus_per_node({gpus_per_node})"
        )

    allocations = []
    for node in nodes:
        train_ids = list(range(train_gpus_per_node))
        rollout_ids = list(range(train_gpus_per_node, train_gpus_per_node + rollout_gpus_per_node))
        allocations.append(NodeGPUAllocation(
            node=node,
            train_gpu_ids=train_ids,
            rollout_gpu_ids=rollout_ids,
            gpus_per_node=gpus_per_node,
        ))

    return allocations


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def build_vllm_endpoints(
    allocations: list[NodeGPUAllocation],
    tp_size: int = 1,
    base_port: int = 8000,
) -> list[str]:
    """Build vllm endpoints."""
    endpoints = []
    port = base_port

    for alloc in allocations:
        n_instances = len(alloc.rollout_gpu_ids) // tp_size
        for i in range(n_instances):
            endpoints.append(f"{alloc.node}:{port}")
            port += 1

    return endpoints


def build_vllm_launch_commands(
    allocations: list[NodeGPUAllocation],
    model_path: str,
    tp_size: int = 1,
    base_port: int = 8000,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    extra_args: str = "",
) -> list[dict]:
    """Build vllm launch commands."""
    commands = []
    port = base_port

    for alloc in allocations:
        n_instances = len(alloc.rollout_gpu_ids) // tp_size
        for i in range(n_instances):
            start_idx = i * tp_size
            gpu_ids = alloc.rollout_gpu_ids[start_idx:start_idx + tp_size]
            gpu_ids_str = ",".join(str(g) for g in gpu_ids)

            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_ids_str} "
                f"python3 -m vllm.entrypoints.openai.api_server "
                f"--model {model_path} "
                f"--host 0.0.0.0 "
                f"--port {port} "
                f"--tensor-parallel-size {tp_size} "
                f"--max-model-len {max_model_len} "
                f"--dtype bfloat16 "
                f"--gpu-memory-utilization {gpu_memory_utilization} "
                f"--enforce-eager "
                f"--trust-remote-code"
            )
            if extra_args:
                cmd += f" {extra_args}"

            commands.append({
                "node": alloc.node,
                "port": port,
                "gpu_ids": gpu_ids,
                "gpu_ids_str": gpu_ids_str,
                "command": cmd,
            })
            port += 1

    return commands


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def build_torchrun_command(
    allocations: list[NodeGPUAllocation],
    train_script: str,
    master_addr: str = "",
    master_port: int = 29500,
    extra_args: str = "",
) -> dict:
    """Build torchrun command."""
    num_nodes = len(allocations)
    nproc_per_node = len(allocations[0].train_gpu_ids) if allocations else 1

    if not master_addr:
        master_addr = allocations[0].node if allocations else "localhost"


    train_gpu_ids_str = ",".join(str(g) for g in allocations[0].train_gpu_ids) if allocations else "0"

    cmd = (
        f"torchrun "
        f"--nnodes={num_nodes} "
        f"--nproc_per_node={nproc_per_node} "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint={master_addr}:{master_port} "
        f"{train_script}"
    )
    if extra_args:
        cmd += f" {extra_args}"

    return {
        "num_nodes": num_nodes,
        "nproc_per_node": nproc_per_node,
        "master_addr": master_addr,
        "master_port": master_port,
        "train_gpu_ids_str": train_gpu_ids_str,
        "command": cmd,
    }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def auto_plan_cluster(
    train_gpus: int,
    rollout_gpus: int,
    tp_size: int = 1,
    base_port: int = 8000,
    gpus_per_node: int = 4,
    nodes: list[str] | None = None,
) -> dict:
    """Auto plan cluster."""
    if nodes is None:
        nodes = get_slurm_nodes()

    num_nodes = len(nodes)
    train_per_node = train_gpus // num_nodes
    rollout_per_node = rollout_gpus // num_nodes

    allocations = compute_gpu_allocation(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        train_gpus_per_node=train_per_node,
        rollout_gpus_per_node=rollout_per_node,
    )

    endpoints = build_vllm_endpoints(
        allocations=allocations,
        tp_size=tp_size,
        base_port=base_port,
    )

    return {
        "nodes": nodes,
        "num_nodes": num_nodes,
        "allocations": allocations,
        "endpoints": endpoints,
        "endpoints_str": ",".join(endpoints),
        "is_slurm": is_slurm_env(),
    }


def print_cluster_plan(plan: dict):
    """Print cluster plan."""
    print("=" * 60)
    print("Cluster deployment plan")
    print("=" * 60)
    print(f"  Nodes: {plan['num_nodes']}")
    print(f"  Slurm environment: {plan['is_slurm']}")
    print(f"  Node list: {plan['nodes']}")

    for alloc in plan["allocations"]:
        print(f"\n  Node {alloc.node}:")
        print(f"    Training GPUs: {alloc.train_gpu_ids}")
        print(f"    Rollout GPU: {alloc.rollout_gpu_ids}")

    print(f"\n  vLLMendpoints: {plan['endpoints']}")
    print("=" * 60)

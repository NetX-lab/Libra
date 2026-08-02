#!/usr/bin/env python3
"""Elastic hybrid worker process.

This process is launched on a borrowed rollout GPU. It loads one distributed
model+optimizer snapshot while joining, then exchanges gradients with the Core
over a decoupled TCP domain. Active workers apply the Core's post-AllReduce
gradient locally, keeping model and optimizer state in lockstep without
per-step checkpoint reloads.

The initial implementation supports a file-queue contract:
  * payload files are torch-saved dictionaries with ``tensors`` or a
    ``GradientPayload``-compatible shape;
  * each processed file is renamed to ``*.done``;
  * a file with ``{"shutdown": True}`` exits the worker.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from RL_Framework.infra.elastic.gradient_ipc import (
    ElasticGradientClient,
    GradientEndpoint,
)
from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--target-core-id", required=True)
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--gradient-host", required=True)
    parser.add_argument("--gradient-port", type=int, required=True)
    parser.add_argument("--authkey", default="")
    parser.add_argument("--gradient-transport", choices=["tcp", "native_rdma"], default="tcp")
    parser.add_argument("--native-rdma-device", default="mlx5_0")
    parser.add_argument("--native-rdma-gid-index", type=int, default=0)
    parser.add_argument("--native-rdma-ib-port", type=int, default=1)
    parser.add_argument("--native-rdma-max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--task-dir", default="")
    parser.add_argument("--payload-file", default="")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--idle", action="store_true")
    parser.add_argument(
        "--worker-mode",
        choices=["transport", "megatron_core"],
        default="transport",
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--gradient-endpoint-dir", default="")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Initialize the Megatron replica without loading state or serving tasks.",
    )
    return parser.parse_args()


def _load_payload_file(
    path: Path,
    *,
    worker_id: str,
    target_core_id: str,
) -> GradientPayload | None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and data.get("shutdown"):
        return None
    if isinstance(data, GradientPayload):
        return data
    tensors = data["tensors"] if isinstance(data, dict) else data
    return GradientPayload(
        replica_id=str(data.get("replica_id", worker_id)) if isinstance(data, dict) else worker_id,
        target_core_id=str(data.get("target_core_id", target_core_id)) if isinstance(data, dict) else target_core_id,
        tensors=tuple(t.detach().cpu() for t in tensors),
        zero_placeholder=bool(data.get("zero_placeholder", False)) if isinstance(data, dict) else False,
        step=int(data.get("step", -1)) if isinstance(data, dict) else -1,
        state_version=int(data.get("state_version", -1)) if isinstance(data, dict) else -1,
        membership_epoch=int(data.get("membership_epoch", 0)) if isinstance(data, dict) else 0,
    )


def _touch_ready(task_dir: Path, worker_id: str, snapshot_path: str):
    task_dir.mkdir(parents=True, exist_ok=True)
    ready = task_dir / f"{worker_id}.ready"
    ready.write_text(f"snapshot={snapshot_path}\npid={os.getpid()}\n", encoding="utf-8")


def _initialize_megatron_worker(args):
    if not args.config:
        raise ValueError("--config is required for --worker-mode megatron_core")
    if not torch.cuda.is_available():
        raise RuntimeError("Megatron elastic worker requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    from RL_Framework.config import AsyncRLConfig
    from RL_Framework.engine.train_factory import create_train_engine

    config = AsyncRLConfig.from_yaml(args.config)
    if config.train_backend != "megatron_core":
        raise ValueError("elastic hybrid workers currently require Megatron-Core")
    cpu_initialization = os.environ.get("LIBRA_HYBRID_CPU_INITIALIZATION", "")
    if cpu_initialization:
        config.megatron_use_cpu_initialization = cpu_initialization.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    engine = create_train_engine(config)
    engine.initialize(max_seq_length=config.max_seq_length)
    if not args.initialize_only:
        engine.load_elastic_state_snapshot(args.snapshot_path)
    engine.hybrid_lockstep_gradient_sync = bool(
        getattr(
            config.global_resource_planner,
            "hybrid_lockstep_gradient_sync",
            True,
        )
    )
    return engine


def _endpoint_for_lane(args, engine):
    endpoint_dir = Path(args.gradient_endpoint_dir)
    if not endpoint_dir.exists():
        raise FileNotFoundError(f"gradient endpoint directory is missing: {endpoint_dir}")
    target_dp = int(args.target_core_id.removeprefix("dp"))
    lane = engine.get_elastic_lane_state()
    for path in endpoint_dir.glob("rank_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("data_parallel_rank", -1)) != target_dp:
            continue
        if int(data.get("tensor_parallel_rank", -1)) != lane["tensor_parallel_rank"]:
            continue
        if int(data.get("pipeline_parallel_rank", -1)) != lane["pipeline_parallel_rank"]:
            continue
        if int(data.get("context_parallel_rank", -1)) != lane["context_parallel_rank"]:
            continue
        return data
    raise RuntimeError(
        f"no core gradient endpoint matches target={args.target_core_id} lane={lane}"
    )


def _gradient_client(args, endpoint_data):
    endpoint = GradientEndpoint(
        str(endpoint_data["host"]),
        int(endpoint_data["port"]),
        str(endpoint_data.get("authkey", "")),
    )
    backend = str(endpoint_data.get("backend", args.gradient_transport))
    if backend == "native_rdma":
        from RL_Framework.infra.elastic.native_rdma import (
            NativeRDMAConfig,
            NativeRDMAGradientClient,
        )

        return NativeRDMAGradientClient(
            endpoint,
            rdma_config=NativeRDMAConfig(
                device=args.native_rdma_device,
                gid_index=args.native_rdma_gid_index,
                ib_port=args.native_rdma_ib_port,
                max_bytes=args.native_rdma_max_bytes,
            ),
        )
    return ElasticGradientClient(endpoint)


def main():
    args = parse_args()
    snapshot = Path(args.snapshot_path)
    if not snapshot.exists() and not args.initialize_only:
        raise FileNotFoundError(f"snapshot does not exist: {snapshot}")

    engine = None
    if args.worker_mode == "megatron_core":
        engine = _initialize_megatron_worker(args)
        if args.initialize_only:
            print("[HybridWorker] initialize_only_complete", flush=True)
            return
    else:
        # Transport mode is retained for protocol tests and custom workers.
        torch.load(snapshot, map_location="cpu", weights_only=False)

    endpoint = GradientEndpoint(args.gradient_host, args.gradient_port, args.authkey)
    if engine is not None:
        client = _gradient_client(args, _endpoint_for_lane(args, engine))
    elif args.gradient_transport == "native_rdma":
        from RL_Framework.infra.elastic.native_rdma import (
            NativeRDMAConfig,
            NativeRDMAGradientClient,
        )

        client = NativeRDMAGradientClient(
            endpoint,
            rdma_config=NativeRDMAConfig(
                device=args.native_rdma_device,
                gid_index=args.native_rdma_gid_index,
                ib_port=args.native_rdma_ib_port,
                max_bytes=args.native_rdma_max_bytes,
            ),
        )
    else:
        client = ElasticGradientClient(endpoint)

    if args.payload_file:
        payload = _load_payload_file(
            Path(args.payload_file),
            worker_id=args.worker_id,
            target_core_id=args.target_core_id,
        )
        if payload is not None:
            client.send(payload)
        return

    if args.idle and not args.task_dir:
        while True:
            time.sleep(args.poll_interval)

    if not args.task_dir:
        raise ValueError("--task-dir or --payload-file is required")

    task_dir = Path(args.task_dir)
    _touch_ready(task_dir, args.worker_id, str(snapshot))

    while True:
        tasks = sorted(task_dir.glob(f"{args.worker_id}.*.pt"))
        if not tasks:
            time.sleep(args.poll_interval)
            continue
        for task_path in tasks:
            try:
                data = torch.load(task_path, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and data.get("shutdown"):
                    task_path.rename(task_path.with_suffix(task_path.suffix + ".done"))
                    return
                if engine is not None and isinstance(data, dict) and data.get("reload_snapshot"):
                    engine.load_elastic_state_snapshot(str(data["reload_snapshot"]))
                    snapshot = Path(str(data["reload_snapshot"]))
                    payload = None
                elif engine is not None and isinstance(data, dict) and "trajectories" in data:
                    task_snapshot = str(data.get("snapshot_path", ""))
                    if task_snapshot and task_snapshot != str(snapshot):
                        engine.load_elastic_state_snapshot(task_snapshot)
                        snapshot = Path(task_snapshot)
                    engine.recompute_logprobs(data["trajectories"])
                    payload = engine.compute_elastic_gradient_payload(
                        data["trajectories"],
                        worker_id=args.worker_id,
                        target_core_id=args.target_core_id,
                        step=int(data["step"]),
                        state_version=int(data["state_version"]),
                        membership_epoch=int(data["membership_epoch"]),
                    )
                else:
                    payload = _load_payload_file(
                        task_path,
                        worker_id=args.worker_id,
                        target_core_id=args.target_core_id,
                    )
                if payload is not None:
                    expect_update = bool(
                        engine is not None
                        and getattr(engine, "hybrid_lockstep_gradient_sync", True)
                    )
                    update = client.send(payload, expect_update=expect_update)
                    if update is not None:
                        if (
                            update.replica_id != args.worker_id
                            or update.step != payload.step
                            or update.membership_epoch != payload.membership_epoch
                        ):
                            raise RuntimeError(
                                "received a mismatched post-AllReduce update: "
                                f"worker={update.replica_id}, step={update.step}, "
                                f"epoch={update.membership_epoch}"
                            )
                        engine.apply_elastic_gradient_update(
                            update.tensors,
                            state_version=update.state_version,
                        )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                if int(os.environ.get("RANK", "0")) == 0:
                    task_path.rename(task_path.with_suffix(task_path.suffix + ".done"))
            except Exception as exc:
                error_path = task_path.with_suffix(task_path.suffix + ".error")
                error_path.write_text(str(exc), encoding="utf-8")
                task_path.rename(task_path.with_suffix(task_path.suffix + ".failed"))


if __name__ == "__main__":
    main()

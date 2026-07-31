#!/usr/bin/env python3
"""Elastic hybrid worker process.

This process is launched on a borrowed rollout GPU. It loads the exported
training snapshot for synchronization and sends gradient payloads to the core
training process over the elastic gradient TCP endpoint.

The initial implementation supports a file-queue contract:
  * payload files are torch-saved dictionaries with ``tensors`` or a
    ``GradientPayload``-compatible shape;
  * each processed file is renamed to ``*.done``;
  * a file with ``{"shutdown": True}`` exits the worker.
"""

from __future__ import annotations

import argparse
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
    )


def _touch_ready(task_dir: Path, worker_id: str, snapshot_path: str):
    task_dir.mkdir(parents=True, exist_ok=True)
    ready = task_dir / f"{worker_id}.ready"
    ready.write_text(f"snapshot={snapshot_path}\npid={os.getpid()}\n", encoding="utf-8")


def main():
    args = parse_args()
    snapshot = Path(args.snapshot_path)
    if not snapshot.exists():
        raise FileNotFoundError(f"snapshot does not exist: {snapshot}")

    # Load once to validate synchronization material is visible to this process.
    torch.load(snapshot, map_location="cpu", weights_only=False)

    endpoint = GradientEndpoint(args.gradient_host, args.gradient_port, args.authkey)
    if args.gradient_transport == "native_rdma":
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
                payload = _load_payload_file(
                    task_path,
                    worker_id=args.worker_id,
                    target_core_id=args.target_core_id,
                )
                if payload is not None:
                    client.send(payload)
                task_path.rename(task_path.with_suffix(task_path.suffix + ".done"))
            except Exception as exc:
                error_path = task_path.with_suffix(task_path.suffix + ".error")
                error_path.write_text(str(exc), encoding="utf-8")
                task_path.rename(task_path.with_suffix(task_path.suffix + ".failed"))


if __name__ == "__main__":
    main()

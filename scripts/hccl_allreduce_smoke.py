#!/usr/bin/env python3
"""Minimal multi-node HCCL collective smoke test."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("HCCL smoke requires an available NPU")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    value = torch.tensor([float(rank + 1)], device="npu", dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) // 2)
    actual = float(value.cpu().item())
    if actual != expected:
        raise RuntimeError(
            f"HCCL all-reduce mismatch on {socket.gethostname()} rank={rank}: "
            f"expected={expected}, actual={actual}"
        )
    print(
        f"HCCL_ALLREDUCE_OK host={socket.gethostname()} "
        f"rank={rank}/{world_size} value={actual}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

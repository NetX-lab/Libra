#!/usr/bin/env python3
"""Verify that torch-npu can create and use an isolated HCCL process group."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    timeout = timedelta(seconds=int(os.environ.get("HCCL_PREFLIGHT_TIMEOUT", "300")))
    try:
        dist.init_process_group("hccl", device_id=device, timeout=timeout)
    except TypeError:
        dist.init_process_group("hccl", timeout=timeout)

    ranks = list(range(dist.get_world_size()))
    core_group = dist.new_group(ranks=ranks, backend="hccl")
    elastic_group = dist.new_group(
        ranks=ranks,
        backend="hccl",
        use_local_synchronization=True,
    )
    if elastic_group is core_group:
        raise RuntimeError("isolated HCCL group aliases the core group")

    value = torch.tensor([float(dist.get_rank() + 1)], device=device)
    dist.all_reduce(value, group=elastic_group)
    expected = float(sum(range(1, dist.get_world_size() + 1)))
    if float(value.item()) != expected:
        raise RuntimeError(
            f"isolated HCCL all-reduce mismatch: got={value.item()} expected={expected}"
        )
    torch.npu.synchronize(device)
    print(
        "[ElasticCCL preflight] "
        f"rank={dist.get_rank()} world={dist.get_world_size()} "
        f"core_group={id(core_group)} elastic_group={id(elastic_group)} "
        f"value={value.item():.1f}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group(elastic_group)
    dist.destroy_process_group(core_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

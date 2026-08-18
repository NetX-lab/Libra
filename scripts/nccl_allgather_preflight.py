#!/usr/bin/env python3
"""Cross-node collective health check for NCCL CUDA or HCCL Ascend NPU."""

from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from RL_Framework.engine.device_utils import (
    distributed_backend,
    set_device,
    synchronize,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = distributed_backend(os.environ.get("DIST_BACKEND", "auto"))
    device = set_device(local_rank, os.environ.get("DEVICE_BACKEND", "auto"))
    timeout = timedelta(
            seconds=int(os.environ.get("COLLECTIVE_PREFLIGHT_TIMEOUT", "600"))
    )
    try:
        dist.init_process_group(backend, device_id=device, timeout=timeout)
    except TypeError:
        dist.init_process_group(backend, timeout=timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    elements = int(
        os.environ.get(
            "COLLECTIVE_PREFLIGHT_ELEMENTS",
            os.environ.get("NCCL_PREFLIGHT_ELEMENTS", "16777216"),
        )
    )
    iterations = int(
        os.environ.get(
            "COLLECTIVE_PREFLIGHT_ITERATIONS",
            os.environ.get("NCCL_PREFLIGHT_ITERATIONS", "3"),
        )
    )
    source = torch.full(
        (elements,), rank, dtype=torch.bfloat16, device=device
    )
    gathered = torch.empty(
        (world_size * elements,), dtype=torch.bfloat16, device=device
    )

    dist.all_gather_into_tensor(gathered, source)
    synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        dist.all_gather_into_tensor(gathered, source)
    synchronize(device)
    elapsed = time.perf_counter() - started

    expected = torch.arange(world_size, dtype=torch.bfloat16, device=device)
    observed = gathered[::elements]
    if not torch.equal(observed, expected):
        raise RuntimeError(
            f"{backend.upper()} all-gather data mismatch on rank {rank}: "
            f"{observed.tolist()}"
        )
    bytes_per_iteration = world_size * elements * source.element_size()
    bandwidth_gib_s = bytes_per_iteration * iterations / elapsed / (1024 ** 3)
    print(
        f"[{backend.upper()} preflight] rank={rank}/{world_size} device={device} "
        f"message_mib={source.nbytes / (1024 ** 2):.1f} "
        f"elapsed_s={elapsed:.3f} effective_gib_s={bandwidth_gib_s:.2f}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import os

import torch
import torch.distributed as dist

from RL_Framework.infra.elastic import InterReplicaGradientDomain


def test_chunked_inter_replica_all_reduce_with_torchrun():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        core_ids = [f"dp{i}" for i in range(world_size)]
        domain = InterReplicaGradientDomain(
            core_replica_ids=core_ids,
            process_group=dist.group.WORLD,
        )

        local_core_id = f"dp{rank}"
        local = torch.full((8192,), float(rank + 1), dtype=torch.float32)
        reduced = domain.reduce_core_gradients(
            core_gradients={local_core_id: (local,)},
        )

        expected = torch.full_like(local, (world_size + 1) / 2.0)
        assert torch.allclose(reduced[local_core_id][0], expected)
    finally:
        dist.destroy_process_group()

#!/usr/bin/env python3
"""Two-NPU HCCL test for MindSpeed EHP dense/expert gradient aggregation."""
from types import SimpleNamespace

import torch
import torch.distributed as dist

from RL_Framework.engine.mindspeed_train_engine import MindSpeedTrainEngine
from RL_Framework.infra.elastic import GradientPayload


def main():
    import torch_npu  # noqa: F401

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"this focused gate expects two ranks, got {world}")

    engine = MindSpeedTrainEngine(
        model_path="/unused", train_tp_size=1, train_pp_size=1,
        train_dp_size=2, train_ep_size=2, expert_tensor_parallel_size=1,
        micro_batch_size=1,
    )
    engine.get_elastic_core_process_group = lambda: None
    engine._parallel_state = lambda: SimpleNamespace(
        get_data_parallel_group=lambda *, with_context_parallel, is_expert=False: None
    )
    domain = engine.configure_elastic_training(
        ["model"], decouple_communication_domains=False, replica_world_size=2
    )
    domain.request_join("ehp-test", "model")
    domain.mark_active("ehp-test")
    epoch = domain.membership_epoch("ehp-test")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Parameter(torch.ones(1, device=f"npu:{local_rank}"))
            self.expert = torch.nn.Parameter(torch.ones(1, device=f"npu:{local_rank}"))
            self.expert.allreduce = False

    tiny = Tiny()
    tiny.dense.main_grad = torch.tensor([float(rank + 1)], device=f"npu:{local_rank}")
    tiny.expert.main_grad = torch.tensor([float(rank + 101)], device=f"npu:{local_rank}")
    engine.model = [tiny]
    engine._elastic_training_step = 1
    engine._elastic_step_hybrid_workers = ("ehp-test",)
    engine._elastic_active_gradient_timeout_s = 30.0
    engine.enqueue_hybrid_gradient_payload(GradientPayload(
        replica_id="ehp-test", target_core_id="model",
        tensors=(torch.tensor([float(rank + 10)]), torch.tensor([float(rank + 20)])),
        replica_rank=rank, replica_world_size=world, step=1,
        state_version=0, membership_epoch=epoch,
    ))
    updates = []
    engine.set_elastic_gradient_update_callback(updates.append)
    engine._apply_elastic_inter_replica_gradients()

    torch.testing.assert_close(tiny.dense.main_grad, torch.tensor([12.], device=f"npu:{local_rank}"))
    torch.testing.assert_close(tiny.expert.main_grad, torch.tensor([122.], device=f"npu:{local_rank}"))
    assert len(updates) == 1 and updates[0].replica_id == "ehp-test"
    if rank == 0:
        print("MINDSPEED_EHP_HCCL_GATE_PASS dense=12 expert=122 ranks=2", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Full-weight DeepSeek V4 MindSpeed integration gate (not a benchmark).

Runs a synthetic GRPO update to verify 128-rank model initialization, forward,
backward, optimizer update, and checkpoint metadata without claiming throughput.
"""
import json
import os
from pathlib import Path

# Sparse Flash MLA metadata opens a local HCCL listener. Give each local NPU
# process its own base port to avoid the default 60000 bind collision.
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HCCL_IF_BASE_PORT"] = str(60000 + local_rank * 100)
os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"

import torch
import torch.distributed as dist

from RL_Framework.engine.mindspeed_train_engine import MindSpeedTrainEngine


def main():
    torch.set_num_threads(4)
    engine = MindSpeedTrainEngine(
        model_path=os.environ.get(
            "MODEL_PATH", "/data/l00619320/models/DeepSeek-V4-Flash-DSpark-BF16"
        ),
        train_tp_size=4,
        train_pp_size=2,
        train_ep_size=32,
        expert_tensor_parallel_size=1,
        global_batch_size=48,
        micro_batch_size=1,
        sequence_parallel=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=False,
        use_transformer_engine=False,
        sync_path="/libra-work/checkpoints",
    )
    max_seq_length = 2048
    engine.initialize(max_seq_length=max_seq_length)
    seed_ids = engine.tokenizer(
        "Compute 2 + 3. The answer is 5.", return_tensors="pt"
    )["input_ids"]
    repeats = (max_seq_length + seed_ids.shape[-1] - 1) // seed_ids.shape[-1]
    ids = seed_ids.repeat(1, repeats)[:, :max_seq_length]
    reward_cycle = (1.0, -0.5, 0.7, -0.3)
    local_batch_size = engine.get_local_batch_size(engine.global_batch_size)
    print(
        f"LIBRA_GATE_DATA rank={engine.rank} local_batch={local_batch_size} "
        f"sequence={ids.shape[-1]}",
        flush=True,
    )
    data = [
        dict(
            input_ids=ids.clone(),
            attention_mask=torch.ones_like(ids),
            logprobs=torch.zeros_like(ids, dtype=torch.float32),
            loss_mask=torch.cat(
                [torch.zeros(1, 1), torch.ones(1, ids.shape[-1] - 1)], dim=1
            ),
            rewards=torch.tensor([reward_cycle[index % len(reward_cycle)]]),
            advantages=torch.tensor([reward_cycle[index % len(reward_cycle)]]),
        )
        for index in range(local_batch_size)
    ]
    data = engine.distribute_trajectories(data if engine.is_batch_source() else None)
    data = engine.align_distributed_trajectories(data)
    engine.recompute_logprobs(data)
    for trajectory in data:
        assert torch.isfinite(trajectory["logprobs"]).all()
    samples = [
        (parameter, parameter.detach().reshape(-1)[:32].float().clone())
        for parameter in engine.model[0].parameters()
        if parameter.requires_grad
    ]
    metrics = engine.grpo_update(data)
    changed = torch.tensor(
        [
            sum(
                int(not torch.equal(parameter.detach().reshape(-1)[:32].float(), old))
                for parameter, old in samples
            )
        ],
        device=engine._device(),
    )
    dist.all_reduce(changed)
    assert changed.item() > 0, "Optimizer did not change sampled model weights"
    assert all(
        torch.isfinite(torch.tensor(value))
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    )
    engine.set_version(1)
    payload = dict(
        rank=engine.rank,
        metrics=metrics,
        version=engine.get_version(),
        changed_parameter_samples=int(changed.item()),
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
        peak_reserved_bytes=torch.npu.max_memory_reserved(),
    )
    output = Path("/libra-work/full_v4_gate")
    output.mkdir(exist_ok=True)
    (output / f"rank_{engine.rank}.json").write_text(json.dumps(payload, indent=2))
    print("LIBRA_V4_GRPO_GATE_OK " + json.dumps(payload), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

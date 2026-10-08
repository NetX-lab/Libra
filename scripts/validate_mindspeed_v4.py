"""Full-weight V4 training gate. Synthetic trajectories, no performance claim."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from RL_Framework.engine.mindspeed_train_engine import MindSpeedTrainEngine


def main():
    torch.set_num_threads(4)
    engine = MindSpeedTrainEngine(
        model_path=os.environ.get('MODEL_PATH', '/data/l00619320/models/DeepSeek-V4-Flash-DSpark-BF16'),
        train_tp_size=4, train_pp_size=2, train_ep_size=32, expert_tensor_parallel_size=1,
        global_batch_size=32, micro_batch_size=1, sequence_parallel=True,
        optimizer_cpu_offload=True, optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=False, use_transformer_engine=False,
        sync_path='/libra-work/checkpoints',
    )
    engine.initialize(max_seq_length=128)
    ids = engine.tokenizer('Compute 2 + 3. The answer is 5.', return_tensors='pt')['input_ids']
    data = [dict(input_ids=ids.clone(), attention_mask=torch.ones_like(ids),
                 logprobs=torch.zeros_like(ids, dtype=torch.float32),
                 loss_mask=torch.cat([torch.zeros(1,1),torch.ones(1,ids.shape[-1]-1)],dim=1),
                 rewards=torch.tensor([r]), advantages=torch.tensor([r])) for r in (1.,-.5,.7,-.3)]
    data = engine.distribute_trajectories(data if engine.is_batch_source() else None)
    data = engine.align_distributed_trajectories(data)
    engine.recompute_logprobs(data)
    for trajectory in data:
        assert torch.isfinite(trajectory['logprobs']).all()
    samples = [(p, p.detach().reshape(-1)[:32].float().clone())
               for p in engine.model[0].parameters() if p.requires_grad]
    metrics = engine.grpo_update(data)
    changed = torch.tensor([sum(int(not torch.equal(p.detach().reshape(-1)[:32].float(), old))
                                for p, old in samples)], device=engine._device())
    dist.all_reduce(changed)
    assert changed.item() > 0, "Optimizer did not change sampled model weights"
    assert all(torch.isfinite(torch.tensor(v)) for k,v in metrics.items() if isinstance(v,(int,float)))
    engine.set_version(1)
    payload = dict(rank=engine.rank, metrics=metrics, version=engine.get_version(),
                   changed_parameter_samples=int(changed.item()),
                   peak_allocated_bytes=torch.npu.max_memory_allocated(),
                   peak_reserved_bytes=torch.npu.max_memory_reserved())
    out=Path('/libra-work/full_v4_gate')
    out.mkdir(exist_ok=True)
    (out/f'rank_{engine.rank}.json').write_text(json.dumps(payload,indent=2))
    print('LIBRA_V4_GRPO_GATE_OK '+json.dumps(payload),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

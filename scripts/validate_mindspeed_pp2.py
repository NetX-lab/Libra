"""Two-NPU 1F1B check using tiny 4D stages, not a V4 performance benchmark.

Run with torchrun --standalone --nproc_per_node=2 after mindspeed_v4_env.sh.
Checks stage gradients against the same unsplit network and PP logprob return.
"""
import os
from types import SimpleNamespace
import torch
import torch.distributed as dist
from RL_Framework.engine.mindspeed_train_engine import MindSpeedTrainEngine, configure_mindspeed_runtime


def main():
    engine = MindSpeedTrainEngine(model_path='/data_nv1/models/DeepSeek-V4-Flash-DSpark-BF16',
                                 train_pp_size=2, train_tp_size=1, sequence_parallel=False,
                                 micro_batch_size=1, kl_coef=0.0)
    engine._prepare_accelerator()
    # Use the real adaptor / PP communication, but no model allocations or kernels.
    configure_mindspeed_runtime(engine.build_mindspeed_args(8))
    dist.init_process_group('hccl')
    from megatron.core import parallel_state as mpu
    mpu.initialize_model_parallel(pipeline_model_parallel_size=2)
    from megatron.core.transformer import TransformerConfig
    from megatron.core.enums import ModelType
    cfg = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=1,
                            pipeline_model_parallel_size=2, pipeline_dtype=torch.float32,
                            variable_seq_lengths=True, calculate_per_token_loss=False,
                            deallocate_pipeline_outputs=False, batch_p2p_comm=True)
    cfg.grad_scale_func = lambda x: x
    cfg.finalize_model_grads_func = None
    device = engine._device()
    rank = dist.get_rank()
    torch.manual_seed(17)
    embedding = torch.nn.Embedding(16, 8).to(device)
    head = torch.nn.Linear(8, 16, bias=False).to(device)

    class Stage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.model_type = ModelType.encoder_or_decoder
            self.layer = embedding if rank == 0 else head
            self.input_tensor = None
        def set_input_tensor(self, tensor):
            self.input_tensor = tensor[0] if isinstance(tensor, list) else tensor
        def forward(self, input_ids, **kwargs):
            if rank == 0:
                return self.layer(input_ids).transpose(0, 1).unsqueeze(2).contiguous()
            return self.layer(self.input_tensor.squeeze(2).transpose(0, 1)).contiguous()
        def zero_grad_buffer(self):
            self.zero_grad(set_to_none=True)

    engine.model = [Stage()]
    engine.tokenizer = SimpleNamespace(pad_token_id=0)
    data = [dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                 logprobs=torch.zeros(1, 8), loss_mask=torch.tensor([[0.,1.,1.,1.,1.,1.,1.,1.]]),
                 rewards=torch.tensor([r])) for r in (1., -0.5, 0.7, -0.3)]
    data = engine.distribute_trajectories(data if rank == 0 else None)
    engine._pipeline(data)
    actual = engine.model[0].layer.weight.grad.detach().clone()
    embedding.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    for batch in engine._iter_micro_batches(data):
        loss, _ = engine._loss(batch, head(embedding(batch['input_ids'])))
        (loss / len(data)).backward()
    expected = (embedding if rank == 0 else head).weight.grad
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    engine.recompute_logprobs(data)
    with torch.no_grad():
        ids = data[0]['input_ids'].to(device)
        expected_lp = engine._action_log_probs(head(embedding(ids)), ids).cpu()
    torch.testing.assert_close(data[0]['logprobs'][:,1:], expected_lp, atol=2e-6, rtol=2e-5)
    print(f'LIBRA_PP2_GRAD_LOGPROB_OK rank={rank}', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

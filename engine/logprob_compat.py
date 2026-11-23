"""Chunked vocabulary-parallel logprobs with local backward and bounded scratch."""

import torch
import torch.distributed as dist


class _VocabParallelLogProbs(torch.autograd.Function):
    """FP32 normalization with bounded scratch space for TP action logprobs."""

    chunk_rows = 512

    @staticmethod
    def forward(ctx, logits, targets, vocab_start, tp_group, action_positions=None):
        if action_positions is None:
            batch_size, sequence_length = targets.shape
            batch_rows = torch.arange(batch_size, device=logits.device).repeat_interleave(sequence_length)
            token_rows = torch.arange(sequence_length, device=logits.device).repeat(batch_size)
            local_max = logits[:, :-1].amax(dim=-1).flatten()
        else:
            batch_rows, token_rows = action_positions
            targets = targets[action_positions]
            local_max = logits.new_empty(targets.numel())
            for start in range(0, targets.numel(), _VocabParallelLogProbs.chunk_rows):
                chunk = slice(start, start + _VocabParallelLogProbs.chunk_rows)
                local_max[chunk] = logits[batch_rows[chunk], token_rows[chunk]].amax(dim=-1)
        output_shape = targets.shape
        targets = targets.reshape(-1)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
        denominator = torch.empty_like(local_max, dtype=torch.float32)
        for start in range(0, targets.numel(), _VocabParallelLogProbs.chunk_rows):
            chunk = slice(start, start + _VocabParallelLogProbs.chunk_rows)
            scratch = logits[batch_rows[chunk], token_rows[chunk]].float()
            scratch.sub_(local_max[chunk].unsqueeze(-1)).exp_()
            denominator[chunk] = scratch.sum(dim=-1)
            del scratch
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=tp_group)
        local_mask = (targets >= vocab_start) & (
            targets < vocab_start + logits.shape[-1]
        )
        local_targets = (targets - vocab_start).masked_fill(~local_mask, 0)
        selected = logits[batch_rows, token_rows, local_targets].float()
        selected.masked_fill_(~local_mask, 0.0)
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=tp_group)
        # Save the original storage and token-sized tensors, never FP32 exp/logits.
        ctx.save_for_backward(
            logits, batch_rows, token_rows, local_max, denominator, local_targets, local_mask
        )
        return (selected - local_max - torch.log(denominator)).reshape(output_shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, batch_rows, token_rows, local_max, denominator, local_targets, local_mask = ctx.saved_tensors
        grad_output = grad_output.reshape(-1)
        selected_grad = grad_output.masked_fill(~local_mask, 0.0).to(logits.dtype)
        # Forward produced the same logprob on each rank. The local shard
        # uses its own derivative of that value, with no second TP reduction.
        denominator_grad = -grad_output / denominator
        grad_logits = torch.zeros_like(logits)
        for start in range(0, local_targets.numel(), _VocabParallelLogProbs.chunk_rows):
            chunk = slice(start, start + _VocabParallelLogProbs.chunk_rows)
            scratch = logits[batch_rows[chunk], token_rows[chunk]].float()
            scratch.sub_(local_max[chunk].unsqueeze(-1)).exp_()
            scratch.mul_(denominator_grad[chunk].unsqueeze(-1))
            # Match the native path's separate casts for the two branches.
            chunk_grad = scratch.to(logits.dtype)
            chunk_grad.scatter_add_(
                -1, local_targets[chunk].unsqueeze(-1), selected_grad[chunk].unsqueeze(-1),
            )
            grad_logits[batch_rows[chunk], token_rows[chunk]] = chunk_grad
            del scratch, chunk_grad
        return grad_logits, None, None, None, None

def install(engine):
    """Change only the action-logprob kernel, retaining the upstream TP=1 path."""
    if getattr(engine, '_bounded_logprob_installed', False):
        return
    original = engine._action_log_probs

    def bounded(logits, input_ids, action_positions=None):
        if engine.train_tp_size == 1:
            return original(logits, input_ids, action_positions)
        state = engine._parallel_state()
        vocab_start = state.get_tensor_model_parallel_rank() * logits.shape[-1]
        return _VocabParallelLogProbs.apply(
            logits, input_ids[:, 1:], vocab_start, state.get_tensor_model_parallel_group(),
            action_positions)

    engine._action_log_probs = bounded
    engine._bounded_logprob_installed = True

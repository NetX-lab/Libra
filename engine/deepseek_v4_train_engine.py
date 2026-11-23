"""CUDA DeepSeek V4 training using NVIDIA's native MCore hybrid model."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from math import lcm

import torch
import torch.distributed as dist
import torch.nn.functional as F

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine

_LOGPROB_CHUNK_SIZE = 512


def _gradient_norms(_chunk_size, _overflow, tensor_lists, per_tensor=False):
    # Megatron's DSA split combines views from several gradient buffers.
    # Compute FP32 norms with PyTorch, grouping the foreach calls by dtype.
    groups = defaultdict(list)
    grads = tensor_lists[0]
    for index, grad in enumerate(grads):
        groups[grad.dtype].append((index, grad))
    norms = [None] * len(grads)
    for group in groups.values():
        indices, tensors = zip(*group)
        for index, norm in zip(indices, torch._foreach_norm(tensors, 2, dtype=torch.float32)):
            norms[index] = norm
    per_gradient = torch.stack(norms)
    return torch.linalg.vector_norm(per_gradient).reshape(1), per_gradient if per_tensor else None


def _scale_gradients(_chunk_size, _overflow, tensor_lists, scale):
    # Megatron clips in place and passes the same list as both source and target.
    groups = defaultdict(list)
    for source, target in zip(*tensor_lists):
        if source is not target:
            raise ValueError("Expected in-place Megatron gradient scaling")
        groups[target.dtype].append(target)
    if isinstance(scale, torch.Tensor):
        scale = scale.reshape(())
    for group in groups.values():
        torch._foreach_mul_(group, scale)


@contextmanager
def _pytorch_gradient_clipping():
    from megatron.core.optimizer import clip_grads

    saved = (
        clip_grads.l2_norm_impl,
        clip_grads.multi_tensor_scale_impl,
        clip_grads.multi_tensor_scale_tensor_impl,
    )
    clip_grads.l2_norm_impl = _gradient_norms
    clip_grads.multi_tensor_scale_impl = _scale_gradients
    clip_grads.multi_tensor_scale_tensor_impl = _scale_gradients
    try:
        yield
    finally:
        (
            clip_grads.l2_norm_impl,
            clip_grads.multi_tensor_scale_impl,
            clip_grads.multi_tensor_scale_tensor_impl,
        ) = saved


def _configure_flash_mla_for_cp():
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

    fused_sparse_attention._ensure_flash_mla()
    sparse_forward = fused_sparse_attention._flash_mla_sparse_fwd
    parameters = signature(sparse_forward).parameters
    if "indexer_topk" in parameters or getattr(sparse_forward, "_libra_zero_indexer_compat", False):
        return
    if "topk_length" not in parameters:
        raise RuntimeError("DeepSeek V4 CP requires FlashMLA sparse forward with topk_length")

    @wraps(sparse_forward)
    def forward_without_indexer_lse(*args, indexer_topk=0, **kwargs):
        # Legacy Hopper FlashMLA returns the same three values in zero-indexer
        # mode. Nonzero indexer_topk requires the extended four-value API.
        if indexer_topk:
            raise RuntimeError("FlashMLA does not support returning indexer LSE")
        return sparse_forward(*args, **kwargs)

    forward_without_indexer_lse._libra_zero_indexer_compat = True
    fused_sparse_attention._flash_mla_sparse_fwd = forward_without_indexer_lse


class DeepSeekV4TrainEngine(MegatronCoreTrainEngine):
    """Run Libra's GRPO loss through the native Megatron 1F1B schedule."""

    supports_pipeline_parallelism = True
    dependency_guide = "docs/deepseek_v4_cuda.md"

    def _load_bridge(self, auto_bridge):
        try:
            from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
            from megatron.core.config import set_experimental_flag
        except ImportError as exc:
            raise RuntimeError(
                "DeepSeek V4 requires a CUDA Megatron Bridge/MCore build with native V4 "
                "support. See docs/deepseek_v4_cuda.md for the compatible revisions."
            ) from exc
        set_experimental_flag(True)
        bridge = super()._load_bridge(auto_bridge)
        if not isinstance(bridge._model_bridge, DeepSeekV4Bridge):
            raise ValueError("Expected a DeepSeek V4 checkpoint for the DeepSeek V4 CUDA provider")
        return bridge

    def _configure_provider(self, max_seq_length: int) -> None:
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
            set_deepseek_v4_pipeline_model_parallel_layout,
        )

        provider = self.provider
        router_dtype = provider.moe_router_dtype
        super()._configure_provider(max_seq_length)
        provider.moe_router_dtype = router_dtype
        # Match the Hopper-compatible no-MTP recipe and keep the native hybrid
        # W/CSA/HCA + MoE layout. MTP is not part of Libra's GRPO objective.
        provider.transformer_impl = "transformer_engine"
        provider.attention_backend = None
        # V4's native CP implementation requires contiguous THD and cuDNN CSA.
        provider.dsa_kernel_backend = "cudnn" if self.train_cp_size > 1 else "none"
        if self.train_cp_size > 1:
            provider.attention_cp_layout = "contiguous"
            # HybridStack uses the linear layout at its PP boundaries and MoE
            # blocks. Keep it consistent with the packed V4 attention layout.
            provider.linear_cp_layout = "contiguous"
            _configure_flash_mla_for_cp()
        provider.use_fused_mhc = False
        provider.mhc_norm_eps_inside_sqrt = True
        provider.apply_rope_fusion = True
        provider.dsa_indexer_loss_coeff = 0.0
        provider.dsa_indexer_use_sparse_loss = False
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_force_load_balancing = False
        provider.moe_router_bias_update_rate = 0.0
        provider.recompute_granularity = "selective"
        provider.recompute_modules = ["moe_act", "mhc"]
        provider.recompute_method = None
        provider.recompute_num_layers = None
        provider.cuda_graph_impl = "none"
        provider.mtp_num_layers = None
        provider.mtp_loss_scaling_factor = 0.0
        provider.mtp_hybrid_override_pattern = None
        if provider.csa_compress_ratios is not None:
            provider.csa_compress_ratios = list(provider.csa_compress_ratios[:provider.num_layers])
        set_deepseek_v4_pipeline_model_parallel_layout(provider)

    def get_data_parallel_world_size(self):
        return self._parallel_state().get_data_parallel_world_size(with_context_parallel=False)

    def get_data_parallel_rank(self):
        return self._parallel_state().get_data_parallel_rank(with_context_parallel=False)

    def is_batch_source(self):
        return super().is_batch_source() and (
            self.train_cp_size == 1 or self._parallel_state().get_context_parallel_rank() == 0
        )

    def distribute_trajectories(self, trajectories):
        if not dist.is_initialized():
            return trajectories or []
        parallel_state = self._parallel_state()
        payload = [trajectories if self.is_batch_source() else None]
        if self.train_cp_size > 1 and parallel_state.is_pipeline_first_stage():
            cp_group = parallel_state.get_context_parallel_group()
            dist.broadcast_object_list(payload, src=dist.get_global_rank(cp_group, 0), group=cp_group)
        dist.broadcast_object_list(
            payload,
            src=parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        return payload[0] or []

    def _forward_context_shard(self, batch, model, cp_group, *, forward_only):
        from megatron.core.packed_seq_params import PackedSeqParams

        ids = batch["input_ids"]
        samples, width = ids.shape
        local_tokens = ids.numel() // cp_group.size()
        first_token = local_tokens * cp_group.rank()
        shard = slice(first_token, first_token + local_tokens)
        positions = torch.arange(first_token, first_token + local_tokens, device=ids.device)
        cu_seqlens = torch.arange(0, (samples + 1) * width, width, dtype=torch.int32, device=ids.device)
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens,
                                 cu_seqlens_kv=cu_seqlens, max_seqlen_q=width, max_seqlen_kv=width)
        output = model(input_ids=ids.flatten()[shard].unsqueeze(0),
                       position_ids=(positions % width).unsqueeze(0),
                       attention_mask=None, labels=None, packed_seq_params=packed)

        def loss_func(logits):
            if logits.shape[:2] != (1, local_tokens):
                raise RuntimeError("Unexpected DeepSeek V4 packed CP logits layout")

            def scores(active, target_positions):
                selected = F.log_softmax(logits[0, active].float(), dim=-1)
                return selected.gather(-1, ids.flatten()[target_positions].unsqueeze(-1)).squeeze(-1)

            if forward_only:
                local = logits.new_zeros(local_tokens, dtype=torch.float32)
                active = (positions % width < batch["length"] - 1).nonzero(as_tuple=True)[0]
                for indices in active.split(_LOGPROB_CHUNK_SIZE):
                    local[indices] = scores(indices, positions[indices] + 1)
                gathered = torch.empty_like(ids, dtype=local.dtype).view(-1)
                dist.all_gather_into_tensor(gathered, local, group=cp_group)
                return logits.new_zeros(()), gathered.view_as(ids)[:, :batch["length"] - 1]

            # Shift in global document order before slicing. The last query on a
            # shard can predict the first token on the next rank, never the next document.
            candidates = (positions % width < width - 1).nonzero(as_tuple=True)[0]
            candidate_targets = positions[candidates] + 1
            mask = batch["loss_mask"]
            selected = mask.flatten()[candidate_targets].nonzero(as_tuple=True)[0]
            active = candidates[selected]
            target_positions = candidate_targets[selected]
            action_rows = target_positions // width
            anchor = batch["old_logprobs"].flatten()[target_positions]
            weights = mask.flatten()[target_positions]
            if "behavior_logprobs" in batch:
                behavior = batch["behavior_logprobs"].flatten()[target_positions]
                weights = torch.exp(anchor - behavior)
            loss, stats = self._loss_from_actions(
                scores(active, target_positions), anchor, batch["advantages"][action_rows],
                weights, mask[:, 1:].sum(-1).clamp_min(1.0)[action_rows], batch["rewards"],
            )
            # Megatron's two-value loss callback scales by CP before the DP×CP
            # gradient average. Do not scale the local contribution a second time.
            return loss, stats

        return output, loss_func

    def _pipeline(self, trajectories, *, forward_only: bool):
        from megatron.core.distributed import finalize_model_grads
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.utils import get_model_config

        micro_batch_size = self.recompute_micro_batch_size if forward_only else self.micro_batch_size
        # 1F1B uses one activation shape for the entire schedule.
        if not trajectories or len(trajectories) % micro_batch_size:
            raise ValueError("DeepSeek V4 pipeline requires nonempty, complete microbatches")
        cp_group = None
        width = self.max_seq_length
        if self.train_cp_size > 1:
            cp_group = self._parallel_state().get_context_parallel_group()
            compression = max(1, *self.provider.csa_compress_ratios)
            boundary = max(self.provider.csa_window_size, compression)
            alignment = lcm(cp_group.size(), compression)
            width = max(width, boundary * cp_group.size())
            width = ((width + alignment - 1) // alignment) * alignment

        def make_batches():
            for start in range(0, len(trajectories), micro_batch_size):
                rows = trajectories[start:start + micro_batch_size]
                batch = self._pad_micro_batch(rows, require_anchor=not forward_only)
                batch["length"] = batch["input_ids"].shape[1]
                if batch["length"] > self.max_seq_length:
                    raise ValueError("DeepSeek V4 trajectory exceeds the configured sequence length")
                padding = width - batch["length"]
                for key in ("input_ids", "loss_mask", "old_logprobs", "behavior_logprobs"):
                    if key in batch:
                        value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                        batch[key] = F.pad(batch[key], (0, padding), value=value)
                yield batch

        def forward_step(iterator, model):
            batch = next(iterator)
            if cp_group is not None:
                return self._forward_context_shard(
                    batch, model, cp_group, forward_only=forward_only,
                )
            ids = batch["input_ids"]
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
            output = model(input_ids=ids, position_ids=positions, attention_mask=None, labels=None)

            def loss_func(logits):
                if logits.shape[:2] != ids.shape:
                    raise RuntimeError("Unexpected DeepSeek V4 pipeline logits layout")
                if forward_only:
                    length = batch["length"]
                    logprobs = self._action_log_probs(logits[:, :length], ids[:, :length])
                    return logits.new_zeros(()), logprobs
                return self._loss(batch, logits)

            return output, loss_func

        config = get_model_config(self.model[0])
        if not forward_only:
            config.grad_scale_func = self.optimizer.scale_loss
            config.finalize_model_grads_func = finalize_model_grads
        return get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=make_batches(),
            model=self.model,
            num_microbatches=len(trajectories) // micro_batch_size,
            seq_length=width if cp_group is None else width * micro_batch_size,
            micro_batch_size=micro_batch_size if cp_group is None else 1,
            forward_only=forward_only,
        )

    def recompute_logprobs(self, trajectories):
        parallel_state = self._parallel_state()
        self._set_train_mode(False)
        try:
            with torch.no_grad():
                batches = self._pipeline(trajectories, forward_only=True)
                logprobs = torch.zeros(
                    (len(trajectories), self.max_seq_length), dtype=torch.float32, device=self._device()
                )
                if parallel_state.is_pipeline_last_stage():
                    offset = 0
                    for batch in batches:
                        count, width = batch.shape
                        logprobs[offset:offset + count, 1:width + 1] = batch
                        offset += count
                dist.broadcast(
                    logprobs,
                    src=parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                for row, trajectory in zip(logprobs.cpu(), trajectories):
                    length = trajectory["input_ids"].shape[-1]
                    trajectory["logprobs"] = row[:length].unsqueeze(0).clone()
        finally:
            self._set_train_mode(True)

    def grpo_update(self, trajectories, ppo_epochs=1):
        parallel_state = self._parallel_state()
        epoch_stats = []
        for _ in range(ppo_epochs):
            self._zero_grad()
            self._set_train_mode(True)
            stats = self._pipeline(trajectories, forward_only=False)
            # The schedule calls finalize_model_grads before returning.
            # Both Apex and TE 2.20 hit illegal accesses calculating V4's
            # DSA split norm on H20. Preserve Megatron's reductions and clipping.
            with _pytorch_gradient_clipping():
                successful, grad_norm, _ = self.optimizer.step()
            if not successful:
                raise FloatingPointError("DeepSeek V4 optimizer rejected the update")
            self.optimizer_revision += 1
            result = None
            if parallel_state.is_pipeline_last_stage():
                result = self._merge_stats(stats)
                result["grad_norm"] = float(grad_norm) if grad_norm is not None else 0.0
                if self.train_cp_size > 1:
                    keys = ("loss", "pg_loss", "kl")
                    values = torch.tensor([result[key] for key in keys], device=self._device())
                    dist.all_reduce(values, group=parallel_state.get_context_parallel_group())
                    result.update(zip(keys, values.tolist()))
                    result = self._reduce_stats(result, with_context_parallel=False)
                else:
                    result = self._reduce_stats(result)
            payload = [result]
            dist.broadcast_object_list(
                payload,
                src=parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )
            epoch_stats.append(payload[0])
        result = self._merge_stats(epoch_stats)
        result["version"] = self.current_version
        return result

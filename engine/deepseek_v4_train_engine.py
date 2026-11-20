"""CUDA DeepSeek V4 training using NVIDIA's native MCore hybrid model."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


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
        provider.dsa_kernel_backend = "none"
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

    def distribute_trajectories(self, trajectories):
        if not dist.is_initialized():
            return trajectories or []
        parallel_state = self._parallel_state()
        payload = [trajectories if self.is_batch_source() else None]
        dist.broadcast_object_list(
            payload,
            src=parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        return payload[0] or []

    def _pipeline(self, trajectories, *, forward_only: bool):
        from megatron.core.distributed import finalize_model_grads
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.utils import get_model_config

        micro_batch_size = self.recompute_micro_batch_size if forward_only else self.micro_batch_size
        # 1F1B uses one activation shape for the entire schedule.
        if not trajectories or len(trajectories) % micro_batch_size:
            raise ValueError("DeepSeek V4 pipeline requires nonempty, complete microbatches")

        def make_batches():
            for start in range(0, len(trajectories), micro_batch_size):
                rows = trajectories[start:start + micro_batch_size]
                batch = self._pad_micro_batch(rows, require_anchor=not forward_only)
                batch["length"] = batch["input_ids"].shape[1]
                width = self.max_seq_length - batch["length"]
                if width < 0:
                    raise ValueError("DeepSeek V4 trajectory exceeds the configured sequence length")
                for key in ("input_ids", "loss_mask", "old_logprobs"):
                    if key in batch:
                        value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                        batch[key] = F.pad(batch[key], (0, width), value=value)
                yield batch

        def forward_step(iterator, model):
            batch = next(iterator)
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
            seq_length=self.max_seq_length,
            micro_batch_size=micro_batch_size,
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

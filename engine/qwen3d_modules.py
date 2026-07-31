"""Support code for Qwen3d modules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from RL_Framework.engine.train_parallel import TrainParallelRuntime


def divide_evenly(total: int, num_parts: int) -> list[tuple[int, int]]:
    """Divide evenly."""
    base = total // num_parts
    remainder = total % num_parts
    result = []
    start = 0
    for idx in range(num_parts):
        length = base + (1 if idx < remainder else 0)
        result.append((start, start + length))
        start += length
    return result


def split_range_for_rank(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Split range for rank."""
    if total % world_size != 0:
        raise ValueError(
            f"Dimension {total} is not divisible by world_size={world_size}; "
            "the current implementation requires even partitioning"
        )
    shard = total // world_size
    start = rank * shard
    return start, start + shard


def _all_gather_cat(
    tensor: torch.Tensor,
    *,
    dim: int,
    group,
) -> torch.Tensor:
    if group is None or not dist.is_initialized():
        return tensor
    gathered = dist_nn.all_gather(tensor, group=group)
    return torch.cat(gathered, dim=dim)


def _all_reduce_sum(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is None or not dist.is_initialized():
        return tensor
    return dist_nn.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def _all_reduce_max(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is None or not dist.is_initialized():
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX, group=group)
    return reduced


@dataclass
class StageSpec:
    """Stage spec implementation."""

    stage_id: int
    layer_start: int
    layer_end: int
    is_first_stage: bool
    is_last_stage: bool

    @property
    def layer_indices(self) -> list[int]:
        return list(range(self.layer_start, self.layer_end))


class VocabParallelEmbedding(nn.Module):
    """Vocab parallel embedding implementation."""

    def __init__(self, embedding: nn.Embedding, runtime: TrainParallelRuntime):
        super().__init__()
        self.runtime = runtime
        weight = embedding.weight.detach().clone()
        vocab_size, hidden_size = weight.shape
        start, end = split_range_for_rank(
            vocab_size,
            runtime.tensor_parallel_size,
            runtime.tensor_parallel_rank,
        )
        self.vocab_start_index = start
        self.vocab_end_index = end
        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        self.padding_idx = embedding.padding_idx
        self.weight = nn.Parameter(weight[start:end].contiguous())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        local_ids = input_ids.clone() - self.vocab_start_index
        local_ids = local_ids.masked_fill(mask, 0)
        output = F.embedding(
            local_ids,
            self.weight,
            padding_idx=self.padding_idx,
        )
        output = output.masked_fill(mask.unsqueeze(-1), 0.0)
        return _all_reduce_sum(output, self.runtime.tp_group)

    def full_weight(self) -> torch.Tensor:
        return _all_gather_cat(self.weight.data, dim=0, group=self.runtime.tp_group)


class ColumnParallelLinear(nn.Module):
    """Column parallel linear implementation."""

    def __init__(
        self,
        linear: nn.Linear,
        runtime: TrainParallelRuntime,
        *,
        gather_output: bool = True,
    ):
        super().__init__()
        self.runtime = runtime
        self.gather_output = gather_output
        out_features, in_features = linear.weight.shape
        start, end = split_range_for_rank(
            out_features,
            runtime.tensor_parallel_size,
            runtime.tensor_parallel_rank,
        )
        self.out_features = out_features
        self.in_features = in_features
        self.output_start = start
        self.output_end = end
        self.weight = nn.Parameter(linear.weight.detach().clone()[start:end].contiguous())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone()[start:end].contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = _all_gather_cat(out, dim=-1, group=self.runtime.tp_group)
        return out

    def full_weight(self) -> torch.Tensor:
        return _all_gather_cat(self.weight.data, dim=0, group=self.runtime.tp_group)

    def full_bias(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return _all_gather_cat(self.bias.data, dim=0, group=self.runtime.tp_group)


class RowParallelLinear(nn.Module):
    """Row parallel linear implementation."""

    def __init__(
        self,
        linear: nn.Linear,
        runtime: TrainParallelRuntime,
    ):
        super().__init__()
        self.runtime = runtime
        out_features, in_features = linear.weight.shape
        start, end = split_range_for_rank(
            in_features,
            runtime.tensor_parallel_size,
            runtime.tensor_parallel_rank,
        )
        self.out_features = out_features
        self.in_features = in_features
        self.input_start = start
        self.input_end = end
        self.weight = nn.Parameter(linear.weight.detach().clone()[:, start:end].contiguous())
        self.bias = (
            nn.Parameter(linear.bias.detach().clone())
            if linear.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_x = x[..., self.input_start:self.input_end]
        out = F.linear(local_x, self.weight, None)
        out = _all_reduce_sum(out, self.runtime.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out

    def full_weight(self) -> torch.Tensor:
        return _all_gather_cat(self.weight.data, dim=1, group=self.runtime.tp_group)

    def full_bias(self) -> torch.Tensor | None:
        return self.bias.data if self.bias is not None else None


class VocabParallelLMHead(nn.Module):
    """Vocab parallel l m head implementation."""

    def __init__(self, linear: nn.Linear, runtime: TrainParallelRuntime):
        super().__init__()
        self.runtime = runtime
        vocab_size, hidden_size = linear.weight.shape
        start, end = split_range_for_rank(
            vocab_size,
            runtime.tensor_parallel_size,
            runtime.tensor_parallel_rank,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_start_index = start
        self.vocab_end_index = end
        self.weight = nn.Parameter(linear.weight.detach().clone()[start:end].contiguous())
        self.bias = (
            nn.Parameter(linear.bias.detach().clone()[start:end].contiguous())
            if linear.bias is not None
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)

    def full_weight(self) -> torch.Tensor:
        return _all_gather_cat(self.weight.data, dim=0, group=self.runtime.tp_group)

    def full_bias(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return _all_gather_cat(self.bias.data, dim=0, group=self.runtime.tp_group)


def vocab_parallel_token_logprobs(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    runtime: TrainParallelRuntime,
) -> torch.Tensor:
    """Vocab parallel token logprobs."""
    vocab_start = split_range_for_rank(
        local_logits.shape[-1] * runtime.tensor_parallel_size,
        runtime.tensor_parallel_size,
        runtime.tensor_parallel_rank,
    )[0]
    vocab_end = vocab_start + local_logits.shape[-1]

    local_max = local_logits.max(dim=-1).values
    global_max = _all_reduce_max(local_max.clone(), runtime.tp_group)
    exp_logits = torch.exp(local_logits - global_max.unsqueeze(-1))
    exp_sum = exp_logits.sum(dim=-1)
    global_exp_sum = _all_reduce_sum(exp_sum.clone(), runtime.tp_group)

    local_mask = (target_ids >= vocab_start) & (target_ids < vocab_end)
    local_target = (target_ids - vocab_start).masked_fill(~local_mask, 0)
    local_selected = torch.gather(
        local_logits,
        -1,
        local_target.unsqueeze(-1),
    ).squeeze(-1)
    local_selected = local_selected.masked_fill(~local_mask, 0.0)
    global_selected = _all_reduce_sum(local_selected.clone(), runtime.tp_group)
    return global_selected - torch.log(global_exp_sum) - global_max


def _patch_qwen_mlp_for_tp(
    module: nn.Module,
    runtime: TrainParallelRuntime,
    visited: set[int] | None = None,
):
    """Patch qwen mlp for tp."""
    if visited is None:
        visited = set()
    if id(module) in visited:
        return
    visited.add(id(module))

    if hasattr(module, "gate_proj") and isinstance(module.gate_proj, nn.Linear):
        module.gate_proj = ColumnParallelLinear(
            module.gate_proj, runtime, gather_output=True
        )
    if hasattr(module, "up_proj") and isinstance(module.up_proj, nn.Linear):
        module.up_proj = ColumnParallelLinear(
            module.up_proj, runtime, gather_output=True
        )
    if hasattr(module, "down_proj") and isinstance(module.down_proj, nn.Linear):
        module.down_proj = RowParallelLinear(module.down_proj, runtime)

    for child in module.children():
        _patch_qwen_mlp_for_tp(child, runtime, visited)


def patch_qwen_layer_for_tp(layer: nn.Module, runtime: TrainParallelRuntime):
    """Patch qwen layer for tp."""
    if runtime.tensor_parallel_size == 1:
        return layer

    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        if hasattr(attn, "q_proj"):
            attn.q_proj = ColumnParallelLinear(attn.q_proj, runtime, gather_output=True)
        if hasattr(attn, "k_proj"):
            attn.k_proj = ColumnParallelLinear(attn.k_proj, runtime, gather_output=True)
        if hasattr(attn, "v_proj"):
            attn.v_proj = ColumnParallelLinear(attn.v_proj, runtime, gather_output=True)
        if hasattr(attn, "o_proj"):
            attn.o_proj = RowParallelLinear(attn.o_proj, runtime)

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        _patch_qwen_mlp_for_tp(mlp, runtime)
    return layer


def build_stage_spec(num_layers: int, runtime: TrainParallelRuntime) -> StageSpec:
    """Build stage spec."""
    layer_ranges = divide_evenly(num_layers, runtime.pipeline_parallel_size)
    start, end = layer_ranges[runtime.pipeline_parallel_rank]
    return StageSpec(
        stage_id=runtime.pipeline_parallel_rank,
        layer_start=start,
        layer_end=end,
        is_first_stage=runtime.is_pipeline_first_stage,
        is_last_stage=runtime.is_pipeline_last_stage,
    )


class QwenPipelineStage(nn.Module):
    """Qwen pipeline stage implementation."""

    def __init__(
        self,
        hf_model: nn.Module,
        runtime: TrainParallelRuntime,
    ):
        super().__init__()
        self.runtime = runtime
        self.full_model_config = hf_model.config
        self.stage_spec = build_stage_spec(len(hf_model.model.layers), runtime)

        self.embed_tokens = None
        self.norm = None
        self.lm_head = None

        if self.stage_spec.is_first_stage:
            self.embed_tokens = (
                VocabParallelEmbedding(hf_model.model.embed_tokens, runtime)
                if runtime.tensor_parallel_size > 1
                else copy.deepcopy(hf_model.model.embed_tokens)
            )

        self.layers = nn.ModuleList(
            [
                patch_qwen_layer_for_tp(copy.deepcopy(hf_model.model.layers[idx]), runtime)
                for idx in self.stage_spec.layer_indices
            ]
        )

        if self.stage_spec.is_last_stage:
            self.norm = copy.deepcopy(hf_model.model.norm)
            self.lm_head = (
                VocabParallelLMHead(hf_model.lm_head, runtime)
                if runtime.tensor_parallel_size > 1
                else copy.deepcopy(hf_model.lm_head)
            )

        self.rotary_emb = copy.deepcopy(hf_model.model.rotary_emb)

    def _build_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids

    def _build_causal_mask(
        self,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        del cache_position
        batch_size, sequence_length = attention_mask.shape
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full(
            (sequence_length, sequence_length),
            min_dtype,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.view(1, 1, sequence_length, sequence_length)
        causal_mask = causal_mask.expand(batch_size, 1, -1, -1).clone()
        padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return causal_mask.masked_fill(~padding_mask, min_dtype)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ):
        if attention_mask is None:
            raise ValueError("attention_mask cannot be None")

        if self.stage_spec.is_first_stage:
            if input_ids is None:
                raise ValueError("The first stage requires input_ids")
            hidden_states = self.embed_tokens(input_ids)
        elif hidden_states is None:
            raise ValueError("Non-first stages require hidden_states")

        position_ids = self._build_position_ids(
            input_ids if input_ids is not None else attention_mask.long(),
            attention_mask,
        )
        cache_position = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        )
        causal_mask = self._build_causal_mask(
            attention_mask=attention_mask,
            inputs_embeds=hidden_states,
            cache_position=cache_position,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if not self.stage_spec.is_last_stage:
            return hidden_states

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def export_stage_state_dict_for_hf(stage: QwenPipelineStage) -> dict[str, torch.Tensor]:
    """Export stage state dict for hf."""
    state: dict[str, torch.Tensor] = {}
    spec = stage.stage_spec

    if spec.is_first_stage:
        if isinstance(stage.embed_tokens, VocabParallelEmbedding):
            state["model.embed_tokens.weight"] = stage.embed_tokens.full_weight().cpu()
        else:
            state["model.embed_tokens.weight"] = stage.embed_tokens.weight.detach().cpu()

    for local_idx, layer_idx in enumerate(spec.layer_indices):
        layer = stage.layers[local_idx]
        prefix = f"model.layers.{layer_idx}"
        layer_state = layer.state_dict()
        for name, tensor in layer_state.items():
            submodule_name, param_name = name.rsplit(".", 1)
            module = layer.get_submodule(submodule_name)
            full_key = f"{prefix}.{name}"
            if isinstance(module, ColumnParallelLinear):
                if param_name == "weight":
                    state[full_key] = module.full_weight().cpu()
                elif param_name == "bias" and module.full_bias() is not None:
                    state[full_key] = module.full_bias().cpu()
            elif isinstance(module, RowParallelLinear):
                if param_name == "weight":
                    state[full_key] = module.full_weight().cpu()
                elif param_name == "bias" and module.full_bias() is not None:
                    state[full_key] = module.full_bias().cpu()
            else:
                state[full_key] = tensor.detach().cpu()

    if spec.is_last_stage:
        norm_prefix = "model.norm"
        for name, tensor in stage.norm.state_dict().items():
            state[f"{norm_prefix}.{name}"] = tensor.detach().cpu()
        if isinstance(stage.lm_head, VocabParallelLMHead):
            state["lm_head.weight"] = stage.lm_head.full_weight().cpu()
            if stage.lm_head.full_bias() is not None:
                state["lm_head.bias"] = stage.lm_head.full_bias().cpu()
        else:
            for name, tensor in stage.lm_head.state_dict().items():
                state[f"lm_head.{name}"] = tensor.detach().cpu()

    return state

"""PyTorch SDPA attention adapter for Megatron-Core local layer specs."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule


class TorchSequenceParallelNorm:
    """PyTorch LayerNorm/RMSNorm with MCore sequence-parallel metadata."""

    def __new__(
        cls,
        config,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ):
        del cls, persist_layer_norm, normalization
        if zero_centered_gamma or config.layernorm_zero_centered_gamma:
            raise ValueError(
                "zero_centered_gamma is not supported by PyTorch norm"
            )
        if config.persist_layer_norm:
            raise ValueError(
                "persist_layer_norm is not supported by PyTorch norm"
            )
        if config.memory_efficient_layer_norm:
            raise ValueError(
                "memory_efficient_layer_norm is not supported by PyTorch norm"
            )

        if config.normalization == "LayerNorm":
            norm = torch.nn.LayerNorm(hidden_size, eps=eps)
        elif config.normalization == "RMSNorm":
            norm = torch.nn.RMSNorm(hidden_size, eps=eps)
        else:
            raise ValueError(
                "PyTorch sequence-parallel norm supports LayerNorm and RMSNorm"
            )
        for parameter in norm.parameters():
            setattr(parameter, "sequence_parallel", config.sequence_parallel)
        return norm


def apply_megatron_core_014_compatibility() -> None:
    """Patch MCore 0.14 compatibility gaps for the no-TE code path."""
    import os

    from packaging.version import Version as PkgVersion
    from megatron.core import optimizer as optimizer_pkg
    from megatron.core import utils as core_utils
    from megatron.core.optimizer import optimizer_config
    from megatron.core.transformer.moe import moe_utils

    original_is_te_min_version = optimizer_config.is_te_min_version
    if not getattr(
        original_is_te_min_version,
        "_rl_framework_missing_te_compat",
        False,
    ):
        def is_te_min_version(version):
            te_version = core_utils.get_te_version()
            if te_version is None:
                return False
            return te_version >= PkgVersion(version)

        is_te_min_version._rl_framework_missing_te_compat = True
        optimizer_config.is_te_min_version = is_te_min_version
        core_utils.is_te_min_version = is_te_min_version
        if hasattr(optimizer_pkg, "optimizer_config"):
            optimizer_pkg.optimizer_config.is_te_min_version = is_te_min_version

    if not hasattr(moe_utils, "te_general_gemm"):
        moe_utils.te_general_gemm = None

    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )

    original_offload_groups = (
        HybridDeviceOptimizer._get_sub_optimizer_param_groups
    )
    if not getattr(
        original_offload_groups,
        "_rl_framework_pageable_cpu_compat",
        False,
    ):
        def _get_sub_optimizer_param_groups(self, offload_fraction):
            params = [
                param
                for group in self.param_groups
                for param in group["params"]
            ]
            gpu_numel = sum(
                param.numel() for param in params if param.is_cuda
            )
            offload_threshold = gpu_numel * offload_fraction
            offloaded_numel = 0
            cpu_param_groups = []
            gpu_param_groups = []
            gpu_params_map_cpu_copy = {}
            cpu_copys_map_gpu_param = {}
            param_to_fp32_param = {}

            for group in self.param_groups:
                gpu_group = group.copy()
                cpu_group = group.copy()
                gpu_group["params"] = []
                cpu_group["params"] = []
                for original_param in group["params"]:
                    param = original_param
                    cpu_copy = False
                    if (
                        offloaded_numel < offload_threshold
                        and param.is_cuda
                    ):
                        # MCore 0.14 ignores pin_cpu_params and first clones on
                        # GPU. Copy directly to host so full offload works for
                        # models whose weights already occupy most of the GPU.
                        param = param.detach().to("cpu", copy=True)
                        if self.pin_cpu_params:
                            param = param.pin_memory()
                        offloaded_numel += param.numel()
                        cpu_copy = True
                    if self.param_update_in_fp32 and param.dtype != torch.float32:
                        param = param.detach().float()
                        param_to_fp32_param[original_param] = param
                    if cpu_copy:
                        gpu_params_map_cpu_copy[original_param] = param
                        cpu_copys_map_gpu_param[param] = original_param
                    if param.is_cuda:
                        gpu_group["params"].append(param)
                    else:
                        cpu_group["params"].append(param)
                if gpu_group["params"]:
                    gpu_param_groups.append(gpu_group)
                if cpu_group["params"]:
                    cpu_param_groups.append(cpu_group)

            return (
                cpu_param_groups,
                gpu_param_groups,
                gpu_params_map_cpu_copy,
                cpu_copys_map_gpu_param,
                param_to_fp32_param,
            )

        _get_sub_optimizer_param_groups._rl_framework_pageable_cpu_compat = (
            True
        )
        HybridDeviceOptimizer._get_sub_optimizer_param_groups = (
            _get_sub_optimizer_param_groups
        )

    original_update_fp32 = (
        HybridDeviceOptimizer._update_fp32_params_by_new_state
    )
    if not getattr(
        original_update_fp32,
        "_rl_framework_dist_optimizer_compat",
        False,
    ):
        def _update_fp32_params_by_new_state(self):
            if not self.param_update_in_fp32:
                return
            for original_param, state in self.state.items():
                master_param = state.get("master_param")
                if master_param is None:
                    continue
                target_param = self.param_to_fp32_param.get(original_param)
                if target_param is None:
                    # DistributedOptimizer already supplies FP32 main params,
                    # so HybridDeviceOptimizer has no bf16-to-fp32 mapping.
                    target_param = self.param_to_inner_param.get(
                        original_param
                    )
                if target_param is not None:
                    target_param.data.copy_(master_param)

        _update_fp32_params_by_new_state._rl_framework_dist_optimizer_compat = (
            True
        )
        HybridDeviceOptimizer._update_fp32_params_by_new_state = (
            _update_fp32_params_by_new_state
        )

    from megatron.core.dist_checkpointing.strategies.common import (
        COMMON_STATE_FNAME,
        TorchCommonLoadStrategy,
    )

    original_load_common = TorchCommonLoadStrategy.load_common
    if not getattr(
        original_load_common,
        "_rl_framework_torch_26_compat",
        False,
    ):
        def load_common(self, checkpoint_dir):
            load_path = os.path.join(checkpoint_dir, COMMON_STATE_FNAME)
            try:
                return torch.load(
                    load_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except FileNotFoundError:
                return original_load_common(self, checkpoint_dir)

        load_common._rl_framework_torch_26_compat = True
        TorchCommonLoadStrategy.load_common = load_common

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    # Megatron-Core's native-Adam DistributedOptimizer calls state_dict() while
    # constructing a distributed-checkpoint load template.  A freshly created
    # optimizer has no per-parameter state yet, so upstream asserts that its
    # empty set of Adam steps has length one.  During an elastic DP resize this
    # is exactly the normal path.  Emit the same optimizer metadata with a
    # step-zero placeholder so MCore can allocate the incoming sharded state.
    original_state_dict = DistributedOptimizer.state_dict
    if not getattr(original_state_dict, "_rl_framework_empty_state_compat", False):
        def state_dict(self):
            inner_state_dict = self.optimizer.state_dict()
            if inner_state_dict.get("state"):
                return original_state_dict(self)

            optimizer_state = {
                key: value
                for key, value in inner_state_dict.items()
                if key != "state"
            }
            for param_group in optimizer_state.get("param_groups", []):
                param_group.pop("params", None)
                # Native PyTorch Adam expects the global iteration in every
                # group; its real per-parameter state is populated by load.
                param_group.setdefault("step", 0)
            result = {"optimizer": optimizer_state}
            if self.grad_scaler:
                result["grad_scaler"] = self.grad_scaler.state_dict()
            return result

        state_dict._rl_framework_empty_state_compat = True
        DistributedOptimizer.state_dict = state_dict

    original_set_state = (
        DistributedOptimizer._set_main_param_and_optimizer_states
    )
    if not getattr(original_set_state, "_rl_framework_step_compat", False):
        def _set_main_param_and_optimizer_states(self, model_param, tensors):
            # fully_sharded_model_space stores Torch Adam's scalar step in
            # param-group metadata, not in each parameter tensor shard.
            if (
                "step" not in tensors
                and not self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            ):
                group_index, group_order = self.model_param_group_index_map[
                    model_param
                ]
                main_param = self.optimizer.param_groups[group_index]["params"][
                    group_order
                ]
                optimizer_state = self.optimizer.state[main_param]
                step = optimizer_state.pop("step", None)
                try:
                    return original_set_state(self, model_param, tensors)
                finally:
                    if step is not None:
                        optimizer_state["step"] = step
            return original_set_state(self, model_param, tensors)

        _set_main_param_and_optimizer_states._rl_framework_step_compat = True
        DistributedOptimizer._set_main_param_and_optimizer_states = (
            _set_main_param_and_optimizer_states
        )

    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
    from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
    from RL_Framework.engine.megatron_core_moe_mapping import (
        stacked_expert_mapping_class,
    )

    original_mapping_registry = Qwen3MoEBridge.mapping_registry
    original_provider_bridge = Qwen3MoEBridge.provider_bridge
    original_dense_mapping_registry = Qwen3Bridge.mapping_registry

    if not getattr(
        original_provider_bridge,
        "_rl_framework_grouped_moe_compat",
        False,
    ):
        def provider_bridge(self, hf_pretrained):
            config = hf_pretrained.config
            num_experts = getattr(
                config,
                "num_experts",
                getattr(config, "num_local_experts", None),
            )
            if num_experts is None:
                raise ValueError("Qwen3 MoE config does not define num_experts")
            # AutoBridge resolves a fresh registered bridge instance for each
            # property access, so provider metadata must live on the class.
            type(self)._rl_framework_num_experts = int(num_experts)
            return original_provider_bridge(self, hf_pretrained)

        provider_bridge._rl_framework_grouped_moe_compat = True
        Qwen3MoEBridge.provider_bridge = provider_bridge

    if not getattr(original_mapping_registry, "_rl_framework_moe_compat", False):
        def mapping_registry(self):
            registry = original_mapping_registry(self)
            num_experts = getattr(
                type(self),
                "_rl_framework_num_experts",
                None,
            )
            if num_experts is None:
                raise RuntimeError(
                    "Qwen3 MoE provider must be configured before mappings"
                )
            StackedExpertMapping = stacked_expert_mapping_class()
            return MegatronMappingRegistry(
                StackedExpertMapping(
                    "decoder.layers.*.mlp.experts.weight1",
                    kind="weight1",
                    num_experts=num_experts,
                ),
                StackedExpertMapping(
                    "decoder.layers.*.mlp.experts.weight2",
                    kind="weight2",
                    num_experts=num_experts,
                ),
                AutoMapping(
                    megatron_param=(
                        "decoder.layers.*.input_layernorm.weight"
                    ),
                    hf_param="model.layers.*.input_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param=(
                        "decoder.layers.*.pre_mlp_layernorm.weight"
                    ),
                    hf_param="model.layers.*.post_attention_layernorm.weight",
                ),
                *registry.get_all_mappings(),
            )

        mapping_registry._rl_framework_moe_compat = True
        Qwen3MoEBridge.mapping_registry = mapping_registry

    if not getattr(
        original_dense_mapping_registry,
        "_rl_framework_dense_qwen3_compat",
        False,
    ):
        def dense_mapping_registry(self):
            registry = original_dense_mapping_registry(self)
            return MegatronMappingRegistry(
                AutoMapping(
                    megatron_param=(
                        "decoder.layers.*.input_layernorm.weight"
                    ),
                    hf_param="model.layers.*.input_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param=(
                        "decoder.layers.*.pre_mlp_layernorm.weight"
                    ),
                    hf_param="model.layers.*.post_attention_layernorm.weight",
                ),
                *registry.get_all_mappings(),
            )

        dense_mapping_registry._rl_framework_dense_qwen3_compat = True
        Qwen3Bridge.mapping_registry = dense_mapping_registry


class TorchSDPADotProductAttention(MegatronModule):
    """MCore core-attention interface backed by PyTorch Flash SDPA."""

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        model_comm_pgs: Any = None,
    ):
        del layer_number, attention_type, cp_comm_type, model_comm_pgs
        super().__init__(config=config)
        if config.context_parallel_size != 1:
            raise ValueError(
                "Torch SDPA attention requires context_parallel_size=1"
            )
        self.attn_mask_type = attn_mask_type
        self.dropout_p = (
            config.attention_dropout
            if attention_dropout is None
            else attention_dropout
        )
        head_dim = config.kv_channels
        self.softmax_scale = (
            1.0 / math.sqrt(head_dim)
            if softmax_scale is None
            else softmax_scale
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: Any = None,
    ) -> torch.Tensor:
        if attention_bias is not None:
            raise ValueError("Torch SDPA adapter does not support attention_bias")
        if packed_seq_params is not None:
            raise ValueError("Torch SDPA adapter does not support packed sequences")

        # MCore uses [sequence, batch, heads, head_dim]; SDPA uses
        # [batch, heads, sequence, head_dim].
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        mask_type = attn_mask_type or self.attn_mask_type
        is_causal = (
            attention_mask is None
            and mask_type == AttnMaskType.causal
        )
        sdpa_mask = attention_mask
        if sdpa_mask is not None and sdpa_mask.dtype == torch.bool:
            # MCore marks blocked positions as True; SDPA marks allowed
            # positions as True.
            sdpa_mask = ~sdpa_mask

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.softmax_scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        context = context.permute(2, 0, 1, 3).contiguous()
        return context.view(context.shape[0], context.shape[1], -1)


def torch_sdpa_layer_spec(config):
    """Return a local MCore block spec with Flash SDPA and PyTorch norms."""
    from megatron.bridge.models.gpt_provider import local_layer_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import (
        TransformerBlock,
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )

    layer_spec = local_layer_spec(config)
    layer_spec.submodules.input_layernorm = TorchSequenceParallelNorm
    layer_spec.submodules.pre_mlp_layernorm = TorchSequenceParallelNorm
    attention_submodules = layer_spec.submodules.self_attention.submodules
    attention_submodules.q_layernorm = TorchSequenceParallelNorm
    attention_submodules.k_layernorm = TorchSequenceParallelNorm
    attention_submodules.core_attention = TorchSDPADotProductAttention

    return ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(
            layer_specs=[layer_spec] * get_num_layers_to_build(config),
            layer_norm=TorchSequenceParallelNorm,
        ),
    )

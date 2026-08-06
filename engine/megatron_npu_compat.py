"""NPU adaptations for the CUDA-assuming Megatron-Core 0.14 runtime.

Only the standard training/model-parallel path is adapted here. CUDA graphs,
CUDA streams, NVTX and CUDA-only fused kernels stay disabled by configuration.
"""

from __future__ import annotations

import functools
import inspect
import sys
import types

import torch


def _rewrite_cuda_constants(code: types.CodeType) -> types.CodeType:
    """Rewrite device/type literals in Megatron functions to native NPU names."""

    def rewrite(value):
        if isinstance(value, types.CodeType):
            return _rewrite_cuda_constants(value)
        if isinstance(value, str):
            if value == "cuda":
                return "npu"
            if value.startswith("cuda:"):
                return "npu:" + value[5:]
            if "torch.cuda." in value:
                return value.replace("torch.cuda.", "torch.npu.")
            return value
        if isinstance(value, tuple):
            return tuple(rewrite(item) for item in value)
        if isinstance(value, frozenset):
            return frozenset(rewrite(item) for item in value)
        return value

    changed = False
    constants = []
    for value in code.co_consts:
        replacement = rewrite(value)
        changed = changed or replacement != value
        constants.append(replacement)
    return code.replace(co_consts=tuple(constants)) if changed else code


def _rewrite_module_cuda_literals(module) -> None:
    """Patch only functions owned by a selected Megatron module."""
    module_name = module.__name__
    for value in vars(module).values():
        candidates = []
        if inspect.isfunction(value) and value.__module__ == module_name:
            candidates.append(value)
        elif inspect.isclass(value) and value.__module__ == module_name:
            for member in vars(value).values():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    for accessor in (member.fget, member.fset, member.fdel):
                        if accessor is not None:
                            candidates.append(accessor)
                    continue
                if inspect.isfunction(member):
                    candidates.append(member)
        for function in candidates:
            function.__code__ = _rewrite_cuda_constants(function.__code__)


def apply_megatron_npu_compatibility() -> None:
    """Adapt MCore 0.14's CUDA-shaped Python paths to native torch-npu."""
    import torch_npu  # noqa: F401
    from megatron.core.tensor_parallel import random as tp_random

    def get_rng_state(device="npu", clone: bool = False, graph_safe: bool = False):
        if graph_safe:
            raise RuntimeError("CUDA graph-safe RNG is unsupported on Ascend")
        state = torch.npu.get_rng_state(device=device)
        return state.clone() if clone else state

    def set_rng_state(
        new_state,
        device: int | str | torch.device = -1,
        graph_safe: bool = False,
    ):
        if graph_safe:
            raise RuntimeError("CUDA graph-safe RNG is unsupported on Ascend")
        if device == -1:
            device = torch.npu.current_device()
        torch.npu.set_rng_state(new_state, device=device)

    tp_random._get_cuda_rng_state = get_rng_state
    tp_random._set_cuda_rng_state = set_rng_state
    torch.cuda.set_device = torch.npu.set_device
    torch.cuda.get_rng_state = torch.npu.get_rng_state
    torch.cuda.set_rng_state = torch.npu.set_rng_state
    torch.cuda.manual_seed = torch.npu.manual_seed
    torch.cuda.manual_seed_all = torch.npu.manual_seed_all
    torch.cuda.initial_seed = torch.npu.initial_seed
    torch.cuda.is_available = torch.npu.is_available
    torch.cuda.device_count = torch.npu.device_count
    torch.cuda.current_device = lambda: torch.device(
        "npu", torch.npu.current_device()
    )
    torch.cuda.empty_cache = torch.npu.empty_cache
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.current_stream = torch.npu.current_stream
    torch.cuda.default_stream = torch.npu.default_stream
    torch.cuda.stream = torch.npu.stream
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.Event = torch.npu.Event
    torch.cuda.memory_allocated = torch.npu.memory_allocated
    torch.cuda.memory_reserved = torch.npu.memory_reserved
    torch.cuda.get_device_properties = torch.npu.get_device_properties

    def translate_device(device):
        if isinstance(device, str):
            if device == "cuda":
                return "npu"
            if device.startswith("cuda:"):
                return "npu:" + device[5:]
        if isinstance(device, torch.device) and device.type == "cuda":
            return torch.device("npu", device.index)
        return device

    def wrap_factory(factory):
        if getattr(factory, "_rl_framework_npu_factory", False):
            return factory

        @functools.wraps(factory)
        def wrapped(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = translate_device(kwargs["device"])
            return factory(*args, **kwargs)

        wrapped._rl_framework_npu_factory = True
        return wrapped

    for factory_name in (
        "tensor",
        "as_tensor",
        "scalar_tensor",
        "zeros",
        "ones",
        "empty",
        "full",
        "arange",
        "rand",
        "randn",
        "randint",
        "linspace",
        "logspace",
        "eye",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
        "rand_like",
        "randn_like",
    ):
        if hasattr(torch, factory_name):
            setattr(torch, factory_name, wrap_factory(getattr(torch, factory_name)))

    def module_cuda(module, device=None):
        return module.npu(device=device)

    def tensor_cuda(
        tensor,
        device=None,
        non_blocking: bool = False,
        memory_format=torch.preserve_format,
    ):
        return tensor.npu(
            device=device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    torch.nn.Module.cuda = module_cuda
    torch.Tensor.cuda = tensor_cuda

    from megatron.core import utils as core_utils
    from megatron.core.distributed import distributed_data_parallel
    from megatron.core.distributed import param_and_grad_buffer
    from megatron.core.optimizer import (
        clip_grads,
        distrib_optimizer,
        grad_scaler,
        optimizer,
    )
    from megatron.core.pipeline_parallel import p2p_communication, schedules

    for module in (
        optimizer,
        grad_scaler,
        clip_grads,
        distrib_optimizer,
        distributed_data_parallel,
        param_and_grad_buffer,
        p2p_communication,
        schedules,
        core_utils,
    ):
        _rewrite_module_cuda_literals(module)

    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("megatron.core"):
            continue
        try:
            _rewrite_module_cuda_literals(module)
        except (AttributeError, TypeError, ValueError):
            continue

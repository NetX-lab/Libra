"""Bound temporary storage in the native PyTorch Adam fallback."""

from functools import wraps

import torch
from torch.optim import optimizer as optimizer_module


def _native_streaming_eligible(optimizer):
    seen = set()
    if (getattr(optimizer, '_optimizer_step_pre_hooks', {})
            or getattr(optimizer, '_optimizer_step_post_hooks', {})
            or getattr(optimizer_module, '_global_optimizer_pre_hooks', {})
            or getattr(optimizer_module, '_global_optimizer_post_hooks', {})):
        return False
    for group in optimizer.param_groups:
        if (group.get('foreach') is not False or group.get('fused')
                or group.get('capturable') or group.get('differentiable')):
            return False
        for parameter in group['params']:
            if id(parameter) in seen or (parameter.grad is not None and parameter.grad.is_sparse):
                return False
            seen.add(id(parameter))
    return True


def _materialize_gradient(parameter, state):
    parameter.grad = state['sources'][parameter].float()
    coefficient = state['coefficient']
    if coefficient is not None and coefficient < 1.0:
        # Preserve local_multi_tensor_scale's src * scale followed by copy_.
        state['scale'](None, [[parameter.grad], [parameter.grad]], coefficient)


def _stream_main_gradient_step(optimizer):
    """Materialize deferred gradients and release each copy after its Adam update."""
    if getattr(optimizer, '_libra_streamed_main_gradient_step', False):
        return
    original_step = optimizer.step

    @wraps(original_step)
    def step(*args, **kwargs):
        # Closures, graph capture and custom hooks can observe the entire
        # optimizer call. Preserve their native behavior instead of splitting it.
        groups = optimizer.param_groups
        state = getattr(optimizer, '_libra_bf16_gradient_state', None)
        streaming = state is not None and state['active']
        has_gradient = streaming or any(p.grad is not None for g in groups for p in g['params'])
        eligible = not args and not kwargs and _native_streaming_eligible(optimizer)
        if not eligible or not has_gradient:
            if streaming:
                for parameter in state['sources']:
                    _materialize_gradient(parameter, state)
                state['active'] = False
                state['sources'].clear()
            return original_step(*args, **kwargs)

        try:
            for group in groups:
                for parameter in group['params']:
                    if streaming and parameter in state['sources']:
                        _materialize_gradient(parameter, state)
                    if parameter.grad is None:
                        continue
                    # Keep the original parameter/state objects and every group
                    # hyperparameter. MCore's global norm determines clipping;
                    # deferred gradients receive that FP32 clip just above.
                    optimizer.param_groups = [dict(group, params=[parameter])]
                    original_step()
                    parameter.grad = None
            if streaming:
                state['active'] = False
                state['sources'].clear()
        finally:
            # MCore's group-range maps and checkpoint layout refer to these
            # original group objects, including when the native update fails.
            optimizer.param_groups = groups

    optimizer.step = step
    optimizer._libra_streamed_main_gradient_step = True


def _install_mcore_bf16_streaming(root, wrappers):
    """Defer MCore's FP32 copies, only with its exact local norm/scale backend."""
    try:
        from megatron.core import utils
        from megatron.core.optimizer import clip_grads
        from megatron.core.optimizer import optimizer as mcore_optimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        return
    if not (
        clip_grads.multi_tensor_applier is utils.local_multi_tensor_applier
        and clip_grads.l2_norm_impl is utils.local_multi_tensor_l2_norm
        and clip_grads.multi_tensor_scale_impl is utils.local_multi_tensor_scale
    ):
        return

    def eligible(wrapper):
        config = getattr(wrapper, 'config', None)
        return (
            type(wrapper) is DistributedOptimizer
            and type(wrapper.optimizer) in (torch.optim.Adam, torch.optim.AdamW)
            and _native_streaming_eligible(wrapper.optimizer)
            and bool(getattr(config, 'bf16', False))
            and getattr(wrapper, 'grad_scaler', None) is None
            and not getattr(config, 'log_num_zeros_in_grad', False)
            and not getattr(getattr(root, 'config', None), 'log_num_zeros_in_grad', False)
            and not getattr(config, 'use_precision_aware_optimizer', False)
            and not getattr(config, 'use_precision_aware_optimizer_no_fp8_or_ds_fp8', False)
            and not getattr(config, 'optimizer_cpu_offload', False)
            and not getattr(getattr(wrapper, 'ddp_config', None), 'use_megatron_fsdp', False)
            and hasattr(wrapper, 'model_float16_groups')
        )

    for wrapper in wrappers:
        if eligible(wrapper) and not getattr(wrapper, '_libra_streamed_bf16_main_gradients', False):
            _install_mcore_bf16_child(wrapper, eligible, clip_grads, mcore_optimizer)
    # ChainedOptimizer computes a common norm before clipping any child. Keep
    # that original call and its collectives, then share its exact coefficient.
    for wrapper in wrappers:
        if (any(getattr(child, '_libra_streamed_bf16_main_gradients', False) for child in wrappers)
                and getattr(wrapper, 'chained_optimizers', None) is not None
                and not getattr(wrapper, '_libra_streamed_bf16_norm', False)):
            _record_gradient_norm(wrapper)


def _record_gradient_norm(wrapper):
    original_norm = wrapper.get_grad_norm

    @wraps(original_norm)
    def norm():
        value = original_norm()
        pending = [wrapper]
        while pending:
            child = pending.pop()
            children = getattr(child, 'chained_optimizers', None)
            if children is not None:
                pending.extend(children)
                continue
            state = getattr(getattr(child, 'optimizer', None), '_libra_bf16_gradient_state', None)
            if state is not None and state['active']:
                state['coefficient'] = (
                    child.config.clip_grad / (value + 1.0e-6)
                    if child.config.clip_grad > 0 else None
                )
        return value

    wrapper.get_grad_norm = norm
    wrapper._libra_streamed_bf16_norm = True


def _install_mcore_bf16_child(wrapper, eligible, clip_grads, mcore_optimizer):
    original_copy = wrapper._copy_model_grads_to_main_grads
    original_norm_grads = wrapper.get_main_grads_for_grad_norm
    original_clip = wrapper.clip_grad_norm
    state = dict(active=False, sources={}, coefficient=None,
        scale=lambda flag, tensors, coefficient: clip_grads.multi_tensor_applier(
            clip_grads.multi_tensor_scale_impl, flag, tensors, coefficient))
    wrapper.optimizer._libra_bf16_gradient_state = state

    @torch.no_grad()
    def copy_views():
        state.update(active=False, sources={}, coefficient=None)
        if not eligible(wrapper) or wrapper.is_stub_optimizer:
            return original_copy()
        sources = {}
        for model_groups, main_groups in (
            (wrapper.model_float16_groups, wrapper.shard_fp32_from_float16_groups),
            (wrapper.model_fp32_groups, wrapper.shard_fp32_groups),
        ):
            for models, masters in zip(model_groups, main_groups):
                for model, master in zip(models, masters):
                    interval = wrapper._get_model_param_range_map(model)['param']
                    assert interval.size == master.nelement()
                    source = model.main_grad.view(-1)[interval.start:interval.end]
                    if source.dtype not in (torch.bfloat16, torch.float32):
                        return original_copy()
                    sources[master] = source
        for master in sources:
            master.grad = None
        state.update(active=True, sources=sources)

    @torch.no_grad()
    def norm_grads():
        if not state['active']:
            return original_norm_grads()
        # The exact local fallback first norms every FP32 tensor, then norms
        # torch.tensor(the scalar norms). Norming a nonnegative scalar again
        # is exact, allowing the original aggregate/collective to remain intact.
        return [torch.norm(state['sources'][parameter].float())
            for parameter in wrapper.get_parameters()
            if parameter in state['sources']
            and mcore_optimizer.param_is_not_shared(parameter)
            and mcore_optimizer.tensor_parallel.param_is_not_tensor_parallel_duplicate(parameter)]

    @torch.no_grad()
    def clip(max_norm):
        if not state['active']:
            return original_clip(max_norm)
        value = wrapper.get_grad_norm()
        state['coefficient'] = max_norm / (value + 1.0e-6)
        return value

    wrapper._copy_model_grads_to_main_grads = copy_views
    wrapper.get_main_grads_for_grad_norm = norm_grads
    wrapper.clip_grad_norm = clip
    wrapper._libra_streamed_bf16_main_gradients = True
    _record_gradient_norm(wrapper)


def bound_default_torch_adam_memory(optimizer, *, release_main_gradients=False) -> int:
    """Disable automatic foreach selection in native Adam/AdamW leaves.

    CUDA's automatic foreach Adam path creates a full parameter-sized list of
    temporaries (for example, sqrt of every second moment at once). MCore's
    non-fused fallback can therefore exceed memory even when model, master
    parameters, gradients and Adam moments fit. The per-tensor implementation
    retains the same precision and bounds those temporaries to one parameter.

    Explicit foreach choices and fused implementations are preserved. This
    walks the MCore chained/single optimizer wrappers without accessing a
    ChainedOptimizer's single-child-only ``optimizer`` property. Return the
    number of parameter groups changed; repeated calls are idempotent.

    With release_main_gradients=True, each FP32 copy is released after its
    native Adam update. For BF16 DistributedOptimizer instances using MCore's
    exact local norm/scale fallback (without grad scaling, zero counting,
    precision-aware optimization, FSDP or CPU offload), prepare_grads also
    defers conversion: retain BF16 shard views, compute scalar FP32 norms one
    at a time, and leave the original aggregate norm/collectives in place.
    Each Adam update then materializes and clips only its own FP32 gradient.
    BF16 accumulation, FP32 clipping/Adam arithmetic and checkpoint state stay
    unchanged; existing moments no longer overlap a full FP32 gradient copy
    on the second or subsequent update. Unsupported paths retain eager copy.
    """
    pending = [optimizer]
    visited = set()
    wrappers = []
    changed = 0
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if type(current) in (torch.optim.Adam, torch.optim.AdamW):
            for group in current.param_groups:
                if group.get("foreach") is None and group.get("fused") is not True:
                    group["foreach"] = False
                    changed += 1
            if (current.defaults.get("foreach") is None
                    and current.defaults.get("fused") is not True):
                current.defaults["foreach"] = False
            if release_main_gradients:
                _stream_main_gradient_step(current)
            continue
        children = getattr(current, "chained_optimizers", None)
        wrappers.append(current)
        if children is not None:
            pending.extend(children)
        else:
            pending.append(getattr(current, "optimizer", None))
    if release_main_gradients and wrappers:
        _install_mcore_bf16_streaming(optimizer, wrappers)
    return changed

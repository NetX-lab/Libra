"""The fallback optimizer must fit without full-model foreach temporaries."""
import copy
import importlib
import weakref
from types import SimpleNamespace

import pytest
import torch

from RL_Framework.engine.optimizer_memory import bound_default_torch_adam_memory


def _parameters():
    return [torch.nn.Parameter(torch.linspace(-1., 1., size)) for size in (19, 31)]


def test_native_adamw_automatic_foreach_allocates_all_moments_but_bounded_path_does_not(monkeypatch):
    # CPU uses another automatic choice; force CUDA's foreach decision while
    # retaining the real PyTorch optimizer step and tensor implementations.
    adam_module = importlib.import_module('torch.optim.adam')
    monkeypatch.setattr(adam_module, '_default_to_fused_or_foreach',
                        lambda *args, **kwargs: (False, True))
    original_sqrt = torch._foreach_sqrt
    allocations = []

    def measured_sqrt(tensors):
        allocations.append(sum(t.numel() * t.element_size() for t in tensors))
        return original_sqrt(tensors)

    monkeypatch.setattr(torch, '_foreach_sqrt', measured_sqrt)
    automatic_parameters = _parameters()
    automatic = torch.optim.AdamW(automatic_parameters, lr=.01)
    for parameter in automatic_parameters:
        parameter.grad = torch.ones_like(parameter)
    automatic.step()
    assert allocations == [sum(p.numel() * p.element_size() for p in automatic_parameters)]

    masters, reference_masters = _parameters(), _parameters()
    bounded = torch.optim.AdamW(masters, lr=.01, betas=(.9, .95))
    reference = torch.optim.AdamW(reference_masters, lr=.01, betas=(.9, .95), foreach=False)
    assert bound_default_torch_adam_memory(bounded) == 1
    for step in range(3):
        for parameter, expected in zip(masters, reference_masters):
            gradient = torch.linspace(-.3, .7, parameter.numel()) * (step + 1)
            parameter.grad = gradient.clone()
            expected.grad = gradient.clone()
        bounded.step()
        reference.step()
        for parameter, expected in zip(masters, reference_masters):
            assert parameter.grad is not None  # The default policy does not consume gradients.
            assert torch.equal(parameter, expected)
            # MCore's BF16 model copy and FP32 master/moments are unchanged.
            assert torch.equal(parameter.to(torch.bfloat16), expected.to(torch.bfloat16))
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                assert torch.equal(bounded.state[parameter][key], reference.state[expected][key])
    assert len(allocations) == 1


@pytest.mark.parametrize('optimizer_type', [torch.optim.Adam, torch.optim.AdamW])
def test_nested_mcore_chained_and_distributed_wrappers_are_configured_once(optimizer_type):
    ChainedOptimizer = pytest.importorskip("megatron.core.optimizer.optimizer").ChainedOptimizer
    DistributedOptimizer = pytest.importorskip("megatron.core.optimizer.distrib_optimizer").DistributedOptimizer

    first, second = optimizer_type(_parameters()), optimizer_type(_parameters())
    distributed = DistributedOptimizer.__new__(DistributedOptimizer)
    distributed.optimizer = first
    inner = ChainedOptimizer.__new__(ChainedOptimizer)
    inner.chained_optimizers = [distributed, SimpleNamespace(optimizer=second)]
    outer = ChainedOptimizer.__new__(ChainedOptimizer)
    # This also proves we never access the asserting .optimizer property of a
    # multi-child chain, and shared references are not counted twice.
    outer.chained_optimizers = [inner, distributed]
    assert bound_default_torch_adam_memory(outer) == 2
    assert all(group['foreach'] is False for opt in (first, second) for group in opt.param_groups)
    assert bound_default_torch_adam_memory(outer) == 0


def test_explicit_foreach_and_fused_choices_and_non_native_optimizers_are_preserved():
    explicit = torch.optim.AdamW([
        {'params': [_parameters()[0]], 'foreach': True},
        {'params': [_parameters()[0]], 'foreach': False},
        {'params': [_parameters()[0]], 'foreach': None},
    ])
    fused = torch.optim.AdamW(_parameters(), fused=True)

    class OtherAdam(torch.optim.AdamW):
        pass

    other = OtherAdam(_parameters())
    sgd = torch.optim.SGD(_parameters(), lr=.1)
    tree = SimpleNamespace(chained_optimizers=[explicit, fused, other, sgd, None])
    assert bound_default_torch_adam_memory(tree) == 1
    assert [group['foreach'] for group in explicit.param_groups] == [True, False, False]
    assert fused.param_groups[0]['foreach'] is None
    assert fused.param_groups[0]['fused'] is True
    assert other.param_groups[0]['foreach'] is None
    assert sgd.param_groups[0]['foreach'] is None


def test_old_optimizer_state_restoration_reapplies_only_automatic_policy():
    optimizer = torch.optim.AdamW(_parameters())
    for parameter in optimizer.param_groups[0]['params']:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    assert saved['param_groups'][0]['foreach'] is None
    assert bound_default_torch_adam_memory(optimizer) == 1
    optimizer.load_state_dict(saved)
    assert optimizer.param_groups[0]['foreach'] is None
    assert bound_default_torch_adam_memory(optimizer) == 1
    actual = optimizer.state_dict()
    for key, values in saved['state'].items():
        for field, tensor in values.items():
            assert torch.equal(actual['state'][key][field], tensor)
    assert optimizer.defaults['foreach'] is False
    optimizer.add_param_group({'params': [_parameters()[0]]})
    assert optimizer.param_groups[-1]['foreach'] is False


def _master_groups(parameters):
    return [dict(params=parameters[:2], lr=.003, weight_decay=.01, name='body'),
            dict(params=parameters[2:], lr=.007, weight_decay=0., amsgrad=True, name='head')]


def _assert_adam_equal(actual, expected, parameters, reference):
    for parameter, target in zip(parameters, reference):
        assert torch.equal(parameter, target)
        assert set(actual.state[parameter]) == set(expected.state[target])
        for field, value in actual.state[parameter].items():
            assert torch.equal(value, expected.state[target][field])


@pytest.mark.parametrize('optimizer_type', [torch.optim.Adam, torch.optim.AdamW])
def test_streamed_main_gradients_preserve_multigroup_updates_and_reloadable_state(optimizer_type):
    parameters = _parameters() + [torch.nn.Parameter(torch.linspace(-.4, .8, 7))]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    actual = optimizer_type(_master_groups(parameters), betas=(.9, .95))
    expected = optimizer_type(_master_groups(reference), betas=(.9, .95), foreach=False)
    original_groups = actual.param_groups
    group_objects = list(original_groups)
    bound_default_torch_adam_memory(actual, release_main_gradients=True)
    streamed_step = actual.step
    bound_default_torch_adam_memory(actual, release_main_gradients=True)
    assert actual.step is streamed_step
    for step in range(4):
        for index, (parameter, target) in enumerate(zip(parameters, reference)):
            gradient = torch.linspace(-.3, .7, parameter.numel()) * (step + 1)
            parameter.grad = gradient.clone() if (step, index) != (2, 1) else None
            target.grad = gradient.clone() if (step, index) != (2, 1) else None
        # MCore computes and applies the common global clipping before Adam.
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        torch.nn.utils.clip_grad_norm_(reference, 1.)
        actual.step()
        expected.step()
        assert all(parameter.grad is None for parameter in parameters)
        assert actual.param_groups is original_groups
        assert all(group is original for group, original in zip(actual.param_groups, group_objects))
        _assert_adam_equal(actual, expected, parameters, reference)

    saved = copy.deepcopy(actual.state_dict())
    assert [len(group['params']) for group in saved['param_groups']] == [2, 1]
    assert [group['name'] for group in saved['param_groups']] == ['body', 'head']
    reloaded_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    reloaded = optimizer_type(_master_groups(reloaded_parameters), betas=(.9, .95), foreach=False)
    reloaded.load_state_dict(saved)
    bound_default_torch_adam_memory(reloaded, release_main_gradients=True)
    for parameter, target in zip(reloaded_parameters, reference):
        parameter.grad = torch.ones_like(parameter)
        target.grad = torch.ones_like(target)
    reloaded.step()
    expected.step()
    _assert_adam_equal(reloaded, expected, reloaded_parameters, reference)


def test_streamed_state_initialization_releases_each_main_gradient_before_the_next(monkeypatch):
    parameters = _parameters() + [torch.nn.Parameter(torch.zeros(7))]
    optimizer = torch.optim.AdamW(parameters, foreach=False)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    bound_default_torch_adam_memory(optimizer, release_main_gradients=True)
    allocate = torch.zeros_like
    resident_gradients = []

    def measured_zeros(tensor, *args, **kwargs):
        if any(tensor is parameter for parameter in parameters):
            resident_gradients.append(sum(p.grad.numel() for p in parameters if p.grad is not None))
        return allocate(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, 'zeros_like', measured_zeros)
    optimizer.step()
    # Each parameter initializes two Adam moments. Later moments must not
    # overlap the FP32 main gradients of parameters already updated.
    assert resident_gradients == [57, 57, 38, 38, 7, 7]
    assert all(parameter.grad is None for parameter in parameters)


def test_streamed_step_restores_original_groups_and_propagates_allocation_failure(monkeypatch):
    parameters = _parameters() + [torch.nn.Parameter(torch.zeros(7))]
    optimizer = torch.optim.AdamW(_master_groups(parameters), foreach=False)
    original_groups, group_objects = optimizer.param_groups, list(optimizer.param_groups)
    gradients = [torch.ones_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    bound_default_torch_adam_memory(optimizer, release_main_gradients=True)
    allocate = torch.zeros_like

    def failing_zeros(tensor, *args, **kwargs):
        if tensor is parameters[1]:
            raise MemoryError('injected moment allocation')
        return allocate(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, 'zeros_like', failing_zeros)
    with pytest.raises(MemoryError, match='injected moment allocation'):
        optimizer.step()
    assert optimizer.param_groups is original_groups
    assert all(group is original for group, original in zip(optimizer.param_groups, group_objects))
    assert parameters[0].grad is None
    assert parameters[1].grad is gradients[1] and parameters[2].grad is gradients[2]


def test_closure_uses_one_native_step_and_keeps_main_gradients():
    parameters = _parameters()
    optimizer = torch.optim.AdamW(parameters, foreach=False)
    gradients = [torch.ones_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    bound_default_torch_adam_memory(optimizer, release_main_gradients=True)
    calls = []

    def closure():
        calls.append(True)
        return torch.tensor(4.)

    assert optimizer.step(closure).item() == 4.
    assert calls == [True]
    assert all(p.grad is gradient for p, gradient in zip(parameters, gradients))


@pytest.mark.parametrize('unsupported', ['foreach', 'fused', 'differentiable', 'capturable'])
def test_unsupported_streaming_configuration_delegates_without_clearing_gradients(unsupported):
    parameters = _parameters()
    optimizer = torch.optim.AdamW([
        dict(params=parameters[:1], foreach=False),
        dict(params=parameters[1:], **{unsupported: True}),
    ])
    gradients = [torch.ones_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    original_groups = optimizer.param_groups
    calls, sentinel = [], object()

    def native_step(*args, **kwargs):
        # Spy at the native boundary so unsupported CUDA-only modes can be
        # checked on CPU without silently exercising another implementation.
        calls.append(([len(group['params']) for group in optimizer.param_groups], args, kwargs))
        return sentinel

    optimizer.step = native_step
    bound_default_torch_adam_memory(optimizer, release_main_gradients=True)
    assert optimizer.step() is sentinel
    assert calls == [([1, 1], (), {})]
    assert optimizer.param_groups is original_groups
    assert all(p.grad is gradient for p, gradient in zip(parameters, gradients))


@pytest.fixture
def mcore_cpu(monkeypatch):
    """Run MCore's real optimizer flow/math, redirecting CUDA-only test plumbing."""
    module = pytest.importorskip('megatron.core.optimizer.optimizer')
    distributed = pytest.importorskip('megatron.core.optimizer.distrib_optimizer')
    layers = pytest.importorskip('megatron.core.tensor_parallel.layers')
    tensor, tensor_type = torch.tensor, torch.Tensor.type
    reductions = []

    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*args, **kwargs)

    def cuda_type_for_clip(value, *args, **kwargs):
        if not args and not kwargs and value.dtype == torch.float32:
            return 'torch.cuda.FloatTensor'
        return tensor_type(value, *args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    monkeypatch.setattr(torch.Tensor, 'type', cuda_type_for_clip)
    monkeypatch.setattr(layers, 'get_tensor_model_parallel_rank', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'all_reduce',
        lambda value, **kwargs: reductions.append((value.clone(), kwargs['group'])))

    def make(children=1):
        wrappers = []
        group = object()
        for child_index in range(children):
            masters = _parameters() + [torch.nn.Parameter(torch.linspace(-.4, .8, 7))]
            models = []
            for index, master in enumerate(masters):
                master.tensor_model_parallel = index != 2
                master.shared = index == 1
                model = torch.nn.Parameter(torch.zeros(master.numel() + 4, dtype=torch.bfloat16))
                model.main_grad = torch.zeros_like(model)
                models.append(model)
            wrapper = distributed.DistributedOptimizer.__new__(distributed.DistributedOptimizer)
            wrapper.optimizer = torch.optim.AdamW(_master_groups(masters), betas=(.9, .95), foreach=False)
            wrapper.config = SimpleNamespace(bf16=True, use_precision_aware_optimizer=False,
                use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, optimizer_cpu_offload=False,
                log_num_zeros_in_grad=False, clip_grad=.7 / (child_index + 1), timers=None,
                reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather_with_optimizer_step=False)
            wrapper.grad_scaler, wrapper.is_stub_optimizer = None, False
            wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=False)
            wrapper.grad_stats_parallel_group = group
            wrapper.model_float16_groups = [models[:2], models[2:]]
            wrapper.shard_fp32_from_float16_groups = [masters[:2], masters[2:]]
            wrapper.model_fp32_groups = wrapper.shard_fp32_groups = [[], []]
            wrapper._get_model_param_range_map = lambda model: {
                'param': SimpleNamespace(start=2, end=model.numel()-2, size=model.numel()-4)}

            @torch.no_grad()
            def copy_models(models=models, masters=masters):
                for model, master in zip(models, masters):
                    model[2:-2].copy_(master)

            wrapper._copy_main_params_to_model_params = copy_models
            wrapper.model_chunks = [SimpleNamespace(start_param_sync=lambda: None)]
            wrappers.append(wrapper)
        chain = module.ChainedOptimizer.__new__(module.ChainedOptimizer)
        chain.chained_optimizers = wrappers
        chain.config = wrappers[0].config
        return SimpleNamespace(root=chain, children=wrappers,
            parameters=[p for child in wrappers for group in child.optimizer.param_groups for p in group['params']],
            models=[p for child in wrappers for group in child.model_float16_groups for p in group])

    return SimpleNamespace(make=make, reductions=reductions)


@pytest.mark.parametrize('children', [1, 2])
def test_bf16_prepare_streams_second_step_with_exact_global_norm_clip_and_state(mcore_cpu, monkeypatch, children):
    actual, expected = mcore_cpu.make(children), mcore_cpu.make(children)
    bound_default_torch_adam_memory(actual.root, release_main_gradients=True)
    assert all(child._libra_streamed_bf16_main_gradients for child in actual.children)
    copy_method = actual.children[0]._copy_model_grads_to_main_grads
    bound_default_torch_adam_memory(actual.root, release_main_gradients=True)
    assert actual.children[0]._copy_model_grads_to_main_grads is copy_method
    for step in range(3):
        # The second pass starts with all Adam moments already resident.
        for index, (model, reference) in enumerate(zip(actual.models, expected.models)):
            gradient = torch.linspace(-3., 7., model.numel()).to(torch.bfloat16) * (step + index + 1)
            model.main_grad.copy_(gradient)
            reference.main_grad.copy_(gradient)
        source_copies = [model.main_grad.clone() for model in actual.models]
        if step == 2:
            for child in actual.children:
                child.optimizer.load_state_dict(copy.deepcopy(child.optimizer.state_dict()))
            bound_default_torch_adam_memory(actual.root, release_main_gradients=True)
        expected_result = expected.root.step()
        references, live_counts = [], []
        original_float = torch.Tensor.float

        def counted_float(value, *args, **kwargs):
            result = original_float(value, *args, **kwargs)
            if value.dtype == torch.bfloat16:
                references.append(weakref.ref(result))
                live_counts.append(sum(ref() is not None for ref in references))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, 'float', counted_float)
            actual_result = actual.root.step()
        assert actual_result == expected_result
        assert actual_result[1] > 1  # Clipping is exercised, across all children.
        assert max(live_counts) == 1
        assert not any(ref() is not None for ref in references)
        assert all(parameter.grad is None for parameter in actual.parameters)
        for model, source in zip(actual.models, source_copies):
            assert torch.equal(model.main_grad, source)  # Never clip the BF16 source.
        for child, reference in zip(actual.children, expected.children):
            parameters = [p for group in child.optimizer.param_groups for p in group['params']]
            targets = [p for group in reference.optimizer.param_groups for p in group['params']]
            _assert_adam_equal(child.optimizer, reference.optimizer, parameters, targets)
        assert all(torch.equal(a, b) for a, b in zip(actual.models, expected.models))
        # Original MCore performs one global SUM for each chain, using its own group.
        reductions = mcore_cpu.reductions[-2:]
        assert len(reductions) == 2
        assert torch.equal(reductions[0][0], reductions[1][0])
        assert reductions[0][1] is expected.root.get_grad_stats_parallel_group()
        assert reductions[1][1] is actual.root.get_grad_stats_parallel_group()


@pytest.mark.parametrize('unsupported', ['grad_scaler', 'zeros', 'precision_aware', 'fsdp', 'backend'])
def test_bf16_streaming_leaves_unsupported_mcore_preparation_untouched(mcore_cpu, monkeypatch, unsupported):
    actual = mcore_cpu.make()
    child = actual.children[0]
    original = child._copy_model_grads_to_main_grads
    if unsupported == 'grad_scaler':
        child.grad_scaler = object()
    elif unsupported == 'zeros':
        child.config.log_num_zeros_in_grad = True
    elif unsupported == 'precision_aware':
        child.config.use_precision_aware_optimizer = True
    elif unsupported == 'fsdp':
        child.ddp_config.use_megatron_fsdp = True
    else:
        from megatron.core.optimizer import clip_grads
        monkeypatch.setattr(clip_grads, 'l2_norm_impl', lambda *args: None)
    bound_default_torch_adam_memory(actual.root, release_main_gradients=True)
    assert not getattr(child, '_libra_streamed_bf16_main_gradients', False)
    assert child._copy_model_grads_to_main_grads == original

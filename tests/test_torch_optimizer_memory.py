"""The fallback optimizer must fit without full-model foreach temporaries."""
import copy
import importlib
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

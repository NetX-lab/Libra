"""Hybrid cache release happens once, without changing gradient/update semantics."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


class DeviceGradient:
    def __init__(self, calls):
        self.calls = calls
        self.device = torch.device('cuda:1')
        self.dtype = torch.bfloat16
        self.tensor = torch.zeros(2, dtype=self.dtype)

    def copy_(self, value):
        self.calls.append('copy')
        self.tensor.copy_(value)


class Source:
    def __init__(self, value):
        self.tensor = torch.tensor(value, dtype=torch.float32)

    def to(self, *, device, dtype):
        assert device == torch.device('cuda:1')
        return self.tensor.to(dtype=dtype)


def cuda_engine(monkeypatch):
    calls = []
    gradients = [DeviceGradient(calls), DeviceGradient(calls)]
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.rank, engine.current_version = 0, 1
    engine._model_main_gradients = lambda: gradients
    engine.optimizer = SimpleNamespace(step=lambda: (calls.append('step') or (True, 0., 0)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    @contextmanager
    def device_context(device):
        assert device == torch.device('cuda:1')
        calls.append('enter')
        yield
        calls.append('exit')

    monkeypatch.setattr(torch.cuda, 'device', device_context)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('empty'))
    return engine, gradients, calls


def test_first_cuda_hybrid_update_releases_after_copy_and_before_step(monkeypatch, capsys):
    engine, gradients, calls = cuda_engine(monkeypatch)
    sources = (Source([1.25, 2.5]), Source([3., 4.]))
    engine.apply_elastic_gradient_update(sources, state_version=2)
    assert calls == ['copy', 'copy', 'enter', 'empty', 'exit', 'step']
    assert engine.current_version == 2 and engine._hybrid_optimizer_cuda_cache_released
    for gradient, source in zip(gradients, sources):
        assert torch.equal(gradient.tensor, source.tensor.bfloat16())
    assert not capsys.readouterr().out
    calls.clear()
    engine.apply_elastic_gradient_update(sources, state_version=3)
    assert calls == ['copy', 'copy', 'step'] and engine.current_version == 3
    assert not capsys.readouterr().out


def test_optimizer_exception_is_preserved_and_cache_release_can_retry(monkeypatch):
    engine, _gradients, calls = cuda_engine(monkeypatch)
    failure = RuntimeError('original NCCL failure')

    def failed_step():
        calls.append('step')
        raise failure

    engine.optimizer.step = failed_step
    with pytest.raises(RuntimeError) as error:
        engine.apply_elastic_gradient_update((Source([1., 2.]), Source([3., 4.])), state_version=2)
    assert error.value is failure and engine.current_version == 1
    assert not getattr(engine, '_hybrid_optimizer_cuda_cache_released', False)
    engine.optimizer.step = lambda: (True, 0., 0)
    engine.apply_elastic_gradient_update((Source([1., 2.]), Source([3., 4.])), state_version=2)
    assert calls.count('empty') == 2 and engine.current_version == 2


def test_rejected_optimizer_update_does_not_mark_cache_or_revision_complete(monkeypatch):
    engine, _gradients, calls = cuda_engine(monkeypatch)
    engine.optimizer.step = lambda: (False, 0., 0)
    with pytest.raises(FloatingPointError, match='rejected the update'):
        engine.apply_elastic_gradient_update((Source([1., 2.]), Source([3., 4.])), state_version=2)
    assert calls.count('empty') == 1 and engine.current_version == 1
    assert not getattr(engine, '_hybrid_optimizer_cuda_cache_released', False)


@pytest.mark.parametrize('cuda_available', [False, True])
def test_cpu_gradients_keep_original_copy_dtype_and_optimizer_update(monkeypatch, cuda_available):
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    gradient = torch.zeros(2, dtype=torch.bfloat16)
    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    source_before = source.clone()
    weight = torch.tensor([10., 20.])
    engine.current_version = 1
    engine._model_main_gradients = lambda: [gradient]
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: pytest.fail('CPU path released CUDA cache'))

    def step():
        weight.sub_(gradient.float(), alpha=.1)
        return True, 0., 0

    engine.optimizer = SimpleNamespace(step=step)
    engine.apply_elastic_gradient_update((source,), state_version=2)
    assert torch.equal(source, source_before)
    assert torch.equal(gradient, source.bfloat16())
    assert torch.equal(weight, torch.tensor([10., 20.]) - source.bfloat16().float() * .1)
    assert engine.current_version == 2
    assert not getattr(engine, '_hybrid_optimizer_cuda_cache_released', False)


def test_tensor_count_failure_happens_before_copy_or_cache_release(monkeypatch):
    engine, _gradients, calls = cuda_engine(monkeypatch)
    with pytest.raises(ValueError, match='tensor count mismatch'):
        engine.apply_elastic_gradient_update((Source([1., 2.]),), state_version=2)
    assert calls == [] and engine.current_version == 1

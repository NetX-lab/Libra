"""Shared rendezvous waits preserve application timeout=0 and subsecond limits."""
from datetime import timedelta
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch.distributed as dist

from RL_Framework.infra.sync import nccl_weight_sync
from RL_Framework.infra.sync.nccl_weight_sync import NcclReloadSpec


@pytest.fixture
def create_group(monkeypatch):
    calls = []
    utils = ModuleType('vllm.distributed.utils')
    utils.StatelessProcessGroup = SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or kwargs)
    for name in ('vllm', 'vllm.distributed'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, 'vllm.distributed.utils', utils)
    return calls


@pytest.mark.parametrize('timeout', [0.125, 1, 29.5, 1200])
def test_positive_nccl_store_timeout_preserves_seconds_without_minimum_or_truncation(create_group, timeout):
    nccl_weight_sync._group(NcclReloadSpec('127.0.0.1', 12345, 2, 0, 'cpu', timeout_s=timeout))
    assert create_group[0]['store_timeout'] == timeout


def test_zero_nccl_store_timeout_uses_safe_unbounded_sentinel(create_group):
    nccl_weight_sync._group(NcclReloadSpec('127.0.0.1', 12345, 2, 0, 'cpu', timeout_s=0))
    timeout = create_group[0]['store_timeout']
    assert timeout > 365 * 24 * 60 * 60
    assert timeout * 1_000_000_000 < 2**63
    # Exercise the same timedelta conversion performed by vLLM.
    assert timedelta(seconds=timeout).total_seconds() == timeout


@pytest.mark.parametrize('timeout', [-1, float('nan'), float('inf')])
def test_invalid_nccl_store_timeout_fails_before_group_creation(create_group, timeout):
    with pytest.raises(ValueError, match='non-negative seconds'):
        nccl_weight_sync._group(NcclReloadSpec('127.0.0.1', 12345, 2, 0, 'cpu', timeout_s=timeout))
    assert not create_group


def test_unbounded_sentinel_is_accepted_by_real_torch_tcp_store():
    # No peer or wait is needed: this verifies PyTorch's C++ duration conversion
    # accepts the sentinel and the resulting store remains functional.
    timeout = timedelta(seconds=nccl_weight_sync._UNBOUNDED_STORE_TIMEOUT_S)
    store = dist.TCPStore('127.0.0.1', 0, 1, True, timeout=timeout)
    store.set('probe', b'ready')
    assert store.get('probe') == b'ready'

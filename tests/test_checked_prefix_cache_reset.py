"""A failed or missing V1 utility cannot produce a cache-clear ACK."""

import asyncio
from types import SimpleNamespace

import pytest

from RL_Framework.infra.sync.prefix_cache_reset import (
    install_checked_prefix_cache_reset, reset_prefix_cache_verified,
)


@pytest.mark.parametrize('success', [True, False])
def test_checked_core_utility_preserves_scheduler_outcome(success):
    class Core:
        def reset_prefix_cache(self):
            # Mirrors vLLM 0.9.2's original wrapper, which discards the bool.
            self.scheduler.reset_prefix_cache()

    original = Core.reset_prefix_cache
    install_checked_prefix_cache_reset(Core)
    core = Core()
    core.scheduler = SimpleNamespace(reset_prefix_cache=lambda: success)
    assert Core.reset_prefix_cache is original
    if success:
        assert core.libra_reset_prefix_cache() is True
    else:
        with pytest.raises(RuntimeError, match='active KV blocks'):
            core.libra_reset_prefix_cache()


@pytest.mark.parametrize('outcome', [True, False, 'missing'])
def test_verified_reset_accepts_only_the_real_utility_success(outcome):
    async def utility(name):
        assert name == 'libra_reset_prefix_cache'
        if outcome == 'missing':
            raise AttributeError('Unknown utility method')
        return outcome

    client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
    if outcome is True:
        assert asyncio.run(reset_prefix_cache_verified(client, require_verified=True)) is True
    else:
        with pytest.raises((RuntimeError, AttributeError)):
            asyncio.run(reset_prefix_cache_verified(client, require_verified=True))


def test_legacy_none_return_is_preserved_without_claiming_verified_reset():
    async def reset():
        return None

    assert asyncio.run(reset_prefix_cache_verified(SimpleNamespace(reset_prefix_cache=reset))) is None

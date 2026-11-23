"""Opt-in V1 utility preserving the scheduler's prefix-reset success signal."""


def install_checked_prefix_cache_reset(core_class=None):
    """Install in each spawned core process, never in installed source files.

    vLLM 0.9.2's EngineCore and AsyncLLM reset wrappers discard the scheduler's
    boolean return. A distinct utility makes missing child-process installation
    an explicit RPC error instead of silently accepting a discarded failure.
    """
    if core_class is None:
        import vllm
        if vllm.__version__.split('+', 1)[0] != '0.9.2':
            raise RuntimeError('Checked prefix reset requires vLLM 0.9.2')
        from vllm.v1.engine.core import EngineCore
        core_class = EngineCore
    if getattr(core_class, '_libra_checked_prefix_reset', False):
        return

    def reset(self):
        if self.scheduler.reset_prefix_cache() is not True:
            raise RuntimeError('Prefix cache reset failed: active KV blocks remain')
        return True

    core_class.libra_reset_prefix_cache = reset
    core_class._libra_checked_prefix_reset = True


async def reset_prefix_cache_verified(client, *, require_verified=False):
    if require_verified:
        core_client = getattr(client, 'engine_core', None)
        if core_client is None or not hasattr(core_client, 'call_utility_async'):
            raise RuntimeError('Checked prefix reset requires the V1 utility client')
        if await core_client.call_utility_async('libra_reset_prefix_cache') is not True:
            raise RuntimeError('Prefix cache reset utility did not attest success')
        return True
    result = await client.reset_prefix_cache()
    if result is False:
        raise RuntimeError('Prefix cache reset failed')
    return True if result is True else None

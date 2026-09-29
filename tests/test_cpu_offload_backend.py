import asyncio
import unittest

from RL_Framework.engine.cmlfq_backend import CMLFQGenerationBackend


class FakeCPUBackend:
    def __init__(self):
        self.calls = []
        self.released = []
        self.ready = asyncio.Event()

    async def generate(self, prompt, kv_transfer_params, **kwargs):
        self.calls.append(kv_transfer_params)
        return {"text": "answer", "kv_transfer_params": {
            "libra_offload_key": kv_transfer_params["libra_offload_key"],
            "source_tp": 1, "cached_tokens": 128,
            "libra_imported_tokens": 128 if "libra_load_key" in kv_transfer_params else 0,
        }}

    async def wait_for_cpu_offload(self, key, source_tp):
        await self.ready.wait()
        return {"ready": True}

    async def release_cpu_offload(self, key):
        self.released.append(key)


class TestCPUOffloadBackend(unittest.IsolatedAsyncioTestCase):
    async def test_first_completion_does_not_wait_for_offload_and_next_reuses_cache(self):
        src, dst = FakeCPUBackend(), FakeCPUBackend()
        backend = CMLFQGenerationBackend("cpu_offload", [(1, 1)])
        first = await asyncio.wait_for(backend.generate(src, src, "prompt", "stay", "r"), 1)
        self.assertFalse(src.ready.is_set())
        key = first.output["kv_transfer_params"]["libra_offload_key"]
        resumed = asyncio.create_task(backend.generate(src, dst, "prompt+tool", "transfer", "r"))
        await asyncio.sleep(0)
        self.assertFalse(resumed.done())
        src.ready.set()
        result = await resumed
        self.assertEqual(dst.calls[0]["libra_load_key"], key)
        self.assertEqual(result.path, "cpu_reload")
        self.assertEqual(result.output["_cpu_offload_info"]["imported_tokens"], 128)
        await backend.close()
        self.assertEqual(len(src.released), 1)
        self.assertEqual(len(dst.released), 1)

    async def test_recompute_path_does_not_wait_or_load_prior_cache(self):
        src, dst = FakeCPUBackend(), FakeCPUBackend()
        backend = CMLFQGenerationBackend("cpu_offload")
        await backend.generate(src, src, "prompt", "stay", "r")
        result = await asyncio.wait_for(backend.generate(src, dst, "next", "recompute", "r"), 1)
        self.assertEqual(result.path, "recompute")
        self.assertNotIn("libra_load_key", dst.calls[0])
        await backend.close()

    async def test_cache_failure_recomputes_and_cleans_up(self):
        class FailedCache(FakeCPUBackend):
            async def wait_for_cpu_offload(self, key, source_tp):
                raise TimeoutError("offload failed")
        src, dst = FailedCache(), FakeCPUBackend()
        backend = CMLFQGenerationBackend("cpu_offload")
        await backend.generate(src, src, "prompt", "stay", "r")
        result = await backend.generate(src, dst, "next", "transfer", "r")
        self.assertEqual(result.fallback_reason, "TimeoutError")
        self.assertNotIn("libra_load_key", dst.calls[0])
        await backend.close()

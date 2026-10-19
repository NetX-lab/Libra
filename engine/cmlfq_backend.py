"""Execution paths for C-MLFQ cost routing over real completions endpoints.

NIXL delegates cache layout/TP mapping and network movement to vLLM, never
interpreting an instance index as a CUDA device. Ordinary servers use prefill
recomputation. The NIXL path follows vLLM's non-streaming P/D protocol.
"""

from __future__ import annotations

import logging
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    output: dict[str, Any]
    path: str
    fallback_reason: str = ""


class CMLFQGenerationBackend:
    def __init__(self, mode: str = "recompute", transfer_tp_pairs=()):
        if mode not in {"recompute", "nixl", "cpu_offload"}:
            raise ValueError(f"unknown CMLFQ KV backend: {mode}")
        self.mode = mode
        self._offloads = {}
        self._release_tasks = set()
        self.transfer_tp_pairs = {tuple(int(tp) for tp in pair) for pair in transfer_tp_pairs}
        if any(len(pair) != 2 or min(pair) <= 0 for pair in self.transfer_tp_pairs):
            raise ValueError("KV transfer TP pairs must contain two positive TP degrees")

    def supports_transfer(self, source: dict, target: dict) -> bool:
        return self.mode in {"nixl", "cpu_offload"} and (
            source["tp_degree"], target["tp_degree"]
        ) in self.transfer_tp_pairs

    async def generate(self, source, target, prompt: str, path: str, request_id: str = "", **kwargs) -> BackendResult:
        # The full workflow transcript is always passed, so recomputation
        # remains correct if the remote cache is absent, expired or incompatible.
        kwargs.pop("kv_transfer_params", None)
        if self.mode == "cpu_offload":
            return await self._generate_cpu(target, prompt, path, request_id, kwargs)
        if path != "transfer":
            return BackendResult(await target.generate(prompt=prompt, **kwargs), path)
        if self.mode != "nixl":
            raise RuntimeError("transfer selected on a recompute-only backend")
        try:
            # Materialize the exact next prompt on the source. Its generated
            # token is discarded; the target samples the user's full output.
            # Profile transfer_ms must include this preparation overhead.
            prepared = await source.generate(
                prompt=prompt, max_new_tokens=1, temperature=0.0, top_p=1.0, n=1,
                kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
            )
            params = prepared.get("kv_transfer_params")
            required = ("remote_engine_id", "remote_request_id", "remote_host", "remote_port")
            if not isinstance(params, dict) or any(not params.get(key) for key in required):
                raise RuntimeError("source did not export a usable KV descriptor")
            blocks = params.get("remote_block_ids")
            if not isinstance(blocks, list) or not any(blocks):
                raise RuntimeError("source exported no KV blocks")
            params = dict(params)
            params.update(do_remote_prefill=True, do_remote_decode=False)
            output = await target.generate(prompt=prompt, kv_transfer_params=params, **kwargs)
            return BackendResult(output, "transfer")
        except Exception as error:
            # Non-streaming requests: no partial output has reached the caller.
            # Cancellation (BaseException) is deliberately not retried.
            logger.warning("CMLFQ KV transfer failed; recomputing: %s", error)
            output = await target.generate(prompt=prompt, **kwargs)
            return BackendResult(output, "recompute", type(error).__name__)

    async def _generate_cpu(self, target, prompt, path, request_id, kwargs):
        if not request_id:
            raise ValueError("CPU offload requires a managed request id")
        previous = self._offloads.get(request_id)
        key = uuid.uuid4().hex
        params = {"libra_offload_key": key}
        fallback = "cache_unavailable" if path == "transfer" and previous is None else ""
        wait_started = time.perf_counter()
        status = None
        if previous and path in {"stay", "transfer"}:
            old_engine, descriptor = previous
            try:
                status = await old_engine.wait_for_cpu_offload(
                    descriptor["libra_offload_key"], descriptor["source_tp"],
                )
                params["libra_load_key"] = descriptor["libra_offload_key"]
            except Exception as error:
                fallback = type(error).__name__
        wait_ms = (time.perf_counter() - wait_started) * 1000
        try:
            try:
                output = await target.generate(prompt=prompt, kv_transfer_params=params, **kwargs)
            except Exception as error:
                if "libra_load_key" not in params:
                    raise
                fallback = type(error).__name__
                params.pop("libra_load_key")
                output = await target.generate(prompt=prompt, kv_transfer_params=params, **kwargs)
        except BaseException:
            self._release(target, key)
            raise
        descriptor = output.get("kv_transfer_params", {})
        if descriptor.get("libra_offload_key"):
            self._offloads[request_id] = (target, descriptor)
        else:
            self._offloads.pop(request_id, None)
            self._release(target, key)
        if previous:
            self._release(previous[0], previous[1]["libra_offload_key"])
        imported = int(descriptor.get("libra_imported_tokens", 0))
        output["_cpu_offload_info"] = {
            "visible_wait_ms": wait_ms, "imported_tokens": imported,
            "source_status": status,
        }
        actual = "cpu_reload" if imported else "recompute"
        if previous and path in {"stay", "transfer"} and not imported and not fallback:
            fallback = "cache_prefix_miss"
        return BackendResult(output, actual, fallback)

    def _release(self, engine, key):
        async def cleanup():
            try:
                await engine.release_cpu_offload(key)
            except Exception:
                logger.exception("Failed to release CPU offload %s", key)
        task = asyncio.create_task(cleanup())
        self._release_tasks.add(task)
        task.add_done_callback(self._release_tasks.discard)

    def release_request(self, request_id):
        previous = self._offloads.pop(request_id, None)
        if previous:
            self._release(previous[0], previous[1]["libra_offload_key"])

    async def close(self):
        for request_id in list(self._offloads):
            self.release_request(request_id)
        if self._release_tasks:
            await asyncio.gather(*list(self._release_tasks))

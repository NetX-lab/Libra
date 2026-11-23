"""Request-completion CPU offload connector for vLLM V1.

The completion is returned before the next worker tick stages its retained KV
blocks into pinned CPU memory. That tick overlaps external tool execution.
Only complete computed blocks are exported; consumers verify token-prefix
identity before claiming external tokens. Shared storage transports CPU shards
between workers (use tmpfs on one host, shared storage across hosts).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole,
)

logger = logging.getLogger(__name__)


def cache_directory(root: str, key: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("invalid offload key")
    return Path(root) / key


def atomic_json(path: Path, value):
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


@dataclass
class OffloadMetadata(KVConnectorMetadata):
    saves: list[dict] = field(default_factory=list)
    loads: list[dict] = field(default_factory=list)


class LibraCPUOffloadConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        if kv_cache_config is None:
            super().__init__(vllm_config, role)
        else:
            super().__init__(vllm_config, role, kv_cache_config)
        groups = getattr(kv_cache_config, "kv_cache_groups", None)
        if groups is not None and len(groups) != 1:
            raise ValueError("CPU offload requires exactly one KV cache group")
        extra = vllm_config.kv_transfer_config
        self.verify_reload = bool(extra.get_from_extra_config("verify_reload", False))
        self.root = extra.get_from_extra_config("storage_path", "/dev/shm/libra-kv")
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self.block_size = vllm_config.cache_config.block_size
        self.tp = vllm_config.parallel_config.tensor_parallel_size
        self.num_heads = vllm_config.model_config.hf_config.num_key_value_heads
        if self.num_heads % self.tp:
            raise ValueError("CPU connector requires non-replicated, divisible KV heads")
        self.rank = get_tensor_model_parallel_rank() if role == KVConnectorRole.WORKER else 0
        self.caches = {}
        self.saves, self.loads = [], []
        self.matches = {}
        self.imported = {}
        self.pending = {}
        self.completed_ids = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libra-kv-store")
        self.stream = None

    def register_kv_caches(self, kv_caches):
        caches = {}
        local_heads = self.num_heads // self.tp
        import vllm
        vllm_version = getattr(vllm, "__version__", "unknown")

        for name, tensor in kv_caches.items():
            shape = tuple(tensor.shape)
            backend_name = "unresolved"

            def fail(message, cause=None):
                error = ValueError(
                    f"cannot resolve KV layout for layer={name}, "
                    f"backend={backend_name}, shape={shape}, "
                    f"vLLM={vllm_version}: {message}"
                )
                if cause is None:
                    raise error
                raise error from cause

            if tensor.ndim != 5:
                fail("expected a five-dimensional cache")
            if shape[2] != self.block_size or shape[3] != local_heads:
                fail(
                    f"expected block_size={self.block_size} and "
                    f"local_heads={local_heads}"
                )

            axis0_is_kv = shape[0] == 2
            axis1_is_kv = shape[1] == 2
            if axis0_is_kv and not axis1_is_kv:
                canonical = tensor
            elif axis1_is_kv and not axis0_is_kv:
                canonical = tensor.movedim(1, 0)
            elif axis0_is_kv and axis1_is_kv:
                try:
                    context = self._vllm_config.compilation_config.static_forward_context
                    layer = context[name]
                except (AttributeError, KeyError, TypeError) as exc:
                    fail("ambiguous leading axes require layer metadata", exc)

                backend = getattr(layer, "attn_backend", None)
                if backend is None:
                    try:
                        from vllm.attention.selector import get_attn_backend
                        backend = get_attn_backend(
                            layer.head_size, layer.dtype, layer.kv_cache_dtype,
                            self.block_size, is_attention_free=False,
                        )
                    except Exception as exc:
                        fail("could not resolve the layer attention backend", exc)

                    backend_name = getattr(backend, "__name__", type(backend).__name__)
                    try:
                        implementation = backend.get_impl_cls()
                        compatible_impl = isinstance(layer.impl, implementation)
                    except Exception as exc:
                        fail("could not verify the resolved backend implementation", exc)
                    if not compatible_impl:
                        fail(
                            "backend implementation mismatch",
                            TypeError(
                                f"backend implementation {implementation!r} does not "
                                f"match layer implementation {type(layer.impl)!r}"
                            ),
                        )
                else:
                    backend_name = getattr(backend, "__name__", type(backend).__name__)

                try:
                    backend_shape = tuple(backend.get_kv_cache_shape(
                        3, self.block_size, local_heads, tensor.shape[-1],
                    ))
                except Exception as exc:
                    fail("backend KV shape probe failed", exc)

                if backend_shape == (
                    2, 3, self.block_size, local_heads, tensor.shape[-1],
                ):
                    canonical = tensor
                elif backend_shape == (
                    3, 2, self.block_size, local_heads, tensor.shape[-1],
                ):
                    canonical = tensor.movedim(1, 0)
                else:
                    fail(
                        f"unsupported backend KV shape probe: {backend_shape}",
                        TypeError(f"unsupported KV cache shape {backend_shape}"),
                    )
            else:
                fail("neither of the first two axes has the KV dimension of 2")

            caches[name] = canonical

        self.caches = caches
        self.stream = torch.cuda.Stream()

    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params or {}
        key = params.get("libra_offload_key")
        loaded = self.imported.pop(request.request_id, 0)
        self.matches.pop(request.request_id, None)
        if not key or request.status.name not in {"FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"}:
            return False, None
        directory = cache_directory(self.root, key)
        n = min(request.num_computed_tokens, len(request.all_token_ids))
        n = n // self.block_size * self.block_size
        if not n:
            return False, {"libra_imported_tokens": loaded}
        directory.mkdir(parents=True, exist_ok=True)
        self.saves.append({
            "request_id": request.request_id, "key": key,
            "block_ids": list(block_ids)[:n // self.block_size],
            "tokens": list(request.all_token_ids[:n]), "source_tp": self.tp,
        })
        return True, {"libra_offload_key": key, "source_tp": self.tp,
                      "cached_tokens": n, "libra_imported_tokens": loaded}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        key = params.get("libra_load_key")
        if not key:
            return 0, False
        directory = cache_directory(self.root, key)
        try:
            manifest = json.loads((directory / "rank_0.json").read_text())
            if manifest.get("block_size") != self.block_size:
                return 0, False
            source_tp = int(manifest["source_tp"])
            if any(not (directory / f"rank_{rank}.json").exists() for rank in range(source_tp)):
                return 0, False
            tokens = manifest["tokens"]
            prompt_tokens = request.prompt_token_ids
            limit = min(len(tokens), len(prompt_tokens) - 1)
            limit -= limit % self.block_size
            if num_computed_tokens % self.block_size:
                return 0, False
            n = 0
            for end in range(self.block_size, limit + 1, self.block_size):
                if tokens[n:end] != prompt_tokens[n:end]:
                    break
                n = end
            if n <= num_computed_tokens:
                return 0, False
            if self.num_heads % source_tp:
                return 0, False
            self.matches[request.request_id] = {
                "key": key, "source_tp": source_tp, "num_tokens": n,
                "num_local_tokens": num_computed_tokens,
                "request_id": request.request_id,
            }
            return n - num_computed_tokens, False
        except (FileNotFoundError, json.JSONDecodeError):
            return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            match = dict(self.matches.pop(request.request_id))
            if num_external_tokens != match["num_tokens"] - match["num_local_tokens"]:
                raise RuntimeError("CPU KV allocation does not match the available prefix")
            match["block_ids"] = list(blocks.get_block_ids()[0])
            self.loads.append(match)
            self.imported[request.request_id] = num_external_tokens

    def build_connector_meta(self, scheduler_output):
        meta = OffloadMetadata(self.saves, self.loads)
        self.saves, self.loads = [], []
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, OffloadMetadata):
            return
        for item in metadata.saves:
            self._start_offload(item)
        for item in metadata.loads:
            self._reload(item)

    def _start_offload(self, item):
        started = time.time()
        host_cache, gathered = {}, []
        # The source blocks stay owned by the request until get_finished
        # observes this event. No .cpu().pin_memory() synchronous intermediate.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            indices = torch.tensor(item["block_ids"], device="cuda", dtype=torch.long)
            for name, cache in self.caches.items():
                packed = cache.index_select(1, indices)
                host = torch.empty(packed.shape, dtype=packed.dtype, device="cpu", pin_memory=True)
                host.copy_(packed, non_blocking=True)
                host_cache[name] = host
                gathered.append(packed)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.pending[item["request_id"]] = (event, gathered, host_cache, item, started)

    def get_finished(self, finished_req_ids):
        self.completed_ids.update(set(finished_req_ids).intersection(self.pending))
        done = set()
        for request_id, (event, gathered, host, item, started) in list(self.pending.items()):
            if request_id not in self.completed_ids:
                continue
            # The scheduler may stop ticking when only finished requests remain.
            # Complete DMA on this post-completion tick; the client/tool runs
            # independently. Disk publication stays in the background thread.
            event.synchronize()
            dma_done = time.time()
            future = self.pool.submit(self._publish, host, item, started, dma_done)
            def report_error(result, key=item["key"]):
                error = result.exception()
                directory = cache_directory(self.root, key)
                if error is not None and directory.exists():
                    atomic_json(directory / f"rank_{self.rank}.error.json", {"error": str(error)})
            future.add_done_callback(report_error)
            del self.pending[request_id]
            self.completed_ids.discard(request_id)
            done.add(request_id)
        return done or None, None

    def _publish(self, host, item, started, dma_done):
        directory = cache_directory(self.root, item["key"])
        if not directory.exists() or (directory / "released").exists():
            return
        temporary = directory / f"rank_{self.rank}.pt.tmp"
        torch.save(host, temporary)
        if (directory / "released").exists():
            temporary.unlink(missing_ok=True)
            return
        temporary.replace(directory / f"rank_{self.rank}.pt")
        atomic_json(directory / f"rank_{self.rank}.json", {
            "tokens": item["tokens"], "source_tp": item["source_tp"],
            "block_size": self.block_size,
            "offload_started_at": started, "dma_done_at": dma_done,
            "ready_at": time.time(), "bytes": sum(t.numel() * t.element_size() for t in host.values()),
        })
        if (directory / "released").exists():
            for path in directory.glob(f"rank_{self.rank}.*"):
                path.unlink(missing_ok=True)

    def _reload(self, item):
        directory = cache_directory(self.root, item["key"])
        source_heads = self.num_heads // item["source_tp"]
        target_heads = self.num_heads // self.tp
        first_head, last_head = self.rank * target_heads, (self.rank + 1) * target_heads
        # Read only source ranks intersecting this target's global head range.
        # mmap shares file-backed pages instead of duplicating a full cache in
        # every target process before its own rank's slice is selected.
        shards = []
        for rank in range(first_head // source_heads, (last_head - 1) // source_heads + 1):
            tensors = torch.load(directory / f"rank_{rank}.pt", map_location="cpu",
                                 weights_only=True, mmap=True)
            start = max(first_head, rank * source_heads) - rank * source_heads
            end = min(last_head, (rank + 1) * source_heads) - rank * source_heads
            shards.append((tensors, start, end))
        local_tokens = item.get("num_local_tokens", 0)
        n = item["num_tokens"]
        if local_tokens % self.block_size or n % self.block_size:
            raise RuntimeError("CPU KV reload boundaries must align to whole blocks")
        first_block, end_block = local_tokens // self.block_size, n // self.block_size
        blocks = item["block_ids"][first_block:end_block]
        if len(blocks) != end_block - first_block:
            raise RuntimeError("CPU KV allocation is missing external prefix blocks")
        indices = torch.tensor(blocks, device="cuda", dtype=torch.long)
        bytes_copied = 0
        verified_layers = 0
        for name, destination in self.caches.items():
            # Canonical full head axis is assembled on CPU before slicing the
            # target rank. Source/target TP ranks are never CUDA device ids.
            pieces = [part[name][:, first_block:end_block, :, start:end, :]
                      for part, start, end in shards]
            rank_cache = (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)).contiguous()
            rank_cache = rank_cache.pin_memory()
            loaded = rank_cache.to(destination.device, non_blocking=True)
            destination.index_copy_(1, indices, loaded)
            bytes_copied += rank_cache.numel() * rank_cache.element_size()
            if self.verify_reload:
                if not torch.equal(destination.index_select(1, indices), loaded):
                    raise RuntimeError(f"CPU KV reload verification failed for {name}")
                verified_layers += 1
        if self.verify_reload:
            logger.info("CPU_KV_RELOAD_VERIFIED %s", json.dumps({
                "event": "CPU_KV_RELOAD_VERIFIED", "verified": True,
                "key": item["key"], "request_id": item["request_id"],
                "rank": self.rank, "source_tp": item["source_tp"],
                "target_tp": self.tp, "num_tokens": n - local_tokens,
                "bytes": bytes_copied, "layers_verified": verified_layers,
            }, sort_keys=True))

    def wait_for_layer_load(self, layer_name):
        # Reload and model execution share the current CUDA stream.
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        # No offload on the generation critical path; completion schedules it.
        return

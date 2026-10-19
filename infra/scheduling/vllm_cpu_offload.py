"""Request-completion CPU offload connector for vLLM V1 0.9.2.

The completion is returned before the next worker tick stages its retained KV
blocks into pinned CPU memory. That tick overlaps external tool execution.
Only complete computed blocks are exported; consumers verify token-prefix
identity before claiming external tokens. Shared storage transports CPU shards
between workers (use tmpfs on one host, shared storage across hosts).
"""

from __future__ import annotations

import json
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
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config
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
        for name, tensor in kv_caches.items():
            if tensor.ndim != 5 or tensor.shape[0] != 2:
                raise ValueError(f"unsupported KV layout for {name}: {tuple(tensor.shape)}")
        self.caches = kv_caches
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
            n = min(len(tokens), len(request.prompt_token_ids) - 1)
            n = n // self.block_size * self.block_size
            if tokens[:n] != request.prompt_token_ids[:n] or n <= num_computed_tokens:
                return 0, False
            if self.num_heads % source_tp:
                return 0, False
            self.matches[request.request_id] = {
                "key": key, "source_tp": source_tp, "num_tokens": n,
                "request_id": request.request_id,
            }
            return n - num_computed_tokens, False
        except (FileNotFoundError, json.JSONDecodeError):
            return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            match = dict(self.matches.pop(request.request_id))
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
            # vLLM 0.9.2 may stop ticking when only finished requests remain.
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
        n = item["num_tokens"]
        blocks = item["block_ids"][:n // self.block_size]
        indices = torch.tensor(blocks, device="cuda", dtype=torch.long)
        for name, destination in self.caches.items():
            # Canonical full head axis is assembled on CPU before slicing the
            # target rank. Source/target TP ranks are never CUDA device ids.
            pieces = [part[name][:, :len(blocks), :, start:end, :]
                      for part, start, end in shards]
            rank_cache = (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)).contiguous()
            rank_cache = rank_cache.pin_memory()
            destination.index_copy_(1, indices, rank_cache.to(destination.device, non_blocking=True))

    def wait_for_layer_load(self, layer_name):
        # Reload and model execution share the current CUDA stream.
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        # No offload on the generation critical path; completion schedules it.
        return

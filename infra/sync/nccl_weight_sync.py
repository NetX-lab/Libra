"""Direct GPU/NCCL weight transport between Megatron and vLLM workers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch


_COMMUNICATOR_CACHE: dict[tuple[str, int, int, int, str], object] = {}
_COMMUNICATOR_CACHE_LOCK = threading.Lock()
_PIPELINE_DEPTH = 2


@dataclass(frozen=True)
class NcclReloadSpec:
    """Connection parameters for one reload generation."""

    host: str
    port: int
    world_size: int
    rank: int
    device: str | int
    timeout_s: float = 1200.0
    chunk_bytes: int = 256 * 1024 * 1024
    rate_limit_gbps: float = 0.0


def _group(spec: NcclReloadSpec):
    from vllm.distributed.utils import StatelessProcessGroup

    return StatelessProcessGroup.create(
        host=spec.host,
        port=int(spec.port),
        rank=int(spec.rank),
        world_size=int(spec.world_size),
        store_timeout=max(30, int(spec.timeout_s)),
    )


def _communicator(spec: NcclReloadSpec):
    # vLLM 0.9.x disables cuMem for its worker communicators. All ranks in
    # this independent communicator must use the same transport mode.
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    class _BroadcastCommunicator(PyNcclCommunicator):
        """Keep vLLM's constructor warm-up output alive until it synchronizes."""

        def all_reduce(self, in_tensor, op=None, stream=None):
            # vLLM 0.9.x discards the out-of-place all-reduce result in
            # PyNcclCommunicator.__init__. Retain it so the CUDA allocator does
            # not recycle the receive buffer while the warm-up is in flight.
            if op is None:
                result = super().all_reduce(in_tensor, stream=stream)
            else:
                result = super().all_reduce(in_tensor, op=op, stream=stream)
            self._constructor_warmup_output = result
            return result

    device = torch.device(spec.device)
    device_key = str(device)
    key = (
        str(spec.host),
        int(spec.port),
        int(spec.world_size),
        int(spec.rank),
        device_key,
    )
    with _COMMUNICATOR_CACHE_LOCK:
        communicator = _COMMUNICATOR_CACHE.get(key)
        if communicator is None:
            # vLLM may have changed the current device while serving requests.
            # Bind the CUDA context before creating this independent NCCL
            # communicator; otherwise DeepSeek's first warm-up collective can
            # execute on a stale device and report an asynchronous launch error.
            if device.type == "cuda" and torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.synchronize(device)
                    communicator = _BroadcastCommunicator(
                        _group(spec), device=device
                    )
            else:
                communicator = _BroadcastCommunicator(_group(spec), device=device)
            communicator._constructor_warmup_output = None
            _COMMUNICATOR_CACHE[key] = communicator
        return communicator


def clear_communicator_cache() -> None:
    """Drop cached communicators, typically during orderly process shutdown."""
    with _COMMUNICATOR_CACHE_LOCK:
        _COMMUNICATOR_CACHE.clear()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported NCCL weight dtype: {name}")
    return value


def _cuda_stream(device: str | int):
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.Stream(device=resolved)


def _wait_for_event(event, device: str | int) -> None:
    if event is None:
        return
    consumer_stream = torch.cuda.current_stream(device=torch.device(device))
    consumer_stream.wait_event(event)


def send_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    spec: NcclReloadSpec,
    *,
    on_tensor: Callable[[str, int], None] | None = None,
    keepalive: list[object] | None = None,
) -> int:
    """Broadcast full HF-format tensors from rank zero to vLLM workers.

    The caller must invoke this on every Megatron rank because the bridge
    export performs TP/PP collectives on every rank. Only rank zero enters the
    NCCL sender; other ranks still consume the generator and discard tensors.
    """
    if spec.rank != 0:
        raise ValueError("send_weights must run with rank=0")
    comm = _communicator(spec)
    if keepalive is not None:
        # The receivers may still be consuming the final metadata marker after
        # this function returns. Keep the TCPStore alive until their RPC acks.
        keepalive.append(comm)
    stream = _cuda_stream(spec.device)
    pending: list[tuple[torch.Tensor, object]] = []

    def drain_one() -> None:
        if pending:
            _tensor, event = pending.pop(0)
            event.synchronize()

    count = 0
    for name, tensor in weights:
        if tensor.device.type != "cuda":
            tensor = tensor.to(device=spec.device, non_blocking=True)
        tensor = tensor.contiguous()
        chunk_numel = max(1, int(spec.chunk_bytes) // tensor.element_size())
        metadata = {
            "done": False,
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": _dtype_name(tensor.dtype),
            "chunk_numel": chunk_numel,
        }
        comm.group.broadcast_obj(metadata, src=0)
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), chunk_numel):
            chunk = flat[start : start + chunk_numel]
            chunk_started = time.perf_counter()
            if stream is None:
                comm.broadcast(chunk, src=0)
            else:
                comm.broadcast(chunk, src=0, stream=stream)
            if spec.rate_limit_gbps > 0:
                if stream is not None:
                    stream.synchronize()
                target_seconds = (
                    chunk.numel() * chunk.element_size() * 8
                ) / (spec.rate_limit_gbps * 1e9)
                elapsed = time.perf_counter() - chunk_started
                if elapsed < target_seconds:
                    time.sleep(target_seconds - elapsed)
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
            pending.append((tensor, event))
            if len(pending) >= _PIPELINE_DEPTH:
                drain_one()
        count += 1
        if on_tensor is not None:
            on_tensor(str(name), int(tensor.numel() * tensor.element_size()))
    while pending:
        drain_one()
    comm.group.broadcast_obj({"done": True}, src=0)
    return count


def receive_weights(spec: NcclReloadSpec) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield NCCL-received HF tensors for vLLM's ``load_weights`` method."""
    comm = _communicator(spec)
    stream = _cuda_stream(spec.device)
    while True:
        metadata = comm.group.broadcast_obj(None, src=0)
        if metadata.get("done", False):
            return
        shape = tuple(int(value) for value in metadata["shape"])
        tensor = torch.empty(
            shape,
            dtype=_dtype_from_name(str(metadata["dtype"])),
            device=spec.device,
        )
        flat = tensor.view(-1)
        chunk_numel = max(1, int(metadata["chunk_numel"]))
        for start in range(0, flat.numel(), chunk_numel):
            chunk = flat[start : start + chunk_numel]
            if stream is None:
                comm.broadcast(chunk, src=0)
            else:
                comm.broadcast(chunk, src=0, stream=stream)
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
            _wait_for_event(event, spec.device)
            tensor.record_stream(torch.cuda.current_stream(device=torch.device(spec.device)))
        yield str(metadata["name"]), tensor

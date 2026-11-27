"""Direct GPU/NCCL weight transport between Megatron and vLLM workers."""

from __future__ import annotations

import os
import ctypes
import ipaddress
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from contextlib import nullcontext
from typing import Callable, Iterable, Iterator

import torch


_COMMUNICATOR_CACHE: dict[tuple[str, int, int, int, str, str], object] = {}
_COMMUNICATOR_CACHE_LOCK = threading.Lock()
_BASE_STORE_CACHE: dict[tuple[str, int, bool], object] = {}
_BASE_STORE_CACHE_LOCK = threading.Lock()
_PIPELINE_DEPTH = 2
# vLLM constructs datetime.timedelta(seconds=store_timeout), so None/inf and
# timedelta.max are unsafe. A century represents the application's unbounded
# wait while staying below signed int64 even after conversion to nanoseconds.
_UNBOUNDED_STORE_TIMEOUT_S = 100 * 365 * 24 * 60 * 60
_LOGGER = logging.getLogger(__name__)


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
    session_id: str = ""

    def __post_init__(self):
        if not self.host.strip():
            raise ValueError("NCCL rendezvous host must be a reachable sender hostname or IP")
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError:
            return  # DNS hostnames are resolved by the transport.
        if address.is_unspecified:
            raise ValueError("NCCL rendezvous cannot advertise a wildcard address: " + self.host)


def _group(spec: NcclReloadSpec):
    from vllm.distributed.utils import StatelessProcessGroup

    timeout_s = float(spec.timeout_s)
    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("NCCL store timeout must be finite non-negative seconds")
    timeout_s = timeout_s if timeout_s > 0 else _UNBOUNDED_STORE_TIMEOUT_S
    if spec.session_id:
        # Keep one TCP listener across member replacement. A new namespace must
        # reset metadata counters/NCCL IDs without rebinding the occupied port.
        key = (str(spec.host), int(spec.port), int(spec.rank) == 0)
        timeout = timedelta(seconds=timeout_s)
        with _BASE_STORE_CACHE_LOCK:
            store = _BASE_STORE_CACHE.get(key)
            if store is None:
                store = torch.distributed.TCPStore(
                    host_name=spec.host, port=int(spec.port), world_size=None,
                    is_master=key[2], timeout=timeout, wait_for_workers=False,
                    use_libuv=False,
                )
                _BASE_STORE_CACHE[key] = store
            store.set_timeout(timeout)
            namespaced = torch.distributed.PrefixStore(spec.session_id, store)
            namespaced.set_timeout(timeout)
        return StatelessProcessGroup(
            rank=int(spec.rank), world_size=int(spec.world_size),
            store=namespaced, socket=None,
        )
    group = StatelessProcessGroup.create(
        host=spec.host,
        port=int(spec.port),
        rank=int(spec.rank),
        world_size=int(spec.world_size),
        store_timeout=timeout_s,
    )
    # vLLM 0.9.2 keeps a Python socket for the master_listen_fd handed to
    # TCPStore. TCPStore owns and closes that descriptor. Letting the Python
    # socket also close it can abort the store's daemon during cache eviction
    # or interpreter shutdown. Transfer ownership while both objects are alive;
    # receiver groups have no listening socket.
    listen_socket = getattr(group, "socket", None)
    if spec.rank == 0 and listen_socket is not None:
        listen_socket.detach()
    return group


def _log_initialization_failure(spec, stage, error):
    """Best-effort failure evidence; never synchronize or allocate CUDA memory."""
    route = dict(host=spec.host, port=spec.port, session_id=spec.session_id,
                 world_size=spec.world_size, rank=spec.rank, device=str(spec.device))
    cuda = {}

    def probe(name, read):
        try:
            cuda[name] = read()
        except Exception as exc:
            cuda[name + '_error'] = repr(exc)

    probe('available', torch.cuda.is_available)
    if torch.device(spec.device).type == 'cuda' and cuda.get('available'):
        probe('current_device', torch.cuda.current_device)
        probe('current_stream_device', lambda: str(torch.cuda.current_stream().device))
        probe('allocated_bytes', lambda: torch.cuda.memory_allocated(spec.device))
        probe('reserved_bytes', lambda: torch.cuda.memory_reserved(spec.device))
        probe('free_total_bytes', lambda: torch.cuda.mem_get_info(spec.device))
    _LOGGER.error('NCCL communicator initialization failed stage=%s route=%r cuda=%r',
                  stage, route, cuda, exc_info=(type(error), error, error.__traceback__))


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
        spec.session_id,
    )
    with _COMMUNICATOR_CACHE_LOCK:
        communicator = _COMMUNICATOR_CACHE.get(key)
        if communicator is None:
            # vLLM may have changed the current device while serving requests.
            # Bind the CUDA context before creating this independent NCCL
            # communicator; otherwise DeepSeek's first warm-up collective can
            # execute on a stale device and report an asynchronous launch error.
            stage = 'cuda_context'
            try:
                use_cuda = device.type == "cuda" and torch.cuda.is_available()
                with torch.cuda.device(device) if use_cuda else nullcontext():
                    if use_cuda:
                        stage = 'cuda_synchronize'
                        torch.cuda.synchronize(device)
                    stage = 'stateless_group'
                    group = _group(spec)
                    stage = 'pynccl_initialize_and_warmup'
                    communicator = _BroadcastCommunicator(group, device=device)
            except Exception as exc:
                try:
                    _log_initialization_failure(spec, stage, exc)
                except Exception:
                    pass  # Diagnostic failures must preserve the transport error.
                raise
            communicator._constructor_warmup_output = None
            _COMMUNICATOR_CACHE[key] = communicator
        # A loan can change world size and a surviving receiver's rank. They
        # describe membership, not ownership of this host/port/device route.
        # Retire superseded sessions only after the new group warmed up.
        if spec.session_id:
            for previous in list(_COMMUNICATOR_CACHE):
                if (previous != key and previous[:2] == key[:2]
                        and previous[4] == key[4]):
                    _abort_communicator(_COMMUNICATOR_CACHE[previous])
                    del _COMMUNICATOR_CACHE[previous]
        return communicator


def _abort_communicator(communicator) -> None:
    """Release a local NCCL handle without waiting for an exited member."""
    if getattr(communicator, "_libra_retired", False):
        return
    handle = getattr(communicator, "comm", None)
    if handle is not None:
        nccl = communicator.nccl
        # vLLM 0.9.2 does not wrap abort. Bind the library symbol locally with
        # the same ncclResult_t(ncclComm_t) ABI as its ncclCommDestroy wrapper.
        abort = nccl.lib.ncclCommAbort
        abort.restype = ctypes.c_int
        abort.argtypes = [ctypes.c_void_p]
        device = getattr(communicator, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(device):
                result = abort(handle)
        else:
            result = abort(handle)
        nccl.NCCL_CHECK(result)
        communicator.comm = None
    communicator.disabled = True
    communicator.available = False
    communicator.group = None
    communicator._libra_retired = True


def clear_communicator_cache(*, keys=None) -> None:
    """Abort owned communicators; optionally invalidate selected route prefixes.

    Five-field keys remain compatible with engine-local publication callers.
    Targeted cleanup retains base stores for subsequent incarnations; whole
    shutdown aborts all communicators before releasing their TCPStore ownership.
    Failed aborts propagate and retain ownership so cleanup can be retried.
    """
    prefixes = None if keys is None else [tuple(key) for key in keys]
    with _COMMUNICATOR_CACHE_LOCK:
        for key in list(_COMMUNICATOR_CACHE):
            if prefixes is None or any(key[:len(prefix)] == prefix for prefix in prefixes):
                _abort_communicator(_COMMUNICATOR_CACHE[key])
                del _COMMUNICATOR_CACHE[key]
        if prefixes is None:
            with _BASE_STORE_CACHE_LOCK:
                _BASE_STORE_CACHE.clear()


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
        if stream is not None:
            # Export and nonblocking copies run on the producer stream.
            stream.wait_stream(torch.cuda.current_stream(device=torch.device(spec.device)))
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
        if stream is not None:
            # The allocator can reuse memory with work pending on this stream.
            stream.wait_stream(torch.cuda.current_stream(device=torch.device(spec.device)))
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

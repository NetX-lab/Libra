"""Device and distributed backend helpers."""

from __future__ import annotations

import os
from typing import Any

import torch


def _normalize_backend(value: str | None = None) -> str:
    backend = (
        value
        or os.environ.get("DEVICE_BACKEND")
        or os.environ.get("ACCELERATOR_BACKEND")
        or os.environ.get("ACCELERATOR_TYPE")
        or "auto"
    )
    return str(backend).strip().lower()


def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu")


def accelerator_backend(value: str | None = None) -> str:
    """Return cuda, npu, or cpu."""
    requested = _normalize_backend(value)
    if requested in {"cuda", "gpu"}:
        return "cuda"
    if requested in {"npu", "ascend", "hccl"}:
        return "npu"
    if requested == "cpu":
        return "cpu"

    if _try_import_torch_npu():
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def distributed_backend(value: str | None = None) -> str:
    requested = str(os.environ.get("DIST_BACKEND", value or "auto")).strip().lower()
    if requested not in {"", "auto"}:
        return requested
    backend = accelerator_backend()
    if backend == "npu":
        return "hccl"
    if backend == "cuda":
        return "nccl"
    return "gloo"


def device_for_local_rank(local_rank: int = 0, backend: str | None = None) -> torch.device:
    backend = accelerator_backend(backend)
    if backend == "npu":
        return torch.device(f"npu:{local_rank}")
    if backend == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def set_device(local_rank: int = 0, backend: str | None = None) -> torch.device:
    backend = accelerator_backend(backend)
    if backend == "npu":
        _try_import_torch_npu()
        torch.npu.set_device(local_rank)
    elif backend == "cuda":
        torch.cuda.set_device(local_rank)
    return device_for_local_rank(local_rank, backend)


def synchronize(device: torch.device | None = None) -> None:
    if device is None:
        backend = accelerator_backend()
        if backend == "npu":
            torch.npu.synchronize()
        elif backend == "cuda":
            torch.cuda.synchronize()
        return
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def device_count(backend: str | None = None) -> int:
    backend = accelerator_backend(backend)
    if backend == "npu":
        _try_import_torch_npu()
        return int(torch.npu.device_count())
    if backend == "cuda":
        return int(torch.cuda.device_count())
    return 0


def memory_stats(local_rank: int = 0, backend: str | None = None) -> dict[str, float]:
    backend = accelerator_backend(backend)
    if backend == "cuda" and torch.cuda.is_available():
        return {
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(local_rank) / 1e9,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved(local_rank) / 1e9,
        }
    if backend == "npu":
        _try_import_torch_npu()
        stats: dict[str, float] = {}
        for name, fn_name in {
            "npu_memory_allocated_gb": "memory_allocated",
            "npu_memory_reserved_gb": "memory_reserved",
        }.items():
            fn: Any = getattr(torch.npu, fn_name, None)
            if callable(fn):
                try:
                    stats[name] = float(fn(local_rank)) / 1e9
                except Exception:
                    pass
        return stats
    return {}

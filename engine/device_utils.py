"""CUDA, Ascend NPU and CPU device/distributed helpers."""

from __future__ import annotations

import os
from typing import Any

import torch


def _normalize_backend(value: str | None = None) -> str:
    backend = (
        value
        or os.environ.get("ACCELERATOR_BACKEND")
        or "auto"
    )
    return str(backend).strip().lower()


def accelerator_backend(value: str | None = None) -> str:
    """Return the configured CUDA or CPU backend."""
    requested = _normalize_backend(value)
    if requested in {"cuda", "gpu"}:
        return "cuda"
    if requested in {"npu", "ascend"}:
        import torch_npu  # noqa: F401
        return "npu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def distributed_backend(value: str | None = None) -> str:
    """Return NCCL for CUDA and Gloo for CPU."""
    requested = str(os.environ.get("DIST_BACKEND", value or "auto")).strip().lower()
    if requested not in {"", "auto"}:
        if requested not in {"nccl", "gloo", "hccl"}:
            raise ValueError("Only NCCL, HCCL and Gloo distributed backends are supported")
        return requested
    return {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}[accelerator_backend()]


def device_for_local_rank(local_rank: int = 0, backend: str | None = None) -> torch.device:
    backend = accelerator_backend(backend)
    if backend in {"cuda", "npu"}:
        return torch.device(f"{backend}:{local_rank}")
    return torch.device("cpu")


def set_device(local_rank: int = 0, backend: str | None = None) -> torch.device:
    backend = accelerator_backend(backend)
    if backend in {"cuda", "npu"}:
        getattr(torch, backend).set_device(local_rank)
    return device_for_local_rank(local_rank, backend)


def synchronize(device: torch.device | None = None) -> None:
    if device is None:
        backend = accelerator_backend()
        if backend in {"cuda", "npu"}:
            getattr(torch, backend).synchronize()
        return
    if device.type in {"cuda", "npu"}:
        getattr(torch, device.type).synchronize(device)


def device_count(backend: str | None = None) -> int:
    selected = accelerator_backend(backend)
    return int(getattr(torch, selected).device_count()) if selected in {"cuda", "npu"} else 0


def memory_stats(local_rank: int = 0, backend: str | None = None) -> dict[str, float]:
    selected = accelerator_backend(backend)
    api = getattr(torch, selected, None)
    if selected not in {"cuda", "npu"} or not api.is_available():
        return {}
    return {
        "gpu_memory_allocated_gb": api.memory_allocated(local_rank) / 1e9,
        "gpu_memory_reserved_gb": api.memory_reserved(local_rank) / 1e9,
    }

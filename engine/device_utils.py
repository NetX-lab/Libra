"""CUDA device and distributed backend helpers."""

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
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def distributed_backend(value: str | None = None) -> str:
    """Return NCCL for CUDA and Gloo for CPU."""
    requested = str(os.environ.get("DIST_BACKEND", value or "auto")).strip().lower()
    if requested not in {"", "auto"}:
        if requested not in {"nccl", "gloo"}:
            raise ValueError("Only NCCL and Gloo distributed backends are supported")
        return requested
    return "nccl" if accelerator_backend() == "cuda" else "gloo"


def device_for_local_rank(local_rank: int = 0, backend: str | None = None) -> torch.device:
    backend = accelerator_backend(backend)
    if backend == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def set_device(local_rank: int = 0, backend: str | None = None) -> torch.device:
    backend = accelerator_backend(backend)
    if backend == "cuda":
        torch.cuda.set_device(local_rank)
    return device_for_local_rank(local_rank, backend)


def synchronize(device: torch.device | None = None) -> None:
    if device is None:
        if accelerator_backend() == "cuda":
            torch.cuda.synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_count(backend: str | None = None) -> int:
    return int(torch.cuda.device_count()) if accelerator_backend(backend) == "cuda" else 0


def memory_stats(local_rank: int = 0, backend: str | None = None) -> dict[str, float]:
    if accelerator_backend(backend) != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(local_rank) / 1e9,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved(local_rank) / 1e9,
    }

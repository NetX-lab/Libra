"""Runtime checks for CANN operators required by the MindSpeed DS4 provider."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


REQUIRED_SPARSE_MLA_SYMBOLS = (
    "aclnnSparseFlashMlaMetadata",
    "aclnnSparseFlashMlaMetadataGetWorkspaceSize",
    "aclnnSparseFlashMla",
    "aclnnSparseFlashMlaGetWorkspaceSize",
    "aclnnSparseFlashMlaGrad",
    "aclnnSparseFlashMlaGradGetWorkspaceSize",
)

# MindSpeed's ops-transformer wrapper returns ``None`` for this API on
# non-Ascend-950 devices. In particular, 910B3 does not need these symbols.
OPTIONAL_SPARSE_MLA_SYMBOLS = (
    "aclnnSparseFlashMlaGradMetadata",
    "aclnnSparseFlashMlaGradMetadataGetWorkspaceSize",
)


def default_cann_root() -> Path:
    """Honor the isolated CANN ops root used by the cluster gate."""
    return Path(os.environ.get("LIBRA_CANN_OPS_ROOT", "/usr/local/Ascend"))


def discover_cann_libraries(root: Path | None = None) -> list[Path]:
    root = root or default_cann_root()
    candidates = {
        *root.glob("cann-*/aarch64-linux/lib64/libopapi.so"),
        *root.glob("cann-*/aarch64-linux/lib64/libopapi_transformer.so"),
        *root.glob("aarch64-linux/lib64/libopapi.so"),
        *root.glob("aarch64-linux/lib64/libopapi_transformer.so"),
        *root.glob("lib64/libopapi.so"),
        *root.glob("lib64/libopapi_transformer.so"),
    }
    return sorted(path for path in candidates if path.is_file())


def inspect_sparse_mla_symbols(libraries: list[Path]) -> tuple[set[str], list[str]]:
    found: set[str] = set()
    errors: list[str] = []
    for library in libraries:
        try:
            handle = ctypes.CDLL(str(library))
        except OSError as exc:
            errors.append(f"cannot load {library}: {exc}")
            continue
        for symbol in (*REQUIRED_SPARSE_MLA_SYMBOLS, *OPTIONAL_SPARSE_MLA_SYMBOLS):
            try:
                getattr(handle, symbol)
            except AttributeError:
                continue
            found.add(symbol)
    return found, errors


def require_mindspeed_v4_ops(root: Path | None = None) -> None:
    root = root or default_cann_root()
    libraries = discover_cann_libraries(root)
    if not libraries:
        raise RuntimeError(f"No CANN libopapi libraries found under {root}")
    found, errors = inspect_sparse_mla_symbols(libraries)
    missing = [symbol for symbol in REQUIRED_SPARSE_MLA_SYMBOLS if symbol not in found]
    if missing:
        detail = ", ".join(missing)
        load_errors = f"; library load errors: {errors}" if errors else ""
        raise RuntimeError(
            "MindSpeed DeepSeek-V4 requires sparse flash MLA CANN APIs missing from "
            f"the current runtime: {detail}{load_errors}. Install a CANN/ops image "
            "with these APIs and a matching host driver/firmware."
        )

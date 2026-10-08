"""Fail-fast check for the CANN APIs required by MindSpeed-LLM DeepSeek-V4."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from RL_Framework.engine.mindspeed_v4_ops import (
    OPTIONAL_SPARSE_MLA_SYMBOLS,
    REQUIRED_SPARSE_MLA_SYMBOLS,
    default_cann_root,
    discover_cann_libraries,
    inspect_sparse_mla_symbols,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cann-root",
        type=Path,
        default=default_cann_root(),
        help="CANN ops install root (default: LIBRA_CANN_OPS_ROOT or /usr/local/Ascend)",
    )
    args = parser.parse_args(argv)

    libraries = discover_cann_libraries(args.cann_root)
    if not libraries:
        print(f"ERROR: no CANN libopapi libraries found under {args.cann_root}", file=sys.stderr)
        return 2

    found, errors = inspect_sparse_mla_symbols(libraries)
    missing = [symbol for symbol in REQUIRED_SPARSE_MLA_SYMBOLS if symbol not in found]
    print("CANN operator libraries:")
    for library in libraries:
        print(f"  {library}")
    if errors:
        for error in errors:
            print(f"WARNING: {error}", file=sys.stderr)
    if missing:
        print("ERROR: MindSpeed-LLM DeepSeek-V4 sparse MLA APIs are incomplete:", file=sys.stderr)
        for symbol in missing:
            print(f"  missing {symbol}", file=sys.stderr)
        print(
            "Use an ops package exposing the required 910B3 sparse flash MLA APIs. "
            "GradMetadata symbols are optional on 910B3 (the pinned MindSpeed "
            "wrapper uses them only on Ascend 950).",
            file=sys.stderr,
        )
        return 1

    optional_missing = [symbol for symbol in OPTIONAL_SPARSE_MLA_SYMBOLS if symbol not in found]
    if optional_missing:
        print("Optional Ascend 950-only APIs absent on this runtime:")
        for symbol in optional_missing:
            print(f"  {symbol}")
    print("MINDSPEED_V4_SPARSE_MLA_APIS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

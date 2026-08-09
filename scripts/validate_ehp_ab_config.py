#!/usr/bin/env python3
"""Validate that a production EHP A/B pair differs only in EHP controls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from materialize_6node48_config import load_inherited_config
except ModuleNotFoundError:  # Imported as scripts.validate_ehp_ab_config in tests.
    from scripts.materialize_6node48_config import load_inherited_config


ALLOWED_DIFFERENCES = {
    "wandb_run_name",
    "history_experiment_name",
    "global_resource_planner.elastic_hybrid_planning_enabled",
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(item, path))
    return flattened


def validate(no_ehp_path: Path, ehp_path: Path) -> list[str]:
    no_ehp = load_inherited_config(no_ehp_path)
    ehp = load_inherited_config(ehp_path)
    left = _flatten(no_ehp)
    right = _flatten(ehp)
    differences = sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )
    unexpected = [key for key in differences if key not in ALLOWED_DIFFERENCES]
    if unexpected:
        details = "\n".join(
            f"  {key}: no-EHP={left.get(key)!r}, EHP={right.get(key)!r}"
            for key in unexpected
        )
        raise ValueError(f"non-EHP variables differ:\n{details}")

    required = ALLOWED_DIFFERENCES - {"wandb_run_name", "history_experiment_name"}
    missing = sorted(required - set(differences))
    if missing:
        raise ValueError(f"expected EHP controls do not differ: {missing}")

    batch_size = int(no_ehp["batch_size"])
    dp_size = int(no_ehp["train_dp_size"])
    n_samples = int(no_ehp["n_samples"])
    if batch_size % dp_size:
        raise ValueError("batch_size must be divisible by train_dp_size")
    local_batch = batch_size // dp_size
    if local_batch % n_samples:
        raise ValueError("local batch per DP replica must contain complete GRPO groups")

    print("EHP A/B config validation passed")
    print(f"global trajectories/step={batch_size}")
    print(f"prompt groups/step={batch_size // n_samples}")
    print(f"local trajectories/DP={local_batch}")
    print(f"local prompt groups/DP={local_batch // n_samples}")
    print("differences:")
    for key in differences:
        print(f"  {key}: no-EHP={left.get(key)!r}, EHP={right.get(key)!r}")
    return differences


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-ehp",
        type=Path,
        default=project
        / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_no_ehp.yaml",
    )
    parser.add_argument(
        "--ehp",
        type=Path,
        default=project
        / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml",
    )
    args = parser.parse_args()
    validate(args.no_ehp, args.ehp)


if __name__ == "__main__":
    main()

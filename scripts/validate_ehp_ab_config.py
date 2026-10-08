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
    "global_resource_planner.runtime_online_replanning",
    "global_resource_planner.runtime_dynamic_reconfiguration_enabled",
    "global_resource_planner.runtime_reconfigure_training",
    "global_resource_planner.hybrid_worker_launch_enabled",
    "global_resource_planner.hybrid_worker_remote_control_enabled",
    "global_resource_planner.elastic_hybrid_require_isolated_ccl",
}

EHP_RUNTIME_CONTROLS = {
    "elastic_hybrid_planning_enabled": True,
    "runtime_online_replanning": True,
    "runtime_dynamic_reconfiguration_enabled": True,
    "runtime_reconfigure_training": True,
    "hybrid_worker_launch_enabled": True,
    "hybrid_worker_remote_control_enabled": True,
    "elastic_hybrid_require_isolated_ccl": True,
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

    for name, ehp_expected in EHP_RUNTIME_CONTROLS.items():
        no_ehp_expected = not ehp_expected
        no_ehp_actual = bool(no_ehp["global_resource_planner"].get(name, False))
        ehp_actual = bool(ehp["global_resource_planner"].get(name, False))
        if (no_ehp_actual, ehp_actual) != (no_ehp_expected, ehp_expected):
            raise ValueError(
                f"invalid runtime control {name}: "
                f"no-EHP={no_ehp_actual}, EHP={ehp_actual}"
            )

    for label, config in (("no-EHP", no_ehp), ("EHP", ehp)):
        if int(config["total_steps"]) != 200:
            raise ValueError(f"{label} total_steps must be 200")
        if int(config["sync_interval"]) != 5:
            raise ValueError(f"{label} sync_interval must be 5")
        planner = config["global_resource_planner"]
        if str(planner.get("initial_allocation_strategy")) != "grp":
            raise ValueError(f"{label} initial allocation must be owned by GRP")
        if int(planner.get("fixed_train_gpus", -1)) != 0:
            raise ValueError(f"{label} fixed_train_gpus must be zero")

    batch_size = int(no_ehp["batch_size"])
    dp_size = int(no_ehp["train_dp_size"])
    n_samples = int(no_ehp["n_samples"])
    if batch_size % dp_size:
        raise ValueError("batch_size must be divisible by train_dp_size")
    local_batch = batch_size // dp_size
    if local_batch % n_samples:
        raise ValueError("local batch per DP replica must contain complete GRPO groups")

    print("EHP A/B config validation passed")
    print("steps/arm=200, rollout-training sync interval=5")
    print("initial allocation=GRP for both arms")
    print("runtime allocation=no-EHP fixed, EHP planner-elastic")
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
        / "configs/r2e_gym_qwen3_14b_mcore_gpu_6node48_production_no_ehp.yaml",
    )
    parser.add_argument(
        "--ehp",
        type=Path,
        default=project
        / "configs/r2e_gym_qwen3_14b_mcore_gpu_6node48_production_ehp.yaml",
    )
    args = parser.parse_args()
    validate(args.no_ehp, args.ehp)


if __name__ == "__main__":
    main()

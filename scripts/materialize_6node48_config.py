#!/usr/bin/env python3
"""Materialize a six-node config for an arbitrary idle-node placement."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def _absolute_reference(value: object, source_dir: Path) -> object:
    if not isinstance(value, str) or os.path.isabs(value):
        return value
    return str((source_dir / value).resolve())


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge config mappings without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_inherited_config(template: Path, seen: set[Path] | None = None) -> dict:
    """Load a YAML config and flatten its recursive ``base_config`` chain.

    The production EHP and no-EHP arms intentionally inherit one common
    training config.  Materialization must therefore resolve inheritance before
    rewriting node- and run-specific fields.
    """
    template = template.resolve()
    seen = set() if seen is None else set(seen)
    if template in seen:
        chain = " -> ".join(str(path) for path in [*seen, template])
        raise ValueError(f"cyclic base_config chain: {chain}")
    seen.add(template)

    with template.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {template}")

    source_dir = template.parent
    base_value = raw.pop("base_config", None)
    base = {}
    if base_value:
        base_path = Path(_absolute_reference(base_value, source_dir))
        base = load_inherited_config(base_path, seen)

    # These references are resolved relative to the file that declares them,
    # not relative to the eventual run directory containing effective_config.
    for key in ("hardware_config", "model_arch_config"):
        if key in raw:
            raw[key] = _absolute_reference(raw[key], source_dir)
    return _deep_merge(base, raw)


def materialize(
    template: Path,
    output: Path,
    master_addr: str,
    master_port: int,
    rollout_hosts: list[str],
    run_root: Path,
    gradient_port: int,
    model_path: str | None = None,
) -> None:
    if not rollout_hosts or len(set(rollout_hosts)) != len(rollout_hosts):
        raise ValueError("at least one distinct rollout host is required")

    config = load_inherited_config(template)

    if model_path:
        config["model_path"] = model_path

    config["master_addr"] = master_addr
    config["master_port"] = master_port
    config["rollout_weight_sync_control_dir"] = str(run_root / "rollout_weight_sync")
    config["sync_path"] = str(run_root / "weights")
    config["log_dir"] = str(run_root / "logs")
    config["history_output_dir"] = str(run_root / "history")

    rollout = config.setdefault("heterogeneous_rollout", {})
    rollout["vllm_host"] = rollout_hosts[0]
    instances = rollout.get("instances") or []
    planner = config.setdefault("global_resource_planner", {})
    node_pattern = [int(tp) for tp in planner.get("rollout_node_tp_pattern", [])]
    if not node_pattern:
        raise ValueError("global_resource_planner.rollout_node_tp_pattern is required")
    if any(tp <= 0 for tp in node_pattern) or sum(node_pattern) != 8:
        raise ValueError(
            "rollout_node_tp_pattern must contain positive TP degrees totaling 8 GPUs: "
            f"{node_pattern}"
        )
    expected_instances = len(rollout_hosts) * len(node_pattern)
    if len(instances) != expected_instances:
        raise ValueError(
            "planned rollout instance count does not match the per-node pattern: "
            f"instances={len(instances)} hosts={len(rollout_hosts)} "
            f"pattern={node_pattern}"
        )

    legacy_suffixes = {
        (1, 0): "short_tp1_0",
        (1, 1): "short_tp1_1",
        (2, 0): "medium_tp2",
        (4, 0): "long_tp4",
    }
    for host_index, host in enumerate(rollout_hosts):
        node_id = f"n{host.rsplit('.', 1)[-1]}"
        device_offset = 0
        tp_occurrences: dict[int, int] = {}
        start = host_index * len(node_pattern)
        host_instances = instances[start:start + len(node_pattern)]
        actual_pattern = [
            max(1, int(instance.get("tp", len(instance.get("gpus", [])) or 1)))
            for instance in host_instances
        ]
        if actual_pattern != node_pattern:
            raise ValueError(
                f"rollout host {host} receives pattern {actual_pattern}, "
                f"expected {node_pattern}"
            )
        for local_index, (instance, tp) in enumerate(zip(host_instances, node_pattern)):
            occurrence = tp_occurrences.get(tp, 0)
            if node_pattern == [1, 1, 2, 4]:
                suffix = legacy_suffixes[(tp, occurrence)]
            else:
                suffix = f"tp{tp}_{occurrence}"
            tp_occurrences[tp] = occurrence + 1
            instance["host"] = host
            instance["gpus"] = list(range(device_offset, device_offset + tp))
            instance["port"] = 8000 + local_index
            instance["instance_id"] = f"{node_id}_{suffix}"
            device_offset += tp

    scheduling = rollout.setdefault("scheduling", {})
    scheduling["cmlfq_tree_path"] = str(run_root / "cmlfq_tree.json")
    scheduling["cmlfq_shared_load_dir"] = str(run_root / "cmlfq_shared_load")

    planner["gradient_server_public_host"] = master_addr
    planner["gradient_server_port"] = gradient_port
    if "hybrid_worker_task_dir" in planner:
        planner["hybrid_worker_task_dir"] = str(run_root / "elastic_training_tasks")
    if "hybrid_worker_remote_control_dir" in planner:
        planner["hybrid_worker_remote_control_dir"] = str(
            run_root / "elastic_training_tasks"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--rollout-host", action="append", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gradient-port", required=True, type=int)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    materialize(
        template=args.template,
        output=args.output,
        master_addr=args.master_addr,
        master_port=args.master_port,
        rollout_hosts=args.rollout_host,
        run_root=args.run_root,
        gradient_port=args.gradient_port,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()

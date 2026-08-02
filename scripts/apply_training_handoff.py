#!/usr/bin/env python3
"""Materialize a supervised Megatron training handoff into a runtime config.

The parent Slurm allocation owns all nodes.  This helper is intentionally
small: it rewrites only topology and rollout placement after all old training
ranks have checkpointed and exited.  The next torchrun invocation can then
load the same distributed checkpoint with a new data-parallel degree.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import yaml


def _pack_instances(tp_list: list[int], rollout_nodes: list[str], base_port: int):
    remaining = {node: list(range(4)) for node in rollout_nodes}
    instances = []
    for index, tp in enumerate(tp_list):
        node = next((name for name in rollout_nodes if len(remaining[name]) >= tp), None)
        if node is None:
            raise ValueError(
                f"cannot place TP={tp} on rollout nodes {rollout_nodes}; "
                f"remaining={remaining}"
            )
        gpus = remaining[node][:tp]
        remaining[node] = remaining[node][tp:]
        instances.append(
            {
                "instance_id": f"handoff_tp{tp}_{index}",
                "tp": int(tp),
                "gpus": gpus,
                "host": node,
                "port": int(base_port) + index,
                "description": "supervised_training_handoff",
            }
        )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--nodes", required=True, help="comma-separated Slurm allocation")
    parser.add_argument("--output-env", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    handoff_path = Path(args.handoff)
    nodes = [node.strip() for node in args.nodes.split(",") if node.strip()]
    if not nodes:
        raise ValueError("--nodes must not be empty")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    plan = handoff["plan"]
    train = plan["train"]
    rollout = plan["rollout"]
    train_gpus = int(train["n_gpus"])
    tp = int(train["tp"])
    pp = int(train["pp"])
    if train_gpus <= 0 or train_gpus % 4:
        raise ValueError(
            "supervised handoff currently requires a whole-node training pool; "
            f"got train_gpus={train_gpus}"
        )
    if train_gpus % max(1, tp * pp):
        raise ValueError("target train GPUs must divide TP*PP")

    train_node_count = train_gpus // 4
    if train_node_count >= len(nodes):
        raise ValueError(
            f"target train pool consumes all nodes: train_nodes={train_node_count}, "
            f"allocated={nodes}"
        )
    train_nodes = nodes[:train_node_count]
    rollout_nodes = nodes[train_node_count:]
    tp_list = [int(value) for value in rollout["tp_list"]]
    if sum(tp_list) != len(rollout_nodes) * 4:
        raise ValueError(
            f"target rollout TP list {tp_list} does not fill available rollout "
            f"nodes {rollout_nodes}"
        )

    hetero = config["heterogeneous_rollout"]
    instances = _pack_instances(tp_list, rollout_nodes, int(hetero["vllm_base_port"]))
    config["train_gpus"] = train_gpus
    config["rollout_gpus"] = sum(tp_list)
    config["n_total_gpus"] = train_gpus + sum(tp_list)
    config["train_tp_size"] = tp
    config["tp_size"] = tp
    config["train_pp_size"] = pp
    config["train_dp_size"] = train_gpus // max(1, tp * pp)
    config["micro_batch_size"] = int(train["b_micro"])
    config["max_concurrent_rollouts"] = int(plan["max_concurrent_rollouts"])
    hetero["total_gpus"] = sum(tp_list)
    hetero["available_gpus"] = list(range(4))
    hetero["instances"] = instances

    planner = config["global_resource_planner"]
    # The handoff target is a one-shot physical transition, not a permanent
    # clamp.  Clearing these guards lets the next online GRP decision grow the
    # training pool again when rollout pressure subsides.
    planner["fixed_train_gpus"] = 0
    planner["runtime_training_pool_target_gpus"] = 0
    planner["runtime_forced_train_gpus"] = 0
    planner["runtime_forced_rollout_tp_list"] = []
    planner["runtime_force_reconfigure"] = False

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # Prefix relaunch variables so sourcing this file cannot clobber the shell
    # arrays that describe the currently running pool before they are rebuilt.
    env_lines = {
        "NEXT_ELASTIC_RESUME_STEP": int(handoff["resume_step"]),
        "NEXT_ELASTIC_RESUME_VERSION": int(handoff["checkpoint_version"]),
        "NEXT_TRAIN_NODE_COUNT": train_node_count,
        "NEXT_TRAIN_NODE_LIST": " ".join(train_nodes),
        "NEXT_ROLLOUT_NODE_LIST": " ".join(rollout_nodes),
        "NEXT_ROLLOUT_INSTANCE_NODE_LIST": " ".join(item["host"] for item in instances),
        "NEXT_ROLLOUT_COUNT": len(instances),
        "NEXT_ROLLOUT_TP_LIST": " ".join(str(item["tp"]) for item in instances),
        "NEXT_ROLLOUT_PORT_LIST": " ".join(str(item["port"]) for item in instances),
        "NEXT_ROLLOUT_NAME_LIST": " ".join(item["instance_id"] for item in instances),
        "NEXT_ROLLOUT_GPU_LIST": " ".join(",".join(str(gpu) for gpu in item["gpus"]) for item in instances),
        "NEXT_HETERO_INSTANCE_HOSTS": ",".join(item["host"] for item in instances),
        "NEXT_MASTER_ADDR": train_nodes[0],
        "NEXT_ROLLOUT_MODEL_PATH": str(handoff["checkpoint_path"]),
    }
    output = Path(args.output_env)
    output.write_text(
        "\n".join(
            f"{key}={shlex.quote(str(value))}" for key, value in env_lines.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"train_nodes": train_nodes, "instances": instances}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plan and materialize an NPU-level GRP or fixed control placement."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner
from RL_Framework.infra.cost_model.preflight_planner import (
    PreflightPlanner,
    load_history_jsonl,
)


def _pack_instances(
    hosts: list[str],
    gpus_per_host: int,
    tp_list: list[int],
) -> tuple[list[dict], dict[str, list[int]]]:
    free = {host: list(range(gpus_per_host)) for host in hosts}
    local_ports: dict[str, int] = defaultdict(int)
    instances: list[dict] = []

    # TP sizes are powers of two on the supported vLLM backend. Best-fit
    # packing preserves the planner's exact NPU count without node rounding.
    for index, tp in enumerate(sorted((int(value) for value in tp_list), reverse=True)):
        candidates = [
            (len(devices) - tp, host)
            for host, devices in free.items()
            if len(devices) >= tp
        ]
        if not candidates:
            raise RuntimeError(f"cannot place rollout TP={tp} on the selected hosts")
        _, host = min(candidates)
        devices = free[host][:tp]
        free[host] = free[host][tp:]
        port = 8000 + local_ports[host]
        local_ports[host] += 1
        instances.append(
            {
                "instance_id": f"n{host.rsplit('.', 1)[-1]}_grp_tp{tp}_{index}",
                "host": host,
                "tp": tp,
                "gpus": devices,
                "port": port,
            }
        )
    return instances, free


def _fixed_equal_placement(
    config: AsyncRLConfig,
    hosts: list[str],
    gpus_per_host: int,
) -> tuple[list[dict], dict[str, list[int]]]:
    train_gpus = int(config.train_gpus)
    rollout_gpus = int(config.rollout_gpus)
    if train_gpus + rollout_gpus != len(hosts) * gpus_per_host:
        raise ValueError("fixed allocation must consume the complete NPU budget")

    flat = [(host, device) for host in hosts for device in range(gpus_per_host)]
    train_devices = flat[:train_gpus]
    rollout_devices = flat[train_gpus:]
    rollout_by_host: dict[str, list[int]] = defaultdict(list)
    for host, device in rollout_devices:
        rollout_by_host[host].append(device)

    tp_list = [int(tp) for tp in config.heterogeneous_rollout.tp_list]
    if sum(tp_list) != rollout_gpus:
        raise ValueError(
            f"fixed rollout topology uses {sum(tp_list)} NPUs, expected {rollout_gpus}"
        )
    instances, remaining = _pack_instances(
        [host for host in hosts if rollout_by_host.get(host)],
        gpus_per_host,
        tp_list,
    )
    allowed = {(host, device) for host, devices in rollout_by_host.items() for device in devices}
    used = {
        (instance["host"], device)
        for instance in instances
        for device in instance["gpus"]
    }
    if used != allowed:
        raise RuntimeError("fixed rollout placement did not match the 24/24 split")
    train_map: dict[str, list[int]] = defaultdict(list)
    for host, device in train_devices:
        train_map[host].append(device)
    return instances, dict(train_map)


def _write_materialized(
    config: AsyncRLConfig,
    instances: list[dict],
    train_devices: dict[str, list[int]],
    output_config: Path,
    placement_json: Path,
    run_root: Path,
    master_port: int,
    strategy: str,
    decision: dict | None,
) -> None:
    train_flat = [
        {"rank": rank, "host": host, "device": device}
        for rank, (host, device) in enumerate(
            (pair for host, devices in train_devices.items() for pair in ((host, d) for d in devices))
        )
    ]
    if len(train_flat) != int(config.train_gpus):
        raise RuntimeError(
            f"training placement has {len(train_flat)} NPUs, expected {config.train_gpus}"
        )
    if not train_flat or not instances:
        raise RuntimeError("both training and rollout must receive at least one NPU")

    master_addr = train_flat[0]["host"]
    config.master_addr = master_addr
    config.master_port = master_port
    config.num_nodes = len(train_devices)
    config.train_gpus_per_node = 0
    config.rollout_gpus_per_node = 0
    config.rollout_weight_sync_control_dir = str(run_root / "rollout_weight_sync")
    config.sync_path = str(run_root / "weights")
    config.log_dir = str(run_root / "logs")
    config.history_output_dir = str(run_root / "history")

    hetero = config.heterogeneous_rollout
    hetero.enabled = True
    hetero.total_gpus = sum(int(item["tp"]) for item in instances)
    hetero.vllm_host = instances[0]["host"]
    hetero.instances = [HeterogeneousInstanceConfig.from_dict(dict(item)) for item in instances]
    hetero.scheduling.cmlfq_tree_path = str(run_root / "cmlfq_tree.json")
    hetero.scheduling.cmlfq_shared_load_dir = str(run_root / "cmlfq_shared_load")

    planner = config.global_resource_planner
    planner.gradient_server_public_host = master_addr
    planner.runtime_length_profile_enabled = True
    planner.runtime_length_profile_jsonl = str(run_root / "history" / "runtime_length_profile.jsonl")

    output_config.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(output_config))
    payload = {
        "strategy": strategy,
        "total_npus": int(config.n_total_gpus),
        "train_gpus": int(config.train_gpus),
        "rollout_gpus": int(config.rollout_gpus),
        "train_topology": {
            "tp": int(config.train_tp_size),
            "pp": int(config.train_pp_size),
            "cp": int(config.train_cp_size),
            "dp": int(config.train_dp_size),
            "micro_batch_size": int(config.micro_batch_size),
        },
        "train_devices": train_flat,
        "rollout_instances": instances,
        "decision": decision,
    }
    placement_json.parent.mkdir(parents=True, exist_ok=True)
    placement_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "[DevicePlacement] "
        f"strategy={strategy} train_gpus={config.train_gpus} "
        f"rollout_gpus={config.rollout_gpus} "
        f"train_tp={config.train_tp_size} train_pp={config.train_pp_size} "
        f"train_dp={config.train_dp_size} "
        f"rollout_tp={[item['tp'] for item in instances]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("grp", "fixed"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--placement-json", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--gpus-per-host", type=int, default=8)
    parser.add_argument("--history-jsonl", type=Path)
    args = parser.parse_args()

    hosts = list(dict.fromkeys(args.host))
    if len(hosts) != len(args.host):
        raise ValueError("hosts must be distinct")
    config = AsyncRLConfig.from_yaml(str(args.config))
    config.n_total_gpus = len(hosts) * args.gpus_per_host

    decision = None
    if args.mode == "grp":
        if not args.history_jsonl:
            raise ValueError("GRP mode requires --history-jsonl")
        planner = config.global_resource_planner
        planner.enabled = True
        planner.initial_allocation_strategy = "grp"
        planner.initial_allocation_applied = False
        planner.fixed_train_gpus = 0
        history = load_history_jsonl(args.history_jsonl)
        global_planner = GlobalResourcePlanner.from_config(config)
        global_planner.optimizer.max_rollout_instances = config.n_total_gpus
        result = PreflightPlanner(
            config,
            planner=global_planner,
            apply_best_candidate=True,
        ).run(history)
        config = result.planned_config
        decision = result.to_dict()
        tp_list = [int(tp) for tp in config.heterogeneous_rollout.tp_list]
        instances, remaining = _pack_instances(hosts, args.gpus_per_host, tp_list)
        train_devices = {host: devices for host, devices in remaining.items() if devices}
        strategy = "grp_unrestricted_npu"
    else:
        instances, train_devices = _fixed_equal_placement(
            config, hosts, args.gpus_per_host
        )
        strategy = "fixed_equal_24_24"

    _write_materialized(
        config=config,
        instances=instances,
        train_devices=train_devices,
        output_config=args.output_config,
        placement_json=args.placement_json,
        run_root=args.run_root,
        master_port=args.master_port,
        strategy=strategy,
        decision=decision,
    )


if __name__ == "__main__":
    main()

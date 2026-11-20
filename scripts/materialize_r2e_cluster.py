#!/usr/bin/env python3
"""Plan disjoint Core/rollout GPU placement within a Slurm allocation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from RL_Framework.config import HeterogeneousInstanceConfig, load_config
from RL_Framework.infra.cost_model.preflight_planner import PreflightPlanner, load_history_jsonl
from RL_Framework.infra.cost_model.resource_alloc import split_gpu_slots


def _hosts(hosts, gpus_per_host):
    result = list(hosts)
    if (not result or any(not isinstance(host, str) or not host.strip() for host in result)
            or len(set(result)) != len(result)):
        raise ValueError("Expected distinct, nonempty allocation hostnames")
    if not isinstance(gpus_per_host, int) or gpus_per_host < 1:
        raise ValueError("gpus_per_host must be positive")
    return result


def _materialize(config, hosts, gpus_per_host):
    """Bind a selected logical partition to disjoint physical GPU slots."""
    planned = deepcopy(config)
    hosts = _hosts(hosts, gpus_per_host)
    total = len(hosts) * gpus_per_host
    if (int(planned.n_total_gpus) != total
            or int(planned.train_gpus) + int(planned.rollout_gpus) != total):
        raise ValueError("The selected resource plan must consume the complete allocation")
    if int(planned.train_gpus) < 1 or int(planned.rollout_gpus) < 1:
        raise ValueError("Core and rollout must each have at least one GPU")
    if planned.train_gpus > gpus_per_host and planned.train_gpus % gpus_per_host:
        raise ValueError("Multi-node Core allocations must contain complete nodes")
    training_width = (int(planned.train_tp_size) * int(planned.train_pp_size)
        * int(planned.train_dp_size) * int(planned.train_cp_size))
    if training_width != int(planned.train_gpus):
        raise ValueError("The Core parallel dimensions do not cover its selected GPUs")
    train_gpu_ids, pools = split_gpu_slots(hosts, gpus_per_host, int(planned.train_gpus))
    train_hosts = list(train_gpu_ids)
    rollout_gpu_ids = deepcopy(pools)
    rollout_hosts = list(pools)
    instances = planned.heterogeneous_rollout.instances
    if not instances or sum(int(instance.tp) for instance in instances) != planned.rollout_gpus:
        raise ValueError("Rollout TP degrees do not cover the selected rollout GPUs")
    if len({instance.instance_id for instance in instances}) != len(instances):
        raise ValueError("Rollout instance IDs must be globally unique")
    # Allocate large engines first so a valid TP4/TP2/TP1 partition cannot be
    # fragmented by the optimizer's presentation order. Keep instance order/ID.
    for instance in sorted(instances, key=lambda item: int(item.tp), reverse=True):
        tp = int(instance.tp)
        if tp < 1 or tp > gpus_per_host:
            raise ValueError("Every rollout TP engine must fit within one allocated host")
        host = next((host for host, available in pools.items() if len(available) >= tp), None)
        if host is None:
            raise ValueError("The rollout TP partition cannot be packed into allocated GPU slots")
        instance.host, instance.gpus = host, pools[host][:tp]
        del pools[host][:tp]
    if any(pools.values()):
        raise ValueError("The rollout plan left allocated GPUs unused")
    # Ports only need to be unique within a host; assign globally unique ones
    # to preserve the old endpoint/index convention used by the controllers.
    base_port = int(planned.heterogeneous_rollout.vllm_base_port)
    if not 0 < base_port <= 65535 - len(instances) + 1:
        raise ValueError("The configured rollout port range is invalid")
    for index, instance in enumerate(instances):
        instance.port = base_port + index
    planned.num_nodes = len(train_hosts)  # torchrun Core nodes, not Slurm nodes.
    planned.gpus_per_node_override = gpus_per_host
    planned.train_gpus_per_node = min(gpus_per_host, int(planned.train_gpus))
    planned.rollout_gpus_per_node = gpus_per_host
    planned.hardware.gpus_per_node = gpus_per_host
    hetero = planned.heterogeneous_rollout
    hetero.vllm_host = rollout_hosts[0]
    hetero.total_gpus = int(planned.rollout_gpus)
    hetero.available_gpus = list(range(gpus_per_host))  # IDs are host-local.
    planned.vllm_num_instances = len(instances)
    planned.vllm_endpoints = ",".join(f"{item.host}:{item.port}" for item in instances)
    placement = dict(strategy="grp", allocation_hosts=hosts, allocation_nodes=len(hosts),
        n_total_gpus=total, gpus_per_host=gpus_per_host,
        train_hosts=train_hosts, rollout_hosts=rollout_hosts,
        rollout_controller_hosts=list(rollout_hosts),
        train_gpus=int(planned.train_gpus), rollout_gpus=int(planned.rollout_gpus),
        training_world_size=training_width,
        train_gpu_ids=train_gpu_ids,
        rollout_gpu_ids=rollout_gpu_ids,
        rollout_instances=[dict(instance_id=item.instance_id, host=item.host,
            port=item.port, tp=int(item.tp), gpus=list(item.gpus)) for item in instances])
    return planned, placement


def seed_cluster_config(config, hosts, gpus_per_host=4):
    """Create a legal physical seed while leaving the complete GRP search open."""
    seeded = deepcopy(config)
    hosts = _hosts(hosts, gpus_per_host)
    seeded.n_total_gpus = len(hosts) * gpus_per_host
    seeded.rollout_gpus = seeded.n_total_gpus - int(seeded.train_gpus)
    if seeded.rollout_gpus < 1:
        raise ValueError("The seed Core allocation must leave rollout GPUs")
    planner = seeded.global_resource_planner
    planner.initial_allocation_strategy = "grp"
    planner.initial_allocation_applied = False
    planner.fixed_train_gpus = 0
    core_unit = min(gpus_per_host, int(seeded.train_gpus))
    planner.allocation_granularity_gpus = max(core_unit, int(planner.allocation_granularity_gpus))
    planner.min_train_gpus = max(core_unit, int(planner.min_train_gpus))
    # TP degrees greater than a physical node were never launchable in this
    # cluster entry; reject them rather than silently shrinking the search.
    allowed = planner.allowed_rollout_tp or [1, 2, 4, 8]
    if not any(0 < int(tp) <= gpus_per_host for tp in allowed):
        raise ValueError("The rollout search has no host-local TP degree")
    if any(int(tp) > gpus_per_host for tp in allowed):
        raise ValueError("The rollout TP search includes degrees wider than one host")
    existing = list(seeded.heterogeneous_rollout.instances)
    pattern = []
    if existing:
        first_host = existing[0].host
        pattern = [int(item.tp) for item in existing if item.host == first_host]
    if sum(pattern) != gpus_per_host:
        preferred = max(int(tp) for tp in allowed if 0 < int(tp) <= gpus_per_host
                        and gpus_per_host % int(tp) == 0)
        pattern = [preferred] * (gpus_per_host // preferred)
    if existing and sum(int(item.tp) for item in existing) == seeded.rollout_gpus:
        degrees = [int(item.tp) for item in existing]
    else:
        count, remaining = divmod(seeded.rollout_gpus, gpus_per_host)
        degrees = pattern * count
        while remaining:
            tp = max((int(tp) for tp in allowed if 0 < int(tp) <= remaining), default=0)
            if not tp:
                raise ValueError("The rollout search cannot cover the remaining GPU slots")
            degrees.append(tp)
            remaining -= tp
    seeded.heterogeneous_rollout.instances = [HeterogeneousInstanceConfig(
        instance_id=f"seed_tp{tp}_{index}", tp=tp, gpus=list(range(tp)))
        for index, tp in enumerate(degrees)]
    return _materialize(seeded, hosts, gpus_per_host)[0]


def materialize_planned_cluster(config, hosts, gpus_per_host=4):
    """Materialize a GRP-selected configuration without changing its decisions."""
    return _materialize(config, hosts, gpus_per_host)


def materialize_cluster(config, hosts, *, history, gpus_per_host=4):
    """Run the complete initial GRP once, then bind its plan to actual hosts."""
    hosts = tuple(hosts)
    seeded = seed_cluster_config(config, hosts, gpus_per_host)
    rows = list(history)
    if not rows:
        raise ValueError("Initial GRP requires a nonempty workload profile")
    result = PreflightPlanner(seeded, apply_best_candidate=True).run(rows, allocation_hosts=hosts)
    if not result.applied or result.applied_plan is None:
        raise RuntimeError("Initial GRP did not produce a feasible resource plan")
    planned, placement = materialize_planned_cluster(result.planned_config, hosts, gpus_per_host)
    placement["decision"] = result.to_dict()
    return planned, placement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--placement-json", required=True)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--gpus-per-host", type=int, default=4)
    parser.add_argument("--history-jsonl", required=True)
    args = parser.parse_args(argv)
    planned, placement = materialize_cluster(load_config(args.config), args.host,
        history=load_history_jsonl(args.history_jsonl), gpus_per_host=args.gpus_per_host)
    planned.to_yaml(args.output_config)
    output = Path(args.placement_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(placement, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

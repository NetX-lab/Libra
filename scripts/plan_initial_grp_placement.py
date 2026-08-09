#!/usr/bin/env python3
"""Choose the initial train/rollout split and whole-node placement with GRP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.preflight_planner import (
    PreflightPlanner,
    load_history_jsonl,
    synthetic_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--placement-json", required=True)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--gpus-per-host", type=int, default=8)
    parser.add_argument("--history-jsonl", default="")
    parser.add_argument("--synthetic-requests", type=int, default=32)
    parser.add_argument("--synthetic-input-len", type=int, default=1024)
    parser.add_argument("--synthetic-output-len", type=int, default=2048)
    args = parser.parse_args()

    hosts = list(dict.fromkeys(args.host))
    if len(hosts) != len(args.host):
        raise ValueError("candidate hosts must be unique")
    if args.gpus_per_host <= 0:
        raise ValueError("gpus-per-host must be positive")

    config = AsyncRLConfig.from_yaml(args.config)
    physical_total = len(hosts) * args.gpus_per_host
    if config.n_total_gpus != physical_total:
        config.n_total_gpus = physical_total
    planner_cfg = config.global_resource_planner
    planner_cfg.initial_allocation_strategy = "grp"
    planner_cfg.fixed_train_gpus = 0
    planner_cfg.allocation_granularity_gpus = max(
        args.gpus_per_host, int(planner_cfg.allocation_granularity_gpus)
    )
    planner_cfg.min_train_gpus = max(args.gpus_per_host, planner_cfg.min_train_gpus)
    planner_cfg.min_rollout_gpus = max(
        args.gpus_per_host, planner_cfg.min_rollout_gpus
    )

    history = (
        load_history_jsonl(args.history_jsonl)
        if args.history_jsonl
        else synthetic_history(
            num_requests=args.synthetic_requests,
            input_len=args.synthetic_input_len,
            output_len=args.synthetic_output_len,
        )
    )
    result = PreflightPlanner(config, apply_best_candidate=True).run(history)
    planned = result.planned_config
    if planned.train_gpus % args.gpus_per_host:
        raise RuntimeError("GRP produced a non-node-aligned training allocation")
    if planned.rollout_gpus % args.gpus_per_host:
        raise RuntimeError("GRP produced a non-node-aligned rollout allocation")

    train_nodes = planned.train_gpus // args.gpus_per_host
    rollout_nodes = planned.rollout_gpus // args.gpus_per_host
    if train_nodes + rollout_nodes != len(hosts):
        raise RuntimeError("GRP placement does not consume the candidate host pool")

    train_hosts = hosts[:train_nodes]
    rollout_hosts = hosts[train_nodes:train_nodes + rollout_nodes]
    planned.num_nodes = train_nodes
    planned.train_gpus_per_node = args.gpus_per_host
    planned.rollout_gpus_per_node = args.gpus_per_host
    planned.to_yaml(args.output_config)

    payload = {
        "strategy": "grp",
        "gpus_per_host": args.gpus_per_host,
        "train_gpus": planned.train_gpus,
        "rollout_gpus": planned.rollout_gpus,
        "train_hosts": train_hosts,
        "rollout_hosts": rollout_hosts,
        "decision": result.to_dict(),
    }
    placement_path = Path(args.placement_json)
    placement_path.parent.mkdir(parents=True, exist_ok=True)
    placement_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "[InitialGRPPlacement] "
        f"train_gpus={planned.train_gpus} train_hosts={','.join(train_hosts)} "
        f"rollout_gpus={planned.rollout_gpus} "
        f"rollout_hosts={','.join(rollout_hosts)}"
    )


if __name__ == "__main__":
    main()

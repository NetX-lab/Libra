#!/usr/bin/env python3
"""Create and validate the GRP startup config for DeepSeek-V4 R2E runs."""
import argparse
import json
from pathlib import Path

from RL_Framework import AsyncRLConfig
from RL_Framework.infra.cost_model.preflight_planner import (
    PreflightPlanner,
    synthetic_history,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

source = AsyncRLConfig.from_yaml(args.config)
expected_rollout = [
    (str(item.host), int(item.port), str(item.instance_id))
    for item in source.heterogeneous_rollout.instances
]
result = PreflightPlanner(source).run(
    synthetic_history(num_requests=32, input_len=1024, output_len=128)
)
if not result.applied:
    raise SystemExit("GRP preflight did not produce an applied plan")

planned = result.planned_config
# GRP plans the train/rollout split. The benchmark harness owns the already
# deployed rollout APIs, whose endpoints must stay fixed across experiment arms.
planned.heterogeneous_rollout = source.heterogeneous_rollout
actual_rollout = [
    (str(item.host), int(item.port), str(item.instance_id))
    for item in planned.heterogeneous_rollout.instances
]
if actual_rollout != expected_rollout:
    raise SystemExit(
        f"preflight changed rollout endpoints: expected={expected_rollout}, "
        f"actual={actual_rollout}"
    )
if planned.train_gpus + planned.rollout_gpus != planned.n_total_gpus:
    raise SystemExit("planned train/rollout allocation does not sum to total GPUs")
if not planned.global_resource_planner.initial_allocation_applied:
    raise SystemExit("preflight plan is missing initial_allocation_applied proof")

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
planned.to_yaml(str(out))
# Validate the serialized artifact too; the trainer will load this exact file.
roundtrip = AsyncRLConfig.from_yaml(str(out))
roundtrip_rollout = [
    (str(item.host), int(item.port), str(item.instance_id))
    for item in roundtrip.heterogeneous_rollout.instances
]
if roundtrip_rollout != expected_rollout:
    raise SystemExit(
        f"serialized planned config changed rollout endpoints: "
        f"expected={expected_rollout}, actual={roundtrip_rollout}"
    )
if not roundtrip.global_resource_planner.initial_allocation_applied:
    raise SystemExit("serialized planned config lost GRP startup proof")

Path(str(out) + ".decision.json").write_text(
    json.dumps(result.to_dict(), indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "train_gpus": roundtrip.train_gpus,
            "rollout_gpus": roundtrip.rollout_gpus,
            "n_total_gpus": roundtrip.n_total_gpus,
            "rollout_instances": roundtrip_rollout,
            "initial_allocation_applied": True,
            "planned_config": str(out),
        },
        indent=2,
    ),
    flush=True,
)

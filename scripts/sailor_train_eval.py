"""Evaluate one training candidate with Sailor's training simulator.

Command contract used by GlobalResourcePlanner:
  python scripts/sailor_train_eval.py INPUT_JSON OUTPUT_JSON --sailor-path PATH

INPUT_JSON is written by ``SailorTrainingSimulatorAdapter`` and contains a
TrainParallelConfig plus RL_Framework model/hardware metadata.  This wrapper
converts the candidate into a homogeneous Aceso-style plan and calls Sailor's
``SimulatorOP.simulate_iteration_time``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _uniform_stages(num_ops: int, pp: int) -> list[list[int]]:
    pp = max(1, min(pp, num_ops))
    base = num_ops // pp
    rem = num_ops % pp
    stages = []
    start = 0
    for idx in range(pp):
        size = base + (1 if idx < rem else 0)
        stages.append(list(range(start, start + size)))
        start += size
    return stages


def _build_plan(payload: dict[str, Any], training_config: dict[str, Any], args) -> dict[str, Any]:
    tc = payload["train_config"]
    tp = max(1, int(tc["tp"]))
    pp = max(1, int(tc["pp"]))
    dp = max(1, int(tc["dp"]))
    per_gpu_micro = max(1, int(tc.get("b_micro", 1)))
    num_ops = int(training_config.get("num_all_layers") or training_config["num_layers"])
    stages = _uniform_stages(num_ops, pp)

    return {
        "name": "Aceso",
        "gpu_type": args.gpu_type,
        "num_gpus_per_node": args.gpus_per_node,
        "num_stages": len(stages),
        "num_ops_in_each_stage": [len(stage) for stage in stages],
        "model_parallel_size_of_each_op": [[tp for _ in stage] for stage in stages],
        "data_parallel_size_of_each_op": [[dp for _ in stage] for stage in stages],
        "algo_of_each_op": [[0 for _ in stage] for stage in stages],
        "micro_batch_size": per_gpu_micro * dp,
        "global_batch_size": int(payload["batch_size"]),
        "num_gpus": [tp * dp for _ in stages],
        "used_gpus": {args.gpu_type: tp * dp * len(stages)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    parser.add_argument("--sailor-path", required=True)
    parser.add_argument("--model-name", default="OPT-350")
    parser.add_argument("--gpu-type", default="A100-40")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--training-config-json", default="")
    parser.add_argument("--profile-json", default="")
    parser.add_argument("--llm-info-json", default="")
    args = parser.parse_args()

    sailor_path = Path(args.sailor_path).expanduser().resolve()
    sys.path.insert(0, str(sailor_path))
    os.environ.setdefault("PYTHONPATH", str(sailor_path))
    # Sailor's SimulatorOP currently reads this file from ~/sailor.
    os.environ["HOME"] = str(sailor_path.parent)

    payload = _load_json(Path(args.input_json))
    planner_dir = sailor_path / "sailor" / "Planner"
    training_config_path = (
        Path(args.training_config_json)
        if args.training_config_json
        else planner_dir / "simulations" / "configs" / "training_config_opt_350.json"
    )
    profile_path = (
        Path(args.profile_json)
        if args.profile_json
        else planner_dir / "simulations" / "profiles_tmp_aceso.json"
    )
    llm_info_path = (
        Path(args.llm_info_json)
        if args.llm_info_json
        else planner_dir / "llm_info_aceso.json"
    )

    training_config = _load_json(training_config_path)
    training_config["model"] = args.model_name
    training_config["global_batch_size"] = int(payload["batch_size"])
    profiles = _load_json(profile_path)
    llm_info = _load_json(llm_info_path)

    from sailor.Planner.simulations.runtime_simulator_op import SimulatorOP

    plan = _build_plan(payload, training_config, args)
    simulator = SimulatorOP(training_config, llm_info, args.fp16, profiles)
    fits = simulator.check_config_fits(plan)
    if not fits:
        result = {
            "t_train": math.inf,
            "is_oom": True,
            "backend": "sailor",
            "plan": plan,
        }
    else:
        iteration_time, comm_cost, reformed_plan = simulator.simulate_iteration_time(plan)
        result = {
            "t_train": float(iteration_time),
            "iteration_time_s": float(iteration_time),
            "communication_cost": float(comm_cost),
            "max_memory_mb": float(simulator.get_max_memory(plan)),
            "backend": "sailor",
            "plan": reformed_plan,
            "input_plan": plan,
        }

    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

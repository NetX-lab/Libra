"""Show a complete Global Resource Planner flow.

This example exercises the real planner path:

1. collect trajectory length history from a training batch
2. evaluate the current plan
3. search candidate train/rollout allocations
4. call Sailor/Vidur adapter command contracts for each evaluation
5. decide whether to reconfigure
6. apply the selected plan back to AsyncRLConfig

The bundled mini Sailor/Vidur commands are deterministic command-contract
simulators. They stand in for full Sailor/Vidur installations while using the
same adapter path that production commands use:

  Sailor command: reads {input_json}, writes {output_json}
  Vidur command:  reads {trace_csv}, writes {output_json}
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from RL_Framework.config import (
    AsyncRLConfig,
    HeterogeneousInstanceConfig,
    HeterogeneousRolloutConfig,
    SchedulingConfig,
)
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlanner,
)


def _write_command_contract_simulators(work_dir: Path) -> tuple[Path, Path, Path, Path]:
    sailor_root = work_dir / "sailor"
    vidur_root = work_dir / "vidur"
    sailor_root.mkdir()
    vidur_root.mkdir()

    sailor_cmd = work_dir / "sailor_train_contract.py"
    sailor_cmd.write_text(
        r'''
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
tc = payload["train_config"]
gpus = max(1, tc["tp"] * tc["pp"] * tc["dp"])
seq = max(1, payload["avg_sequence_length"])
batch = max(1, payload["batch_size"])

# A tiny Sailor-like training model: more training GPUs reduce compute time,
# while larger PP adds a small pipeline bubble.
iteration_time = (0.0010 * seq * batch) / gpus
iteration_time *= 1.0 + 0.08 * max(0, tc["pp"] - 1)
iteration_time *= 1.0 / max(1, tc["b_micro"])

json.dump(
    {
        "t_train": iteration_time,
        "simulator": "sailor-command-contract",
        "train_gpus": gpus,
        "tp": tc["tp"],
        "pp": tc["pp"],
        "dp": tc["dp"],
        "b_micro": tc["b_micro"],
    },
    open(sys.argv[2], "w", encoding="utf-8"),
)
'''.strip(),
        encoding="utf-8",
    )

    vidur_cmd = work_dir / "vidur_rollout_contract.py"
    vidur_cmd.write_text(
        r'''
import csv
import json
import sys

trace_csv, output_json, tp_list_raw = sys.argv[1], sys.argv[2], sys.argv[3]
tp_list = [int(x) for x in tp_list_raw.split(",") if x]
total_gpus = max(1, sum(tp_list))
max_tp = max(tp_list) if tp_list else 1
num_instances = max(1, len(tp_list))

work = 0.0
max_request = 0
with open(trace_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        prompt = int(float(row["num_prefill_tokens"]))
        decode = int(float(row["num_decode_tokens"]))
        tokens = prompt + decode
        max_request = max(max_request, tokens)
        # A tiny Vidur-like rollout model: decode dominates, larger TP helps
        # long requests, and more instances improve throughput.
        work += prompt * 0.0008 + decode * 0.006

makespan = work / (total_gpus * (1.0 + 0.18 * (max_tp - 1)))
makespan /= num_instances ** 0.25

json.dump(
    {
        "makespan": makespan,
        "simulator": "vidur-command-contract",
        "tp_list": tp_list,
        "rollout_gpus": total_gpus,
        "num_instances": num_instances,
        "max_request_tokens": max_request,
    },
    open(output_json, "w", encoding="utf-8"),
)
'''.strip(),
        encoding="utf-8",
    )

    return sailor_root, vidur_root, sailor_cmd, vidur_cmd


def _build_config(
    sailor_root: Path,
    vidur_root: Path,
    sailor_cmd: Path,
    vidur_cmd: Path,
) -> AsyncRLConfig:
    cfg = AsyncRLConfig(
        model_path="/tmp/fake-qwen",
        train_gpus=2,
        rollout_gpus=2,
        train_tp_size=1,
        train_pp_size=1,
        train_dp_size=2,
        micro_batch_size=1,
        batch_size=8,
        n_total_gpus=4,
        max_seq_length=32768,
        max_new_tokens=32768,
        max_concurrent_rollouts=8,
        heterogeneous_rollout=HeterogeneousRolloutConfig(
            enabled=True,
            total_gpus=2,
            available_gpus=[2, 3],
            instances=[
                HeterogeneousInstanceConfig(instance_id="short_tp1_0", tp=1, gpus=[2]),
                HeterogeneousInstanceConfig(instance_id="short_tp1_1", tp=1, gpus=[3]),
            ],
            scheduling=SchedulingConfig(
                scheduler_type="cmlfq",
                cmlfq_buckets={
                    "short": {"tp_degrees": [1], "max_tokens": 5000},
                    "long": {"tp_degrees": [2], "max_tokens": 50000},
                },
            ),
        ),
    )

    planner_cfg = cfg.global_resource_planner
    planner_cfg.enabled = True
    planner_cfg.train_backend = "sailor"
    planner_cfg.rollout_backend = "vidur"
    planner_cfg.plan_interval = 1
    planner_cfg.warmup_steps = 0
    planner_cfg.min_history_size = 4
    planner_cfg.min_gain_ratio = 0.0
    planner_cfg.reconfiguration_cost_s = 0.0
    planner_cfg.allowed_train_tp = [1, 2]
    planner_cfg.allowed_train_pp = [1]
    planner_cfg.allowed_rollout_tp = [1, 2]
    planner_cfg.micro_batch_sizes = [1, 2]
    planner_cfg.verbose = True
    planner_cfg.simulator_allow_fallback = False
    planner_cfg.sailor_path = str(sailor_root)
    planner_cfg.vidur_path = str(vidur_root)
    planner_cfg.sailor_train_command = (
        f"{sys.executable} {sailor_cmd} {{input_json}} {{output_json}}"
    )
    planner_cfg.vidur_rollout_command = (
        f"{sys.executable} {vidur_cmd} {{trace_csv}} {{output_json}} {{tp_list}}"
    )
    return cfg


def _print_plan(label: str, plan) -> None:
    print(f"\n[{label}]")
    if plan is None:
        print("  <none>")
        return
    d = plan.to_dict()
    print(
        "  train: "
        f"TP={d['train']['tp']} PP={d['train']['pp']} DP={d['train']['dp']} "
        f"micro={d['train']['b_micro']} GPUs={d['train']['n_gpus']}"
    )
    print(
        "  rollout: "
        f"tp_list={d['rollout']['tp_list']} GPUs={d['rollout']['n_gpus']} "
        f"instances={d['rollout']['n_instances']}"
    )
    print(
        "  predicted: "
        f"T_train={d['t_train']:.4f}s "
        f"T_rollout={d['t_rollout']:.4f}s "
        f"T_global={d['t_global']:.4f}s"
    )
    if d["expected_gain_ratio"]:
        print(
            "  gain: "
            f"net={d['expected_gain_s']:.4f}s "
            f"ratio={d['expected_gain_ratio']:.2%}"
        )
    metadata = d.get("metadata", {})
    if metadata:
        print(f"  metadata keys: {sorted(metadata.keys())}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="grp_full_flow_") as td:
        work_dir = Path(td)
        sailor_root, vidur_root, sailor_cmd, vidur_cmd = (
            _write_command_contract_simulators(work_dir)
        )
        cfg = _build_config(sailor_root, vidur_root, sailor_cmd, vidur_cmd)

        batch = [
            {"prompt_id": "r2e-0", "input_len": 1024, "output_len": 18000},
            {"prompt_id": "r2e-1", "input_len": 1024, "output_len": 22000},
            {"prompt_id": "r2e-2", "input_len": 2048, "output_len": 26000},
            {"prompt_id": "r2e-3", "input_len": 2048, "output_len": 32000},
        ]

        print("=== Global Resource Planner Full Flow ===")
        print("1. History batch:")
        for item in batch:
            print(
                "  "
                f"{item['prompt_id']}: input={item['input_len']} "
                f"output={item['output_len']}"
            )

        print("\n2. Backend configuration:")
        print(f"  train_backend={cfg.global_resource_planner.train_backend}")
        print(f"  rollout_backend={cfg.global_resource_planner.rollout_backend}")
        print(f"  sailor_command={cfg.global_resource_planner.sailor_train_command}")
        print(f"  vidur_command={cfg.global_resource_planner.vidur_rollout_command}")

        planner = GlobalResourcePlanner.from_config(cfg)
        decision = planner.plan_if_needed(step=1, config=cfg, batch=batch)

        print("\n3. Planner decision:")
        print(f"  reason={decision.reason}")
        print(f"  should_reconfigure={decision.should_reconfigure}")
        print(f"  history_requests={decision.num_requests}")

        _print_plan("current plan", decision.current_plan)
        _print_plan("candidate plan", decision.candidate_plan)

        print("\n4. Example evaluated candidates:")
        evaluated = (
            decision.candidate_plan.metadata.get("evaluated", [])
            if decision.candidate_plan
            else []
        )
        for idx, item in enumerate(evaluated[-5:], 1):
            print(
                "  "
                f"#{idx}: train={item['train']} rollout={item['rollout']} "
                f"T_train={item['t_train']:.4f}s "
                f"T_rollout={item['t_rollout']:.4f}s "
                f"T_global={item['t_global']:.4f}s"
            )

        print("\n5. Apply plan to runtime config:")
        if decision.candidate_plan is not None:
            planner.apply_plan_to_config(decision.candidate_plan, cfg)
        print(
            "  config train: "
            f"gpus={cfg.train_gpus} tp={cfg.train_tp_size} "
            f"pp={cfg.train_pp_size} dp={cfg.train_dp_size} "
            f"micro={cfg.micro_batch_size}"
        )
        print(
            "  config rollout: "
            f"gpus={cfg.rollout_gpus} tp_list={cfg.heterogeneous_rollout.tp_list}"
        )
        for inst in cfg.heterogeneous_rollout.instances:
            print(
                "    "
                f"{inst.instance_id}: tp={inst.tp} gpus={inst.gpus} "
                f"{inst.description}"
            )

        print("\nFLOW_OK")


if __name__ == "__main__":
    main()

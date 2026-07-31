#!/usr/bin/env python3
"""Run Global Resource Planner before launching training."""

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Input YAML config")
    parser.add_argument("--output-config", required=True, help="Planned YAML path")
    parser.add_argument("--decision-json", default="", help="Optional decision JSON path")
    parser.add_argument("--history-jsonl", default="", help="JSONL with input_len/output_len")
    parser.add_argument("--synthetic-requests", type=int, default=32)
    parser.add_argument("--synthetic-input-len", type=int, default=1024)
    parser.add_argument("--synthetic-output-len", type=int, default=2048)
    parser.add_argument(
        "--respect-threshold",
        action="store_true",
        help="Only apply when the runtime reconfiguration threshold would trigger",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = AsyncRLConfig.from_yaml(args.config)

    if args.history_jsonl:
        history = load_history_jsonl(args.history_jsonl)
    else:
        history = synthetic_history(
            num_requests=args.synthetic_requests,
            input_len=args.synthetic_input_len,
            output_len=args.synthetic_output_len,
        )

    result = PreflightPlanner(
        config,
        apply_best_candidate=not args.respect_threshold,
    ).run(history)

    result.planned_config.to_yaml(args.output_config)
    if args.decision_json:
        Path(args.decision_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.decision_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    plan = result.applied_plan
    if plan is None:
        print(
            "[PreflightPlanner] no plan applied "
            f"reason={result.decision.reason} output={args.output_config}"
        )
    else:
        print(
            "[PreflightPlanner] applied "
            f"train={plan.train_config.tp}x{plan.train_config.pp}x{plan.train_config.dp} "
            f"rollout_tp={plan.rollout_tp_list} "
            f"T={plan.t_global:.3f}s output={args.output_config}"
        )


if __name__ == "__main__":
    main()

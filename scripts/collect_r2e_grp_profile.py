#!/usr/bin/env python3
"""Collect real R2E-Gym length samples for startup GRP planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.startup_profile import (
    load_jsonl_rows,
    load_profile_summary,
    profile_jsonl_has_records,
    profile_metadata_matches,
    startup_profile_metadata,
    summarize_length_profile,
    write_profile_jsonl,
    write_profile_summary,
)
from RL_Framework.workflow.r2e_gym_profile import build_r2e_gym_length_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="AsyncRL YAML config")
    parser.add_argument("--dataset-jsonl", required=True, help="R2E-Gym index JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Planner history JSONL")
    parser.add_argument("--summary-json", default="", help="Optional profile summary JSON")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--strategy", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing output JSONL when it already contains usable profile rows",
    )
    parser.add_argument(
        "--row-load-limit",
        type=int,
        default=0,
        help="Optional cap on loaded dataset rows before sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AsyncRLConfig.from_yaml(args.config)
    planner_cfg = config.global_resource_planner

    dataset_path = Path(args.dataset_jsonl)
    if not dataset_path.exists():
        raise FileNotFoundError(f"R2E-Gym profile dataset not found: {dataset_path}")

    sample_size = (
        args.sample_size
        if args.sample_size > 0
        else int(planner_cfg.startup_profile_sample_size)
    )
    strategy = args.strategy or planner_cfg.startup_profile_strategy
    seed = (
        int(args.seed)
        if args.seed is not None
        else int(planner_cfg.startup_profile_seed)
    )
    samples_per_prompt = (
        args.samples_per_prompt
        if args.samples_per_prompt > 0
        else int(planner_cfg.startup_profile_samples_per_prompt)
    )

    reuse_existing = args.reuse_existing or planner_cfg.startup_profile_reuse_existing
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else None
    expected_metadata = startup_profile_metadata(
        dataset_jsonl=dataset_path,
        model_path=config.model_path,
        tokenizer_path=(
            config.tokenizer_path if config.tokenizer_path else config.model_path
        ),
        sample_size=sample_size,
        strategy=strategy,
        seed=seed,
        samples_per_prompt=samples_per_prompt,
        max_new_tokens=config.max_new_tokens,
        max_seq_length=config.max_seq_length,
        r2e_max_turns=config.r2e_max_turns,
        r2e_max_prompt_tokens=config.r2e_max_prompt_tokens,
    )
    if reuse_existing and profile_jsonl_has_records(output_path):
        existing_summary = load_profile_summary(summary_path) if summary_path else None
        if summary_path is None or existing_summary is None or profile_metadata_matches(existing_summary, expected_metadata):
            records = load_jsonl_rows(output_path)
            if records:
                summary = summarize_length_profile(records)
                summary.update(expected_metadata)
                summary["reused"] = True
                if summary_path:
                    write_profile_summary(summary, summary_path)
                print(
                    "[R2EGRPProfile] "
                    f"reused records={summary['records']} "
                    f"input_mean={summary['input_len']['mean']:.1f} "
                    f"output_mean={summary['output_len']['mean']:.1f} "
                    f"output_max={summary['output_len']['max']:.0f} "
                    f"output={args.output_jsonl}"
                )
                print(json.dumps(summary, ensure_ascii=False))
                return

    rows = load_jsonl_rows(dataset_path, limit=args.row_load_limit)
    if not rows:
        raise ValueError(f"No R2E-Gym rows loaded from {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = build_r2e_gym_length_profile(
        rows,
        tokenizer=tokenizer,
        sample_size=sample_size,
        strategy=strategy,
        seed=seed,
        samples_per_prompt=samples_per_prompt,
        max_turns=config.r2e_max_turns,
        max_new_tokens=config.max_new_tokens,
        max_seq_length=config.max_seq_length,
        max_prompt_tokens=config.r2e_max_prompt_tokens or None,
    )
    if not records:
        raise ValueError("R2E-Gym startup profile produced no records")

    write_profile_jsonl(records, output_path)
    summary = summarize_length_profile(records)
    summary.update(expected_metadata)
    if summary_path:
        write_profile_summary(summary, summary_path)
    print(
        "[R2EGRPProfile] "
        f"records={summary['records']} "
        f"input_mean={summary['input_len']['mean']:.1f} "
        f"output_mean={summary['output_len']['mean']:.1f} "
        f"output_max={summary['output_len']['max']:.0f} "
        f"output={args.output_jsonl}"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

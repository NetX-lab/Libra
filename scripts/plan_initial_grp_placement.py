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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--placement-json", required=True)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--gpus-per-host", type=int, default=8)
    parser.add_argument("--history-jsonl", default="")
    parser.add_argument("--profile-dataset-jsonl", default="")
    parser.add_argument("--profile-output-jsonl", default="")
    parser.add_argument("--profile-summary-json", default="")
    parser.add_argument("--profile-sample-size", type=int, default=0)
    parser.add_argument("--profile-strategy", default="")
    parser.add_argument("--profile-seed", type=int, default=None)
    parser.add_argument("--profile-samples-per-prompt", type=int, default=0)
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Use synthetic lengths only when no real startup profile is provided",
    )
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

    history_source = "real_profile"
    history_path = args.history_jsonl
    profile_summary = {}
    if args.history_jsonl:
        history = load_history_jsonl(args.history_jsonl)
        if not history:
            raise ValueError(f"history-jsonl is empty: {args.history_jsonl}")
        profile_summary = summarize_length_profile(history)
    elif planner_cfg.startup_profile_enabled and (
        args.profile_dataset_jsonl or planner_cfg.startup_profile_dataset_jsonl
    ):
        profile_dataset = (
            args.profile_dataset_jsonl or planner_cfg.startup_profile_dataset_jsonl
        )
        sample_size = (
            args.profile_sample_size
            if args.profile_sample_size > 0
            else int(planner_cfg.startup_profile_sample_size)
        )
        strategy = args.profile_strategy or planner_cfg.startup_profile_strategy
        seed = (
            int(args.profile_seed)
            if args.profile_seed is not None
            else int(planner_cfg.startup_profile_seed)
        )
        samples_per_prompt = (
            args.profile_samples_per_prompt
            if args.profile_samples_per_prompt > 0
            else int(planner_cfg.startup_profile_samples_per_prompt)
        )
        profile_history_path = (
            args.profile_output_jsonl or planner_cfg.startup_profile_history_jsonl
        )
        profile_summary_path = (
            args.profile_summary_json or planner_cfg.startup_profile_summary_json
        )
        profile_metadata = startup_profile_metadata(
            dataset_jsonl=profile_dataset,
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
        if profile_history_path:
            profile_history_file = Path(profile_history_path)
            if profile_jsonl_has_records(profile_history_file):
                existing_summary = (
                    load_profile_summary(profile_summary_path)
                    if profile_summary_path
                    else None
                )
                if (
                    not profile_summary_path
                    or existing_summary is None
                    or profile_metadata_matches(existing_summary, profile_metadata)
                ):
                    history = load_history_jsonl(profile_history_file)
                    if not history:
                        raise ValueError(
                            f"startup profile JSONL is empty: {profile_history_path}"
                        )
                    history_source = "cached_startup_profile"
                    profile_summary = summarize_length_profile(history)
                    profile_summary.update(profile_metadata)
                    profile_summary["reused"] = True
                    history_path = profile_history_path
                    if profile_summary_path:
                        write_profile_summary(profile_summary, profile_summary_path)
        if history is None:
            from transformers import AutoTokenizer

            from RL_Framework.workflow.r2e_gym_profile import build_r2e_gym_length_profile

            rows = load_jsonl_rows(profile_dataset)
            if not rows:
                raise ValueError(f"profile dataset is empty: {profile_dataset}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path if config.tokenizer_path else config.model_path,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            history = build_r2e_gym_length_profile(
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
            profile_summary = summarize_length_profile(history)
            profile_summary.update(profile_metadata)
            history_path = profile_history_path
            if profile_history_path:
                write_profile_jsonl(history, profile_history_path)
            if profile_summary_path:
                write_profile_summary(profile_summary, profile_summary_path)
    elif args.allow_synthetic_fallback or planner_cfg.startup_profile_allow_synthetic_fallback:
        history_source = "synthetic_fallback"
        history = synthetic_history(
            num_requests=args.synthetic_requests,
            input_len=args.synthetic_input_len,
            output_len=args.synthetic_output_len,
        )
        profile_summary = summarize_length_profile(history)
    else:
        raise ValueError(
            "initial GRP placement requires a real startup profile; pass "
            "--history-jsonl or --profile-dataset-jsonl, set "
            "global_resource_planner.startup_profile_dataset_jsonl, or explicitly "
            "enable --allow-synthetic-fallback"
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
        "history_source": history_source,
        "history_jsonl": history_path,
        "profile_summary": profile_summary,
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
        f"rollout_hosts={','.join(rollout_hosts)} "
        f"history_source={history_source}"
    )


if __name__ == "__main__":
    main()

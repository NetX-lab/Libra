"""Full AsyncRLTrainer entrypoint for R2E-Gym with C-MLFQ rollout."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from RL_Framework import AsyncRLTrainer, parse_args_and_load_config
from RL_Framework.env.r2e_gym_reward import r2e_gym_reward_fn
from RL_Framework.workflow.r2e_gym import R2EGymWorkflow


def _normalize_modified_files(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return [str(value)]
    return parsed if isinstance(parsed, list) else [str(parsed)]


def main():
    config = parse_args_and_load_config()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (
        config.train_backend != "megatron_core"
        and not dist.is_initialized()
        and world_size > 1
    ):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        distributed_timeout = timedelta(
            seconds=int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", "3600"))
        )
        try:
            dist.init_process_group(
                backend="nccl",
                device_id=device,
                timeout=distributed_timeout,
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                timeout=distributed_timeout,
            )
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0

    data_path = os.environ.get(
        "R2E_GYM_INDEX",
        os.path.join(_PROJECT_ROOT, "data", "r2e_gym_v1", "index.jsonl"),
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"R2E-Gym index not found: {data_path}")

    if is_main_process:
        print("=" * 60)
        print("R2E-Gym Async RL post-training with C-MLFQ")
        print("=" * 60)
        print(f"Model: {config.model_path}")
        print(f"Dataset index: {data_path}")
        print(f"Train GPUs: {config.train_gpus}, rollout GPUs: {config.rollout_gpus}")
        print(f"Scheduler: {config.heterogeneous_rollout.scheduling.scheduler_type}")

    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(example):
        prompt = (example.get("prompt") or "").strip()
        target_issue = (
            example.get("task_text")
            or example.get("problem_statement")
            or ""
        ).strip()
        prompt_id = f"{example.get('repo_name', 'repo')}:{example.get('commit_hash', '')}"
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "task_text": target_issue,
            "target_issue": target_issue,
            "repo_name": example.get("repo_name", ""),
            "commit_hash": example.get("commit_hash", ""),
            "docker_image": example.get("docker_image", ""),
            "expected_output_json": example.get("expected_output_json", "{}"),
            "modified_files": _normalize_modified_files(example.get("modified_files")),
        }

    dataset = dataset.map(preprocess)
    if is_main_process:
        repo_counts = Counter(dataset["repo_name"])
        print(f"Loaded R2E-Gym rows: {len(dataset)}")
        print(f"Repositories: {dict(sorted(repo_counts.items()))}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    workflow = R2EGymWorkflow(
        reward_fn=r2e_gym_reward_fn,
        tokenizer=tokenizer,
        max_turns=int(os.environ.get("R2E_MAX_TURNS", "3")),
        max_new_tokens=config.max_new_tokens,
        max_seq_length=config.max_seq_length,
        max_prompt_tokens=(
            int(os.environ["R2E_MAX_PROMPT_TOKENS"])
            if os.environ.get("R2E_MAX_PROMPT_TOKENS")
            else None
        ),
        temperature=config.temperature,
        top_p=config.top_p,
        n_samples=config.n_samples,
    )

    trainer = AsyncRLTrainer(config)
    trainer.train(workflow=workflow, dataset=dataset)


if __name__ == "__main__":
    main()

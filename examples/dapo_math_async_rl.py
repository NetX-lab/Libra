"""AsyncRLTrainer entrypoint for DAPO-Math-17K with C-MLFQ scheduling."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from RL_Framework import AsyncRLTrainer, parse_args_and_load_config
from RL_Framework.env.dapo_math_reward import dapo_math_reward_fn, extract_dapo_answer
from RL_Framework.workflow.dapo_math import DAPOMathWorkflow


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com/spec/"
        )
    except Exception:
        return False


def _find_dataset_file() -> tuple[Path, str]:
    env_path = os.environ.get("DAPO_MATH_PATH")
    candidates = [Path(env_path)] if env_path else []
    root = Path(os.environ.get("DAPO_MATH_ROOT", Path(_PROJECT_ROOT) / "data" / "DAPO-Math-17K"))
    split = os.environ.get("DAPO_MATH_SPLIT", "all")
    candidates.extend([
        root / split / "train.jsonl",
        root / split / "train-00000-of-00001.parquet",
        root / "all" / "train.jsonl",
        root / "all" / "train-00000-of-00001.parquet",
        root / "en" / "train.jsonl",
        root / "en" / "train-00000-of-00001.parquet",
    ])
    for path in candidates:
        if not path or not path.exists():
            continue
        if _is_lfs_pointer(path):
            if env_path and path == Path(env_path):
                raise RuntimeError(
                    f"DAPO-Math file is still a Git LFS pointer, not real data: {path}"
                )
            continue
        if path.suffix == ".jsonl":
            return path, "json"
        if path.suffix == ".parquet":
            return path, "parquet"
    raise FileNotFoundError(
        f"No DAPO-Math train file found under {root}. Expected train.jsonl or parquet."
    )


def _extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(part for part in parts if part).strip()
    return str(prompt or "").strip()


def _extract_ground_truth(example: dict[str, Any]) -> Any:
    reward_model = example.get("reward_model")
    if isinstance(reward_model, dict):
        for key in ("ground_truth", "answer", "target"):
            if reward_model.get(key):
                return reward_model[key]
    extra = example.get("extra_info")
    if isinstance(extra, dict):
        for key in ("answer", "ground_truth", "target"):
            if extra.get(key):
                return extra[key]
    for key in ("ground_truth", "answer", "target", "final_answer"):
        if example.get(key):
            return example[key]
    solution = example.get("solution", "")
    return extract_dapo_answer(str(solution)) if solution else ""


def _preprocess(example: dict[str, Any], idx: int) -> dict[str, Any]:
    question = (
        _extract_prompt_text(example.get("prompt"))
        or str(example.get("question") or example.get("problem") or example.get("input") or "").strip()
    )
    answer = _extract_ground_truth(example)
    extra = example.get("extra_info")
    extra_index = extra.get("index") if isinstance(extra, dict) else None
    prompt_id = str(
        example.get("prompt_id")
        or example.get("uid")
        or example.get("id")
        or extra_index
        or f"dapo_math:{idx}"
    )
    return {
        "prompt_id": prompt_id,
        "question": question,
        "ground_truth": answer,
        "answer": answer,
        "data_source": example.get("data_source", "dapo_math"),
    }


def main():
    config = parse_args_and_load_config()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist.is_initialized() and world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    data_file, data_format = _find_dataset_file()
    if is_main:
        print("=" * 60)
        print("DAPO-Math Async RL post-training with C-MLFQ")
        print("=" * 60)
        print(f"Model: {config.model_path}")
        print(f"Dataset file: {data_file} ({data_format})")
        print(f"Scheduler: {config.heterogeneous_rollout.scheduling.scheduler_type}")

    dataset = load_dataset(data_format, data_files=str(data_file), split="train")
    dataset = dataset.map(_preprocess, with_indices=True)
    dataset = dataset.filter(lambda row: bool(row.get("question")) and bool(row.get("ground_truth")))
    if os.environ.get("DAPO_DEDUPLICATE", "1").lower() not in {"0", "false", "no"}:
        seen_prompt_ids: set[str] = set()

        def keep_first(row: dict[str, Any]) -> bool:
            prompt_id = str(row.get("prompt_id") or "")
            if prompt_id in seen_prompt_ids:
                return False
            seen_prompt_ids.add(prompt_id)
            return True

        before = len(dataset)
        dataset = dataset.filter(keep_first)
        if is_main:
            print(f"Deduplicated DAPO-Math rows by prompt_id: {before} -> {len(dataset)}")
    if is_main:
        print(f"Loaded DAPO-Math rows: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        print(f"First prompt_id: {dataset[0]['prompt_id'] if len(dataset) else '<empty>'}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    workflow = DAPOMathWorkflow(
        reward_fn=dapo_math_reward_fn,
        tokenizer=tokenizer,
        max_turns=int(os.environ.get("DAPO_MAX_TURNS", "2")),
        max_new_tokens=config.max_new_tokens,
        max_seq_length=config.max_seq_length,
        max_prompt_tokens=(
            int(os.environ["DAPO_MAX_PROMPT_TOKENS"])
            if os.environ.get("DAPO_MAX_PROMPT_TOKENS")
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

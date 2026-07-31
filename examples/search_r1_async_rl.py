"""Support code for Search r1 async rl."""

import os
import sys
from collections import Counter


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from RL_Framework import AsyncRLTrainer, parse_args_and_load_config
from RL_Framework.env.search_r1_reward import (
    search_r1_reward_fn,
    combined_search_r1_reward_fn,
)
from RL_Framework.env.search_tool import SearchTool
from RL_Framework.workflow.search_r1 import SearchR1Workflow


def main():
    """Main."""
    config = parse_args_and_load_config()


    is_distributed = dist.is_initialized()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not is_distributed and world_size > 1:
        dist.init_process_group(backend="nccl")
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    is_main_process = rank == 0

    if is_main_process:
        print("=" * 60)
        print("SearchR1 Asynchronous RL training")
        print("=" * 60)
        print(f"Model: {config.model_path}")
        print(f"Training GPUs: {config.train_gpus}, rollout GPUs: {config.rollout_gpus}")
        print(f"Batch Size: {config.batch_size}, Max Turns: 5")
        print("=" * 60)

    # ================================================================

    # ================================================================
    if is_main_process:
        print("\nLoading dataset...")


    local_data_path = os.path.join(_PROJECT_ROOT, "data", "search_r1_train.jsonl")
    if not os.path.exists(local_data_path):
        raise FileNotFoundError(
            f"Search-R1 data does not exist: {local_data_path}. "
            "This task does not fall back to GSM8K or another dataset."
        )
    if is_main_process:
        print(f"Loading local Search-R1 data: {local_data_path}")
    dataset = load_dataset("json", data_files=local_data_path, split="train")

    def preprocess(example):
        return {
            "question": example["question"].strip(),
            "ground_truth": example["ground_truth"],
            "data_source": example["data_source"],
        }

    dataset = dataset.map(preprocess)
    source_counts = Counter(dataset["data_source"])
    expected_sources = {"nq", "hotpotqa"}
    if set(source_counts) != expected_sources:
        raise ValueError(
            "The loaded data is not the complete Search-R1 training set; "
            f"expected sources {sorted(expected_sources)}, got {sorted(source_counts)}"
        )
    if is_main_process:
        print(f"Loaded dataset: {len(dataset)} samples")
        print(f"Search-R1 source counts: {dict(sorted(source_counts.items()))}")

    # ================================================================

    # ================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    search_tool = SearchTool()


    reward_fn = lambda prompt, completion, **kwargs: combined_search_r1_reward_fn(
        prompt=prompt,
        completion=completion,
        format_weight=0.15,
        **kwargs,
    )


    workflow = SearchR1Workflow(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        search_tool=search_tool,
        max_turns=5,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        n_samples=config.n_samples,
    )

    if is_main_process:
        print("\nCreated the Search-R1 workflow")
        print(f"  Maximum turns: {workflow.max_turns}")
        print(f"  Search backend: {search_tool.backend}")
        if search_tool.backend == "searxng":
            print(f"  SearXNGaddress: {search_tool.searxng_url}")
        else:
            print(
                "  Search tool: "
                f"{'ready' if search_tool.api_key else 'not configured (SERPER_KEY_ID is missing)'}"
            )

    # ================================================================

    # ================================================================
    if is_main_process:
        print("\nInitializing the asynchronous RL trainer...")

    trainer = AsyncRLTrainer(config)

    if is_main_process:
        print("\nStarting asynchronous pipeline training...")
        print("=" * 60)

    trainer.train(workflow=workflow, dataset=dataset)

    if is_main_process:
        print("\nTraining complete.")


if __name__ == "__main__":
    main()

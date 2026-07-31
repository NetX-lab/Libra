"""Support code for Livecodebench code agent."""

from __future__ import annotations

import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from RL_Framework import AsyncRLConfig, AsyncRLTrainer
from RL_Framework.env.code_executor import CodeExecutor
from RL_Framework.env.code_reward import make_code_reward_fn
from RL_Framework.workflow.code_agent import CodeAgentWorkflow


def main():
    """Main."""

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

    # ================================================================

    # ================================================================
    config = AsyncRLConfig(
        model_path="/path/to/Qwen3-4B",
        tokenizer_path="",
        train_gpus=2,
        rollout_gpus=2,
        tp_size=1,
        max_concurrent_rollouts=16,
        max_head_offpolicyness=4,
        queue_size=256,
        enable_rollout_tracing=False,
        sync_interval=1,
        recompute_logprobs=True,
        weight_sync_mode="disk",
        sync_path="/path/to/user/async_rl_weights",
        learning_rate=1e-6,
        ppo_epochs=1,
        batch_size=16,
        micro_batch_size=4,
        kl_coef=0.001,
        clip_epsilon=0.2,
        max_new_tokens=2048,
        n_samples=4,
        temperature=1.0,
        dataset_name="livecodebench_codeforces",
        dataset_split="train",
        max_seq_length=8192,
        total_steps=100,
        eval_interval=10,
        save_interval=10,
        seed=42,
        ray_address="auto",
        vllm_port=8000,
        log_dir="./logs/code_agent",
        wandb_project="RL_Framework_code_agent",
        wandb_run_name="livecodebench_codeforces",
        keep_latest_checkpoints=5,
    )

    if is_main_process:
        print("=" * 60)
        print("LiveCodeBench code-agent asynchronous RL training")
        print("=" * 60)
        print(f"Model: {config.model_path}")
        print(f"Training GPUs: {config.train_gpus}, rollout GPUs: {config.rollout_gpus}")
        print(f"Batch Size: {config.batch_size}, Max Turns: 5")
        print("=" * 60)

    # ================================================================

    # ================================================================
    if is_main_process:
        print("\nLoading dataset...")


    local_data_path = os.path.join(_PROJECT_ROOT, "data", "livecodebench_codeforces_train.jsonl")
    if os.path.exists(local_data_path):
        if is_main_process:
            print(f"Loading local file: {local_data_path}")
        dataset = load_dataset("json", data_files=local_data_path, split="train")
    else:

        if is_main_process:
            print("Local data is unavailable; trying LiveCodeBench on Hugging Face...")
        try:
            raw_dataset = load_dataset("livecodebench/code_generation_lite", split="train")

            dataset = raw_dataset.filter(lambda x: str(x.get("platform", "")).lower() == "codeforces")
            if is_main_process:
                print(f"Loaded {len(dataset)} samples from Hugging Face")
        except Exception as e:
            print(f"Loading failed: {e}")
            print("Run the data preprocessing script first:")
            print("  python data/preprocess_livecodebench.py --platform codeforces")
            return


    def preprocess(example):

        test_cases = example.get("test_cases", {})
        if isinstance(test_cases, str):
            import json
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = {}

        return {
            "question": example.get("question", example.get("problem", "")),
            "test_cases": test_cases,
            "starter_code": example.get("starter_code", ""),
            "platform": example.get("platform", "unknown"),
            "difficulty": example.get("difficulty", "unknown"),
        }

    dataset = dataset.map(preprocess)
    if is_main_process:
        print(f"Loaded dataset: {len(dataset)} samples")

    # ================================================================

    # ================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    executor = CodeExecutor(
        mode="local",
        timeout=10,
        memory_limit_mb=1024,
    )


    reward_fn = make_code_reward_fn(
        executor=executor,
        continuous=True,
        format_weight=0.1,
        compile_bonus=0.0,
    )


    workflow = CodeAgentWorkflow(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        executor=executor,
        max_turns=5,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        n_samples=config.n_samples,
        stop_on_submit=True,
    )

    if is_main_process:
        print("\nCreated the CodeAgent workflow")
        print(f"  Maximum turns: {workflow.max_turns}")
        print(f"  Code executor: {executor.mode} mode")
        print(f"  timed out: {executor.timeout}s")

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

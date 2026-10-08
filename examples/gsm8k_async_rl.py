"""Support code for Gsm8k async rl."""

import os
import torch
import torch.distributed as dist
from RL_Framework import AsyncRLConfig, AsyncRLTrainer, parse_args_and_load_config
from RL_Framework.workflow.rlvr import RLVRWorkflow
from RL_Framework.env.parse_utils_qwen import extract_answer
from RL_Framework.env.grader import math_equal
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
import logging
from RL_Framework.engine.device_utils import distributed_backend, set_device
def gsm8k_reward_fn(
    prompt: str,
    completion: str,
    answer: str = "",
    **kwargs,
) -> float:
    """Gsm8k reward fn."""
    if not answer:
        return 0.0


    model_answer = extract_answer(completion, data_name="gsm8k")
    if not model_answer:
        return 0.0



    if math_equal(model_answer, answer, timeout=True):
        return 1.0
    else:
        return 0.0


def main():
    """Main."""


    #   python gsm8k_async_rl.py --config configs/gsm8k_hetero.yaml

    config = parse_args_and_load_config()


    is_distributed = dist.is_initialized()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not is_distributed and world_size > 1:


        device = set_device(local_rank)
        backend = distributed_backend()
        try:
            dist.init_process_group(backend=backend, device_id=device)
        except TypeError:

            dist.init_process_group(backend=backend)
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        set_device(local_rank)

    is_main_process = rank == 0


    print("Loading the GSM8K dataset...")

    local_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "gsm8k_train.jsonl"
    )
    if os.path.exists(local_data_path):
        print(f"Loading local file: {local_data_path}")
        dataset = load_dataset("json", data_files=local_data_path, split="train")
    else:
        print("Local file does not exist; downloading from Hugging Face Hub...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")


    def preprocess(example):

        full_answer = example["answer"]
        gt_answer = extract_answer(full_answer, data_name="gsm8k")

        return {
            "question": example["question"],
            "answer": gt_answer,
        }

    dataset = dataset.map(preprocess)
    print(f"Loaded dataset: {len(dataset)} samples")


    print("Creating the RLVR workflow...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        n_samples=config.n_samples,
    )


    print("Initializing the trainer...")
    trainer = AsyncRLTrainer(config)

    print("\nstartedAsynchronous RL training...")
    print("=" * 60)

    trainer.train(workflow=workflow, dataset=dataset)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()

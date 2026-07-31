"""Support code for Rlvr."""

import uuid
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from RL_Framework.env.prompt import COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP


class RLVRWorkflow:
    """R l v r workflow implementation."""

    def __init__(
        self,
        reward_fn: Callable[..., float],
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        n_samples: int = 4,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_samples = n_samples

    async def run_episode(
        self,
        engine: Any,  # VLLMRolloutEngine
        data: dict[str, Any],
        version: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Run episode."""

        question = data.get("question", "")




        prompt_str = COT_TASK_DESC + SEP + PROBLEM_FORMAT_STR.format(question=question)


        input_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        input_len = len(input_ids)


        response = await engine.generate(
            prompt=prompt_str,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=1,
            input_tokens=input_len,
        )


        output_text = response["text"]

        # Tokenize output
        output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
        output_logprobs = response.get("logprobs", [])


        full_sequence = input_ids + output_tokens
        input_len = len(input_ids)
        output_len = len(output_tokens)



        if not output_logprobs or len(output_logprobs) != output_len:
            output_logprobs = [0.0] * output_len
        logprobs = [0.0] * input_len + output_logprobs


        loss_mask = [0] * input_len + [1] * output_len


        reward = self.reward_fn(
            prompt=prompt_str,
            completion=output_text,
            input_tokens=input_ids,
            output_tokens=output_tokens,
            **data,
        )


        trajectory = {
            "input_ids": torch.tensor(full_sequence, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(len(full_sequence), dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "versions": torch.tensor([version], dtype=torch.long),
            "input_len": input_len,
            "output_len": output_len,
        }

        return trajectory

    async def run_batch(
        self,
        engine: Any,
        batch_data: list[dict[str, Any]],
        version: int = 0,
    ) -> list[dict[str, torch.Tensor]]:
        """Run batch."""
        import asyncio


        tasks = [
            self.run_episode(engine, data, version=version)
            for data in batch_data
        ]

        trajectories = await asyncio.gather(*tasks, return_exceptions=True)


        valid_trajectories = []
        for i, traj in enumerate(trajectories):
            if isinstance(traj, Exception):
                print(f"WARNING: Episode {i} execution failed: {traj}")
                continue
            valid_trajectories.append(traj)

        return valid_trajectories

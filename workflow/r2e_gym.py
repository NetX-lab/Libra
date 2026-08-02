"""R2E-Gym issue-generation workflow with validator tool feedback."""

from __future__ import annotations

import asyncio
from hashlib import blake2b
import os
import random
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from RL_Framework.env.r2e_gym_reward import (
    evaluate_issue,
    format_validator_feedback,
)


SYSTEM_PROMPT = """You are an expert software engineer writing GitHub issues from commit details and test results.

Write a concise issue inside [ISSUE] and [/ISSUE] tags. Include a title, a minimal reproduction or concrete example, expected behavior, and actual behavior or error. Do not reveal the fix.
Start directly with [ISSUE]. Do not include analysis, planning, chain-of-thought, preambles, or markdown fences.
"""


class R2EGymWorkflow:
    """Multi-turn R2E-Gym workflow.

    The model drafts an issue, receives validator tool feedback, and may revise.
    Tool-return payloads are routed through C-MLFQ when the rollout engine
    supports it.
    """

    def __init__(
        self,
        reward_fn: Callable[..., float],
        tokenizer: PreTrainedTokenizerBase,
        max_turns: int = 3,
        max_new_tokens: int = 2048,
        max_seq_length: int = 8192,
        max_prompt_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n_samples: int = 1,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.max_turns = max(1, max_turns)
        self.max_new_tokens = max(1, max_new_tokens)
        self.max_seq_length = max(512, max_seq_length)
        if max_prompt_tokens is None:
            max_prompt_tokens = min(8192, max(512, self.max_seq_length // 2))
        self.max_prompt_tokens = max(128, min(max_prompt_tokens, self.max_seq_length - 128))
        self.temperature = temperature
        self.top_p = top_p
        self.n_samples = n_samples

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text or "", add_special_tokens=False)

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        max_tokens = max(1, max_tokens)
        tokens = self._encode(text)
        if len(tokens) <= max_tokens:
            return text
        head = max_tokens // 2
        tail = max_tokens - head
        kept = tokens[:head] + tokens[-tail:]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    def _fit_generation_prompt(self, text: str) -> tuple[str, int]:
        max_input_tokens = max(128, self.max_seq_length - 32)
        fitted = self._truncate_text_to_tokens(text, max_input_tokens)
        return fitted, len(self._encode(fitted))

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _generation_budget(
        self,
        prompt_tokens: int,
        used_sequence_tokens: int,
        turn: int,
        reserve_feedback_tokens: int = 128,
    ) -> int:
        request_budget = self.max_seq_length - prompt_tokens - 16
        remaining_turns = max(1, self.max_turns - turn)
        sequence_budget = self.max_seq_length - used_sequence_tokens - reserve_feedback_tokens
        per_turn_budget = sequence_budget // remaining_turns
        return max(1, min(self.max_new_tokens, request_budget, per_turn_budget))

    def _clip_tokens_to_sequence_budget(
        self,
        tokens: list[int],
        logprobs: list[float],
        used_sequence_tokens: int,
    ) -> tuple[list[int], list[float]]:
        remaining = self.max_seq_length - used_sequence_tokens
        if remaining <= 0:
            return [], []
        if len(tokens) <= remaining:
            return tokens, logprobs
        return tokens[:remaining], logprobs[:remaining]

    def _build_initial_prompt(self, row: dict[str, Any]) -> str:
        task_prompt = row.get("prompt") or row.get("problem_statement") or row.get("task_text") or ""
        task_prompt = self._truncate_text_to_tokens(task_prompt, self.max_prompt_tokens)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        return self._apply_chat_template(messages)

    async def run_episode(
        self,
        engine: Any,
        data: dict[str, Any],
        version: int = 0,
    ) -> dict[str, torch.Tensor]:
        prompt_text = self._build_initial_prompt(data)
        prompt_tokens = self._encode(prompt_text)
        segments: list[tuple[list[int], list[float], int]] = [
            (prompt_tokens, [0.0] * len(prompt_tokens), 0)
        ]
        tool_returns: list[dict[str, Any]] = []

        prompt_id = str(
            data.get(
                "prompt_id",
                f"{data.get('repo_name', 'repo')}:{data.get('commit_hash', '')}",
            )
        )
        begin_cmlfq = getattr(engine, "begin_cmlfq_request", None)
        cmlfq_request_id = (
            begin_cmlfq(prompt_id, len(prompt_tokens))
            if callable(begin_cmlfq)
            else ""
        )

        current_text = prompt_text
        final_turn = 0
        final_completion = ""

        try:
            for turn in range(self.max_turns):
                final_turn = turn
                used_tokens = sum(len(tokens) for tokens, _, _ in segments)
                generation_prompt, generation_prompt_tokens = self._fit_generation_prompt(current_text)
                max_tokens_this_turn = self._generation_budget(
                    prompt_tokens=generation_prompt_tokens,
                    used_sequence_tokens=used_tokens,
                    turn=turn,
                    reserve_feedback_tokens=128 if turn < self.max_turns - 1 else 0,
                )
                generate_kwargs = {
                    "prompt": generation_prompt,
                    "max_new_tokens": max_tokens_this_turn,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": 1,
                }
                if cmlfq_request_id:
                    generate_kwargs.update({
                        "request_id": cmlfq_request_id,
                        "prompt_id": prompt_id,
                    })
                response = await engine.generate(**generate_kwargs)

                output_text = response["text"]
                output_tokens = self._encode(output_text)
                output_logprobs = response.get("logprobs", [])
                if not output_logprobs or len(output_logprobs) != len(output_tokens):
                    output_logprobs = [0.0] * len(output_tokens)
                output_tokens, output_logprobs = self._clip_tokens_to_sequence_budget(
                    output_tokens,
                    output_logprobs,
                    used_sequence_tokens=used_tokens,
                )
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                final_completion = output_text
                if output_tokens:
                    segments.append((output_tokens, output_logprobs, 1))

                metrics = evaluate_issue(
                    completion=output_text,
                    target_issue=data.get("target_issue", data.get("task_text", "")),
                    expected_output_json=data.get("expected_output_json"),
                    modified_files=data.get("modified_files"),
                )
                if turn >= self.max_turns - 1 or metrics["reward"] >= 0.92:
                    break

                feedback = format_validator_feedback(metrics)
                feedback_tokens = self._encode(feedback)
                used_after_output = sum(len(tokens) for tokens, _, _ in segments)
                feedback_tokens, feedback_logprobs = self._clip_tokens_to_sequence_budget(
                    feedback_tokens,
                    [0.0] * len(feedback_tokens),
                    used_sequence_tokens=used_after_output,
                )
                if not feedback_tokens:
                    break
                feedback = self.tokenizer.decode(feedback_tokens, skip_special_tokens=True)
                segments.append((feedback_tokens, feedback_logprobs, 0))
                generated_tokens = sum(len(tokens) for tokens, _, _ in segments[1:])
                tool_event = {
                    "tool_type": "r2e_issue_validator",
                    "output": feedback,
                    "status": "success" if metrics["reward"] >= 0.5 else "failure",
                    "payload_tokens": len(feedback_tokens),
                    "token_position": generated_tokens,
                    "turn": turn,
                    "reward": metrics["reward"],
                }
                tool_returns.append(tool_event)

                route_tool_return = getattr(engine, "route_cmlfq_tool_return", None)
                if cmlfq_request_id and callable(route_tool_return):
                    route_tool_return(cmlfq_request_id, tool_event, generated_tokens)

                current_text += output_text + "\n" + feedback + "\n"
        except Exception:
            cancel_cmlfq = getattr(engine, "cancel_cmlfq_request", None)
            if cmlfq_request_id and callable(cancel_cmlfq):
                cancel_cmlfq(cmlfq_request_id)
            raise

        all_input_ids: list[int] = []
        all_logprobs: list[float] = []
        all_loss_masks: list[int] = []
        for toks, lps, mask_val in segments:
            all_input_ids.extend(toks)
            all_logprobs.extend(lps)
            all_loss_masks.extend([mask_val] * len(toks))
        assert len(all_input_ids) == len(all_logprobs) == len(all_loss_masks)
        if len(all_input_ids) > self.max_seq_length:
            all_input_ids = all_input_ids[:self.max_seq_length]
            all_logprobs = all_logprobs[:self.max_seq_length]
            all_loss_masks = all_loss_masks[:self.max_seq_length]

        output_len = sum(len(toks) for toks, _, mask in segments if mask == 1)
        total_output_tokens = sum(len(toks) for toks, _, _ in segments[1:])
        for tool_event in tool_returns:
            position = int(tool_event.get("token_position", total_output_tokens))
            tool_event["remaining_length"] = max(0, total_output_tokens - position)

        finish_cmlfq = getattr(engine, "finish_cmlfq_request", None)
        if cmlfq_request_id and callable(finish_cmlfq):
            finish_cmlfq(cmlfq_request_id, total_output_tokens)

        full_completion = self.tokenizer.decode(all_input_ids, skip_special_tokens=True)
        reward = self.reward_fn(
            prompt=prompt_text,
            completion=final_completion or full_completion,
            target_issue=data.get("target_issue", data.get("task_text", "")),
            expected_output_json=data.get("expected_output_json"),
            modified_files=data.get("modified_files"),
        )

        seq_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        return {
            "input_ids": seq_tensor.unsqueeze(0),
            "attention_mask": torch.ones_like(seq_tensor).unsqueeze(0),
            "logprobs": torch.tensor(all_logprobs, dtype=torch.float32).unsqueeze(0),
            "loss_mask": torch.tensor(all_loss_masks, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "versions": torch.tensor([version], dtype=torch.long),
            "input_len": len(prompt_tokens),
            "output_len": max(0, output_len),
            "total_output_tokens": int(total_output_tokens),
            "prompt_id": prompt_id,
            "tool_returns": tool_returns,
            "cmlfq_request_id": cmlfq_request_id,
            "n_turns": final_turn + 1,
        }

    async def evaluate(
        self,
        engine: Any,
        dataset: Any,
        max_samples: int | None = None,
        concurrency: int = 16,
        accuracy_threshold: float = 0.5,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate issue quality on a fixed R2E-Gym subset without C-MLFQ tree updates."""
        n_total = len(dataset)
        n_eval = n_total if max_samples is None or max_samples <= 0 else min(max_samples, n_total)
        semaphore = asyncio.Semaphore(max(1, concurrency))
        eval_max_new_tokens = max_new_tokens if max_new_tokens and max_new_tokens > 0 else self.max_new_tokens
        use_feedback = os.environ.get("R2E_EVAL_MULTI_TURN", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        record_limit = max(0, int(os.environ.get("R2E_EVAL_RECORD_SAMPLES", "8")))
        eval_strategy = os.environ.get("R2E_EVAL_STRATEGY", "spread").lower()

        if n_eval >= n_total:
            eval_indices = list(range(n_total))
        elif eval_strategy == "first":
            eval_indices = list(range(n_eval))
        elif eval_strategy == "random":
            rng = random.Random(int(os.environ.get("R2E_EVAL_SEED", "0")))
            eval_indices = sorted(rng.sample(range(n_total), n_eval))
        else:
            eval_strategy = "spread"
            if n_eval == 1:
                eval_indices = [0]
            else:
                eval_indices = [
                    min(n_total - 1, round(i * (n_total - 1) / (n_eval - 1)))
                    for i in range(n_eval)
                ]

        async def eval_one(index: int) -> dict[str, Any]:
            row = dataset[int(index)]
            async with semaphore:
                prompt_text = self._build_initial_prompt(row)
                try:
                    current_text = prompt_text
                    completion = ""
                    turn_metrics = []
                    eval_turns = self.max_turns if use_feedback else 1
                    for turn in range(eval_turns):
                        generation_prompt, prompt_len = self._fit_generation_prompt(current_text)
                        max_tokens_this_call = max(
                            1,
                            min(eval_max_new_tokens, self.max_seq_length - prompt_len - 16),
                        )
                        response = await engine.generate(
                            prompt=generation_prompt,
                            max_new_tokens=max_tokens_this_call,
                            temperature=0.0,
                            top_p=1.0,
                            n=1,
                            prompt_id=str(row.get("prompt_id", index)),
                        )
                        completion = response.get("text", "")
                        metrics = evaluate_issue(
                            completion=completion,
                            target_issue=row.get("target_issue", row.get("task_text", "")),
                            expected_output_json=row.get("expected_output_json"),
                            modified_files=row.get("modified_files"),
                        )
                        turn_metrics.append(metrics)
                        if (
                            not use_feedback
                            or turn >= eval_turns - 1
                            or metrics["reward"] >= 0.92
                        ):
                            break
                        current_text += completion + "\n" + format_validator_feedback(metrics) + "\n"
                    metrics = evaluate_issue(
                        completion=completion,
                        target_issue=row.get("target_issue", row.get("task_text", "")),
                        expected_output_json=row.get("expected_output_json"),
                        modified_files=row.get("modified_files"),
                    )
                    return {
                        "ok": True,
                        "index": int(index),
                        "prompt_id": str(row.get("prompt_id", index)),
                        "reward": float(metrics["reward"]),
                        "accurate": float(metrics["reward"] >= accuracy_threshold),
                        "completion": completion,
                        "target_issue": row.get("target_issue", row.get("task_text", "")),
                        "metrics": metrics,
                        "turn_metrics": turn_metrics,
                    }
                except Exception as exc:
                    return {
                        "ok": False,
                        "index": int(index),
                        "prompt_id": str(row.get("prompt_id", index)),
                        "reward": 0.0,
                        "accurate": 0.0,
                        "error": str(exc),
                    }

        results = await asyncio.gather(*(eval_one(i) for i in eval_indices))
        rewards = [r["reward"] for r in results]
        accurate = [r["accurate"] for r in results]
        failures = [r for r in results if not r["ok"]]
        stats = {
            "eval_samples": n_eval,
            "eval_total_rows": n_total,
            "eval_index_strategy": eval_strategy,
            "eval_indices": eval_indices,
            "eval_index_digest": blake2b(
                ",".join(str(index) for index in eval_indices).encode("ascii"),
                digest_size=8,
            ).hexdigest(),
            "eval_accuracy": sum(accurate) / max(1, len(accurate)),
            "eval_accuracy_threshold": float(accuracy_threshold),
            "eval_reward_ge_0_3": sum(float(r >= 0.3) for r in rewards) / max(1, len(rewards)),
            "eval_reward_ge_0_4": sum(float(r >= 0.4) for r in rewards) / max(1, len(rewards)),
            "eval_reward_ge_0_5": sum(float(r >= 0.5) for r in rewards) / max(1, len(rewards)),
            "eval_reward_mean": sum(rewards) / max(1, len(rewards)),
            "eval_reward_min": min(rewards) if rewards else 0.0,
            "eval_reward_max": max(rewards) if rewards else 0.0,
            "eval_lexical_f1": (
                sum(r.get("metrics", {}).get("lexical_f1", 0.0) for r in results)
                / max(1, len(results))
            ),
            "eval_test_coverage": (
                sum(r.get("metrics", {}).get("test_coverage", 0.0) for r in results)
                / max(1, len(results))
            ),
            "eval_file_coverage": (
                sum(r.get("metrics", {}).get("file_coverage", 0.0) for r in results)
                / max(1, len(results))
            ),
            "eval_format_score": (
                sum(r.get("metrics", {}).get("format_score", 0.0) for r in results)
                / max(1, len(results))
            ),
            "eval_failures": len(failures),
            "eval_first_error": failures[0].get("error", "") if failures else "",
            "eval_mode": "multi_turn" if use_feedback else "single_turn",
        }
        if record_limit:
            stats["eval_records"] = results[:record_limit]
        return stats

    async def run_batch(
        self,
        engine: Any,
        batch_data: list[dict[str, Any]],
        version: int = 0,
    ) -> list[dict[str, torch.Tensor]]:
        tasks = [self.run_episode(engine, data, version=version) for data in batch_data]
        trajectories = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        for i, traj in enumerate(trajectories):
            if isinstance(traj, Exception):
                print(f"WARNING: R2E-Gym episode {i} failed: {traj}")
                continue
            valid.append(traj)
        return valid

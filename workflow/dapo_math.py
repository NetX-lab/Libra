"""DAPO-Math workflow with validator tool feedback for C-MLFQ routing."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from RL_Framework.env.dapo_math_reward import (
    dapo_math_reward_fn,
    evaluate_math_completion,
    extract_dapo_answer,
    format_math_validator_feedback,
)

SYSTEM_PROMPT = """You are a careful mathematical reasoner.
Solve the problem step by step. End your response with the final answer in \\boxed{}.
"""


class DAPOMathWorkflow:
    def __init__(
        self,
        reward_fn: Callable[..., float] = dapo_math_reward_fn,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_turns: int = 2,
        max_new_tokens: int = 4096,
        max_seq_length: int = 8192,
        max_prompt_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n_samples: int = 1,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.max_turns = max(1, max_turns)
        self.max_new_tokens = max(1, max_new_tokens)
        self.max_seq_length = max(512, max_seq_length)
        if max_prompt_tokens is None:
            max_prompt_tokens = min(8192, max(512, self.max_seq_length // 3))
        self.max_prompt_tokens = max(128, min(max_prompt_tokens, self.max_seq_length - 128))
        self.temperature = temperature
        self.top_p = top_p
        self.n_samples = n_samples

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text or "", add_special_tokens=False)

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._encode(text)
        if len(tokens) <= max_tokens:
            return text
        head = max_tokens // 2
        tail = max_tokens - head
        return self.tokenizer.decode(tokens[:head] + tokens[-tail:], skip_special_tokens=True)

    def _extract_question(self, row: dict[str, Any]) -> str:
        prompt = row.get("prompt")
        if isinstance(prompt, list):
            parts = []
            for msg in prompt:
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", "")))
                else:
                    parts.append(str(msg))
            return "\n".join(part for part in parts if part).strip()
        if prompt:
            return str(prompt).strip()
        return str(row.get("question") or row.get("problem") or row.get("input") or "").strip()

    def _extract_answer(self, row: dict[str, Any]) -> str | list[str]:
        reward_model = row.get("reward_model")
        if isinstance(reward_model, dict):
            for key in ("ground_truth", "answer", "target"):
                if reward_model.get(key):
                    return reward_model[key]
        extra = row.get("extra_info")
        if isinstance(extra, dict):
            for key in ("answer", "ground_truth", "target"):
                if extra.get(key):
                    return extra[key]
        for key in ("ground_truth", "answer", "target", "final_answer"):
            if row.get(key):
                return row[key]
        solution = row.get("solution")
        return extract_dapo_answer(str(solution or "")) if solution else ""

    def _prompt_id(self, row: dict[str, Any]) -> str:
        return str(
            row.get("prompt_id")
            or row.get("uid")
            or row.get("id")
            or row.get("data_source")
            or abs(hash(self._extract_question(row)))
        )

    def _build_prompt(self, row: dict[str, Any]) -> str:
        question = self._truncate_text_to_tokens(self._extract_question(row), self.max_prompt_tokens)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _fit_generation_prompt(self, text: str) -> tuple[str, int]:
        max_input_tokens = max(128, self.max_seq_length - 32)
        fitted = self._truncate_text_to_tokens(text, max_input_tokens)
        return fitted, len(self._encode(fitted))

    def _generation_budget(self, prompt_tokens: int, used_sequence_tokens: int, turn: int) -> int:
        request_budget = self.max_seq_length - prompt_tokens - 16
        remaining_turns = max(1, self.max_turns - turn)
        sequence_budget = self.max_seq_length - used_sequence_tokens - 128
        return max(1, min(self.max_new_tokens, request_budget, sequence_budget // remaining_turns))

    def _clip(self, tokens: list[int], logprobs: list[float], used: int) -> tuple[list[int], list[float]]:
        remaining = self.max_seq_length - used
        if remaining <= 0:
            return [], []
        if len(tokens) <= remaining:
            return tokens, logprobs
        return tokens[:remaining], logprobs[:remaining]

    async def run_episode(self, engine: Any, data: dict[str, Any], version: int = 0) -> dict[str, torch.Tensor]:
        prompt_text = self._build_prompt(data)
        prompt_tokens = self._encode(prompt_text)
        answer = self._extract_answer(data)
        prompt_id = self._prompt_id(data)
        segments: list[tuple[list[int], list[float], int]] = [(prompt_tokens, [0.0] * len(prompt_tokens), 0)]
        tool_returns: list[dict[str, Any]] = []

        begin_cmlfq = getattr(engine, "begin_cmlfq_request", None)
        cmlfq_request_id = begin_cmlfq(prompt_id, len(prompt_tokens)) if callable(begin_cmlfq) else ""
        current_text = prompt_text
        final_completion = ""
        final_turn = 0
        try:
            for turn in range(self.max_turns):
                final_turn = turn
                used = sum(len(toks) for toks, _, _ in segments)
                generation_prompt, prompt_len = self._fit_generation_prompt(current_text)
                max_tokens = self._generation_budget(prompt_len, used, turn)
                kwargs = {
                    "prompt": generation_prompt,
                    "max_new_tokens": max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": 1,
                }
                if cmlfq_request_id:
                    kwargs.update({"request_id": cmlfq_request_id, "prompt_id": prompt_id})
                response = await engine.generate(**kwargs)
                output_text = response["text"]
                output_tokens = self._encode(output_text)
                output_logprobs = response.get("logprobs", [])
                if not output_logprobs or len(output_logprobs) != len(output_tokens):
                    output_logprobs = [0.0] * len(output_tokens)
                output_tokens, output_logprobs = self._clip(output_tokens, output_logprobs, used)
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                final_completion = output_text
                if output_tokens:
                    segments.append((output_tokens, output_logprobs, 1))

                metrics = evaluate_math_completion(output_text, answer)
                if turn >= self.max_turns - 1 or metrics["accuracy"] >= 1.0:
                    break

                feedback = format_math_validator_feedback(metrics)
                feedback_tokens = self._encode(feedback)
                used_after_output = sum(len(toks) for toks, _, _ in segments)
                feedback_tokens, feedback_logprobs = self._clip(
                    feedback_tokens, [0.0] * len(feedback_tokens), used_after_output
                )
                if not feedback_tokens:
                    break
                feedback = self.tokenizer.decode(feedback_tokens, skip_special_tokens=True)
                segments.append((feedback_tokens, feedback_logprobs, 0))
                generated_tokens = sum(len(toks) for toks, _, _ in segments[1:])
                tool_event = {
                    "tool_type": "dapo_math_validator",
                    "output": feedback,
                    "status": "success" if metrics["accuracy"] >= 1.0 else "failure",
                    "payload_tokens": len(feedback_tokens),
                    "token_position": generated_tokens,
                    "turn": turn,
                    "reward": metrics["reward"],
                    "accuracy": metrics["accuracy"],
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

        input_ids: list[int] = []
        logprobs: list[float] = []
        loss_mask: list[int] = []
        for toks, lps, mask in segments:
            input_ids.extend(toks)
            logprobs.extend(lps)
            loss_mask.extend([mask] * len(toks))
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            logprobs = logprobs[: self.max_seq_length]
            loss_mask = loss_mask[: self.max_seq_length]
        total_output_tokens = sum(len(toks) for toks, _, _ in segments[1:])
        output_len = sum(len(toks) for toks, _, mask in segments if mask == 1)
        for tr in tool_returns:
            pos = int(tr.get("token_position", total_output_tokens))
            tr["remaining_length"] = max(0, total_output_tokens - pos)
        finish_cmlfq = getattr(engine, "finish_cmlfq_request", None)
        if cmlfq_request_id and callable(finish_cmlfq):
            finish_cmlfq(cmlfq_request_id, total_output_tokens)
        reward = self.reward_fn(prompt=prompt_text, completion=final_completion, ground_truth=answer, answer=answer)
        seq = torch.tensor(input_ids, dtype=torch.long)
        return {
            "input_ids": seq.unsqueeze(0),
            "attention_mask": torch.ones_like(seq).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "versions": torch.tensor([version], dtype=torch.long),
            "input_len": len(prompt_tokens),
            "output_len": int(output_len),
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
        accuracy_threshold: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        n_total = len(dataset)
        n_eval = n_total if max_samples is None or max_samples <= 0 else min(max_samples, n_total)
        semaphore = asyncio.Semaphore(max(1, concurrency))
        eval_max_new = max_new_tokens if max_new_tokens and max_new_tokens > 0 else self.max_new_tokens

        async def eval_one(i: int) -> dict[str, Any]:
            row = dataset[int(i)]
            async with semaphore:
                prompt = self._build_prompt(row)
                fitted, prompt_len = self._fit_generation_prompt(prompt)
                max_tokens = max(1, min(eval_max_new, self.max_seq_length - prompt_len - 16))
                try:
                    response = await engine.generate(
                        prompt=fitted,
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        n=1,
                        prompt_id=self._prompt_id(row),
                    )
                    metrics = evaluate_math_completion(response.get("text", ""), self._extract_answer(row))
                    return {"ok": True, "reward": float(metrics["reward"]), "accurate": float(metrics["accuracy"])}
                except Exception as exc:
                    return {"ok": False, "reward": 0.0, "accurate": 0.0, "error": str(exc)}

        results = await asyncio.gather(*(eval_one(i) for i in range(n_eval)))
        rewards = [r["reward"] for r in results]
        accurate = [r["accurate"] for r in results]
        failures = [r for r in results if not r["ok"]]
        return {
            "eval_samples": n_eval,
            "eval_total_rows": n_total,
            "eval_accuracy": sum(accurate) / max(1, len(accurate)),
            "eval_reward_mean": sum(rewards) / max(1, len(rewards)),
            "eval_reward_min": min(rewards) if rewards else 0.0,
            "eval_reward_max": max(rewards) if rewards else 0.0,
            "eval_failures": len(failures),
            "eval_first_error": failures[0].get("error", "") if failures else "",
        }

    async def run_batch(self, engine: Any, batch_data: list[dict[str, Any]], version: int = 0):
        tasks = [self.run_episode(engine, data, version=version) for data in batch_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"WARNING: DAPO-Math episode {i} failed: {result}")
                continue
            valid.append(result)
        return valid

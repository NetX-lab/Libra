"""R2E-Gym issue-generation workflow with validator tool feedback."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
from RL_Framework.env.r2e_patch_harness import (
    R2EPatchExecutionHarness,
    R2ESingularityPatchHarness,
)

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
        stop_reward: float = 0.5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n_samples: int = 1,
        patch_harness: R2EPatchExecutionHarness | R2ESingularityPatchHarness | None = None,
        grpo_objective: str = "legacy",
        run_seed: int = 0,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.max_turns = max(1, max_turns)
        self.max_new_tokens = max(1, max_new_tokens)
        self.max_seq_length = max(512, max_seq_length)
        if max_prompt_tokens is None:
            max_prompt_tokens = min(8192, max(512, self.max_seq_length // 2))
        self.max_prompt_tokens = max(128, min(max_prompt_tokens, self.max_seq_length - 128))
        if not 0.0 <= stop_reward <= 1.0:
            raise ValueError("stop_reward must be between 0 and 1")
        self.stop_reward = float(stop_reward)
        self.temperature = temperature
        self.top_p = top_p
        self.n_samples = n_samples
        self.patch_harness = patch_harness
        self.configure_objective(grpo_objective, seed=run_seed)

    def configure_objective(self, objective: str, *, seed: int = 0):
        """Select the sampling probabilities needed by the training objective."""
        if objective not in {"legacy", "decoupled"}:
            raise ValueError(f"Unknown grpo_objective: {objective}")
        if type(seed) is not int:
            raise TypeError("run_seed must be an integer")
        if objective == "decoupled" and (
            not math.isfinite(self.temperature) or self.temperature <= 0
            or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1
        ):
            raise ValueError("Decoupled R2E sampling requires positive temperature and valid top_p")
        self.grpo_objective = objective
        self.run_seed = seed

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
        task_prompt = (
            row.get("problem_statement") or row.get("task_text") or ""
            if self.patch_harness
            else row.get("prompt") or row.get("problem_statement") or row.get("task_text") or ""
        )
        task_prompt = self._truncate_text_to_tokens(task_prompt, self.max_prompt_tokens)
        system_prompt = (
            "Return only a unified git diff that fixes the reported issue. "
            "Start with diff --git. Do not use markdown fences."
            if self.patch_harness else SYSTEM_PROMPT
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        return self._apply_chat_template(messages)

    async def _score(self, completion: str, row: dict[str, Any]) -> dict[str, Any]:
        if self.patch_harness:
            required = ["test_command", "docker_image"]
            if self.patch_harness.requires_repo_path:
                required.extend(("repo_path", "commit_hash"))
            missing = [key for key in required if not row.get(key)]
            if missing:
                raise ValueError(f"R2E patch mode missing task fields: {missing}")
            execution = await asyncio.to_thread(
                self.patch_harness.execute, completion, row.get("repo_path"),
                row["test_command"], row["docker_image"], row.get("commit_hash", ""),
            )
            return {"reward": float(execution["tests_passed"]), "execution": execution}
        return evaluate_issue(
            completion=completion,
            target_issue=row.get("target_issue", row.get("task_text", "")),
            expected_output_json=row.get("expected_output_json"),
            modified_files=row.get("modified_files"),
        )

    async def run_episode(
        self,
        engine: Any,
        data: dict[str, Any],
        version: int = 0,
        rollout_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        from RL_Framework.workflow.behavior import (
            SamplingEvidenceError, ValidatedBehaviorTrace, validate_behavior_trace,
        )

        exact = self.grpo_objective == "decoupled"
        prompt_text = self._build_initial_prompt(data)
        ids = self._encode(prompt_text)
        initial_length = len(ids)
        if exact and (not ids or initial_length + 17 > self.max_seq_length):
            raise SamplingEvidenceError("Initial prompt leaves no safe generation budget")
        probabilities, masks = [0.0] * initial_length, [0] * initial_length
        prompt_id = str(data.get("prompt_id", f"{data.get('repo_name', 'repo')}:{data.get('commit_hash', '')}"))
        begin_async = getattr(engine, "begin_cmlfq_request_when_available", None)
        begin = getattr(engine, "begin_cmlfq_request", None)
        if callable(begin_async):
            request_id = await begin_async(prompt_id, initial_length)
        else:
            request_id = begin(prompt_id, initial_length) if callable(begin) else ""
        record_tool_return = getattr(engine, "record_cmlfq_tool_return", None)
        route_tool_return = (
            record_tool_return if callable(record_tool_return)
            else getattr(engine, "route_cmlfq_tool_return", None)
        )
        tool_returns, current_text, final_completion = [], prompt_text, ""
        final_turn = 0
        sample_revision = None
        generated_tokens = 0
        try:
            for turn in range(self.max_turns):
                if exact and len(ids) + 17 > self.max_seq_length:
                    break
                final_turn = turn
                prompt_ids = list(ids) if exact else None
                generation_prompt, prompt_length = (
                    (self.tokenizer.decode(prompt_ids, skip_special_tokens=False), len(prompt_ids))
                    if exact else self._fit_generation_prompt(current_text)
                )
                limit = self._generation_budget(
                    prompt_tokens=prompt_length, used_sequence_tokens=len(ids), turn=turn,
                    reserve_feedback_tokens=128 if turn < self.max_turns - 1 else 0,
                )
                seed = (
                    self.sampling_seed(data, version=version, rollout_index=rollout_index, turn=turn)
                    if exact else int.from_bytes(hashlib.sha256(
                        f"{prompt_id}:{version}:{rollout_index}:{turn}".encode()).digest()[:8],
                        "big") % (2**31 - 1)
                )
                kwargs = dict(
                    prompt_id=prompt_id, prompt=generation_prompt,
                    max_new_tokens=limit, input_tokens=prompt_length,
                    temperature=self.temperature, top_p=self.top_p, n=1,
                    stop=[] if self.patch_harness else ["[/ISSUE]"],
                    seed=seed,
                )
                if exact:
                    kwargs.update(prompt_token_ids=prompt_ids, capture_behavior=True,
                        include_stop_str_in_output=True)
                if request_id:
                    kwargs["request_id"] = request_id
                response = await engine.generate(**kwargs)
                if exact:
                    sampled = response.get("behavior_trace")
                    if not isinstance(sampled, ValidatedBehaviorTrace):
                        sampled = validate_behavior_trace(
                            response, prompt_ids, seed=seed, temperature=self.temperature,
                            top_p=self.top_p, output_limit=limit,
                        )
                    output_tokens, output_logprobs = sampled["output_token_ids"], sampled["logprobs"]
                    final_completion = response["text"]
                else:
                    output_tokens = self._encode(response["text"])
                    output_logprobs = response.get("logprobs", [])
                    if not output_logprobs or len(output_logprobs) != len(output_tokens):
                        output_logprobs = [0.0] * len(output_tokens)
                    output_tokens, output_logprobs = self._clip_tokens_to_sequence_budget(
                        output_tokens, output_logprobs, used_sequence_tokens=len(ids),
                    )
                    final_completion = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if output_tokens:
                    revision = response.get("sample_revision")
                    if exact and (type(revision) is not int or revision < 0):
                        raise SamplingEvidenceError("Action omitted a valid actual sampling revision")
                    if type(revision) is int:
                        sample_revision = revision if sample_revision is None else min(sample_revision, revision)
                ids.extend(output_tokens)
                generated_tokens += len(output_tokens)
                probabilities.extend(output_logprobs)
                masks.extend([1] * len(output_tokens))
                metrics = await self._score(final_completion, data)
                if turn >= self.max_turns - 1 or metrics["reward"] >= self.stop_reward:
                    break
                feedback = self._format_feedback(metrics)
                feedback_tokens = self._encode("\n" + feedback + "\n" if exact else feedback)
                if exact:
                    if len(ids) + len(feedback_tokens) + 17 > self.max_seq_length:
                        break
                else:
                    feedback_tokens, _ = self._clip_tokens_to_sequence_budget(
                        feedback_tokens, [0.0] * len(feedback_tokens), used_sequence_tokens=len(ids),
                    )
                    if not feedback_tokens:
                        break
                    feedback = self.tokenizer.decode(feedback_tokens, skip_special_tokens=True)
                    current_text += final_completion + "\n" + feedback + "\n"
                ids.extend(feedback_tokens)
                probabilities.extend([0.0] * len(feedback_tokens))
                masks.extend([0] * len(feedback_tokens))
                total_output = len(ids) - initial_length
                event = dict(
                    tool_type="r2e_patch_executor" if self.patch_harness else "r2e_issue_validator",
                    output=feedback,
                    status="success" if metrics["reward"] >= self.stop_reward else "failure",
                    payload_tokens=len(feedback_tokens), token_position=total_output, turn=turn,
                    generated_token_position=generated_tokens,
                    reward=metrics["reward"],
                )
                tool_returns.append(event)
                if request_id and callable(route_tool_return):
                    route_tool_return(request_id, event, total_output)
            total_output = len(ids) - initial_length
            for event in tool_returns:
                event["remaining_length"] = (
                    generated_tokens - event["generated_token_position"] if exact
                    else total_output - event["token_position"]
                )
            output_length = sum(masks)
            if not exact:
                ids = ids[:self.max_seq_length]
                probabilities = probabilities[:self.max_seq_length]
                masks = masks[:self.max_seq_length]
            completion = final_completion if exact else (
                final_completion or self.tokenizer.decode(ids, skip_special_tokens=True))
            reward = metrics["reward"] if self.patch_harness else self.reward_fn(
                prompt=prompt_text, completion=completion,
                target_issue=data.get("target_issue", data.get("task_text", "")),
                expected_output_json=data.get("expected_output_json"),
                modified_files=data.get("modified_files"),
            )
            if exact and not output_length:
                raise SamplingEvidenceError("Consumed action mask/count must include sampled tokens")
            ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            trajectory = dict(
                input_ids=ids_tensor, attention_mask=torch.ones_like(ids_tensor),
                loss_mask=torch.tensor(masks, dtype=torch.long).unsqueeze(0),
                rewards=torch.tensor([reward], dtype=torch.float32),
                versions=torch.tensor([version], dtype=torch.long), input_len=initial_length,
                output_len=output_length, total_output_tokens=total_output, prompt_id=prompt_id,
                sequence_length=len(ids),
                tool_returns=tool_returns, cmlfq_request_id=request_id, n_turns=final_turn + 1,
            )
            if sample_revision is not None:
                trajectory["sample_revision"] = sample_revision
            trajectory["behavior_logprobs" if exact else "logprobs"] = torch.tensor(
                probabilities, dtype=torch.float32).unsqueeze(0)
            finish = getattr(engine, "finish_cmlfq_request", None)
            if request_id and callable(finish):
                if callable(record_tool_return):
                    finish(request_id, total_output, generated_tokens=output_length)
                else:
                    finish(request_id, total_output)
            return trajectory
        except BaseException:
            # Cancellation must retire CMLFQ state too, or idle waits can hang.
            cancel = getattr(engine, "cancel_cmlfq_request", None)
            if request_id and callable(cancel):
                cancel(request_id)
            raise

    def sampling_seed(self, data, *, version, rollout_index, turn):
        prompt_id = str(data.get("prompt_id", f"{data.get('repo_name', 'repo')}:{data.get('commit_hash', '')}"))
        identity = f"{self.run_seed}:{prompt_id}:{version}:{rollout_index}:{turn}"
        return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (2**31 - 1)

    def _format_feedback(self, metrics):
        return (
            f"### Patch execution\n{metrics['execution']}\nRevise the patch."
            if self.patch_harness else format_validator_feedback(metrics)
        )

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
                prompt_id = str(row.get("prompt_id", index))
                begin_cmlfq = getattr(engine, "begin_cmlfq_request", None)
                request_id = (
                    begin_cmlfq(prompt_id, len(self._encode(prompt_text)))
                    if callable(begin_cmlfq)
                    else ""
                )
                try:
                    current_text = prompt_text
                    completion = ""
                    turn_metrics = []
                    eval_turns = self.max_turns if use_feedback else 1
                    generated_tokens = 0
                    for turn in range(eval_turns):
                        generation_prompt, prompt_len = self._fit_generation_prompt(current_text)
                        remaining_budget = eval_max_new_tokens - generated_tokens
                        if remaining_budget <= 0:
                            break
                        remaining_turns = max(1, eval_turns - turn)
                        max_tokens_this_call = max(
                            1,
                            min(
                                remaining_budget,
                                max(1, remaining_budget // remaining_turns),
                                self.max_seq_length - prompt_len - 16,
                            ),
                        )
                        generate_kwargs = {
                            "prompt": generation_prompt,
                            "max_new_tokens": max_tokens_this_call,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "n": 1,
                            "prompt_id": prompt_id,
                            "input_tokens": prompt_len,
                            "stop": [] if self.patch_harness else ["[/ISSUE]"],
                        }
                        if request_id:
                            generate_kwargs["request_id"] = request_id
                        response = await engine.generate(**generate_kwargs)
                        completion = response.get("text", "")
                        generated_tokens += len(self._encode(completion))
                        metrics = await self._score(completion, row)
                        turn_metrics.append(metrics)
                        if (
                            not use_feedback
                            or turn >= eval_turns - 1
                            or metrics["reward"] >= self.stop_reward
                        ):
                            break
                        feedback = (
                            f"### Patch execution\n{metrics['execution']}\nRevise the patch."
                            if self.patch_harness else format_validator_feedback(metrics)
                        )
                        feedback_tokens = len(self._encode(feedback))
                        generated_tokens += feedback_tokens
                        route_tool_return = getattr(engine, "route_cmlfq_tool_return", None)
                        if request_id and callable(route_tool_return):
                            route_tool_return(
                                request_id,
                                {
                                    "tool_type": "r2e_patch_executor" if self.patch_harness else "r2e_issue_validator",
                                    "output": feedback,
                                    "status": (
                                        "success"
                                        if metrics["reward"] >= self.stop_reward
                                        else "failure"
                                    ),
                                    "payload_tokens": feedback_tokens,
                                    "token_position": generated_tokens,
                                    "turn": turn,
                                    "reward": metrics["reward"],
                                },
                                generated_tokens,
                            )
                        current_text += completion + "\n" + feedback + "\n"
                    finish_cmlfq = getattr(engine, "finish_cmlfq_request", None)
                    if request_id and callable(finish_cmlfq):
                        finish_cmlfq(request_id, generated_tokens)
                    metrics = turn_metrics[-1] if turn_metrics else await self._score(completion, row)
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
                except asyncio.CancelledError:
                    # Task cancellation must propagate, not be converted
                    # into a (false) evaluation result row.
                    cancel_cmlfq = getattr(engine, "cancel_cmlfq_request", None)
                    if request_id and callable(cancel_cmlfq):
                        cancel_cmlfq(request_id)
                    raise
                except Exception as exc:
                    cancel_cmlfq = getattr(engine, "cancel_cmlfq_request", None)
                    if request_id and callable(cancel_cmlfq):
                        cancel_cmlfq(request_id)
                    return {
                        "ok": False,
                        "index": int(index),
                        "prompt_id": prompt_id,
                        "reward": 0.0,
                        "accurate": 0.0,
                        "error": str(exc),
                    }

        results = await asyncio.gather(*(eval_one(i) for i in eval_indices))
        rewards = [r["reward"] for r in results]
        accurate = [r["accurate"] for r in results]
        failures = [r for r in results if not r["ok"]]
        dense_metric_keys = (
            "lexical_f1",
            "test_coverage",
            "file_coverage",
            "format_score",
        )
        stats = {
            "eval_samples": n_eval,
            "eval_total_rows": n_total,
            "eval_index_strategy": eval_strategy,
            "eval_indices": eval_indices,
            "eval_index_digest": hashlib.sha256(
                ",".join(str(index) for index in eval_indices).encode("utf-8")
            ).hexdigest(),
            "eval_accuracy": sum(accurate) / max(1, len(accurate)),
            "eval_accuracy_threshold": float(accuracy_threshold),
            "eval_reward_ge_0_3": sum(float(r >= 0.3) for r in rewards) / max(1, len(rewards)),
            "eval_reward_ge_0_4": sum(float(r >= 0.4) for r in rewards) / max(1, len(rewards)),
            "eval_reward_ge_0_5": sum(float(r >= 0.5) for r in rewards) / max(1, len(rewards)),
            "eval_reward_mean": sum(rewards) / max(1, len(rewards)),
            "eval_reward_min": min(rewards) if rewards else 0.0,
            "eval_reward_max": max(rewards) if rewards else 0.0,
            "eval_failures": len(failures),
            "eval_first_error": failures[0].get("error", "") if failures else "",
            "eval_mode": "multi_turn" if use_feedback else "single_turn",
        }
        for metric_key in dense_metric_keys:
            stats[f"eval_{metric_key}"] = sum(
                float(result.get("metrics", {}).get(metric_key, 0.0))
                for result in results
            ) / max(1, len(results))
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

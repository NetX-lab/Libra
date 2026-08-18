"""Support code for Code agent."""

from __future__ import annotations

import re
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from RL_Framework.env.code_agent_prompt import (
    CODE_AGENT_SYSTEM_PROMPT,
    SUBMIT_TAG,
    format_execution_result,
    format_problem,
)
from RL_Framework.env.code_executor import CodeExecutor


class CodeAgentWorkflow:
    """Code agent workflow implementation."""

    def __init__(
        self,
        reward_fn: Callable[..., float],
        tokenizer: PreTrainedTokenizerBase,
        executor: CodeExecutor | None = None,
        max_turns: int = 5,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        n_samples: int = 4,
        stop_on_submit: bool = True,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.executor = executor or CodeExecutor(mode="local", timeout=10)
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_samples = n_samples
        self.stop_on_submit = stop_on_submit

    # ------------------------------------------------------------------
    # Code block extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Extract code block."""
        if "```python" in text:
            code = text.split("```python")[-1].split("```")[0]
            return code.strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                code = parts[1]
                if "\n" in code:
                    first_line, rest = code.split("\n", 1)
                    if first_line.strip().isalpha():
                        return rest.strip()
                return code.strip()
        return None

    def _has_submit_tag(self, text: str) -> bool:
        """Has submit tag."""
        return SUBMIT_TAG in text

    # ------------------------------------------------------------------
    # Initial prompt building
    # ------------------------------------------------------------------

    def _build_initial_messages(self, question: str, starter_code: str | None = None) -> list[dict[str, str]]:
        """Build initial messages."""
        user_content = format_problem(question, starter_code)
        messages = [
            {"role": "system", "content": CODE_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Messages to prompt."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # Core episode execution
    # ------------------------------------------------------------------

    async def run_episode(
        self,
        engine: Any,
        data: dict[str, Any],
        version: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Run episode."""
        question = data.get("question", "")
        test_cases = data.get("test_cases", {})
        starter_code = data.get("starter_code", None)


        messages = self._build_initial_messages(question, starter_code)
        prompt_text = self._messages_to_prompt(messages)
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)


        segments: list[tuple[list[int], list[float], int]] = []


        segments.append((prompt_tokens, [0.0] * len(prompt_tokens), 0))

        prompt_id = str(
            data.get("prompt_id", data.get("id", question))
        )
        begin_cmlfq = getattr(engine, "begin_cmlfq_request", None)
        cmlfq_request_id = (
            begin_cmlfq(prompt_id, len(prompt_tokens))
            if callable(begin_cmlfq)
            else ""
        )


        current_text = prompt_text
        n_executions = 0
        final_turn = 0
        all_passed = False
        tool_returns: list[dict[str, Any]] = []

        for turn in range(self.max_turns):
            final_turn = turn


            generate_kwargs = {
                "prompt": current_text,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "n": 1,
            }
            if cmlfq_request_id:
                generate_kwargs.update({
                    "request_id": cmlfq_request_id,
                    "prompt_id": prompt_id,
                })
            try:
                response = await engine.generate(**generate_kwargs)
            except Exception:
                cancel_cmlfq = getattr(
                    engine, "cancel_cmlfq_request", None
                )
                if cmlfq_request_id and callable(cancel_cmlfq):
                    cancel_cmlfq(cmlfq_request_id)
                raise

            output_text = response["text"]
            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            output_logprobs = response.get("logprobs", [])


            if not output_logprobs or len(output_logprobs) != len(output_tokens):
                output_logprobs = [0.0] * len(output_tokens)


            if self.stop_on_submit and self._has_submit_tag(output_text):
                segments.append((output_tokens, output_logprobs, 1))
                break


            code = self._extract_code_block(output_text)

            if code and turn < self.max_turns - 1:

                try:
                    pass_rate, metadata = await self.executor.execute(code, test_cases)
                except Exception as e:
                    pass_rate = 0.0
                    metadata = {"error": str(e)}

                n_executions += 1
                passed = int(metadata.get("passed", 0))
                total = int(metadata.get("total", 0))


                execution_result = format_execution_result(
                    passed=passed,
                    total=total,
                    metadata_list=metadata.get("metadata_list", []),
                )
                result_tokens = self.tokenizer.encode(execution_result, add_special_tokens=False)


                segments.append((output_tokens, output_logprobs, 1))
                segments.append((result_tokens, [0.0] * len(result_tokens), 0))

                generated_tokens = sum(
                    len(tokens) for tokens, _, _ in segments[1:]
                )
                tool_status = (
                    "success"
                    if "error" not in metadata
                    else "failure"
                )
                tool_event = {
                    "tool_type": "code_executor",
                    "output": execution_result,
                    "status": tool_status,
                    "payload_tokens": len(result_tokens),
                    "token_position": generated_tokens,
                    "turn": turn,
                }
                tool_returns.append(tool_event)

                route_tool_return = getattr(
                    engine, "route_cmlfq_tool_return", None
                )
                if cmlfq_request_id and callable(route_tool_return):
                    route_tool_return(
                        cmlfq_request_id,
                        tool_event,
                        generated_tokens,
                    )


                current_text += output_text + execution_result


                if pass_rate >= 1.0 - 1e-6 and total > 0:
                    all_passed = True


                    break
            else:

                segments.append((output_tokens, output_logprobs, 1))
                break

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        all_input_ids: list[int] = []
        all_logprobs: list[float] = []
        all_loss_masks: list[int] = []

        for toks, lps, mask_val in segments:
            all_input_ids.extend(toks)
            all_logprobs.extend(lps)
            all_loss_masks.extend([mask_val] * len(toks))

        assert len(all_input_ids) == len(all_logprobs) == len(all_loss_masks)


        output_len = sum(len(toks) for toks, _, mask in segments if mask == 1)
        total_output_tokens = sum(len(toks) for toks, _, _ in segments[1:])
        for tool_event in tool_returns:
            position = int(tool_event.get("token_position", total_output_tokens))
            tool_event["remaining_length"] = max(0, total_output_tokens - position)

        finish_cmlfq = getattr(engine, "finish_cmlfq_request", None)
        if cmlfq_request_id and callable(finish_cmlfq):
            finish_cmlfq(cmlfq_request_id, total_output_tokens)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        full_completion = self.tokenizer.decode(all_input_ids, skip_special_tokens=True)

        reward_kwargs = dict(data)

        for key in ["question", "test_cases", "starter_code"]:
            reward_kwargs.pop(key, None)

        reward = self.reward_fn(
            prompt=prompt_text,
            completion=full_completion,
            test_cases=test_cases,
            **reward_kwargs,
        )

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        seq_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        trajectory = {
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
            "n_executions": n_executions,
            "all_passed": all_passed,
        }

        return trajectory

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

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
                print(f"WARNING: CodeAgent episode {i} failed: {traj}")
                continue
            valid_trajectories.append(traj)

        return valid_trajectories

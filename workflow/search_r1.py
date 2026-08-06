"""Support code for Search r1."""

from __future__ import annotations

import re
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from env.search_r1_prompt import (
    SEARCH_R1_SYSTEM_PROMPT,
    SEARCH_R1_USER_PREFIX,
    TOOL_RESPONSE_TEMPLATE,
)
from env.search_tool import SearchTool


class SearchR1Workflow:
    """Search r1 workflow implementation."""

    def __init__(
        self,
        reward_fn: Callable[..., float],
        tokenizer: PreTrainedTokenizerBase,
        search_tool: SearchTool | None = None,
        max_turns: int = 5,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        n_samples: int = 4,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.search_tool = search_tool or SearchTool()
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_samples = n_samples

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_call(text: str) -> str | None:
        """Extract tool call."""
        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if match:
            query = match.group(1).strip()
            return query if query else None
        return None

    def _has_answer_tag(self, text: str) -> bool:
        """Has answer tag."""
        return "<answer>" in text and "</answer>" in text

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _build_initial_prompt(self, question: str) -> str:
        """Build initial prompt."""
        user_content = SEARCH_R1_USER_PREFIX + question
        messages = [
            {"role": "system", "content": SEARCH_R1_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def run_episode(
        self,
        engine: Any,
        data: dict[str, Any],
        version: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Run episode."""
        question = data.get("question", "")
        ground_truth = data.get("ground_truth", data.get("answer", ""))


        prompt_text = self._build_initial_prompt(question)
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)



        segments: list[tuple[list[int], list[float], int]] = []


        segments.append((
            prompt_tokens,
            [0.0] * len(prompt_tokens),
            0,
        ))

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
        n_searches = 0
        final_turn = 0
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


            tool_query = self._extract_tool_call(output_text)
            has_answer = self._has_answer_tag(output_text)

            if tool_query and turn < self.max_turns - 1:

                try:
                    search_result = await self.search_tool.search(tool_query)
                    tool_status = "success"
                except Exception as exc:
                    search_result = f"Error: {exc}"
                    tool_status = "failure"
                n_searches += 1


                tool_response = TOOL_RESPONSE_TEMPLATE.format(results=search_result)
                tool_tokens = self.tokenizer.encode(
                    tool_response, add_special_tokens=False
                )


                segments.append((output_tokens, output_logprobs, 1))
                segments.append((
                    tool_tokens,
                    [0.0] * len(tool_tokens),
                    0,
                ))

                generated_tokens = sum(
                    len(tokens) for tokens, _, _ in segments[1:]
                )
                tool_event = {
                    "tool_type": "web_search",
                    "output": search_result,
                    "status": tool_status,
                    "payload_tokens": len(tool_tokens),
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


                current_text += output_text + tool_response
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
        reward_kwargs.pop("ground_truth", None)
        reward = self.reward_fn(
            prompt=prompt_text,
            completion=full_completion,
            ground_truth=ground_truth,
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
            "n_searches": n_searches,
        }

        return trajectory

    # ------------------------------------------------------------------

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
                print(f"WARNING: SearchR1 episode {i} failed: {traj}")
                continue
            valid_trajectories.append(traj)

        return valid_trajectories

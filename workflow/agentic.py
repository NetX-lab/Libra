"""Support code for Agentic."""

import uuid
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase


class AgenticWorkflow:
    """Agentic workflow implementation."""

    def __init__(
        self,
        reward_fn: Callable[..., float],
        tokenizer: PreTrainedTokenizerBase,
        tools: list[dict[str, Any]] | None = None,
        max_turns: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        discount_factor: float = 0.9,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.tools = tools or []
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.discount_factor = discount_factor

    async def run_episode(
        self,
        engine: Any,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run episode."""
        messages = data.get("messages", []).copy()

        all_input_ids = []
        all_logprobs = []
        all_loss_masks = []
        all_actions = []

        current_turn = 0

        while current_turn < self.max_turns:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            prompt_str = self.tokenizer.decode(input_ids)

            response = await engine.generate(
                prompt=prompt_str,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=1,
            )

            output_text = response["text"]
            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            output_logprobs = response.get("logprobs", [0.0] * len(output_tokens))

            tool_calls = self._parse_tool_calls(output_text)

            if tool_calls:
                tool_results = await self._execute_tools(tool_calls)

                messages.append({"role": "assistant", "content": output_text})
                for tool_result in tool_results:
                    messages.append({
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["tool_call_id"],
                    })

                all_actions.append({
                    "type": "tool_call",
                    "tools": tool_calls,
                    "results": tool_results,
                })
            else:
                messages.append({"role": "assistant", "content": output_text})

                full_sequence = input_ids + output_tokens
                input_len = len(input_ids)
                output_len = len(output_tokens)

                all_input_ids.extend(full_sequence)
                all_logprobs.extend([0.0] * input_len + output_logprobs[:output_len])
                all_loss_masks.extend([0] * input_len + [1] * output_len)

                break

            current_turn += 1

        if current_turn >= self.max_turns and len(all_input_ids) == 0:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            all_input_ids.extend(input_ids)
            all_logprobs.extend([0.0] * len(input_ids))
            all_loss_masks.extend([0] * len(input_ids))

        reward = self.reward_fn(
            prompt=self.tokenizer.decode(data.get("messages", [])),
            completion=self.tokenizer.decode(all_input_ids),
            actions=all_actions,
            n_turns=current_turn,
            **data,
        )

        if self.discount_factor < 1.0:
            reward *= (self.discount_factor ** current_turn)

        sequence_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        trajectory = {
            "input_ids": sequence_tensor.unsqueeze(0),
            "attention_mask": torch.ones_like(sequence_tensor).unsqueeze(0),
            "logprobs": torch.tensor(all_logprobs, dtype=torch.float32).unsqueeze(0),
            "loss_mask": torch.tensor(all_loss_masks, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "versions": torch.tensor([-1] * len(all_input_ids), dtype=torch.long).unsqueeze(0),
            "n_turns": current_turn,
            "n_tool_calls": len(all_actions),
        }

        return trajectory

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls."""
        tool_calls = []


        for tool in self.tools:
            tool_name = tool.get("name", "")
            if tool_name in text:
                tool_calls.append({
                    "name": tool_name,
                    "arguments": {},
                })

        return tool_calls

    async def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tools."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]


            tool_impl = None
            for tool in self.tools:
                if tool.get("name") == tool_name:
                    tool_impl = tool.get("function")
                    break

            if tool_impl:
                try:
                    result = await tool_impl(**tool_call.get("arguments", {}))
                    results.append({
                        "tool_call_id": str(uuid.uuid4()),
                        "content": str(result),
                    })
                except Exception as e:
                    results.append({
                        "tool_call_id": str(uuid.uuid4()),
                        "content": f"Error: {str(e)}",
                    })
            else:
                results.append({
                    "tool_call_id": str(uuid.uuid4()),
                    "content": f"Tool not found: {tool_name}",
                })

        return results

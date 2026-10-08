"""Length profiling for R2E-Gym startup GRP planning."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

from RL_Framework.infra.cost_model.startup_profile import build_length_profile


SYSTEM_PROMPT = """You are an expert software engineer writing GitHub issues from commit details and test results.

Write a concise issue inside [ISSUE] and [/ISSUE] tags. Include a title, a minimal reproduction or concrete example, expected behavior, and actual behavior or error. Do not reveal the fix.
Start directly with [ISSUE]. Do not include analysis, planning, chain-of-thought, preambles, or markdown fences.
"""


def _extract_target_issue(row: dict[str, Any]) -> str:
    return str(
        row.get("target_issue")
        or row.get("task_text")
        or row.get("problem_statement")
        or row.get("prompt")
        or ""
    )


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer.encode(text or "", add_special_tokens=False)


def _truncate_text_to_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int,
) -> str:
    max_tokens = max(1, int(max_tokens))
    tokens = _encode(tokenizer, text)
    if len(tokens) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    kept = tokens[:head] + tokens[-tail:]
    return tokenizer.decode(kept, skip_special_tokens=True)


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _build_initial_prompt(
    tokenizer: PreTrainedTokenizerBase,
    row: dict[str, Any],
    *,
    max_seq_length: int,
    max_prompt_tokens: int | None,
) -> str:
    if max_prompt_tokens is None:
        max_prompt_tokens = min(8192, max(512, max_seq_length // 2))
    max_prompt_tokens = max(128, min(int(max_prompt_tokens), max_seq_length - 128))
    task_prompt = row.get("prompt") or row.get("problem_statement") or row.get("task_text") or ""
    task_prompt = _truncate_text_to_tokens(tokenizer, task_prompt, max_prompt_tokens)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    return _apply_chat_template(tokenizer, messages)


def _estimate_issue_output_tokens(
    tokenizer: PreTrainedTokenizerBase,
    row: dict[str, Any],
) -> int:
    """Estimate rollout completion length from the real R2E target issue text."""

    issue = _extract_target_issue(row).strip()
    if not issue:
        return 1
    if not issue.startswith("[ISSUE]"):
        issue = "[ISSUE]\n" + issue
    if "[/ISSUE]" not in issue:
        issue = issue.rstrip() + "\n[/ISSUE]"
    return len(_encode(tokenizer, issue))


def build_r2e_gym_length_profile(
    rows: list[dict[str, Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    sample_size: int,
    strategy: str = "spread",
    seed: int = 0,
    samples_per_prompt: int = 1,
    max_turns: int = 3,
    max_new_tokens: int = 2048,
    max_seq_length: int = 8192,
    max_prompt_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Build planner-compatible length history from real R2E-Gym samples."""

    return build_length_profile(
        rows,
        prompt_length_fn=lambda row: len(
            _encode(
                tokenizer,
                _build_initial_prompt(
                    tokenizer,
                    row,
                    max_seq_length=max_seq_length,
                    max_prompt_tokens=max_prompt_tokens,
                ),
            )
        ),
        output_length_fn=lambda row: _estimate_issue_output_tokens(tokenizer, row),
        sample_size=sample_size,
        strategy=strategy,
        seed=seed,
        samples_per_prompt=samples_per_prompt,
        max_output_len=max_new_tokens,
    )

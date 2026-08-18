"""Reward helpers for DAPO-Math style mathematical reasoning data."""

from __future__ import annotations

import re
import string
from typing import Any

from RL_Framework.env.grader import math_equal
from RL_Framework.env.parse_utils_qwen import extract_answer

_BOX_RE = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_dapo_answer(text: str) -> str:
    """Extract a final answer from boxed output, answer tags, or Qwen parser."""
    text = text or ""
    tag = _ANSWER_TAG_RE.findall(text)
    if tag:
        return tag[-1].strip()
    boxed = _BOX_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return extract_answer(text, data_name="math") or ""


def _normalize_answer(answer: str) -> str:
    answer = str(answer or "").lower().strip()
    answer = "".join(ch for ch in answer if ch not in string.punctuation.replace("/", ""))
    return " ".join(answer.split())


def is_math_correct(prediction: str, ground_truth: str | list[str]) -> bool:
    if not ground_truth:
        return False
    answers = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    for answer in answers:
        answer = str(answer)
        try:
            if math_equal(prediction, answer, timeout=True):
                return True
        except Exception:
            pass
        if _normalize_answer(prediction) == _normalize_answer(answer):
            return True
    return False


def evaluate_math_completion(completion: str, ground_truth: str | list[str]) -> dict[str, Any]:
    pred = extract_dapo_answer(completion)
    correct = is_math_correct(pred, ground_truth)
    has_boxed = "\\boxed" in (completion or "")
    has_reasoning = len(completion or "") > len(pred) + 20
    format_score = 0.0
    if pred:
        format_score += 0.35
    if has_boxed:
        format_score += 0.45
    if has_reasoning:
        format_score += 0.20
    format_score = min(1.0, format_score)
    reward = 1.0 if correct else 0.05 * format_score
    return {
        "reward": float(reward),
        "accuracy": float(correct),
        "prediction": pred,
        "format_score": float(format_score),
        "has_boxed": bool(has_boxed),
    }


def format_math_validator_feedback(metrics: dict[str, Any]) -> str:
    status = "correct" if metrics.get("accuracy", 0.0) >= 1.0 else "incorrect"
    pred = metrics.get("prediction") or "<empty>"
    return (
        "### DAPO-Math Validator\n"
        f"status={status}\n"
        f"reward={metrics.get('reward', 0.0):.3f}\n"
        f"prediction={pred}\n"
        f"format_score={metrics.get('format_score', 0.0):.3f}\n"
        "Revise the solution if needed. Put only the final answer inside \\boxed{} at the end."
    )


def dapo_math_reward_fn(
    prompt: str,
    completion: str,
    ground_truth: str | list[str] = "",
    answer: str | list[str] = "",
    **_: Any,
) -> float:
    del prompt
    target = ground_truth or answer
    return float(evaluate_math_completion(completion, target)["reward"])

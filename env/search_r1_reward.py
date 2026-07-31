"""Support code for Search r1 reward."""

import re
import string
from typing import List, Union

from RL_Framework.env.grader import math_equal
from RL_Framework.env.parse_utils_qwen import extract_answer


def extract_search_r1_answer(text: str) -> str:
    """Extract search r1 answer."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_search_r1_think(text: str) -> str:
    """Extract search r1 think."""
    for pattern in [r"<think>(.*?)</think>", r"<thinking>(.*?)</thinking>"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def search_r1_reward_fn(
    prompt: str,
    completion: str,
    ground_truth: Union[str, List[str]] = "",
    **kwargs,
) -> float:
    """Search r1 reward fn."""
    if not ground_truth:
        return 0.0

    model_answer = extract_search_r1_answer(completion)
    if not model_answer:
        return 0.0

    answers = ground_truth if isinstance(ground_truth, list) else [ground_truth]

    def normalize_answer(answer: str) -> str:
        answer = answer.lower()
        answer = "".join(char for char in answer if char not in string.punctuation)
        answer = re.sub(r"\b(a|an|the)\b", " ", answer)
        return " ".join(answer.split())

    for answer in answers:

        try:
            if math_equal(model_answer, str(answer), timeout=True):
                return 1.0
        except Exception:
            pass


        if normalize_answer(model_answer) == normalize_answer(str(answer)):
            return 1.0

    return 0.0


def format_reward_fn(completion: str) -> float:
    """Format reward fn."""
    reward = 0.0


    if "<answer>" in completion and "</answer>" in completion:
        reward += 0.1


    if "<think>" in completion or "<thinking>" in completion:
        reward += 0.05

    return min(reward, 0.15)


def combined_search_r1_reward_fn(
    prompt: str,
    completion: str,
    ground_truth: Union[str, List[str]] = "",
    format_weight: float = 0.15,
    **kwargs,
) -> float:
    """Combined search r1 reward fn."""
    correctness = search_r1_reward_fn(prompt, completion, ground_truth, **kwargs)
    fmt = format_reward_fn(completion)
    return correctness + format_weight * fmt

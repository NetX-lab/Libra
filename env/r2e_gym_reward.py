"""Reward and validator helpers for R2E-Gym issue-generation RL."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_ISSUE_RE = re.compile(r"\[ISSUE\](.*?)\[/ISSUE\]", re.DOTALL | re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "but",
        "not", "when", "into", "should", "expected", "actual", "behavior",
        "description", "issue", "test", "tests", "code", "output",
    }
    return {
        token.lower()
        for token in _WORD_RE.findall(text or "")
        if len(token) >= 3 and token.lower() not in stop
    }


def extract_issue_text(completion: str) -> str:
    """Extract the issue body from [ISSUE]...[/ISSUE], or return completion."""
    matches = _ISSUE_RE.findall(completion or "")
    if not matches:
        return completion or ""
    return matches[-1].strip()


def parse_expected_tests(expected_output_json: str | dict | None) -> dict[str, str]:
    if expected_output_json is None:
        return {}
    if isinstance(expected_output_json, dict):
        return {str(k): str(v) for k, v in expected_output_json.items()}
    try:
        parsed = json.loads(expected_output_json)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _f1(pred_tokens: set[str], target_tokens: set[str]) -> float:
    if not pred_tokens or not target_tokens:
        return 0.0
    overlap = len(pred_tokens & target_tokens)
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(target_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_issue(
    completion: str,
    target_issue: str,
    expected_output_json: str | dict | None = None,
    modified_files: list[str] | str | None = None,
) -> dict[str, Any]:
    """Return dense reward components for a generated R2E-Gym issue."""
    issue = extract_issue_text(completion)
    pred_tokens = _tokenize(issue)
    target_tokens = _tokenize(target_issue)
    lexical_f1 = _f1(pred_tokens, target_tokens)

    expected_tests = parse_expected_tests(expected_output_json)
    test_names = {
        part.lower()
        for name in expected_tests
        for part in re.split(r"[.:_\[\]\s]+", name)
        if len(part) >= 3
    }
    test_coverage = (
        len(test_names & pred_tokens) / max(1, len(test_names))
        if test_names
        else 0.0
    )

    if isinstance(modified_files, str):
        try:
            modified_files = json.loads(modified_files)
        except Exception:
            modified_files = [modified_files]
    file_tokens = {
        part.lower()
        for path in (modified_files or [])
        for part in re.split(r"[/_.\-]+", str(path))
        if len(part) >= 3
    }
    file_coverage = (
        len(file_tokens & pred_tokens) / max(1, len(file_tokens))
        if file_tokens
        else 0.0
    )

    has_issue_tags = bool(_ISSUE_RE.search(completion or ""))
    has_title = bool(re.search(r"(title:|^#|\*\*title:\*\*)", issue, re.IGNORECASE | re.MULTILINE))
    has_expected = "expected" in issue.lower()
    has_actual = "actual" in issue.lower() or "error" in issue.lower()
    format_score = sum([has_issue_tags, has_title, has_expected, has_actual]) / 4.0

    reward = (
        0.55 * lexical_f1
        + 0.20 * test_coverage
        + 0.10 * file_coverage
        + 0.15 * format_score
    )
    reward = max(0.0, min(1.0, reward))

    return {
        "reward": reward,
        "lexical_f1": lexical_f1,
        "test_coverage": test_coverage,
        "file_coverage": file_coverage,
        "format_score": format_score,
        "issue_length": len(issue),
        "n_expected_tests": len(expected_tests),
        "missing_sections": [
            name
            for name, ok in [
                ("[ISSUE] tags", has_issue_tags),
                ("title", has_title),
                ("expected behavior", has_expected),
                ("actual behavior/error", has_actual),
            ]
            if not ok
        ],
    }


def format_validator_feedback(metrics: dict[str, Any]) -> str:
    missing = metrics.get("missing_sections", [])
    missing_text = ", ".join(missing) if missing else "none"
    return (
        "### R2E-Gym Issue Validator\n"
        f"reward={metrics['reward']:.3f}\n"
        f"lexical_f1={metrics['lexical_f1']:.3f}\n"
        f"test_coverage={metrics['test_coverage']:.3f}\n"
        f"file_coverage={metrics['file_coverage']:.3f}\n"
        f"format_score={metrics['format_score']:.3f}\n"
        f"missing_sections={missing_text}\n"
        "Revise the issue to be concise, concrete, and faithful to the failing behavior."
    )


def r2e_gym_reward_fn(
    prompt: str,
    completion: str,
    target_issue: str = "",
    expected_output_json: str | dict | None = None,
    modified_files: list[str] | str | None = None,
    **_: Any,
) -> float:
    del prompt
    return float(
        evaluate_issue(
            completion=completion,
            target_issue=target_issue,
            expected_output_json=expected_output_json,
            modified_files=modified_files,
        )["reward"]
    )

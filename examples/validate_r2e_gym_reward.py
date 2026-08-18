"""Validate R2E-Gym pytest logs against a task's expected output."""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-log", type=Path, required=True)
    parser.add_argument("--after-log", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    return parser.parse_args()


def parse_pytest_log(log: str) -> dict[str, str]:
    if "short test summary info" not in log:
        return {}

    summary = log.split("short test summary info", 1)[1]
    statuses = {}
    for line in summary.strip().splitlines():
        if "PASSED" in line:
            name = ".".join(line.split("::")[1:])
            statuses[name] = "PASSED"
        elif "FAILED" in line:
            name = ".".join(line.split("::")[1:]).split(" - ")[0]
            statuses[name] = "FAILED"
        elif "ERROR" in line:
            name = ".".join(line.split("::")[1:]).split(" - ")[0]
            statuses[name] = "ERROR"
    return {re.sub(r"\u001b\[\d+m", "", key): value for key, value in statuses.items()}


def normalize(statuses: dict[str, str]) -> dict[str, str]:
    return {
        key.split(" - ")[0]: statuses[key]
        for key in sorted(statuses)
    }


def main():
    args = parse_args()
    before = normalize(parse_pytest_log(args.before_log.read_text(errors="replace")))
    after = normalize(parse_pytest_log(args.after_log.read_text(errors="replace")))
    expected = normalize(json.loads(args.expected.read_text()))

    if before == expected:
        raise AssertionError("R2E-Gym task unexpectedly passed before applying the patch")
    if after != expected:
        missing = sorted(set(expected) - set(after))
        unexpected = sorted(set(after) - set(expected))
        mismatched = sorted(
            key for key in set(after) & set(expected) if after[key] != expected[key]
        )
        raise AssertionError(
            "R2E-Gym reward mismatch after applying the patch: "
            f"expected={len(expected)} parsed={len(after)} "
            f"missing={missing[:10]} unexpected={unexpected[:10]} "
            f"mismatched={mismatched[:10]}"
        )

    print(
        "R2E_GYM_TASK_OK "
        f"before_matches_expected=false after_matches_expected=true tests={len(expected)}"
    )


if __name__ == "__main__":
    main()

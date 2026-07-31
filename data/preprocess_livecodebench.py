"""Support code for Preprocess livecodebench."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Field mapping helpers
# ---------------------------------------------------------------------------

LIVECODEBENCH_FIELD_CANDIDATES = {
    "question": ["question", "problem", "prompt", "description", "content"],
    "test_cases": ["test", "test_cases", "public_test_cases", "input_output", "in_outs"],
    "starter_code": ["starter_code", "code", "template", "skeleton"],
    "platform": ["platform", "source", "contest", "origin"],
    "difficulty": ["difficulty", "diff", "rating"],
    "title": ["title", "name", "id"],
    "fn_name": ["fn_name", "function_name", "entry_point"],
}


def extract_field(example: dict, candidates: list[str]) -> Any:
    """Extract field."""
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return None


def normalize_test_cases(raw_test: Any) -> dict[str, Any] | None:
    """Normalize test cases."""
    if raw_test is None:
        return None


    if isinstance(raw_test, dict):
        result = {"inputs": [], "outputs": []}
        if "inputs" in raw_test:
            result["inputs"] = raw_test["inputs"]
        if "outputs" in raw_test:
            result["outputs"] = raw_test["outputs"]
        if "fn_name" in raw_test:
            result["fn_name"] = raw_test["fn_name"]

        if "assert_case" in raw_test:
            result["assert_case"] = raw_test["assert_case"]
        return result if result["inputs"] or result["outputs"] else None


    if isinstance(raw_test, str):
        try:
            parsed = json.loads(raw_test)
            return normalize_test_cases(parsed)
        except json.JSONDecodeError:
            return None

    # list of tuples
    if isinstance(raw_test, list):
        inputs = []
        outputs = []
        for item in raw_test:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                inputs.append(item[0])
                outputs.append(item[1])
        return {"inputs": inputs, "outputs": outputs} if inputs else None

    return None


def format_question(raw_question: Any, title: str | None = None) -> str:
    """Format question."""
    if raw_question is None:
        return ""

    if isinstance(raw_question, str):
        question_text = raw_question
    elif isinstance(raw_question, dict):

        parts = []
        for key in ["description", "problem_statement", "statement", "question", "content"]:
            if key in raw_question and raw_question[key]:
                parts.append(str(raw_question[key]))
        for key in ["input_format", "input", "input_spec"]:
            if key in raw_question and raw_question[key]:
                parts.append(f"Input Format:\n{raw_question[key]}")
        for key in ["output_format", "output", "output_spec"]:
            if key in raw_question[key] and raw_question[key]:
                parts.append(f"Output Format:\n{raw_question[key]}")
        for key in ["constraints", "notes", "explanation"]:
            if key in raw_question and raw_question[key]:
                parts.append(f"Constraints/Notes:\n{raw_question[key]}")
        question_text = "\n\n".join(parts)
    else:
        question_text = str(raw_question)

    if title:
        question_text = f"Title: {title}\n\n{question_text}"

    return question_text


def preprocess_livecodebench(example: dict, platform_filter: str | None = None) -> dict | None:
    """Preprocess livecodebench."""

    platform = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["platform"])
    if platform and isinstance(platform, str):
        platform = platform.lower().strip()


    if platform_filter and platform != platform_filter.lower().strip():
        return None


    raw_question = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["question"])
    title = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["title"])
    question = format_question(raw_question, title)

    if not question or not question.strip():
        return None


    raw_test = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["test_cases"])
    test_cases = normalize_test_cases(raw_test)

    if not test_cases or not test_cases.get("inputs") or not test_cases.get("outputs"):

        if "inputs" in example and "outputs" in example:
            test_cases = {
                "inputs": example["inputs"],
                "outputs": example["outputs"],
            }
            if "fn_name" in example:
                test_cases["fn_name"] = example["fn_name"]
        else:
            return None


    starter_code = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["starter_code"])
    if starter_code and not isinstance(starter_code, str):
        starter_code = str(starter_code)


    difficulty = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["difficulty"])


    fn_name = extract_field(example, LIVECODEBENCH_FIELD_CANDIDATES["fn_name"])
    if fn_name and isinstance(fn_name, str) and "fn_name" not in test_cases:
        test_cases["fn_name"] = fn_name

    result = {
        "question": question,
        "test_cases": test_cases,
        "starter_code": starter_code or "",
        "platform": platform or "unknown",
        "difficulty": str(difficulty) if difficulty else "unknown",
        "source": "livecodebench",
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess LiveCodeBench training data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="livecodebench/code_generation_lite",
        help="Hugging Face dataset name or local path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split, such as train, test, or validation",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="codeforces",
        choices=["codeforces", "leetcode", "atcoder", None],
        help="Filter by platform",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/livecodebench_train.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples; -1 means all",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote dataset code (required by some LiveCodeBench versions)",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} (split={args.split})")


    is_streaming = False
    try:
        load_kwargs = {"split": args.split}
        if args.config:
            load_kwargs["name"] = args.config
        if args.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        dataset = load_dataset(args.dataset, **load_kwargs)
    except Exception as e:
        print(f"Failed to load from Hugging Face: {e}")
        print("Trying streaming mode...")
        try:
            load_kwargs = {"split": args.split, "streaming": True}
            if args.config:
                load_kwargs["name"] = args.config
            if args.trust_remote_code:
                load_kwargs["trust_remote_code"] = True
            dataset = load_dataset(args.dataset, **load_kwargs)
            is_streaming = True
        except Exception as e2:
            print(f"Streaming mode also failed: {e2}")
            print("Trying the dataset argument as a local path...")
            if args.dataset.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=args.dataset, split="train")
            elif args.dataset.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=args.dataset, split="train")
            elif Path(args.dataset).is_dir():
                dataset = load_dataset(args.dataset, split=args.split)
            else:
                raise ValueError(f"Unable to load dataset: {args.dataset}")

    if is_streaming:
        print("Collecting samples from the streaming dataset...")
        dataset = list(dataset)

    print(f"Raw dataset size: {len(dataset)}")
    print(f"Platform filter: {args.platform or 'none'}")


    processed = []
    for i, example in enumerate(dataset):
        if args.max_samples > 0 and i >= args.max_samples:
            break

        item = preprocess_livecodebench(example, platform_filter=args.platform)
        if item is not None:
            processed.append(item)

    print(f"Valid samples after preprocessing: {len(processed)}")

    if len(processed) == 0:
        print("Warning: no valid samples. Check the dataset schema and platform filter.")
        print("First three raw samples for debugging:")
        for i, example in enumerate(dataset[:3]):
            print(f"\n--- Raw Sample {i} ---")
            if isinstance(example, dict):
                print(f"Keys: {list(example.keys())}")
                for k, v in example.items():
                    preview = str(v)[:200] if v is not None else "None"
                    print(f"  {k}: {preview}")
            else:
                print(f"  Type: {type(example)}")
                print(f"  Value: {str(example)[:500]}")
        return


    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")


    print("\nFirst two processed samples:")
    for i, item in enumerate(processed[:2], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Question (first 200 chars): {item['question'][:200]}...")
        print(f"Test cases: {len(item['test_cases'].get('inputs', []))} cases")
        print(f"Starter code: {'Yes' if item['starter_code'] else 'No'}")
        print(f"Platform: {item['platform']}, Difficulty: {item['difficulty']}")


if __name__ == "__main__":
    main()

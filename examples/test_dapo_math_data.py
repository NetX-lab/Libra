"""Validate local DAPO-Math-17K files before launching a long training job."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset

from examples.dapo_math_async_rl import _find_dataset_file, _preprocess


def main():
    data_file, data_format = _find_dataset_file()
    dataset = load_dataset(data_format, data_files=str(data_file), split="train")
    dataset = dataset.map(_preprocess, with_indices=True)
    dataset = dataset.filter(lambda row: bool(row.get("question")) and bool(row.get("ground_truth")))
    seen_prompt_ids: set[str] = set()

    def keep_first(row):
        prompt_id = str(row.get("prompt_id") or "")
        if prompt_id in seen_prompt_ids:
            return False
        seen_prompt_ids.add(prompt_id)
        return True

    raw_usable_rows = len(dataset)
    dataset = dataset.filter(keep_first)
    if len(dataset) == 0:
        raise RuntimeError(f"DAPO-Math data has no usable rows: {data_file}")
    min_rows = int(os.environ.get("DAPO_MIN_ROWS", "1000"))
    if len(dataset) < min_rows:
        raise RuntimeError(f"DAPO-Math data has only {len(dataset)} usable rows; expected at least {min_rows}")
    print(f"DAPO-Math data OK: {len(dataset)} unique usable rows from {Path(data_file)}")
    print(f"Raw usable rows before prompt_id deduplication: {raw_usable_rows}")
    print(f"First prompt_id: {dataset[0]['prompt_id']}")
    print(f"First question preview: {dataset[0]['question'][:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()

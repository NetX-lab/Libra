"""Strict integrity test for the official R2E-Gym V1 dataset."""

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "r2e_gym_v1"
EXPECTED_REPOS = {
    "aiohttp",
    "coveragepy",
    "datalad",
    "matplotlib",
    "moto",
    "numpy",
    "orange3",
    "pandas",
    "pillow",
    "pyramid",
    "scrapy",
    "sympy",
    "tornado",
}


def main():
    shards = sorted(DATASET_DIR.glob("train-*-of-00013.parquet"))
    counts = Counter()
    rows = 0

    if shards:
        assert len(shards) == 13, f"Expected 13 R2E-Gym shards, got {len(shards)}"
        row_iterables = []
        for shard in shards:
            table = pq.read_table(
                shard,
                columns=[
                    "repo_name",
                    "docker_image",
                    "prompt",
                    "problem_statement",
                    "expected_output_json",
                ],
            )
            row_iterables.append(table.to_pylist())
    else:
        index_path = DATASET_DIR / "index.jsonl"
        assert index_path.exists(), (
            "Expected R2E-Gym parquet shards or prepared index, "
            f"found neither in {DATASET_DIR}"
        )
        row_iterables = [
            [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        ]

    for row_iterable in row_iterables:
        for row in row_iterable:
            problem_statement = row["problem_statement"] or ""
            prompt = row["prompt"] or ""
            assert problem_statement.strip() or prompt.strip()
            assert ":" in row["docker_image"]
            assert json.loads(row["expected_output_json"])
            counts[row["repo_name"]] += 1
            rows += 1

    assert rows == 8101, f"Expected 8101 R2E-Gym tasks, got {rows}"
    assert set(counts) == EXPECTED_REPOS, (
        f"Expected only official R2E-Gym repositories, got {sorted(counts)}"
    )
    print(f"R2E_GYM_DATASET_OK rows={rows} repos={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

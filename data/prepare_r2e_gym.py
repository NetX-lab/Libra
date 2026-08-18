"""Validate the official R2E-Gym V1 shards and build a compact task index."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_ROWS = 8101
EXPECTED_SHARDS = 13
EXPECTED_REPOSITORIES = {
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
REQUIRED_COLUMNS = {
    "repo_name",
    "docker_image",
    "commit_hash",
    "parsed_commit_content",
    "execution_result_content",
    "modified_files",
    "prompt",
    "problem_statement",
    "expected_output_json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Output compact JSONL index; defaults to dataset_dir/index.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output validation manifest; defaults to dataset_dir/manifest.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    shards = sorted(args.dataset_dir.glob("train-*-of-00013.parquet"))
    if len(shards) != EXPECTED_SHARDS:
        raise ValueError(f"Expected {EXPECTED_SHARDS} shards, found {len(shards)}")

    index_path = args.index or args.dataset_dir / "index.jsonl"
    manifest_path = args.manifest or args.dataset_dir / "manifest.json"
    repo_counts = Counter()
    docker_images = set()
    row_count = 0
    shard_records = []

    with index_path.open("w", encoding="utf-8") as index:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            columns = set(parquet.schema_arrow.names)
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"{shard.name} is missing columns: {sorted(missing)}")

            shard_rows = 0
            for batch in parquet.iter_batches(
                columns=[
                    "repo_name",
                    "docker_image",
                    "commit_hash",
                    "prompt",
                    "problem_statement",
                    "expected_output_json",
                    "modified_files",
                ],
                batch_size=128,
            ):
                for row in batch.to_pylist():
                    problem_statement = row["problem_statement"] or ""
                    prompt = row["prompt"] or ""
                    if not problem_statement.strip() and not prompt.strip():
                        raise ValueError(f"Empty task text in {shard.name}")
                    if ":" not in row["docker_image"]:
                        raise ValueError(f"Invalid Docker image in {shard.name}")
                    json.loads(row["expected_output_json"])

                    index.write(
                        json.dumps(
                            {
                                "data_source": "r2e_gym_v1",
                                "repo_name": row["repo_name"],
                                "docker_image": row["docker_image"],
                                "commit_hash": row["commit_hash"],
                                "problem_statement": problem_statement,
                                "prompt": prompt,
                                "task_text": problem_statement or prompt,
                                "expected_output_json": row["expected_output_json"],
                                "modified_files": row["modified_files"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    repo_counts[row["repo_name"]] += 1
                    docker_images.add(row["docker_image"])
                    row_count += 1
                    shard_rows += 1

            shard_records.append(
                {
                    "file": shard.name,
                    "rows": shard_rows,
                    "bytes": shard.stat().st_size,
                }
            )

    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {row_count}")
    if set(repo_counts) != EXPECTED_REPOSITORIES:
        raise ValueError(
            "Unexpected repository set: "
            f"expected {sorted(EXPECTED_REPOSITORIES)}, found {sorted(repo_counts)}"
        )

    manifest = {
        "dataset": "R2E-Gym/R2E-Gym-V1",
        "data_source": "r2e_gym_v1",
        "rows": row_count,
        "repositories": dict(sorted(repo_counts.items())),
        "unique_docker_images": len(docker_images),
        "shards": shard_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

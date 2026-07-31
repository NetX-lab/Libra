"""Convert the official Search-R1 parquet dataset to the local JSONL format."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_SOURCES = {"nq", "hotpotqa"}
EXPECTED_ROWS = 169615


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Official Search-R1 train.parquet")
    parser.add_argument("output", type=Path, help="Destination JSONL file")
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_ROWS,
        help="Expected row count; set to 0 to disable the row-count check",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    written = 0
    parquet = pq.ParquetFile(args.input)

    with args.output.open("w", encoding="utf-8") as output:
        for batch in parquet.iter_batches(
            columns=["id", "question", "golden_answers", "data_source"],
            batch_size=2048,
        ):
            for row in batch.to_pylist():
                source = row["data_source"]
                question = (row["question"] or "").strip()
                answers = [
                    answer.strip()
                    for answer in (row["golden_answers"] or [])
                    if answer and answer.strip()
                ]
                if source not in EXPECTED_SOURCES:
                    raise ValueError(f"Unexpected data source: {source!r}")
                if not question or not answers:
                    raise ValueError(f"Invalid Search-R1 sample: {row['id']}")

                output.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "question": question,
                            "ground_truth": answers,
                            "data_source": source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                counts[source] += 1
                written += 1

    if set(counts) != EXPECTED_SOURCES:
        raise ValueError(f"Expected both Search-R1 sources, got: {dict(counts)}")
    if args.expected_rows > 0 and written != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} Search-R1 rows, found {written}"
        )

    print(f"Wrote {written} Search-R1 samples to {args.output}")
    print(f"Source counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

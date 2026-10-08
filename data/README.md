# Libra data utilities

This directory contains source-controlled download, conversion, and validation
utilities. Dataset payloads and generated indexes are intentionally excluded
from Git.

- `prepare_r2e_gym.py` validates the R2E-Gym V1 parquet shards and writes a
  compact task index plus a checksum manifest.
- `prepare_search_r1.py` converts the Search-R1 NQ/HotpotQA parquet data into
  Libra's JSONL schema.
- `download_gsm8k.py` prepares the small GSM8K examples.
- `preprocess_livecodebench.py` prepares the code-agent examples.

See [`../docs/data_preparation.md`](../docs/data_preparation.md) for dataset
sources, exact commands, expected outputs, and validation steps.

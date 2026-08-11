"""Startup length profiling helpers for initial GRP planning."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


LengthFn = Callable[[dict[str, Any]], int]


def load_jsonl_rows(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    """Load dictionary rows from a JSONL file."""

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def _length_record_from_payload(
    payload: Mapping[str, Any],
    *,
    step: Any = None,
) -> dict[str, Any] | None:
    prompt = payload.get("input_len", payload.get("prompt_length", 0))
    gen = payload.get("output_len", payload.get("gen_length", 0))
    try:
        prompt_i = int(prompt)
        gen_i = int(gen)
    except (TypeError, ValueError):
        return None
    if prompt_i <= 0 and gen_i <= 0:
        return None

    record = {
        "input_len": max(1, prompt_i),
        "output_len": max(1, gen_i),
    }
    if step is not None:
        record["step"] = step
    for key in (
        "prompt_id",
        "source_index",
        "sample_index",
        "profile_source",
        "total_output_tokens",
        "tool_returns",
        "cmlfq_request_id",
    ):
        if key in payload:
            record[key] = payload[key]
    return record


def _length_records_from_history_payload(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    step = payload.get("step")
    sequences = payload.get("sequences")
    if isinstance(sequences, list):
        records = []
        for seq in sequences:
            if isinstance(seq, Mapping):
                record = _length_record_from_payload(seq, step=step)
                if record is not None:
                    records.append(record)
        if records:
            return records

    prompt_lengths = payload.get("raw_prompt_lengths")
    gen_lengths = payload.get("raw_gen_lengths")
    if isinstance(prompt_lengths, list) and isinstance(gen_lengths, list):
        records = []
        for prompt, gen in zip(prompt_lengths, gen_lengths):
            record = _length_record_from_payload(
                {"input_len": prompt, "output_len": gen},
                step=step,
            )
            if record is not None:
                records.append(record)
        if records:
            return records

    record = _length_record_from_payload(payload, step=step)
    return [record] if record is not None else []


def load_length_profile_records(
    path: str | Path,
    *,
    max_records: int = 0,
) -> list[dict[str, Any]]:
    """Load planner-compatible length records from flat or step-history JSONL."""

    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    records.extend(_length_records_from_history_payload(payload))
    except FileNotFoundError:
        return []
    except OSError:
        return []
    if max_records > 0 and len(records) > max_records:
        return records[-int(max_records) :]
    return records


def profile_jsonl_has_records(path: str | Path) -> bool:
    """Return True when the JSONL already contains usable length profile rows."""

    try:
        with open(path, "r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                input_len = payload.get("input_len", payload.get("prompt_length", 0))
                output_len = payload.get("output_len", payload.get("gen_length", 0))
                try:
                    if int(input_len) > 0 and int(output_len) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return False


def load_profile_summary(path: str | Path) -> dict[str, Any] | None:
    """Load a profile summary JSON file when present."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def startup_profile_metadata(
    *,
    source: str = "r2e_gym_startup_profile",
    dataset_jsonl: str | Path = "",
    model_path: str | Path = "",
    tokenizer_path: str | Path = "",
    sample_size: int = 0,
    strategy: str = "",
    seed: int = 0,
    samples_per_prompt: int = 0,
    max_new_tokens: int = 0,
    max_seq_length: int = 0,
    r2e_max_turns: int = 0,
    r2e_max_prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Return the stable metadata fields used to validate startup profiles."""

    return {
        "source": source,
        "dataset_jsonl": str(dataset_jsonl),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "sample_size": int(sample_size),
        "strategy": str(strategy),
        "seed": int(seed),
        "samples_per_prompt": int(samples_per_prompt),
        "max_new_tokens": int(max_new_tokens),
        "max_seq_length": int(max_seq_length),
        "r2e_max_turns": int(r2e_max_turns),
        "r2e_max_prompt_tokens": int(r2e_max_prompt_tokens),
    }


def profile_metadata_matches(
    summary: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> bool:
    """Return True when a loaded summary matches the expected metadata."""

    if not summary:
        return False
    for key, value in expected.items():
        if summary.get(key) != value:
            return False
    return True


def profile_summary_matches(
    path: str | Path,
    expected: Mapping[str, Any],
) -> bool:
    """Return True when the summary JSON at ``path`` matches ``expected``."""

    return profile_metadata_matches(load_profile_summary(path), expected)


def sample_indices(
    total: int,
    sample_size: int,
    *,
    strategy: str = "spread",
    seed: int = 0,
) -> list[int]:
    """Select row indices for a startup profile sample."""

    total = max(0, int(total))
    sample_size = max(0, int(sample_size))
    if total == 0 or sample_size == 0:
        return []
    if sample_size >= total:
        return list(range(total))

    strategy = (strategy or "spread").lower()
    if strategy == "first":
        return list(range(sample_size))
    if strategy == "random":
        rng = random.Random(int(seed))
        return sorted(rng.sample(range(total), sample_size))
    if strategy != "spread":
        raise ValueError("profile strategy must be one of: spread, random, first")
    if sample_size == 1:
        return [0]
    return [
        min(total - 1, round(i * (total - 1) / (sample_size - 1)))
        for i in range(sample_size)
    ]


def default_prompt_id(row: dict[str, Any], index: int) -> str:
    """Return a stable prompt id for profile metadata."""

    for key in ("prompt_id", "id", "uid"):
        value = row.get(key)
        if value:
            return str(value)
    repo = row.get("repo_name")
    commit = row.get("commit_hash")
    if repo or commit:
        return f"{repo or 'repo'}:{commit or index}"
    return str(index)


def build_length_profile(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_length_fn: LengthFn,
    output_length_fn: LengthFn,
    sample_size: int,
    strategy: str = "spread",
    seed: int = 0,
    samples_per_prompt: int = 1,
    max_output_len: int = 0,
    prompt_id_fn: Callable[[dict[str, Any], int], str] = default_prompt_id,
) -> list[dict[str, Any]]:
    """Build GRP history records from sampled real rows."""

    indices = sample_indices(len(rows), sample_size, strategy=strategy, seed=seed)
    repeats = max(1, int(samples_per_prompt))
    records: list[dict[str, Any]] = []

    for source_index in indices:
        row = rows[source_index]
        prompt_len = max(1, int(prompt_length_fn(row)))
        output_len = max(1, int(output_length_fn(row)))
        if max_output_len > 0:
            output_len = min(output_len, int(max_output_len))
        prompt_id = prompt_id_fn(row, source_index)
        for sample_index in range(repeats):
            records.append(
                {
                    "input_len": prompt_len,
                    "output_len": output_len,
                    "prompt_id": prompt_id,
                    "source_index": int(source_index),
                    "sample_index": int(sample_index),
                    "profile_source": "startup_real_sample",
                }
            )
    return records


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stats(values: Iterable[int]) -> dict[str, float]:
    data = [int(v) for v in values if int(v) > 0]
    if not data:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(data),
        "min": float(min(data)),
        "max": float(max(data)),
        "mean": float(sum(data) / len(data)),
        "p50": _percentile(data, 0.50),
        "p90": _percentile(data, 0.90),
        "p95": _percentile(data, 0.95),
        "p99": _percentile(data, 0.99),
    }


def summarize_length_profile(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize sampled GRP profile records."""

    def as_positive_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    input_lengths = [
        as_positive_int(record.get("input_len", record.get("prompt_length", 0)))
        for record in records
    ]
    output_lengths = [
        as_positive_int(record.get("output_len", record.get("gen_length", 0)))
        for record in records
    ]
    total_lengths = [prompt + output for prompt, output in zip(input_lengths, output_lengths)]
    return {
        "records": len(records),
        "input_len": _stats(input_lengths),
        "output_len": _stats(output_lengths),
        "total_len": _stats(total_lengths),
    }


def write_profile_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write sampled profile records as planner-compatible JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_profile_summary(summary: dict[str, Any], path: str | Path) -> None:
    """Write a profile summary JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

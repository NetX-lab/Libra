"""Evaluate one rollout candidate with Vidur.

Command contract used by GlobalResourcePlanner:
  python scripts/vidur_rollout_eval.py TRACE_CSV OUTPUT_JSON TP_LIST --vidur-path PATH

TRACE_CSV is emitted by ``VidurRolloutSimulatorAdapter`` with columns:
``arrived_at,num_prefill_tokens,num_decode_tokens``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _rewrite_traces(input_trace: Path, work_dir: Path) -> tuple[Path, Path, int]:
    length_trace = work_dir / "vidur_lengths.csv"
    interval_trace = work_dir / "vidur_arrivals.csv"
    rows: list[dict[str, Any]] = []
    with input_trace.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    with length_trace.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["num_prefill_tokens", "num_decode_tokens"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "num_prefill_tokens": int(float(row["num_prefill_tokens"])),
                    "num_decode_tokens": int(float(row["num_decode_tokens"])),
                }
            )

    with interval_trace.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arrival_time"])
        writer.writeheader()
        last_seconds = 1.0
        for row in rows:
            seconds = 1.0 + float(row.get("arrived_at", 0.0) or 0.0)
            last_seconds = seconds
            writer.writerow(
                {"arrival_time": f"1970-01-01 00:00:{seconds:06.3f}"}
            )
        writer.writerow(
            {"arrival_time": f"1970-01-01 00:00:{last_seconds + 1.0:06.3f}"}
        )

    return length_trace, interval_trace, len(rows)


def _parse_metrics(output_dir: Path) -> dict[str, Any]:
    request_files = sorted(output_dir.glob("**/request_metrics.csv"))
    if request_files:
        path = request_files[-1]
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        values = []
        for row in rows:
            for key in (
                "request_e2e_time",
                "request_execution_plus_preemption_time",
                "request_execution_time",
                "request_model_execution_time",
            ):
                if key in row and row[key] not in ("", None):
                    values.append(float(row[key]))
                    break
        if values:
            return {
                "makespan": max(values),
                "mean_request_time": sum(values) / len(values),
                "num_requests": len(values),
                "metrics_file": str(path),
            }

    batch_files = sorted(output_dir.glob("**/batch_metrics.csv"))
    if batch_files:
        path = batch_files[-1]
        values = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key, value in row.items():
                    if "batch_execution_time" in key and value not in ("", None):
                        values.append(float(value))
        if values:
            return {
                "makespan": sum(values),
                "mean_batch_time": sum(values) / len(values),
                "num_batches": len(values),
                "metrics_file": str(path),
            }

    raise RuntimeError(f"No usable Vidur metrics found under {output_dir}")


def _scheduler_batch_size_arg(scheduler: str) -> str:
    normalized = scheduler.lower().replace("-", "_")
    if normalized == "faster_transformer":
        return "--faster_transformer_scheduler_config_batch_size_cap"
    if normalized in {"sarathi", "vllm", "lightllm", "orca"}:
        return f"--{normalized}_scheduler_config_batch_size_cap"
    raise ValueError(f"Unsupported Vidur scheduler: {scheduler}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_csv")
    parser.add_argument("output_json")
    parser.add_argument("tp_list")
    parser.add_argument("--vidur-path", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device", default="a100")
    parser.add_argument("--scheduler", default="sarathi")
    parser.add_argument("--batch-size-cap", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    vidur_path = Path(args.vidur_path).expanduser().resolve()
    output_json = Path(args.output_json)
    work_dir = output_json.parent / "vidur_real"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    length_trace, interval_trace, num_requests = _rewrite_traces(
        Path(args.trace_csv),
        work_dir,
    )
    tp_list = [int(x) for x in args.tp_list.split(",") if x]
    tp = max(tp_list) if tp_list else 1
    num_replicas = max(1, sum(tp_list) // tp)
    sim_output_dir = Path(args.output_dir) if args.output_dir else work_dir / "output"

    cmd = [
        args.python,
        "-m",
        "vidur.main",
        "--replica_config_device",
        args.device,
        "--replica_config_model_name",
        args.model_name,
        "--cluster_config_num_replicas",
        str(num_replicas),
        "--replica_config_tensor_parallel_size",
        str(tp),
        "--replica_config_num_pipeline_stages",
        "1",
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        str(num_requests),
        "--length_generator_config_type",
        "trace",
        "--trace_request_length_generator_config_trace_file",
        str(length_trace),
        "--trace_request_length_generator_config_max_tokens",
        str(args.max_tokens),
        "--interval_generator_config_type",
        "trace",
        "--trace_request_interval_generator_config_trace_file",
        str(interval_trace),
        "--trace_request_interval_generator_config_start_time",
        "1970-01-01 00:00:00",
        "--trace_request_interval_generator_config_end_time",
        "1970-01-01 23:59:59",
        "--replica_scheduler_config_type",
        args.scheduler,
        _scheduler_batch_size_arg(args.scheduler),
        str(args.batch_size_cap),
        "--metrics_config_output_dir",
        str(sim_output_dir),
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_token_completion_metrics",
    ]
    if args.scheduler.lower().replace("-", "_") == "sarathi":
        cmd.extend(
            [
                "--sarathi_scheduler_config_chunk_size",
                str(args.chunk_size),
            ]
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vidur_path)
    proc = subprocess.run(
        cmd,
        cwd=str(vidur_path),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Vidur command failed\n"
            f"cmd={' '.join(cmd)}\nstdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-4000:]}"
        )

    result = _parse_metrics(sim_output_dir)
    result.update(
        {
            "backend": "vidur",
            "tp_list": tp_list,
            "tensor_parallel_size": tp,
            "num_replicas": num_replicas,
            "num_requests": num_requests,
            "output_dir": str(sim_output_dir),
        }
    )
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Simulator adapters for GlobalResourcePlanner.

This module keeps RL_Framework decoupled from external simulator projects.  The
planner can still use the analytic in-repo CostModel, but training and rollout
evaluations may be delegated to Sailor and Vidur through small adapter
interfaces.  External calls are intentionally optional and fall back to the
analytic model unless the config disables fallback.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from RL_Framework.infra.cost_model.model import (
    CostModel,
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)


class SimulatorBackendError(RuntimeError):
    """Raised when an external simulator cannot produce a valid estimate."""


@dataclass
class SimulatorEstimate:
    """Normalized simulator result."""

    seconds: float
    details: dict[str, Any]
    is_oom: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_external_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    return p


def _parse_seconds(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            value = float(payload[key])
            if math.isfinite(value) and value >= 0:
                return value
    if "throughput" in payload and float(payload["throughput"]) > 0:
        return 1.0 / float(payload["throughput"])
    raise SimulatorBackendError(
        f"Simulator output does not contain any time key from {keys}"
    )


def _run_command(
    command: str,
    cwd: Path | None,
    timeout_s: float,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )


class SailorTrainingSimulatorAdapter:
    """Adapter for Sailor-backed training-time estimation.

    The most robust integration point is a command contract:
      command reads ``{input_json}`` and writes ``{output_json}``.

    Output JSON can contain one of:
      ``t_train``, ``iteration_time``, ``iteration_time_s``,
      ``iteration_time_sec``, or ``throughput``.
    """

    def __init__(self, planner_config: Any, rl_config: Any):
        self.planner_config = planner_config
        self.rl_config = rl_config
        self.sailor_path = _resolve_external_path(planner_config.sailor_path)
        self.timeout_s = float(planner_config.simulator_timeout_s)

    def evaluate_training(
        self,
        train_config: TrainParallelConfig,
        B_global: int,
        L: int,
    ) -> SimulatorEstimate:
        command = (self.planner_config.sailor_train_command or "").strip()
        if not command:
            raise SimulatorBackendError(
                "sailor_train_command is empty; configure a command to enable "
                "Sailor-backed training simulation"
            )
        if not self.sailor_path.exists():
            raise SimulatorBackendError(f"Sailor path does not exist: {self.sailor_path}")

        with tempfile.TemporaryDirectory(
            prefix="sailor_train_", dir=self._cache_dir()
        ) as td:
            work = Path(td)
            input_json = work / "input.json"
            output_json = work / "output.json"
            payload = {
                "train_config": asdict(train_config),
                "batch_size": B_global,
                "avg_sequence_length": L,
                "hardware": asdict(self.rl_config.hardware),
                "model_arch": asdict(self.rl_config.model_arch),
                "profiling": asdict(self.rl_config.profiling),
            }
            input_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            formatted = command.format(
                input_json=shlex.quote(str(input_json)),
                output_json=shlex.quote(str(output_json)),
                sailor_path=shlex.quote(str(self.sailor_path)),
                rl_framework_path=shlex.quote(str(_project_root())),
                python=shlex.quote(self.planner_config.sailor_python),
            )
            proc = _run_command(
                formatted,
                cwd=self.sailor_path,
                timeout_s=self.timeout_s,
                extra_env={"PYTHONPATH": str(self.sailor_path)},
            )
            if proc.returncode != 0:
                raise SimulatorBackendError(
                    "Sailor training simulator failed "
                    f"(code={proc.returncode}): {proc.stderr[-1000:]}"
                )
            result = self._read_result(output_json, proc.stdout)
            seconds = _parse_seconds(
                result,
                ("t_train", "iteration_time", "iteration_time_s", "iteration_time_sec"),
            )
            return SimulatorEstimate(
                seconds=seconds,
                is_oom=bool(result.get("is_oom", result.get("oom", False))),
                details={
                    "backend": "sailor",
                    "command": formatted,
                    "raw": result,
                },
            )

    def _cache_dir(self) -> str:
        path = Path(self.planner_config.simulator_cache_dir)
        if not path.is_absolute():
            path = _repo_root() / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _read_result(self, output_json: Path, stdout: str) -> dict[str, Any]:
        if output_json.exists():
            return json.loads(output_json.read_text(encoding="utf-8"))
        marker = "SAILOR_RESULT_JSON="
        for line in stdout.splitlines():
            if line.startswith(marker):
                return json.loads(line[len(marker) :])
        raise SimulatorBackendError(
            "Sailor simulator did not write output_json or SAILOR_RESULT_JSON"
        )


class VidurRolloutSimulatorAdapter:
    """Adapter for Vidur-backed rollout makespan estimation."""

    def __init__(self, planner_config: Any, rl_config: Any):
        self.planner_config = planner_config
        self.rl_config = rl_config
        self.vidur_path = _resolve_external_path(planner_config.vidur_path)
        self.timeout_s = float(planner_config.simulator_timeout_s)

    def evaluate_rollout(
        self,
        rollout_config: RolloutClusterConfig,
        requests: list[RequestInfo],
    ) -> SimulatorEstimate:
        if not requests:
            return SimulatorEstimate(
                seconds=0.0,
                details={"backend": "vidur", "reason": "empty_requests"},
            )
        if not self.vidur_path.exists():
            raise SimulatorBackendError(f"Vidur path does not exist: {self.vidur_path}")

        with tempfile.TemporaryDirectory(
            prefix="vidur_rollout_", dir=self._cache_dir()
        ) as td:
            work = Path(td)
            trace_csv = work / "request_trace.csv"
            output_json = work / "output.json"
            output_dir = work / "vidur_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_trace(trace_csv, requests)

            command = (self.planner_config.vidur_rollout_command or "").strip()
            if command:
                formatted = command.format(
                    trace_csv=shlex.quote(str(trace_csv)),
                    output_json=shlex.quote(str(output_json)),
                    output_dir=shlex.quote(str(output_dir)),
                    vidur_path=shlex.quote(str(self.vidur_path)),
                    rl_framework_path=shlex.quote(str(_project_root())),
                    python=shlex.quote(self.planner_config.vidur_python),
                    tp_list=shlex.quote(",".join(map(str, rollout_config.tp_list))),
                    num_replicas=max(1, rollout_config.n_instances),
                    tensor_parallel_size=max(1, max(rollout_config.tp_list or [1])),
                )
            else:
                formatted = self._default_vidur_command(
                    trace_csv=trace_csv,
                    output_dir=output_dir,
                    rollout_config=rollout_config,
                )

            proc = _run_command(
                formatted,
                cwd=self.vidur_path,
                timeout_s=self.timeout_s,
                extra_env={"PYTHONPATH": str(self.vidur_path)},
            )
            if proc.returncode != 0:
                raise SimulatorBackendError(
                    "Vidur rollout simulator failed "
                    f"(code={proc.returncode}): {proc.stderr[-1000:]}"
                )
            result = self._read_result(output_json, output_dir, proc.stdout)
            seconds = _parse_seconds(
                result,
                ("t_rollout", "makespan", "makespan_s", "runtime", "runtime_s"),
            )
            return SimulatorEstimate(
                seconds=seconds,
                details={
                    "backend": "vidur",
                    "command": formatted,
                    "trace_csv": str(trace_csv),
                    "output_dir": str(output_dir),
                    "raw": result,
                },
            )

    def _cache_dir(self) -> str:
        path = Path(self.planner_config.simulator_cache_dir)
        if not path.is_absolute():
            path = _repo_root() / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _write_trace(self, path: Path, requests: list[RequestInfo]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["arrived_at", "num_prefill_tokens", "num_decode_tokens"],
            )
            writer.writeheader()
            for idx, req in enumerate(requests):
                writer.writerow(
                    {
                        "arrived_at": float(idx) / max(float(self.planner_config.vidur_qps), 1e-9),
                        "num_prefill_tokens": max(1, int(req.prompt_length)),
                        "num_decode_tokens": max(1, int(req.gen_length)),
                    }
                )

    def _default_vidur_command(
        self,
        trace_csv: Path,
        output_dir: Path,
        rollout_config: RolloutClusterConfig,
    ) -> str:
        tp = max(1, max(rollout_config.tp_list or [1]))
        num_replicas = max(1, rollout_config.n_gpus // tp)
        max_tokens = max(2, int(self.rl_config.max_seq_length or self.rl_config.max_new_tokens))
        parts = [
            shlex.quote(self.planner_config.vidur_python),
            "-m",
            "vidur.main",
            "--replica_config_device",
            shlex.quote(self.planner_config.vidur_device),
            "--replica_config_model_name",
            shlex.quote(self.planner_config.vidur_model_name),
            "--cluster_config_num_replicas",
            str(num_replicas),
            "--replica_config_tensor_parallel_size",
            str(tp),
            "--replica_config_num_pipeline_stages",
            "1",
            "--request_generator_config_type",
            "synthetic",
            "--synthetic_request_generator_config_num_requests",
            str(max(1, len(rollout_config.tp_list) * 1024)),
            "--length_generator_config_type",
            "trace",
            "--trace_request_length_generator_config_trace_file",
            shlex.quote(str(trace_csv)),
            "--trace_request_length_generator_config_max_tokens",
            str(max_tokens),
            "--interval_generator_config_type",
            "static",
            "--replica_scheduler_config_type",
            shlex.quote(self.planner_config.vidur_scheduler),
            "--base_replica_scheduler_config_batch_size_cap",
            str(self.planner_config.vidur_batch_size_cap),
            "--sarathi_scheduler_config_chunk_size",
            str(self.planner_config.vidur_chunk_size),
            "--metrics_config_output_dir",
            shlex.quote(str(output_dir)),
            "--metrics_config_enable_chrome_trace",
            "False",
            "--metrics_config_store_plots",
            "False",
            "--metrics_config_store_operation_metrics",
            "False",
            "--metrics_config_store_token_completion_metrics",
            "False",
        ]
        if float(self.planner_config.vidur_time_limit_s) > 0:
            parts.extend(["--time_limit", str(self.planner_config.vidur_time_limit_s)])
        return " ".join(parts)

    def _read_result(self, output_json: Path, output_dir: Path, stdout: str) -> dict[str, Any]:
        if output_json.exists():
            return json.loads(output_json.read_text(encoding="utf-8"))
        marker = "VIDUR_RESULT_JSON="
        for line in stdout.splitlines():
            if line.startswith(marker):
                return json.loads(line[len(marker) :])

        request_metrics = list(output_dir.glob("**/request_metrics.csv"))
        if request_metrics:
            return self._parse_request_metrics(request_metrics[-1])
        batch_metrics = list(output_dir.glob("**/batch_metrics.csv"))
        if batch_metrics:
            return self._parse_batch_metrics(batch_metrics[-1])
        raise SimulatorBackendError(
            "Vidur simulator did not expose output_json, VIDUR_RESULT_JSON, "
            "request_metrics.csv, or batch_metrics.csv"
        )

    def _parse_request_metrics(self, path: Path) -> dict[str, Any]:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SimulatorBackendError(f"Empty Vidur request metrics: {path}")
        candidate_cols = [
            "request_e2e_time",
            "request_execution_plus_preemption_time",
            "request_execution_time",
            "request_model_execution_time",
        ]
        values = []
        for row in rows:
            for col in candidate_cols:
                if col in row and row[col] not in ("", None):
                    values.append(float(row[col]))
                    break
        if not values:
            raise SimulatorBackendError(
                f"No usable request time column in Vidur metrics: {path}"
            )
        return {
            "makespan": max(values),
            "mean_request_time": float(np.mean(values)),
            "num_requests": len(values),
            "metrics_file": str(path),
        }

    def _parse_batch_metrics(self, path: Path) -> dict[str, Any]:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SimulatorBackendError(f"Empty Vidur batch metrics: {path}")
        values = []
        for row in rows:
            for key, value in row.items():
                if "batch_execution_time" in key and value not in ("", None):
                    values.append(float(value))
        if not values:
            raise SimulatorBackendError(
                f"No batch_execution_time column in Vidur metrics: {path}"
            )
        return {
            "makespan": sum(values),
            "mean_batch_time": float(np.mean(values)),
            "num_batches": len(values),
            "metrics_file": str(path),
        }


class HybridSimulatorCostModel:
    """CostModel-compatible facade with optional Sailor/Vidur backends."""

    def __init__(self, config: Any):
        self.config = config
        self.planner_config = config.global_resource_planner
        self.analytic = CostModel(
            hardware=config.hardware,
            model_arch=config.model_arch,
            profiling=config.profiling,
            max_seq_length=config.max_seq_length,
            recompute_logprobs=(
                bool(config.recompute_logprobs)
                and bool(getattr(self.planner_config, "memory_budget_check_enabled", True))
            ),
            recompute_logits_dtype_bytes=int(
                getattr(self.planner_config, "memory_budget_logits_dtype_bytes", 4)
            ),
            recompute_workspace_factor=float(
                getattr(self.planner_config, "memory_budget_workspace_factor", 1.5)
            ),
            memory_safety_margin_bytes=float(
                getattr(self.planner_config, "memory_budget_safety_margin_bytes", 0.0)
            ),
        )
        self.train_backend = (self.planner_config.train_backend or "analytic").lower()
        self.rollout_backend = (
            self.planner_config.rollout_backend or "analytic"
        ).lower()
        self.allow_fallback = bool(self.planner_config.simulator_allow_fallback)
        self.sailor = SailorTrainingSimulatorAdapter(self.planner_config, config)
        self.vidur = VidurRolloutSimulatorAdapter(self.planner_config, config)

    def check_train_oom(
        self,
        config: TrainParallelConfig,
        B_global: int,
        L: int,
    ) -> bool:
        return self.analytic.check_train_oom(config, B_global, L)

    def evaluate_training(
        self,
        config: TrainParallelConfig,
        B_global: int,
        L: int,
    ) -> tuple[float, dict]:
        if self.train_backend == "analytic":
            seconds, details = self.analytic.evaluate_training(config, B_global, L)
            details = dict(details)
            details["backend"] = "analytic"
            return seconds, details
        if self.train_backend != "sailor":
            raise ValueError(f"Unknown train simulator backend: {self.train_backend}")
        try:
            estimate = self.sailor.evaluate_training(config, B_global, L)
            if estimate.is_oom:
                return float("inf"), estimate.details
            return estimate.seconds, estimate.details
        except Exception as exc:
            if not self.allow_fallback:
                raise
            seconds, details = self.analytic.evaluate_training(config, B_global, L)
            details = dict(details)
            details.update(
                {
                    "backend": "analytic",
                    "requested_backend": "sailor",
                    "fallback_reason": str(exc),
                }
            )
            return seconds, details

    def evaluate_rollout(
        self,
        cluster: RolloutClusterConfig,
        requests: list[RequestInfo],
    ) -> tuple[float, dict]:
        if self.rollout_backend == "analytic":
            seconds, details = self.analytic.evaluate_rollout(cluster, requests)
            details = dict(details)
            details["backend"] = "analytic"
            return seconds, details
        if self.rollout_backend != "vidur":
            raise ValueError(
                f"Unknown rollout simulator backend: {self.rollout_backend}"
            )
        try:
            estimate = self.vidur.evaluate_rollout(cluster, requests)
            if estimate.is_oom:
                return float("inf"), estimate.details
            return estimate.seconds, estimate.details
        except Exception as exc:
            if not self.allow_fallback:
                raise
            seconds, details = self.analytic.evaluate_rollout(cluster, requests)
            details = dict(details)
            details.update(
                {
                    "backend": "analytic",
                    "requested_backend": "vidur",
                    "fallback_reason": str(exc),
                }
            )
            return seconds, details

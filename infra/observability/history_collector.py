"""Support code for History collector."""

from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class SequenceLengthStats:
    """Sequence length stats implementation."""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: int = 0
    max: int = 0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @classmethod
    def from_lengths(cls, lengths: list[int]) -> "SequenceLengthStats":
        """From lengths."""
        if not lengths:
            return cls()
        arr = np.array(lengths, dtype=float)
        return cls(
            count=len(lengths),
            mean=round(float(arr.mean()), 2),
            std=round(float(arr.std()), 2),
            min=int(arr.min()),
            max=int(arr.max()),
            p25=round(float(np.percentile(arr, 25)), 1),
            p50=round(float(np.percentile(arr, 50)), 1),
            p75=round(float(np.percentile(arr, 75)), 1),
            p90=round(float(np.percentile(arr, 90)), 1),
            p95=round(float(np.percentile(arr, 95)), 1),
            p99=round(float(np.percentile(arr, 99)), 1),
        )


@dataclass
class TimingRecord:
    """Timing record implementation."""
    rollout_time: float = 0.0
    train_time: float = 0.0
    weight_sync_time: float = 0.0
    advantage_time: float = 0.0
    recompute_logprob_time: float = 0.0
    step_total_time: float = 0.0

    @property
    def overhead_time(self) -> float:
        """Overhead time."""
        accounted = (self.rollout_time + self.train_time +
                     self.weight_sync_time + self.advantage_time +
                     self.recompute_logprob_time)
        return max(0.0, self.step_total_time - accounted)


@dataclass
class ResourceConfig:
    """Resource config implementation."""

    n_total_gpus: int = 0
    train_gpus: int = 0
    rollout_gpus: int = 0
    train_backend: str = "fsdp"

    train_tp: int = 1
    train_pp: int = 1
    train_dp: int = 1

    rollout_tp_list: list[int] = field(default_factory=list)
    vllm_tp_size: int = 1
    vllm_num_instances: int = 1
    # Batch
    batch_size: int = 32
    micro_batch_size: int = 4
    B_global: int = 32

    max_new_tokens: int = 1024
    max_seq_length: int = 2048
    n_samples: int = 4
    temperature: float = 1.0

    model_path: str = ""
    model_params: float = 0.0


@dataclass
class CostModelComparison:
    """Cost model comparison implementation."""
    # Rollout
    predicted_rollout_time: float = 0.0
    actual_rollout_time: float = 0.0
    rollout_ratio: float = 0.0      # predicted / actual
    rollout_abs_error: float = 0.0   # |predicted - actual|
    # Training
    predicted_train_time: float = 0.0
    actual_train_time: float = 0.0
    train_ratio: float = 0.0
    train_abs_error: float = 0.0

    eta_comp: float = 0.5
    eta_mem: float = 0.5
    gamma_overlap: float = 0.3


@dataclass
class PipelineStats:
    """Pipeline stats implementation."""
    running_rollouts: int = 0
    accepted_rollouts: int = 0
    rejected_rollouts: int = 0
    model_version: int = 0


@dataclass
class TrainingMetrics:
    """Training metrics implementation."""
    loss: float = 0.0
    pg_loss: float = 0.0
    kl: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    n_trajectories: int = 0
    grpo_num_groups: int = 0
    grpo_mean_group_size: float = 0.0
    grpo_singleton_groups: int = 0
    grpo_zero_variance_groups: int = 0


@dataclass
class StepRecord:
    """Step record implementation."""

    step: int = 0
    timestamp: float = 0.0
    timestamp_iso: str = ""
    wall_clock_elapsed: float = 0.0


    sequence_stats: dict = field(default_factory=dict)
    timing: TimingRecord = field(default_factory=TimingRecord)
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)
    training_metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    pipeline_stats: PipelineStats = field(default_factory=PipelineStats)
    control_plane: dict[str, Any] = field(default_factory=dict)


    cost_model_comparison: CostModelComparison | None = None



    sequences: list[dict] = field(default_factory=list)


    raw_prompt_lengths: list[int] = field(default_factory=list)
    raw_gen_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the object to a dictionary."""
        d = {}
        d["step"] = self.step
        d["timestamp"] = self.timestamp
        d["timestamp_iso"] = self.timestamp_iso
        d["wall_clock_elapsed"] = round(self.wall_clock_elapsed, 3)


        d["sequence_stats"] = {}
        for key, stats in self.sequence_stats.items():
            if isinstance(stats, SequenceLengthStats):
                d["sequence_stats"][key] = asdict(stats)
            else:
                d["sequence_stats"][key] = stats


        d["sequences"] = self.sequences


        timing_d = asdict(self.timing)
        timing_d["overhead_time"] = round(self.timing.overhead_time, 6)
        d["timing"] = {k: round(v, 6) for k, v in timing_d.items()}


        d["resource_config"] = asdict(self.resource_config)


        d["training_metrics"] = asdict(self.training_metrics)


        d["pipeline_stats"] = asdict(self.pipeline_stats)

        if self.control_plane:
            d["control_plane"] = self.control_plane

        if self.cost_model_comparison is not None:
            d["cost_model_comparison"] = asdict(self.cost_model_comparison)


        if self.raw_prompt_lengths:
            d["raw_prompt_lengths"] = self.raw_prompt_lengths
        if self.raw_gen_lengths:
            d["raw_gen_lengths"] = self.raw_gen_lengths

        return d


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class HistoryDataCollector:
    """History data collector implementation."""

    def __init__(
        self,
        output_dir: str = "./logs/history",
        save_raw_lengths: bool = False,
        flush_interval: int = 10,
        experiment_name: str = "",
    ):
        self.output_dir = output_dir
        self.save_raw_lengths = save_raw_lengths
        self.flush_interval = flush_interval
        self.experiment_name = experiment_name

        self._records: list[StepRecord] = []
        self._start_time: float = 0.0
        self._file_handle = None
        self._lock = threading.Lock()
        self._record_count = 0
        self._initialized = False

    def initialize(self):
        """Initialize."""
        os.makedirs(self.output_dir, exist_ok=True)


        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.experiment_name}_" if self.experiment_name else ""
        self._jsonl_path = os.path.join(
            self.output_dir, f"{prefix}history_{ts}.jsonl"
        )
        self._summary_path = os.path.join(
            self.output_dir, f"{prefix}summary_{ts}.json"
        )

        self._file_handle = open(self._jsonl_path, "w", encoding="utf-8")
        self._start_time = time.time()
        self._initialized = True

        print(f"[HistoryCollector] Initialized; data file: {self._jsonl_path}")



    def record_step(
        self,
        step: int,

        sequence_pairs: list[tuple[int, int]] | None = None,  # [(input_len, output_len), ...]
        sequence_records: list[dict[str, Any]] | None = None,

        prompt_lengths: list[int] | None = None,
        gen_lengths: list[int] | None = None,

        rollout_time: float = 0.0,
        train_time: float = 0.0,
        weight_sync_time: float = 0.0,
        advantage_time: float = 0.0,
        recompute_logprob_time: float = 0.0,
        step_total_time: float = 0.0,

        training_stats: dict[str, Any] | None = None,

        pipeline_running: int = 0,
        pipeline_accepted: int = 0,
        pipeline_rejected: int = 0,
        model_version: int = 0,

        predicted_rollout_time: float = 0.0,
        predicted_train_time: float = 0.0,
        eta_comp: float = 0.5,
        eta_mem: float = 0.5,
        gamma_overlap: float = 0.3,

        resource_config: ResourceConfig | None = None,
        control_plane: dict[str, Any] | None = None,
    ) -> StepRecord:
        """Record step."""
        if not self._initialized:
            self.initialize()

        now = time.time()


        if sequence_records is not None and len(sequence_records) > 0:
            sequences = []
            for seq in sequence_records:
                item = dict(seq)
                item["input_len"] = int(item.get("input_len", 0))
                item["output_len"] = int(item.get("output_len", 0))
                if "total_output_tokens" in item:
                    item["total_output_tokens"] = int(item["total_output_tokens"])
                sequences.append(item)
            prompt_lengths = [seq["input_len"] for seq in sequences]
            gen_lengths = [seq["output_len"] for seq in sequences]
        elif sequence_pairs is not None and len(sequence_pairs) > 0:

            prompt_lengths = [int(p) for p, _ in sequence_pairs]
            gen_lengths = [int(g) for _, g in sequence_pairs]
            sequences = [
                {"input_len": int(p), "output_len": int(g)}
                for p, g in sequence_pairs
            ]
        else:
            prompt_lengths = [int(x) for x in prompt_lengths] if prompt_lengths else []
            gen_lengths = [int(x) for x in gen_lengths] if gen_lengths else []

            if prompt_lengths and gen_lengths and len(prompt_lengths) == len(gen_lengths):
                sequences = [
                    {"input_len": p, "output_len": g}
                    for p, g in zip(prompt_lengths, gen_lengths)
                ]
            else:
                sequences = []

        total_lengths = [p + g for p, g in zip(prompt_lengths, gen_lengths)] if prompt_lengths and gen_lengths else []


        seq_stats = {
            "prompt": SequenceLengthStats.from_lengths(prompt_lengths),
            "generation": SequenceLengthStats.from_lengths(gen_lengths),
            "total": SequenceLengthStats.from_lengths(total_lengths),
        }


        timing = TimingRecord(
            rollout_time=rollout_time,
            train_time=train_time,
            weight_sync_time=weight_sync_time,
            advantage_time=advantage_time,
            recompute_logprob_time=recompute_logprob_time,
            step_total_time=step_total_time,
        )


        stats = training_stats or {}
        rewards = stats.get("rewards", [])
        if isinstance(rewards, (list, np.ndarray)) and len(rewards) > 0:
            r_arr = np.array(rewards)
            r_mean, r_std, r_min, r_max = float(r_arr.mean()), float(r_arr.std()), float(r_arr.min()), float(r_arr.max())
        else:
            r_mean = stats.get("reward_mean", 0.0)
            r_std = stats.get("reward_std", 0.0)
            r_min = stats.get("reward_min", 0.0)
            r_max = stats.get("reward_max", 0.0)

        metrics = TrainingMetrics(
            loss=float(stats.get("loss", 0.0)),
            pg_loss=float(stats.get("pg_loss", 0.0)),
            kl=float(stats.get("kl", 0.0)),
            reward_mean=float(r_mean),
            reward_std=float(r_std),
            reward_min=float(r_min),
            reward_max=float(r_max),
            n_trajectories=int(stats.get("n_trajectories", len(prompt_lengths))),
            grpo_num_groups=int(stats.get("grpo_num_groups", 0)),
            grpo_mean_group_size=float(stats.get("grpo_mean_group_size", 0.0)),
            grpo_singleton_groups=int(stats.get("grpo_singleton_groups", 0)),
            grpo_zero_variance_groups=int(stats.get("grpo_zero_variance_groups", 0)),
        )


        pipe_stats = PipelineStats(
            running_rollouts=pipeline_running,
            accepted_rollouts=pipeline_accepted,
            rejected_rollouts=pipeline_rejected,
            model_version=model_version,
        )


        cost_cmp = None
        actual_rollout = rollout_time
        actual_train = train_time
        if predicted_rollout_time > 0 or predicted_train_time > 0:
            rollout_ratio = (predicted_rollout_time / actual_rollout) if actual_rollout > 0 else 0.0
            train_ratio = (predicted_train_time / actual_train) if actual_train > 0 else 0.0
            cost_cmp = CostModelComparison(
                predicted_rollout_time=predicted_rollout_time,
                actual_rollout_time=actual_rollout,
                rollout_ratio=round(rollout_ratio, 4),
                rollout_abs_error=round(abs(predicted_rollout_time - actual_rollout), 6),
                predicted_train_time=predicted_train_time,
                actual_train_time=actual_train,
                train_ratio=round(train_ratio, 4),
                train_abs_error=round(abs(predicted_train_time - actual_train), 6),
                eta_comp=eta_comp,
                eta_mem=eta_mem,
                gamma_overlap=gamma_overlap,
            )


        if resource_config is not None:
            self._last_resource_config = resource_config
        rc = getattr(self, "_last_resource_config", ResourceConfig())


        record = StepRecord(
            step=step,
            timestamp=now,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            wall_clock_elapsed=now - self._start_time,
            sequence_stats=seq_stats,
            timing=timing,
            resource_config=rc,
            training_metrics=metrics,
            pipeline_stats=pipe_stats,
            control_plane=dict(control_plane or {}),
            cost_model_comparison=cost_cmp,
            sequences=sequences,
            raw_prompt_lengths=prompt_lengths if self.save_raw_lengths else [],
            raw_gen_lengths=gen_lengths if self.save_raw_lengths else [],
        )


        with self._lock:
            self._records.append(record)
            self._record_count += 1
            self._write_record(record)

            if self._record_count % self.flush_interval == 0:
                self._flush()

        return record



    @staticmethod
    def snapshot_resource_config(config) -> ResourceConfig:
        """Snapshot resource config."""
        return ResourceConfig(
            n_total_gpus=getattr(config, "n_total_gpus", 0) or (
                getattr(config, "train_gpus", 0) + getattr(config, "rollout_gpus", 0)
            ),
            train_gpus=getattr(config, "train_gpus", 0),
            rollout_gpus=getattr(config, "rollout_gpus", 0),
            train_backend=getattr(config, "train_backend", "fsdp"),
            train_tp=getattr(
                config,
                "train_tp_size",
                getattr(config, "tp_size", 1),
            ),
            train_pp=getattr(config, "train_pp_size", 1),
            train_dp=getattr(config, "train_dp_size", 1),
            rollout_tp_list=(
                list(getattr(
                    getattr(config, "heterogeneous_rollout", None),
                    "tp_list",
                    [],
                ))
                if getattr(
                    getattr(config, "heterogeneous_rollout", None),
                    "enabled",
                    False,
                )
                else []
            ),
            vllm_tp_size=getattr(config, "vllm_tp_size", 1),
            vllm_num_instances=(
                getattr(
                    getattr(config, "heterogeneous_rollout", None),
                    "n_instances",
                    0,
                )
                if getattr(
                    getattr(config, "heterogeneous_rollout", None),
                    "enabled",
                    False,
                )
                else getattr(config, "vllm_num_instances", 1)
            ),
            batch_size=getattr(config, "batch_size", 32),
            micro_batch_size=getattr(config, "micro_batch_size", 4),
            B_global=getattr(config, "batch_size", 32),
            max_new_tokens=getattr(config, "max_new_tokens", 1024),
            max_seq_length=getattr(config, "max_seq_length", 2048),
            n_samples=getattr(config, "n_samples", 4),
            temperature=getattr(config, "temperature", 1.0),
            model_path=getattr(config, "model_path", ""),
            model_params=getattr(
                getattr(config, "model_arch", None), "num_params", 0.0
            ),
        )



    def get_summary(self) -> dict:
        """Get summary."""
        if not self._records:
            return {"total_steps": 0}

        records = self._records
        n = len(records)


        gen_means = [r.sequence_stats.get("generation", SequenceLengthStats()).mean
                     for r in records if "generation" in r.sequence_stats]
        prompt_means = [r.sequence_stats.get("prompt", SequenceLengthStats()).mean
                        for r in records if "prompt" in r.sequence_stats]


        rollout_times = [r.timing.rollout_time for r in records]
        train_times = [r.timing.train_time for r in records]
        step_times = [r.timing.step_total_time for r in records]


        cost_ratios_rollout = []
        cost_ratios_train = []
        for r in records:
            if r.cost_model_comparison:
                cmp = r.cost_model_comparison
                if cmp.actual_rollout_time > 0 and cmp.predicted_rollout_time > 0:
                    cost_ratios_rollout.append(cmp.rollout_ratio)
                if cmp.actual_train_time > 0 and cmp.predicted_train_time > 0:
                    cost_ratios_train.append(cmp.train_ratio)

        summary = {
            "total_steps": n,
            "total_wall_clock": round(records[-1].wall_clock_elapsed, 2),
            "data_file": self._jsonl_path,
            "sequence_length_trend": {
                "gen_mean_first10": round(float(np.mean(gen_means[:10])), 1) if len(gen_means) >= 10 else None,
                "gen_mean_last10": round(float(np.mean(gen_means[-10:])), 1) if len(gen_means) >= 10 else None,
                "gen_mean_overall": round(float(np.mean(gen_means)), 1) if gen_means else 0,
                "prompt_mean_overall": round(float(np.mean(prompt_means)), 1) if prompt_means else 0,
            },
            "timing_summary": {
                "rollout_mean": round(float(np.mean(rollout_times)), 4) if rollout_times else 0,
                "rollout_std": round(float(np.std(rollout_times)), 4) if rollout_times else 0,
                "train_mean": round(float(np.mean(train_times)), 4) if train_times else 0,
                "train_std": round(float(np.std(train_times)), 4) if train_times else 0,
                "step_mean": round(float(np.mean(step_times)), 4) if step_times else 0,
            },
            "resource_config": asdict(records[-1].resource_config),
        }


        if cost_ratios_rollout:
            arr = np.array(cost_ratios_rollout)
            mape = float(np.mean(np.abs(arr - 1.0))) * 100  # MAPE (%)
            summary["cost_model_accuracy"] = {
                "rollout_mape_pct": round(mape, 2),
                "rollout_ratio_mean": round(float(arr.mean()), 4),
                "rollout_ratio_std": round(float(arr.std()), 4),
                "n_comparisons": len(cost_ratios_rollout),
            }
        if cost_ratios_train:
            arr = np.array(cost_ratios_train)
            mape = float(np.mean(np.abs(arr - 1.0))) * 100
            summary.setdefault("cost_model_accuracy", {}).update({
                "train_mape_pct": round(mape, 2),
                "train_ratio_mean": round(float(arr.mean()), 4),
                "train_ratio_std": round(float(arr.std()), 4),
            })

        return summary

    def save_summary(self):
        """Save summary."""
        summary = self.get_summary()
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[HistoryCollector] Summary saved: {self._summary_path}")

    def get_records(self) -> list[StepRecord]:
        """Get records."""
        return list(self._records)

    def get_sequence_trend(self) -> dict[str, list[float]]:
        """Get sequence trend."""
        trend = {"steps": [], "gen_mean": [], "gen_p95": [],
                 "prompt_mean": [], "total_mean": []}
        for r in self._records:
            trend["steps"].append(r.step)
            gen_s = r.sequence_stats.get("generation", SequenceLengthStats())
            prompt_s = r.sequence_stats.get("prompt", SequenceLengthStats())
            total_s = r.sequence_stats.get("total", SequenceLengthStats())
            trend["gen_mean"].append(gen_s.mean)
            trend["gen_p95"].append(gen_s.p95)
            trend["prompt_mean"].append(prompt_s.mean)
            trend["total_mean"].append(total_s.mean)
        return trend



    def _write_record(self, record: StepRecord):
        """Write record."""
        if self._file_handle is None:
            return
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        self._file_handle.write(line + "\n")

    def _flush(self):
        """Flush."""
        if self._file_handle is not None:
            self._file_handle.flush()

    def finalize(self):
        """Finalize."""
        with self._lock:
            self._flush()
            self.save_summary()
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
        print(f"[HistoryCollector] Complete; recorded {self._record_count} steps")

    def __del__(self):
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def load_history(jsonl_path: str) -> list[dict]:
    """Load history."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_cost_model_accuracy(records: list[dict]) -> dict:
    """Compute cost model accuracy."""
    rollout_errors = []
    train_errors = []
    steps = []

    for r in records:
        cmp = r.get("cost_model_comparison")
        if not cmp:
            continue
        steps.append(r["step"])

        pred_r = cmp.get("predicted_rollout_time", 0)
        act_r = cmp.get("actual_rollout_time", 0)
        if pred_r > 0 and act_r > 0:
            rollout_errors.append(abs(pred_r - act_r) / act_r)

        pred_t = cmp.get("predicted_train_time", 0)
        act_t = cmp.get("actual_train_time", 0)
        if pred_t > 0 and act_t > 0:
            train_errors.append(abs(pred_t - act_t) / act_t)

    result = {"n_records": len(records), "n_with_prediction": len(steps)}

    if rollout_errors:
        arr = np.array(rollout_errors)
        result["rollout"] = {
            "mape_pct": round(float(arr.mean()) * 100, 2),
            "median_ape_pct": round(float(np.median(arr)) * 100, 2),
            "max_ape_pct": round(float(arr.max()) * 100, 2),
        }
    if train_errors:
        arr = np.array(train_errors)
        result["training"] = {
            "mape_pct": round(float(arr.mean()) * 100, 2),
            "median_ape_pct": round(float(np.median(arr)) * 100, 2),
            "max_ape_pct": round(float(arr.max()) * 100, 2),
        }

    return result

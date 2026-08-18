"""Support code for Test history collector."""

import sys
import os
import json
import tempfile
import shutil
import time
import numpy as np


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import types
_pkg = types.ModuleType("RL_Framework")
_pkg.__path__ = [_project_root]
_pkg.__package__ = "RL_Framework"
sys.modules["RL_Framework"] = _pkg

from RL_Framework.infra.observability.history_collector import (
    HistoryDataCollector,
    SequenceLengthStats,
    TimingRecord,
    ResourceConfig,
    CostModelComparison,
    StepRecord,
    load_history,
    compute_cost_model_accuracy,
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_sequence_length_stats():
    separator("Test 1: SequenceLengthStats statistics")


    stats = SequenceLengthStats.from_lengths([])
    assert stats.count == 0
    print("  Empty list: OK")


    lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    stats = SequenceLengthStats.from_lengths(lengths)
    assert stats.count == 10
    assert stats.mean == 550.0
    assert stats.min == 100
    assert stats.max == 1000
    assert abs(stats.p50 - 550.0) < 1.0
    print(f"  Normal list: count={stats.count}, mean={stats.mean}, "
          f"p50={stats.p50}, p95={stats.p95}")


    stats = SequenceLengthStats.from_lengths([42])
    assert stats.count == 1
    assert stats.mean == 42.0
    assert stats.min == 42
    assert stats.max == 42
    print("  Single element: OK")

    print("\n  [PASS] SequenceLengthStats tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_timing_record():
    separator("Test 2: TimingRecord")

    t = TimingRecord(
        rollout_time=5.0,
        train_time=3.0,
        weight_sync_time=0.5,
        advantage_time=0.1,
        recompute_logprob_time=0.2,
        step_total_time=10.0,
    )

    # overhead = 10.0 - (5.0+3.0+0.5+0.1+0.2) = 1.2
    assert abs(t.overhead_time - 1.2) < 0.001
    print(f"  overhead_time={t.overhead_time:.3f}s (expected 1.2s)")

    print("\n  [PASS] TimingRecord tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_basic_recording():
    separator("Test 3: basic recording and JSONL output")

    tmpdir = tempfile.mkdtemp(prefix="history_test_")
    try:
        collector = HistoryDataCollector(
            output_dir=tmpdir,
            save_raw_lengths=True,
            flush_interval=1,
        )
        collector.initialize()


        np.random.seed(42)
        for step in range(3):
            prompt_lengths = list(np.random.randint(100, 500, size=20))
            gen_lengths = list(np.random.randint(50, 1000, size=20))

            record = collector.record_step(
                step=step,
                prompt_lengths=prompt_lengths,
                gen_lengths=gen_lengths,
                rollout_time=5.0 + step * 0.5,
                train_time=3.0 + step * 0.2,
                weight_sync_time=0.3,
                step_total_time=9.0 + step,
                training_stats={"loss": 0.5 - step * 0.1, "kl": 0.01},
                model_version=step + 1,
                resource_config=ResourceConfig(
                    train_gpus=4, rollout_gpus=4, batch_size=32,
                    model_path="test-model",
                ),
            )

            assert record.step == step
            assert record.sequence_stats["prompt"].count == 20
            assert record.sequence_stats["generation"].count == 20
            assert record.sequence_stats["total"].count == 20
            assert len(record.raw_prompt_lengths) == 20

            assert len(record.sequences) == 20
            assert "input_len" in record.sequences[0]
            assert "output_len" in record.sequences[0]

        collector.finalize()


        jsonl_files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(jsonl_files) == 1, f"Expected 1 jsonl file, got {len(jsonl_files)}"

        records = load_history(os.path.join(tmpdir, jsonl_files[0]))
        assert len(records) == 3
        assert records[0]["step"] == 0
        assert records[2]["step"] == 2
        assert "sequence_stats" in records[0]
        assert "timing" in records[0]
        assert "resource_config" in records[0]
        assert records[0]["resource_config"]["train_gpus"] == 4


        assert len(records[0]["raw_prompt_lengths"]) == 20
        assert len(records[0]["raw_gen_lengths"]) == 20


        assert "sequences" in records[0]
        assert len(records[0]["sequences"]) == 20
        seq0 = records[0]["sequences"][0]
        assert "input_len" in seq0 and "output_len" in seq0
        assert seq0["input_len"] > 0 and seq0["output_len"] > 0

        print(f"  JSONL file: {jsonl_files[0]}")
        print(f"  Records: {len(records)}")
        print(f"  Step 0 gen mean: {records[0]['sequence_stats']['generation']['mean']:.1f}")
        print(f"  Step 0 sequence count: {len(records[0]['sequences'])}")
        print(f"  Step 0 seq[0]: {records[0]['sequences'][0]}")


        summary_files = [f for f in os.listdir(tmpdir) if f.startswith("summary")]
        assert len(summary_files) == 1
        with open(os.path.join(tmpdir, summary_files[0]), "r") as f:
            summary = json.load(f)
        assert summary["total_steps"] == 3
        print(f"  Summary total_steps: {summary['total_steps']}")

        print("\n  [PASS] Basic recording and JSONL output tests passed")

    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_cost_model_comparison():
    separator("Test 4: cost-model comparison records")

    tmpdir = tempfile.mkdtemp(prefix="history_cost_")
    try:
        collector = HistoryDataCollector(
            output_dir=tmpdir,
            flush_interval=1,
        )
        collector.initialize()


        np.random.seed(123)
        for step in range(10):
            actual_rollout = 5.0 + np.random.normal(0, 0.5)
            actual_train = 3.0 + np.random.normal(0, 0.3)

            pred_rollout = actual_rollout * (1.0 + np.random.normal(0, 0.2))
            pred_train = actual_train * (1.0 + np.random.normal(0, 0.15))

            collector.record_step(
                step=step,
                prompt_lengths=list(np.random.randint(100, 400, size=15)),
                gen_lengths=list(np.random.randint(50, 800, size=15)),
                rollout_time=actual_rollout,
                train_time=actual_train,
                step_total_time=actual_rollout + actual_train + 1.0,
                predicted_rollout_time=pred_rollout,
                predicted_train_time=pred_train,
                eta_comp=0.45,
                eta_mem=0.50,
                model_version=step + 1,
            )

        collector.finalize()


        jsonl_files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        records = load_history(os.path.join(tmpdir, jsonl_files[0]))


        for r in records:
            assert "cost_model_comparison" in r
            cmp = r["cost_model_comparison"]
            assert cmp["eta_comp"] == 0.45
            assert cmp["predicted_rollout_time"] > 0
            assert cmp["actual_rollout_time"] > 0


        accuracy = compute_cost_model_accuracy(records)
        assert "rollout" in accuracy
        assert "training" in accuracy
        print(f"  Rollout MAPE: {accuracy['rollout']['mape_pct']:.2f}%")
        print(f"  Training MAPE: {accuracy['training']['mape_pct']:.2f}%")
        print(f"  Comparison records: {accuracy['n_with_prediction']}")


        summary_files = [f for f in os.listdir(tmpdir) if f.startswith("summary")]
        with open(os.path.join(tmpdir, summary_files[0]), "r") as f:
            summary = json.load(f)
        assert "cost_model_accuracy" in summary
        print(f"  Summary MAPE: {summary['cost_model_accuracy']['rollout_mape_pct']:.2f}%")

        print("\n  [PASS] Cost-model comparison record tests passed")

    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_sequence_trend():
    separator("Test 5: sequence-length trend tracking (simulated capability emergence)")

    tmpdir = tempfile.mkdtemp(prefix="history_trend_")
    try:
        collector = HistoryDataCollector(
            output_dir=tmpdir,
            flush_interval=5,
        )
        collector.initialize()

        np.random.seed(0)


        for step in range(10):
            gen_lengths = list(np.random.normal(300, 50, size=20).clip(min=16).astype(int))
            collector.record_step(
                step=step,
                prompt_lengths=list(np.random.randint(100, 300, size=20)),
                gen_lengths=gen_lengths,
                rollout_time=5.0,
                train_time=3.0,
                step_total_time=9.0,
            )


        for step in range(10, 20):
            gen_lengths = list(np.random.normal(1500, 300, size=20).clip(min=16).astype(int))
            collector.record_step(
                step=step,
                prompt_lengths=list(np.random.randint(100, 300, size=20)),
                gen_lengths=gen_lengths,
                rollout_time=15.0,
                train_time=5.0,
                step_total_time=22.0,
            )

        collector.finalize()


        trend = collector.get_sequence_trend()
        assert len(trend["steps"]) == 20
        assert len(trend["gen_mean"]) == 20

        first_half_mean = np.mean(trend["gen_mean"][:10])
        second_half_mean = np.mean(trend["gen_mean"][10:])
        print(f"  Phase 1 gen_mean mean: {first_half_mean:.1f}")
        print(f"  Phase 2 gen_mean mean: {second_half_mean:.1f}")
        print(f"  Growth factor: {second_half_mean / first_half_mean:.2f}x")


        assert second_half_mean > first_half_mean * 3, \
            f"Unexpected sequence-length trend: {second_half_mean:.0f} vs {first_half_mean:.0f}"


        summary = collector.get_summary()
        assert summary["sequence_length_trend"]["gen_mean_first10"] is not None
        assert summary["sequence_length_trend"]["gen_mean_last10"] is not None
        print(f"  Summary first10: {summary['sequence_length_trend']['gen_mean_first10']:.1f}")
        print(f"  Summary last10: {summary['sequence_length_trend']['gen_mean_last10']:.1f}")

        print("\n  [PASS] Sequence-length trend tests passed")

    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_resource_config_snapshot():
    separator("Test 6: resource configuration snapshot")


    class MockConfig:
        train_gpus = 4
        rollout_gpus = 4
        n_total_gpus = 0
        train_backend = "megatron3d"
        tp_size = 2
        train_tp_size = 2
        train_pp_size = 2
        train_dp_size = 1
        vllm_tp_size = 2
        vllm_num_instances = 2
        batch_size = 32
        micro_batch_size = 4
        max_new_tokens = 2048
        max_seq_length = 4096
        n_samples = 8
        temperature = 0.7
        model_path = "/models/Qwen3-32B"

        class model_arch:
            num_params = 32e9

    rc = HistoryDataCollector.snapshot_resource_config(MockConfig())
    assert rc.train_gpus == 4
    assert rc.rollout_gpus == 4
    assert rc.n_total_gpus == 8  # auto: 4+4
    assert rc.train_backend == "megatron3d"
    assert rc.train_tp == 2
    assert rc.train_pp == 2
    assert rc.train_dp == 1
    assert rc.vllm_tp_size == 2
    assert rc.batch_size == 32
    assert rc.model_path == "/models/Qwen3-32B"
    assert rc.model_params == 32e9

    print(f"  train_gpus={rc.train_gpus}, rollout_gpus={rc.rollout_gpus}")
    print(f"  n_total_gpus={rc.n_total_gpus}")
    print(f"  train_backend={rc.train_backend}")
    print(f"  vllm_tp_size={rc.vllm_tp_size}")
    print(f"  model_params={rc.model_params:.1e}")

    print("\n  [PASS] Resource configuration snapshot tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_full_simulation():
    separator("Test 7: full simulated training workflow (50 steps, sequence_pairs interface)")

    tmpdir = tempfile.mkdtemp(prefix="history_sim_")
    try:
        collector = HistoryDataCollector(
            output_dir=tmpdir,
            save_raw_lengths=False,
            flush_interval=10,
            experiment_name="gsm8k_test",
        )
        collector.initialize()

        rc = ResourceConfig(
            n_total_gpus=8,
            train_gpus=4,
            rollout_gpus=4,
            train_tp=2,
            train_pp=1,
            train_dp=2,
            vllm_tp_size=2,
            vllm_num_instances=2,
            batch_size=32,
            micro_batch_size=4,
            max_new_tokens=1024,
            max_seq_length=2048,
            model_path="Qwen/Qwen2.5-3B-Instruct",
            model_params=3e9,
        )

        np.random.seed(42)
        total_steps = 50

        for step in range(total_steps):

            gen_mean = 300 + step * 10
            n_seqs = 32


            pairs = [
                (
                    int(np.random.randint(100, 400)),
                    max(16, int(np.random.normal(gen_mean, gen_mean * 0.2))),
                )
                for _ in range(n_seqs)
            ]


            rollout_time = 5.0 + step * 0.1 + np.random.normal(0, 0.3)
            train_time = 3.0 + np.random.normal(0, 0.2)
            sync_time = 0.3 if step % 1 == 0 else 0.0


            pred_rollout = rollout_time * (1.0 + np.random.normal(0, 0.15))
            pred_train = train_time * (1.0 + np.random.normal(0, 0.10))

            record = collector.record_step(
                step=step,
                sequence_pairs=pairs,
                rollout_time=rollout_time,
                train_time=train_time,
                weight_sync_time=sync_time,
                advantage_time=0.05,
                recompute_logprob_time=0.1,
                step_total_time=rollout_time + train_time + sync_time + 0.2,
                training_stats={
                    "loss": 0.5 * np.exp(-step / 20),
                    "pg_loss": 0.3 * np.exp(-step / 20),
                    "kl": 0.01 * (1 + step * 0.01),
                    "reward_mean": 0.1 * min(step, 30),
                },
                predicted_rollout_time=pred_rollout,
                predicted_train_time=pred_train,
                eta_comp=0.45 + step * 0.001,
                eta_mem=0.50,
                model_version=step + 1,
                resource_config=rc,
            )


            assert len(record.sequences) == n_seqs
            assert record.sequences[0]["input_len"] == pairs[0][0]
            assert record.sequences[0]["output_len"] == pairs[0][1]

        collector.finalize()


        jsonl_files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(jsonl_files) == 1
        assert jsonl_files[0].startswith("gsm8k_test_")

        records = load_history(os.path.join(tmpdir, jsonl_files[0]))
        assert len(records) == total_steps


        for r in records:
            assert "sequences" in r
            assert len(r["sequences"]) == 32
            for seq in r["sequences"]:
                assert "input_len" in seq
                assert "output_len" in seq
                assert seq["input_len"] > 0


        accuracy = compute_cost_model_accuracy(records)
        print(f"  Total steps: {total_steps}")
        print(f"  Rollout MAPE: {accuracy['rollout']['mape_pct']:.2f}%")
        print(f"  Training MAPE: {accuracy['training']['mape_pct']:.2f}%")


        trend = collector.get_sequence_trend()
        first_10_gen = np.mean(trend["gen_mean"][:10])
        last_10_gen = np.mean(trend["gen_mean"][-10:])
        print(f"  Gen mean (first 10): {first_10_gen:.0f}")
        print(f"  Gen mean (last 10): {last_10_gen:.0f}")
        assert last_10_gen > first_10_gen, "Sequence length should increase during training"


        step0 = records[0]
        print(f"  Step 0 sequences[0]: {step0['sequences'][0]}")
        print(f"  Step 0 sequences count: {len(step0['sequences'])}")


        jsonl_size = os.path.getsize(os.path.join(tmpdir, jsonl_files[0]))
        print(f"  JSONL file size: {jsonl_size / 1024:.1f} KB")

        print("\n  [PASS] Full simulated training workflow tests passed")

    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_config_default_disabled():
    separator("Test 8: configuration switch (disabled by default)")


    from RL_Framework.config import AsyncRLConfig


    cfg = AsyncRLConfig(model_path="dummy-model")
    assert cfg.enable_history_collection is False, \
        f"enable_history_collection should default to False; got {cfg.enable_history_collection}"
    print(f"  enable_history_collection default value: {cfg.enable_history_collection}")


    cfg_enabled = AsyncRLConfig(model_path="dummy-model", enable_history_collection=True)
    assert cfg_enabled.enable_history_collection is True
    print(f"  Explicitly enabled: {cfg_enabled.enable_history_collection}")



    collector = None
    if cfg.enable_history_collection:
        collector = HistoryDataCollector()
    assert collector is None, "No collector is created when disabled"
    print("  collector is None when disabled: OK")

    print("\n  [PASS] Configuration-switch tests passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  HistoryDataCollector unit tests")
    print("=" * 60)

    test_sequence_length_stats()
    test_timing_record()
    test_basic_recording()
    test_cost_model_comparison()
    test_sequence_trend()
    test_resource_config_snapshot()
    test_full_simulation()
    test_config_default_disabled()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

import json
import os
from types import SimpleNamespace

import torch

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class FakeTrainEngine:
    train_backend = "megatron_core"

    def __init__(self, snapshot_path=None):
        self.snapshot_path = snapshot_path

    def get_elastic_state_snapshot_path(self, version):
        del version
        return str(self.snapshot_path)


def _trainer(export_only: bool = True) -> AsyncRLTrainer:
    trainer = object.__new__(AsyncRLTrainer)
    trainer.train_engine = FakeTrainEngine()
    trainer.config = SimpleNamespace(rollout_weight_sync_export_only=export_only)
    return trainer


def test_restart_weight_sync_sets_megatron_rollout_export_only(monkeypatch):
    trainer = _trainer()
    monkeypatch.delenv("MEGATRON_ROLLOUT_EXPORT_ONLY", raising=False)

    with trainer._rollout_export_only_context("restart"):
        assert "MEGATRON_ROLLOUT_EXPORT_ONLY" in os.environ
        assert os.environ["MEGATRON_ROLLOUT_EXPORT_ONLY"] == "1"

    assert "MEGATRON_ROLLOUT_EXPORT_ONLY" not in os.environ


def test_restart_weight_sync_restores_existing_export_only_env(monkeypatch):
    trainer = _trainer()
    monkeypatch.setenv("MEGATRON_ROLLOUT_EXPORT_ONLY", "0")

    with trainer._rollout_export_only_context("restart"):
        assert os.environ["MEGATRON_ROLLOUT_EXPORT_ONLY"] == "1"

    assert os.environ["MEGATRON_ROLLOUT_EXPORT_ONLY"] == "0"


def test_non_restart_weight_sync_leaves_export_only_env_unchanged(monkeypatch):
    trainer = _trainer()
    monkeypatch.delenv("MEGATRON_ROLLOUT_EXPORT_ONLY", raising=False)

    with trainer._rollout_export_only_context("none"):
        assert "MEGATRON_ROLLOUT_EXPORT_ONLY" not in os.environ


def test_hybrid_batch_uses_exact_step_snapshot(tmp_path):
    task_dir = tmp_path / "tasks"
    membership_dir = task_dir / "membership"
    membership_dir.mkdir(parents=True)
    snapshot = tmp_path / "state" / "v4" / "rank_0.pt"
    snapshot.parent.mkdir(parents=True)
    torch.save({"version": 4}, snapshot)
    (membership_dir / "hybrid0.json").write_text(
        json.dumps(
            {
                "worker_id": "hybrid0",
                "role": "hybrid_training",
                "membership_epoch": 7,
            }
        ),
        encoding="utf-8",
    )
    trainer = object.__new__(AsyncRLTrainer)
    trainer.train_engine = FakeTrainEngine(snapshot)
    trainer.config = SimpleNamespace(
        global_resource_planner=SimpleNamespace(
            hybrid_worker_task_dir=str(task_dir),
            hybrid_lockstep_gradient_sync=False,
        )
    )
    trainer._pending_hybrid_training_batches = {"hybrid0": [{"sample": 1}]}

    trainer._dispatch_hybrid_training_batches(4)

    task = torch.load(
        task_dir / "hybrid0.step_4.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert task["snapshot_path"] == str(snapshot)
    assert task["state_version"] == 4
    assert task["membership_epoch"] == 7


def test_hybrid_batch_rejects_missing_step_snapshot(tmp_path):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.train_engine = FakeTrainEngine(tmp_path / "missing.pt")
    trainer.config = SimpleNamespace(
        global_resource_planner=SimpleNamespace(
            hybrid_worker_task_dir=str(tmp_path / "tasks"),
            hybrid_lockstep_gradient_sync=False,
        )
    )
    trainer._pending_hybrid_training_batches = {"hybrid0": [{"sample": 1}]}

    try:
        trainer._dispatch_hybrid_training_batches(5)
    except FileNotFoundError as exc:
        assert "step-aligned elastic snapshot" in str(exc)
    else:
        raise AssertionError("missing snapshot must reject an active hybrid task")


def test_hybrid_lockstep_batch_does_not_require_per_step_snapshot(tmp_path):
    task_dir = tmp_path / "tasks"
    membership_dir = task_dir / "membership"
    membership_dir.mkdir(parents=True)
    (membership_dir / "hybrid0.json").write_text(
        json.dumps(
            {
                "worker_id": "hybrid0",
                "role": "hybrid_training",
                "membership_epoch": 3,
            }
        ),
        encoding="utf-8",
    )
    trainer = object.__new__(AsyncRLTrainer)
    trainer.train_engine = FakeTrainEngine(tmp_path / "missing.pt")
    trainer.config = SimpleNamespace(
        global_resource_planner=SimpleNamespace(
            hybrid_worker_task_dir=str(task_dir),
            hybrid_lockstep_gradient_sync=True,
        )
    )
    trainer._pending_hybrid_training_batches = {"hybrid0": [{"sample": 1}]}

    trainer._dispatch_hybrid_training_batches(5)

    task = torch.load(
        task_dir / "hybrid0.step_5.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert task["snapshot_path"] == ""
    assert task["lockstep_gradient_sync"] is True


def test_elastic_snapshot_leases_keep_bootstrap_snapshot(tmp_path):
    task_dir = tmp_path / "tasks"
    membership_dir = task_dir / "membership"
    lease_dir = task_dir / "bootstrap_leases"
    membership_dir.mkdir(parents=True)
    lease_dir.mkdir()
    (membership_dir / "hybrid0.json").write_text(
        json.dumps(
            {
                "worker_id": "hybrid0",
                "role": "hybrid_joining",
                "state_version": 9,
            }
        ),
        encoding="utf-8",
    )
    (lease_dir / "hybrid0.json").write_text(
        json.dumps({"worker_id": "hybrid0", "snapshot_version": 4}),
        encoding="utf-8",
    )
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        global_resource_planner=SimpleNamespace(
            hybrid_worker_task_dir=str(task_dir),
        )
    )

    assert trainer._elastic_snapshot_leases() == {4, 9}

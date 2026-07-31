import os
from types import SimpleNamespace

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class FakeTrainEngine:
    train_backend = "megatron_core"


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

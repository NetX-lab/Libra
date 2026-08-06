from types import SimpleNamespace

import pytest
import torch

from RL_Framework.engine.megatron_core_train_engine import (
    MegatronCoreTrainEngine,
)


def test_merge_stats_uses_update_weighted_averages():
    merged = MegatronCoreTrainEngine._merge_stats(
        [
            {
                "loss": 1.0,
                "pg_loss": 2.0,
                "kl": 0.1,
                "reward_sum": 2.0,
                "n_samples": 2.0,
                "n_updates": 1.0,
                "grad_norm": 3.0,
            },
            {
                "loss": 3.0,
                "pg_loss": 4.0,
                "kl": 0.3,
                "reward_sum": 4.0,
                "n_samples": 2.0,
                "n_updates": 3.0,
                "grad_norm": 5.0,
            },
        ]
    )

    assert merged["loss"] == pytest.approx(2.5)
    assert merged["pg_loss"] == pytest.approx(3.5)
    assert merged["kl"] == pytest.approx(0.25)
    assert merged["grad_norm"] == pytest.approx(4.5)
    assert merged["reward_mean"] == pytest.approx(1.5)


def test_loss_uses_non_negative_ppo_kl_approximation(monkeypatch):
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.clip_epsilon = 0.2
    engine.kl_coef = 0.1
    new_logprobs = torch.tensor([[0.3, -0.4]])
    monkeypatch.setattr(
        engine,
        "_action_log_probs",
        lambda logits, input_ids: new_logprobs,
    )
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "old_logprobs": torch.tensor([[0.0, 0.1, -0.2]]),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "advantages": torch.tensor([1.0]),
        "rewards": torch.tensor([1.0]),
    }

    _, stats = engine._loss(micro_batch, torch.empty(0))

    assert stats["kl"] >= 0.0


def test_sequence_parallel_padding_updates_training_tensors():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.sequence_parallel = True
    engine.train_tp_size = 4
    engine.tokenizer = SimpleNamespace(pad_token_id=99)
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "old_logprobs": torch.ones((1, 5)),
        "loss_mask": torch.ones((1, 5)),
    }

    padded = engine._pad_micro_batch_for_sequence_parallel(micro_batch)

    assert padded["input_ids"].shape == (1, 8)
    assert padded["old_logprobs"].shape == (1, 8)
    assert padded["loss_mask"].shape == (1, 8)
    assert padded["input_ids"].tolist()[0][-3:] == [99, 99, 99]
    assert padded["old_logprobs"].tolist()[0][-3:] == [0.0, 0.0, 0.0]
    assert padded["loss_mask"].tolist()[0][-3:] == [0.0, 0.0, 0.0]


def test_conversion_coverage_rejects_unmapped_local_parameters():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.model = [object()]
    engine.bridge = SimpleNamespace(
        get_conversion_tasks=lambda model: [
            SimpleNamespace(
                param_name="decoder.layers.0.mlp.experts.weight",
                megatron_module=None,
                mapping=None,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="1 local parameters unmapped"):
        engine._validate_conversion_coverage()


def test_megatron_core_device_uses_configured_backend():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.device_backend = "npu"
    engine.local_rank = 3

    assert str(engine._device()) == "npu:3"

"""CPU regression checks for Libra's V4 pipeline adapter."""

from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from RL_Framework.config import load_config
from RL_Framework.engine.train_factory import create_train_engine


def test_h20_gate_config_selects_v4_pipeline():
    path = Path(__file__).resolve().parents[1] / "configs/deepseek_v4_flash_h20_32_train_gate.yaml"
    config = load_config(str(path))
    assert (config.train_tp_size, config.train_pp_size, config.train_ep_size, config.train_dp_size) == (1, 4, 8, 8)
    assert create_train_engine(config).supports_pipeline_parallelism
    with pytest.raises(ValueError, match="requires train_pp_size=1"):
        replace(config, megatron_model_family="default")
    with pytest.raises(ValueError, match="complete GRPO groups"):
        replace(config, batch_size=24)


def test_v4_schedule_preserves_actions_when_padding_variable_lengths(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "configs/deepseek_v4_flash_h20_32_train_gate.yaml"
    engine = create_train_engine(load_config(str(path)))
    engine._device = lambda: torch.device("cpu")
    engine.tokenizer = SimpleNamespace(pad_token_id=0)
    engine.max_seq_length = 8
    engine.optimizer = SimpleNamespace(scale_loss=lambda loss: loss)
    model_config = SimpleNamespace()

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 8)

        def forward(self, *, input_ids, position_ids, attention_mask, labels):
            assert input_ids.shape == (1, 8)
            return self.embedding(input_ids)

    model = TinyPolicy()
    engine.model = [model]

    def schedule(*, forward_step_func, data_iterator, model, num_microbatches,
                 seq_length, micro_batch_size, forward_only):
        assert seq_length == 8 and micro_batch_size == 1
        stats = []
        for _ in range(num_microbatches):
            logits, objective = forward_step_func(data_iterator, model[0])
            loss, item = objective(logits)
            if not forward_only:
                (model_config.grad_scale_func(loss) / num_microbatches).backward()
            stats.append(item)
        return stats

    symbols = {
        "megatron.core.distributed": {"finalize_model_grads": lambda model: None},
        "megatron.core.pipeline_parallel": {"get_forward_backward_func": lambda: schedule},
        "megatron.core.utils": {"get_model_config": lambda model: model_config},
    }
    for name, attrs in symbols.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    trajectories = [
        {
            "input_ids": torch.arange(length).unsqueeze(0),
            "loss_mask": torch.tensor([[0] + [1] * (length - 1)]),
            "rewards": torch.tensor([1.0]),
        }
        for length in (3, 5)
    ]
    logprobs = engine._pipeline(trajectories, forward_only=True)
    for row, actual in zip(trajectories, logprobs):
        ids = row["input_ids"]
        expected = model.embedding(ids)[:, :-1].log_softmax(dim=-1)
        expected = expected.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected)
        row["logprobs"] = torch.cat((torch.zeros(1, 1), actual.detach()), dim=1)
    stats = engine._pipeline(trajectories, forward_only=False)
    assert sum(item["n_samples"] for item in stats) == 2
    assert all(torch.isfinite(item["loss"]) for item in stats)
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad[5:].count_nonzero() == 0

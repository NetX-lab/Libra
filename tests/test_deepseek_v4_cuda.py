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


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4, 6, 12])
def test_v4_config_derives_logical_dp_for_48_training_gpus(cp_size):
    path = Path(__file__).resolve().parents[1] / "configs/deepseek_v4_flash_h20_32_train_gate.yaml"
    config = replace(load_config(str(path)), train_gpus=48, train_pp_size=4,
                     train_ep_size=4, train_cp_size=cp_size, train_dp_size=0, batch_size=24)
    assert config.train_dp_size == 12 // cp_size
    assert create_train_engine(config).train_cp_size == cp_size
    with pytest.raises(ValueError, match="must divide DP"):
        replace(config, train_ep_size=8)


@pytest.mark.parametrize("cp_size", [2, 3, 4])
def test_v4_cp_shards_preserve_action_logprobs_and_gradient(monkeypatch, cp_size):
    from RL_Framework.engine import deepseek_v4_train_engine as module

    path = Path(__file__).resolve().parents[1] / "configs/deepseek_v4_flash_h20_32_train_gate.yaml"
    config = replace(load_config(str(path)), train_gpus=48, train_ep_size=4,
                     train_cp_size=cp_size, train_dp_size=0, batch_size=24)
    engine = create_train_engine(config)
    packed_module = ModuleType("megatron.core.packed_seq_params")
    packed_module.PackedSeqParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, packed_module.__name__, packed_module)

    generator = torch.Generator().manual_seed(19)
    ids = torch.randint(0, 7, (2, 12), generator=generator)
    # CP4 rank zero has no actions. CP3 rank zero owns the query for target 8.
    mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    full_logits = torch.randn(2, 12, 7, generator=generator, requires_grad=True)
    batch = dict(input_ids=ids, loss_mask=mask, length=10,
                 old_logprobs=torch.full((2, 12), -2.0),
                 advantages=torch.tensor([1., -1.]), rewards=torch.tensor([1., 2.]))
    reference = engine._action_log_probs(full_logits, ids)
    expected_loss, expected_stats = engine._loss(batch, full_logits)
    expected_grad = torch.autograd.grad(expected_loss, full_logits)[0]
    aligned = torch.nn.functional.pad(reference[:, :9].detach(), (0, 3)).flatten()
    width = ids.numel() // cp_size
    total_loss = 0
    for rank in range(cp_size):
        start = rank * width
        stop = start + width
        group = SimpleNamespace(size=lambda: cp_size, rank=lambda: rank)

        def model(*, input_ids, position_ids, packed_seq_params, **kwargs):
            assert packed_seq_params.qkv_format == "thd"
            assert packed_seq_params.cu_seqlens_q.tolist() == [0, 12, 24]
            torch.testing.assert_close(input_ids[0], ids.flatten()[start:stop])
            torch.testing.assert_close(position_ids[0], torch.arange(start, stop) % 12)
            return full_logits.reshape(1, 24, 7)[:, start:stop]

        def gather(output, local, *, group):
            torch.testing.assert_close(local, aligned[start:stop])
            output.copy_(aligned)

        monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
        with torch.no_grad():
            logits, objective = engine._forward_context_shard(
                batch, model, group, forward_only=True,
            )
            torch.testing.assert_close(objective(logits)[1], reference[:, :9])
        logits, objective = engine._forward_context_shard(
            batch, model, group, forward_only=False,
        )
        loss, stats = objective(logits)
        assert stats["n_samples"] == expected_stats["n_samples"]
        total_loss += loss
    torch.testing.assert_close(total_loss, expected_loss)
    torch.testing.assert_close(torch.autograd.grad(total_loss, full_logits)[0], expected_grad)


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


def test_v4_gradient_clipping_supports_mixed_precision():
    from RL_Framework.engine.deepseek_v4_train_engine import _gradient_norms, _scale_gradients

    grads = [torch.tensor([3., 4.], dtype=torch.bfloat16),
             torch.tensor([0., 12.], dtype=torch.float32)]
    total, norms = _gradient_norms(None, None, [grads], per_tensor=True)
    torch.testing.assert_close(norms, torch.tensor([5., 12.]))
    torch.testing.assert_close(total, torch.tensor([13.]))
    _scale_gradients(None, None, [grads, grads], torch.tensor([0.5]))
    torch.testing.assert_close(grads[0].float(), torch.tensor([1.5, 2.]))
    torch.testing.assert_close(grads[1], torch.tensor([0., 6.]))

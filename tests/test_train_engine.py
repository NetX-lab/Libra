import json

import pytest
import torch

from RL_Framework.engine.train_engine import (
    FSDPTrainEngine,
    _apply_fsdp_activation_checkpointing,
    _checkpoint_only_with_grad,
    _selected_token_logprobs,
)


def test_selected_token_logprobs_matches_log_softmax():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 11)
    targets = torch.randint(0, logits.shape[-1], (2, 3))

    expected = torch.gather(
        torch.log_softmax(logits, dim=-1),
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    actual = _selected_token_logprobs(logits, targets)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_fsdp_local_batch_size_is_sharded():
    engine = FSDPTrainEngine(model_path="unused")
    engine.world_size = 4

    assert engine.get_local_batch_size(32) == 8
    with pytest.raises(ValueError):
        engine.get_local_batch_size(10)


def test_fsdp_activation_checkpointing_wraps_each_transformer_layer():
    class TransformerLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, inputs):
            return torch.relu(self.linear(inputs))

    model = torch.nn.Sequential(TransformerLayer(), TransformerLayer())

    wrapped = _apply_fsdp_activation_checkpointing(model, TransformerLayer)
    output = model(torch.randn(2, 4, requires_grad=True)).sum()
    output.backward()

    assert wrapped == 2
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_checkpoint_wrapper_bypasses_checkpoint_without_grad(monkeypatch):
    calls = []

    def fail_if_checkpointed(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("no-grad forward should bypass checkpoint")

    monkeypatch.setattr(
        "RL_Framework.engine.train_engine.torch_checkpoint",
        fail_if_checkpointed,
    )
    value = torch.tensor(3.0)
    with torch.no_grad():
        result = _checkpoint_only_with_grad(lambda item: item * 2, value)

    assert result.item() == 6.0
    assert calls == []


def test_fsdp_weight_sync_phase_is_recorded_atomically(tmp_path, monkeypatch):
    engine = FSDPTrainEngine.__new__(FSDPTrainEngine)
    engine.rank = 3
    engine.local_rank = 1
    monkeypatch.setenv("FSDP_PROGRESS_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "42")

    weight_file = str(tmp_path / "weights" / "pytorch_model.bin")
    engine._record_weight_sync_phase(5, "after_state_dict", weight_file)

    progress_path = (
        tmp_path
        / "job_42"
        / "update_5"
        / "weight_sync"
        / "rank_3.json"
    )
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["rank"] == 3
    assert payload["local_rank"] == 1
    assert payload["update"] == 5
    assert payload["phase"] == "after_state_dict"
    assert payload["weight_file"] == weight_file

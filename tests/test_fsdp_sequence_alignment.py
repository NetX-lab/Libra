from types import SimpleNamespace

import torch

from RL_Framework.engine.train_engine import FSDPTrainEngine


def _trajectory(length: int) -> dict:
    return {
        "input_ids": torch.arange(length).unsqueeze(0),
        "attention_mask": torch.ones((1, length), dtype=torch.long),
        "logprobs": torch.zeros((1, length), dtype=torch.float32),
        "loss_mask": torch.ones((1, length), dtype=torch.long),
    }


def test_alignment_pads_effective_work_but_masks_loss(monkeypatch):
    engine = FSDPTrainEngine.__new__(FSDPTrainEngine)
    engine.is_distributed = True
    engine.world_size = 2
    engine.rank = 0
    engine.local_rank = 0
    engine.model = torch.nn.Linear(1, 1)
    engine.tokenizer = SimpleNamespace(pad_token_id=0)

    call_count = 0

    def fake_all_reduce(tensor, op=None):
        nonlocal call_count
        call_count += 1
        if tensor.numel() == 2:
            tensor.copy_(torch.tensor([8, 6], dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    aligned = engine.align_distributed_trajectories(
        [_trajectory(4), _trajectory(7)]
    )

    assert call_count == 3
    assert [traj["_original_seq_len"] for traj in aligned] == [7, 4]
    assert [traj["_padded_seq_len"] for traj in aligned] == [8, 6]
    assert aligned[0]["attention_mask"].tolist()[0][-1] == 1
    assert aligned[0]["loss_mask"].tolist()[0][-1] == 0
    assert aligned[1]["attention_mask"].tolist()[0][-2:] == [1, 1]
    assert aligned[1]["loss_mask"].tolist()[0][-2:] == [0, 0]


def test_recompute_logprobs_locksteps_each_microbatch():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            vocab_size = 16
            logits = torch.zeros(
                (*input_ids.shape, vocab_size),
                device=input_ids.device,
            )
            return SimpleNamespace(logits=logits + self.anchor)

    engine = FSDPTrainEngine.__new__(FSDPTrainEngine)
    engine.model = TinyModel()
    engine._fsdp_update_index = 3
    phases = []
    engine._fsdp_lockstep = (
        lambda update, micro, phase, length:
        phases.append((update, micro, phase, length))
    )

    trajectories = [_trajectory(4), _trajectory(6)]
    engine.recompute_logprobs(trajectories)

    assert phases == [
        (3, 0, "recompute_before_forward", 4),
        (3, 0, "recompute_after_forward", 4),
        (3, 0, "recompute_after_logprobs", 4),
        (3, 1, "recompute_before_forward", 6),
        (3, 1, "recompute_after_forward", 6),
        (3, 1, "recompute_after_logprobs", 6),
    ]

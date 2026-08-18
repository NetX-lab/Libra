"""Support code for Test qwen3d modules."""

import os
import sys
import types

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("RL_Framework")
_pkg.__path__ = [_project_root]
_pkg.__package__ = "RL_Framework"
sys.modules.setdefault("RL_Framework", _pkg)

import torch

from RL_Framework.engine.qwen3d_modules import (
    QwenPipelineStage,
    build_stage_spec,
    divide_evenly,
    split_range_for_rank,
    vocab_parallel_token_logprobs,
)
from RL_Framework.engine.train_parallel import TrainParallelRuntime


def test_divide_evenly():
    parts = divide_evenly(10, 3)
    assert parts == [(0, 4), (4, 7), (7, 10)]


def test_split_range():
    assert split_range_for_rank(12, 3, 0) == (0, 4)
    assert split_range_for_rank(12, 3, 1) == (4, 8)
    assert split_range_for_rank(12, 3, 2) == (8, 12)


def test_build_stage_spec():
    runtime = TrainParallelRuntime(
        rank=2,
        world_size=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        data_parallel_size=1,
    )
    spec = build_stage_spec(10, runtime)
    assert spec.stage_id == 2
    assert spec.layer_start == 6
    assert spec.layer_end == 8


def test_vocab_parallel_logprob_tp1():
    runtime = TrainParallelRuntime(
        rank=0,
        world_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.0, -0.2]]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[2, 0]], dtype=torch.long)
    expected = torch.log_softmax(logits, dim=-1)
    expected = torch.gather(expected, -1, targets.unsqueeze(-1)).squeeze(-1)
    actual = vocab_parallel_token_logprobs(logits, targets, runtime)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_causal_mask_with_padding():
    stage = QwenPipelineStage.__new__(QwenPipelineStage)
    attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.long)
    inputs_embeds = torch.zeros(2, 3, 4, dtype=torch.float32)
    cache_position = torch.arange(3)

    mask = stage._build_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
    )
    min_dtype = torch.finfo(inputs_embeds.dtype).min

    assert mask.shape == (2, 1, 3, 3)
    assert mask[0, 0, 2, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() == min_dtype
    assert torch.all(mask[1, 0, :, 0] == min_dtype)


def main():
    test_divide_evenly()
    test_split_range()
    test_build_stage_spec()
    test_vocab_parallel_logprob_tp1()
    test_causal_mask_with_padding()
    print("Qwen 3D module tests passed")


if __name__ == "__main__":
    main()

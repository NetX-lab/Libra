from RL_Framework.config import AsyncRLConfig, HardwareConfig, ModelArchConfig
from RL_Framework.infra.cost_model.memory_budget import (
    estimate_recompute_logprobs_memory,
)
from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig


def test_recompute_memory_uses_local_vocab_and_fp32_workspace():
    estimate = estimate_recompute_logprobs_memory(
        batch_size=2,
        sequence_length=1024,
        vocab_size=1000,
        tensor_parallel_size=4,
        logits_dtype_bytes=4,
        workspace_factor=1.5,
    )

    assert estimate.local_vocab_size == 250
    assert estimate.logits_bytes == 2 * 1023 * 250 * 4
    assert estimate.total_bytes == int(estimate.logits_bytes * 1.5)


def test_train_oom_budget_uses_max_sequence_length_for_recompute():
    config = AsyncRLConfig(
        model_path="/tmp/model",
        train_gpus=1,
        rollout_gpus=1,
        train_tp_size=1,
        train_dp_size=1,
        batch_size=1,
        micro_batch_size=1,
        max_seq_length=4096,
        recompute_logprobs=True,
        hardware=HardwareConfig(mem_capacity=2.0e9),
        model_arch=ModelArchConfig(
            num_params=1.0e8,
            d_model=128,
            n_layers=2,
            vocab_size=10000,
        ),
    )
    model = CostModel(
        hardware=config.hardware,
        model_arch=config.model_arch,
        profiling=config.profiling,
        max_seq_length=config.max_seq_length,
        recompute_logprobs=True,
    )

    assert model.check_train_oom(
        TrainParallelConfig(tp=1, pp=1, dp=1, b_micro=1),
        B_global=1,
        L=128,
    )

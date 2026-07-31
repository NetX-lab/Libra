import pytest

from RL_Framework.config import AsyncRLConfig, ModelArchConfig


def make_config(**overrides):
    values = {
        "model_path": "Qwen/Qwen3-30B-A3B",
        "train_backend": "megatron_core",
        "train_gpus": 8,
        "rollout_gpus": 8,
        "train_tp_size": 1,
        "train_pp_size": 1,
        "train_cp_size": 1,
        "train_ep_size": 8,
        "train_dp_size": 8,
        "batch_size": 32,
        "micro_batch_size": 1,
        "model_arch": ModelArchConfig(
            num_params=30_000_000_000,
            d_model=2048,
            n_layers=48,
            n_heads=32,
            n_kv_heads=4,
            vocab_size=151936,
            intermediate_size=6144,
            is_moe=True,
            num_experts=128,
            num_activated_experts=8,
            expert_intermediate_size=768,
        ),
    }
    values.update(overrides)
    return AsyncRLConfig(**values)


def test_megatron_core_qwen3_moe_topology():
    config = make_config()

    assert config.train_backend == "megatron_core"
    assert config.train_dp_size == 8
    assert config.train_ep_size == 8
    assert config.use_distributed_optimizer is True
    assert config.megatron_checkpoint_format == "torch_dist"
    assert config.megatron_use_transformer_engine is False
    assert config.megatron_grad_reduce_in_fp32 is False
    assert config.megatron_use_precision_aware_optimizer is False


def test_megatron_core_preflight_topology():
    config = make_config(
        train_tp_size=2,
        train_dp_size=4,
        train_ep_size=4,
        expert_tensor_parallel_size=2,
    )

    assert config.train_gpus == 8
    assert config.train_tp_size == 2
    assert config.train_dp_size == 4
    assert config.train_ep_size == 4
    assert config.expert_tensor_parallel_size == 2


def test_megatron_core_rejects_expert_parallel_mismatch():
    with pytest.raises(ValueError, match="train_ep_size must divide"):
        make_config(train_ep_size=3)


def test_megatron_core_rejects_pipeline_parallel_for_initial_backend():
    with pytest.raises(ValueError, match="requires train_pp_size=1"):
        make_config(
            train_pp_size=2,
            train_dp_size=4,
        )


def test_megatron_core_rejects_invalid_expert_tensor_parallelism():
    with pytest.raises(
        ValueError,
        match="expert_tensor_parallel_size must divide",
    ):
        make_config(expert_tensor_parallel_size=3)


def test_megatron_core_optimizer_offload_defaults_to_disabled():
    config = make_config()

    assert config.megatron_optimizer_cpu_offload is False
    assert config.megatron_optimizer_offload_fraction == 0.0


def test_megatron_core_rejects_offload_fraction_without_cpu_offload():
    with pytest.raises(ValueError, match="requires"):
        make_config(megatron_optimizer_offload_fraction=1.0)

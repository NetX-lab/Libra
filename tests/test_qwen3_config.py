from pathlib import Path

from RL_Framework.config import (
    HardwareConfig,
    ModelArchConfig,
    ProfilingConfig,
)
from RL_Framework.infra.cost_model.model import CostModel


CONFIG_DIR = Path(__file__).parents[1] / "configs" / "model_arch_config"


def test_qwen3_14b_architecture_config():
    model = ModelArchConfig.from_file(
        str(CONFIG_DIR / "qwen3-14b.yaml")
    )
    assert model.num_params == 14e9
    assert model.d_model == 5120
    assert model.n_layers == 40
    assert model.intermediate_size == 17408
    assert model.vocab_size == 151936
    assert not model.is_moe


def test_qwen3_30b_a3b_moe_config_and_memory():
    model = ModelArchConfig.from_file(
        str(CONFIG_DIR / "qwen3-30b-a3b.yaml")
    )
    assert model.is_moe
    assert model.d_model == 2048
    assert model.n_layers == 48
    assert model.num_experts == 128
    assert model.num_activated_experts == 8
    assert model.expert_intermediate_size == 768
    assert model.effective_num_params == 3e9

    cost = CostModel(
        hardware=HardwareConfig(),
        model_arch=model,
        profiling=ProfilingConfig(),
    )
    expected = (
        8 * cost.hw.mem_capacity
        - model.num_params * model.dtype_bytes
        - cost.rollout_model.act_workspace
    )
    assert cost.rollout_model.compute_kv_pool(8) == expected

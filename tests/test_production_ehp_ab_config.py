from pathlib import Path

from scripts.materialize_6node48_config import load_inherited_config
from scripts.validate_ehp_ab_config import validate


PROJECT = Path(__file__).resolve().parents[1]
NO_EHP = PROJECT / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_no_ehp.yaml"
EHP = PROJECT / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml"


def test_production_ab_diff_is_ehp_only():
    differences = validate(NO_EHP, EHP)
    assert "global_resource_planner.hybrid_training_prewarm_enabled" in differences
    assert "batch_size" not in differences
    assert "n_samples" not in differences
    assert "seed" not in differences


def test_production_batch_contains_complete_grpo_groups_per_dp_replica():
    config = load_inherited_config(EHP)
    local_batch = config["batch_size"] // config["train_dp_size"]
    assert config["batch_size"] == 32
    assert config["n_samples"] == 4
    assert local_batch == 8
    assert local_batch % config["n_samples"] == 0
    assert config["total_steps"] == 200

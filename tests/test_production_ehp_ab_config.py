from pathlib import Path

from scripts.materialize_6node48_config import load_inherited_config
from scripts.validate_ehp_ab_config import validate


PROJECT = Path(__file__).resolve().parents[1]
NO_EHP = PROJECT / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_no_ehp.yaml"
EHP = PROJECT / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml"


def test_production_ab_diff_is_ehp_only():
    differences = validate(NO_EHP, EHP)
    assert "global_resource_planner.elastic_hybrid_planning_enabled" in differences
    assert "global_resource_planner.runtime_online_replanning" in differences
    assert "global_resource_planner.runtime_dynamic_reconfiguration_enabled" in differences
    assert "global_resource_planner.runtime_reconfigure_training" in differences
    assert "global_resource_planner.hybrid_worker_launch_enabled" in differences
    assert "global_resource_planner.hybrid_worker_remote_control_enabled" in differences
    assert "global_resource_planner.elastic_hybrid_require_isolated_ccl" in differences
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
    assert config["sync_interval"] == 5


def test_control_arm_freezes_initial_grp_allocation():
    no_ehp = load_inherited_config(NO_EHP)
    ehp = load_inherited_config(EHP)
    no_ehp_planner = no_ehp["global_resource_planner"]
    ehp_planner = ehp["global_resource_planner"]

    assert no_ehp_planner["initial_allocation_strategy"] == "grp"
    assert no_ehp_planner["fixed_train_gpus"] == 0
    assert no_ehp_planner["runtime_online_replanning"] is False
    assert no_ehp_planner["runtime_dynamic_reconfiguration_enabled"] is False
    assert no_ehp_planner["runtime_reconfigure_training"] is False
    assert no_ehp_planner["hybrid_worker_launch_enabled"] is False
    assert no_ehp_planner["hybrid_worker_remote_control_enabled"] is False
    assert no_ehp_planner["elastic_hybrid_require_isolated_ccl"] is False

    assert ehp_planner["runtime_online_replanning"] is True
    assert ehp_planner["runtime_dynamic_reconfiguration_enabled"] is True
    assert ehp_planner["runtime_reconfigure_training"] is True
    assert ehp_planner["hybrid_worker_launch_enabled"] is True
    assert ehp_planner["hybrid_worker_remote_control_enabled"] is True
    assert ehp_planner["elastic_hybrid_require_isolated_ccl"] is True

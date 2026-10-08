from pathlib import Path

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.paper_hierarchical_planner import (
    PaperHierarchicalPlanner,
    apply_paper_plan,
)


def test_paper_tree_dp_produces_auditable_feasible_plan():
    root = Path(__file__).resolve().parents[1]
    config = AsyncRLConfig.from_yaml(
        str(root / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_paper_tree_dp.yaml")
    )
    history = [
        {"input_len": 256 + (index % 5) * 128, "output_len": 128 + index * 64}
        for index in range(12)
    ]

    plan = PaperHierarchicalPlanner(
        config, n_total_gpus=48, max_dp_requests=12
    ).plan(history)
    apply_paper_plan(config, plan)

    assert plan.train_gpus + plan.rollout_gpus == 48
    assert sum(plan.rollout.tp_list) == plan.rollout_gpus
    assert plan.trace["tree_nodes"]
    assert plan.trace["rollout_dp"]
    assert any(
        node["status"] == "pruned" for node in plan.trace["tree_nodes"]
    )
    assert config.train_gpus == (
        config.train_tp_size
        * config.train_pp_size
        * config.train_cp_size
        * config.train_dp_size
    )

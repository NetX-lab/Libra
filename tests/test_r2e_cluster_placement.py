"""GRP binds Core ranks and rollout services to disjoint physical GPUs."""

import pytest

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlan,
    GlobalResourcePlanner,
)
from RL_Framework.infra.cost_model.model import RolloutClusterConfig, TrainParallelConfig
from RL_Framework.scripts.materialize_r2e_cluster import (
    materialize_cluster,
    materialize_planned_cluster,
    seed_cluster_config,
)


def config(train_gpus=4, instances=None):
    result = AsyncRLConfig(
        model_path="not-loaded",
        batch_size=16,
        n_samples=4,
        train_gpus=train_gpus,
        train_tp_size=train_gpus,
        train_dp_size=1,
        max_seq_length=4096,
        max_new_tokens=2048,
        max_concurrent_rollouts=32,
    )
    result.heterogeneous_rollout.enabled = True
    result.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f"seed-{index}", tp=tp)
        for index, tp in enumerate(instances if instances is not None else [2, 2])
    ]
    controls = result.global_resource_planner
    controls.enabled = True
    controls.allowed_train_tp = [train_gpus]
    controls.allowed_train_ep = controls.allowed_train_pp = [1]
    controls.allowed_rollout_tp = [1, 2, 4]
    controls.micro_batch_sizes = [1]
    return result


def assert_physical_coverage(planned, placement):
    train = {
        (host, gpu) for host, gpus in placement["train_gpu_ids"].items() for gpu in gpus
    }
    rollout = [
        (instance.host, gpu)
        for instance in planned.heterogeneous_rollout.instances
        for gpu in instance.gpus
    ]
    assert len(rollout) == len(set(rollout)) == planned.rollout_gpus
    assert train.isdisjoint(rollout)
    assert train.union(rollout) == {
        (host, gpu) for host in placement["allocation_hosts"] for gpu in range(4)
    }
    assert planned.num_nodes == len(placement["train_hosts"])
    assert placement["training_world_size"] == planned.train_gpus


def test_cluster_seed_covers_both_rollout_nodes_without_changing_input():
    original = config()
    before = original.to_dict()
    hosts = ["core", "rollout-a", "rollout-b"]
    seeded = seed_cluster_config(original, hosts)
    planned, placement = materialize_planned_cluster(seeded, hosts)
    assert original.to_dict() == before
    assert planned.train_gpus == 4
    assert planned.rollout_gpus == 8
    assert placement["rollout_gpu_ids"] == {
        "rollout-a": [0, 1, 2, 3],
        "rollout-b": [0, 1, 2, 3],
    }
    assert [instance.tp for instance in planned.heterogeneous_rollout.instances] == [2, 2, 2, 2]
    assert_physical_coverage(planned, placement)


def test_startup_plans_shared_node_with_real_cost_and_preserves_global_concurrency():
    original = config(train_gpus=2, instances=[2, 2, 2])
    controls = original.global_resource_planner
    controls.allowed_rollout_tp = [2]
    controls.min_rollout_gpus = 6
    planned, placement = materialize_cluster(
        original,
        ["core", "rollout"],
        history=[{"input_len": 1024, "output_len": 128}] * 16,
    )
    assert planned.global_resource_planner.initial_allocation_applied
    assert placement["train_gpu_ids"] == {"core": [0, 1]}
    assert len(planned.heterogeneous_rollout.instances) == 3
    assert placement["decision"]["applied_plan"]["max_concurrent_rollouts"] == 32
    assert planned.max_concurrent_rollouts == original.max_concurrent_rollouts == 32
    assert_physical_coverage(planned, placement)


def test_materialization_preserves_a_grp_selected_two_node_core():
    hosts = ["n0", "n1", "n2"]
    seeded = seed_cluster_config(config(), hosts)
    seeded.train_gpus, seeded.rollout_gpus, seeded.train_dp_size = 8, 4, 2
    seeded.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id="grp_tp4_0", tp=4)
    ]
    planned, placement = materialize_planned_cluster(seeded, hosts)
    assert placement["train_hosts"] == ["n0", "n1"]
    assert placement["rollout_hosts"] == ["n2"]
    assert planned.num_nodes == planned.train_dp_size == 2
    assert_physical_coverage(planned, placement)


def test_startup_reassigns_released_core_node_to_rollout_pool():
    original = config(train_gpus=8, instances=[4])
    original.train_tp_size, original.train_dp_size = 4, 2
    controls = original.global_resource_planner
    controls.allowed_train_tp = [4]
    controls.min_rollout_gpus = 8
    planned, placement = materialize_cluster(
        original,
        ["n0", "n1", "n2"],
        history=[{"input_len": 1024, "output_len": 128}] * 16,
    )
    assert placement["train_hosts"] == ["n0"]
    assert placement["rollout_hosts"] == ["n1", "n2"]
    assert_physical_coverage(planned, placement)


@pytest.mark.parametrize("degrees", [[2, 4], [4, 2], [1, 1, 4]])
def test_online_repartition_keeps_core_slots_and_fits_largest_service(degrees):
    hosts = ["shared", "rollout"]
    seeded = seed_cluster_config(config(train_gpus=2, instances=[2, 2, 2]), hosts)
    _, placement = materialize_planned_cluster(seeded, hosts)
    leading = next(item for item in seeded.heterogeneous_rollout.instances if item.host == "shared")
    plan = GlobalResourcePlan(
        train_config=TrainParallelConfig(tp=2, dp=1, b_micro=1),
        rollout_config=RolloutClusterConfig(tp_list=degrees),
        t_train=1.0,
        t_rollout=1.0,
        t_global=1.0,
        n_total_gpus=8,
        max_concurrent_rollouts=32,
    )
    GlobalResourcePlanner.from_config(seeded).apply_plan_to_config(plan, seeded)
    assert [item.tp for item in seeded.heterogeneous_rollout.instances] == degrees
    assert next(item for item in seeded.heterogeneous_rollout.instances if item.tp == 4).host == "rollout"
    if 2 in degrees:
        assert leading in seeded.heterogeneous_rollout.instances
    assert_physical_coverage(seeded, placement)

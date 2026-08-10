from pathlib import Path

import yaml

from RL_Framework.infra.scheduling.load_balance import LoadBalanceScheduler
from scripts.materialize_6node48_config import load_inherited_config, materialize


PROJECT = Path(__file__).resolve().parents[1]
HETERO = PROJECT / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_100step.yaml"
HOMOGENEOUS = (
    PROJECT
    / "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_100step_homogeneous_lb.yaml"
)


def test_homogeneous_baseline_changes_only_rollout_policy_fields():
    hetero = load_inherited_config(HETERO)
    homogeneous = load_inherited_config(HOMOGENEOUS)

    for key in (
        "model_path",
        "train_backend",
        "train_gpus",
        "rollout_gpus",
        "batch_size",
        "micro_batch_size",
        "n_samples",
        "total_steps",
        "max_seq_length",
        "max_new_tokens",
        "sync_interval",
        "seed",
    ):
        assert homogeneous.get(key) == hetero.get(key)

    homogeneous_rollout = homogeneous["heterogeneous_rollout"]
    assert homogeneous_rollout["scheduling"]["scheduler_type"] == "load_balance"
    assert [instance["tp"] for instance in homogeneous_rollout["instances"]] == [2] * 8
    planner = homogeneous["global_resource_planner"]
    assert planner["rollout_node_tp_pattern"] == [2, 2, 2, 2]
    assert planner["allowed_rollout_tp"] == [2]
    assert planner["require_heterogeneous_rollout_tp"] is False


def test_materialize_homogeneous_tp2_layout(tmp_path: Path):
    output = tmp_path / "effective.yaml"
    materialize(
        template=HOMOGENEOUS,
        output=output,
        master_addr="192.0.2.10",
        master_port=29901,
        rollout_hosts=["192.0.2.20", "192.0.2.21"],
        run_root=tmp_path / "run",
        gradient_port=29902,
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    instances = config["heterogeneous_rollout"]["instances"]
    assert len(instances) == 8

    for host_index, host in enumerate(("192.0.2.20", "192.0.2.21")):
        host_instances = instances[host_index * 4:(host_index + 1) * 4]
        assert [instance["host"] for instance in host_instances] == [host] * 4
        assert [instance["tp"] for instance in host_instances] == [2] * 4
        assert [instance["gpus"] for instance in host_instances] == [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
        assert [instance["port"] for instance in host_instances] == [8000, 8001, 8002, 8003]
        assert [instance["instance_id"] for instance in host_instances] == [
            f"n{host.rsplit('.', 1)[-1]}_tp2_0",
            f"n{host.rsplit('.', 1)[-1]}_tp2_1",
            f"n{host.rsplit('.', 1)[-1]}_tp2_2",
            f"n{host.rsplit('.', 1)[-1]}_tp2_3",
        ]


def test_least_connections_rotates_equal_load_endpoints():
    scheduler = LoadBalanceScheduler(load_balance_strategy="least_connections")
    for index in range(8):
        scheduler.register_instance(index, f"tp2_{index}", 2)

    routed = []
    for request_index in range(16):
        result = scheduler.schedule(512, prompt_id=str(request_index))
        routed.append(result.instance_index)
        # Model a low-concurrency multi-turn workload: the request finishes
        # before the next turn is scheduled, so all endpoints tie at zero.
        scheduler.on_request_done(result.instance_index)

    assert routed[:8] == list(range(8))
    assert routed[8:] == list(range(8))


def test_least_connections_does_not_skip_endpoints_across_concurrent_waves():
    scheduler = LoadBalanceScheduler(load_balance_strategy="least_connections")
    for index in range(8):
        scheduler.register_instance(index, f"tp2_{index}", 2)

    waves = []
    for _ in range(2):
        wave = [scheduler.schedule(512).instance_index for _ in range(4)]
        waves.append(wave)
        for instance_index in wave:
            scheduler.on_request_done(instance_index)

    assert waves == [list(range(4)), list(range(4, 8))]

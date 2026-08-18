from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.model import TrainParallelConfig
from RL_Framework.infra.cost_model.optimizer import generate_training_configs
from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel
from unittest.mock import patch


def _npu_experiment_config() -> AsyncRLConfig:
    with patch("os.makedirs"):
        return AsyncRLConfig.from_yaml(
            "configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_grp.yaml"
        )


def test_megatron_qwen3_14b_tp1_is_pruned_by_observed_npu_capacity():
    evaluator = HybridSimulatorCostModel(_npu_experiment_config())
    topology = TrainParallelConfig(tp=1, pp=1, dp=2, b_micro=1)

    estimated = evaluator.analytic.training_model.estimate_memory_per_gpu(
        topology, b_micro=1, L=32768
    )

    assert estimated > evaluator.analytic.hw.mem_capacity
    assert evaluator.check_train_oom(topology, B_global=6, L=32768)


def test_megatron_first_adam_step_peak_prunes_tp4_dp1_without_gpu_floor():
    evaluator = HybridSimulatorCostModel(_npu_experiment_config())
    topology = TrainParallelConfig(tp=4, pp=1, dp=1, b_micro=1)

    assert evaluator.check_train_oom(topology, B_global=6, L=3135)


def test_megatron_qwen3_14b_feasible_topology_is_not_overpruned():
    evaluator = HybridSimulatorCostModel(_npu_experiment_config())
    topology = TrainParallelConfig(tp=4, pp=1, dp=2, b_micro=1)

    assert not evaluator.check_train_oom(topology, B_global=6, L=32768)


def test_training_search_includes_non_node_aligned_npu_counts():
    configs = generate_training_configs(
        n_total_gpus=10,
        allowed_tp=[2],
        allowed_pp=[1],
        micro_batch_sizes=[1],
    )

    assert any(config.n_gpus == 6 for config in configs)


def test_context_parallelism_counts_devices_and_shards_activations():
    evaluator = HybridSimulatorCostModel(_npu_experiment_config())
    no_cp = TrainParallelConfig(tp=4, pp=1, cp=1, dp=1, b_micro=1)
    with_cp = TrainParallelConfig(tp=4, pp=1, cp=2, dp=1, b_micro=1)

    no_cp_memory = evaluator.analytic.training_model.estimate_memory_per_gpu(
        no_cp, b_micro=1, L=32768
    )
    with_cp_memory = evaluator.analytic.training_model.estimate_memory_per_gpu(
        with_cp, b_micro=1, L=32768
    )

    assert with_cp.n_gpus == 8
    assert with_cp_memory < no_cp_memory


def test_megatron_backend_rejects_unsupported_pipeline_parallelism():
    evaluator = HybridSimulatorCostModel(_npu_experiment_config())

    seconds, details = evaluator.evaluate_training(
        TrainParallelConfig(tp=1, pp=2, dp=1, b_micro=1),
        B_global=6,
        L=32768,
    )

    assert seconds == float("inf")
    assert details["backend"] == "capability_check"
    assert details["supported"] is False

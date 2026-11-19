"""Finite cohorts retain their last long decode after short requests end."""

import pytest

from RL_Framework.config import AsyncRLConfig, HardwareConfig, ModelArchConfig, ProfilingConfig
from RL_Framework.infra.cost_model.model import CostModel, RequestInfo, RolloutClusterConfig
from RL_Framework.infra.cost_model.optimizer import TwoLevelNestedOptimizer
from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel


def finite_cost(concurrency=2):
    cost = CostModel(
        hardware=HardwareConfig(mem_bw=1., mem_capacity=1e12),
        model_arch=ModelArchConfig(num_params=100., d_model=1, n_layers=1,
                                  n_heads=1, n_kv_heads=1, head_dim=1, dtype_bytes=1),
        profiling=ProfilingConfig(decode_bw_util=1.),
    )
    cost.rollout_model.max_queue_limit = concurrency
    return cost


@pytest.mark.parametrize("outputs,concurrency,want", [
    ([10, 2], 2, 1104.),
    ([10, 2, 2], 2, 1108.),
    ([10, 2], 1, 1304.),
])
def test_finite_decode_preserves_tail_and_reuses_finished_slots(outputs, concurrency, want):
    # Decode costs 100 weight bytes + 2 bytes per active context token.
    # [10,2]: 2*(100+4) + 8*(100+12) = 1104 at concurrency two.
    requests = [RequestInfo(0, generated) for generated in outputs]
    cost = finite_cost(concurrency).rollout_model
    assert cost.compute_instance_makespan(requests, 1) == pytest.approx(want)


def test_rollout_dp_retains_individual_lengths_when_evaluating_segments():
    optimizer = TwoLevelNestedOptimizer(finite_cost(), allowed_rollout_tp=[1])
    config, seconds, _ = optimizer.rollout_dp(1, [RequestInfo(0, 10), RequestInfo(0, 2)])
    assert config.tp_list == [1]
    assert seconds == pytest.approx(1104.)


def test_fixed_rollout_assignment_retains_tail_through_simulator_facade():
    cost = finite_cost()
    config = AsyncRLConfig(model_path="/tmp/model", hardware=cost.hw,
                           model_arch=cost.ma, profiling=cost.prof)
    facade = HybridSimulatorCostModel(config)
    facade.analytic.rollout_model.max_queue_limit = 2
    optimizer = TwoLevelNestedOptimizer(facade, allowed_rollout_tp=[1])
    seconds, _ = optimizer.evaluate_fixed_rollout_config(
        RolloutClusterConfig([1]), [RequestInfo(0, 10), RequestInfo(0, 2)],
    )
    assert seconds == pytest.approx(1104.)

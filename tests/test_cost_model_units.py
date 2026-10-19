"""Analytic estimates use consistent per-worker and per-instance units."""

import math

import pytest

from RL_Framework.infra.cost_model.model import RolloutCostModel
from RL_Framework.infra.scheduling.cmlfq_cost import CMLFQRoutingCosts


def model(**overrides):
    kwargs = dict(
        num_params=14e9, d_model=5120, n_layers=40, n_kv_heads=8,
        head_dim=128, dtype_bytes=2, flops_peak=312e12, mem_bw=2e12,
        mem_capacity=16e9, tp_comm_overhead=1e-5,
    )
    kwargs.update(overrides)
    return RolloutCostModel(**kwargs)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("capacity,bandwidth", [(16e9, 2e12), (80e9, 1.5e12)])
def test_capacity_and_decode_use_consistent_hardware_units(tp, capacity, bandwidth):
    cost = model(mem_capacity=capacity, mem_bw=bandwidth)
    per_worker_kv = 2 * 40 * max(1, math.ceil(8 / tp)) * 128 * 2
    aggregate_pool = max(0, tp * capacity - 14e9 * 2 - 1e9)
    capacity_tokens = int(aggregate_pool / (tp * per_worker_kv) * .96)
    assert cost.compute_kv_pool(tp) == aggregate_pool
    assert cost.compute_token_capacity(tp) == capacity_tokens
    assert cost.check_oom(capacity_tokens + 1, tp)
    assert not cost.check_oom(capacity_tokens, tp)
    active_tokens = 12345
    per_worker_read = 14e9 * 2 / tp + active_tokens * per_worker_kv
    collective_seconds = 2 * 40 * 1e-5 if tp > 1 else 0
    assert cost.compute_decode_step_time(active_tokens, tp) == pytest.approx(
        per_worker_read / (bandwidth * .85) + collective_seconds,
    )


@pytest.mark.parametrize("heads,tp", [(1, 4), (3, 2), (8, 4)])
def test_replicated_or_uneven_kv_heads_are_not_under_counted(heads, tp):
    cost = model(n_kv_heads=heads)
    per_worker_bytes = 2 * 40 * math.ceil(heads / tp) * 128 * 2
    assert cost.compute_token_kv_bytes(tp) == per_worker_bytes
    assert cost.compute_token_capacity(tp) == int(
        cost.compute_kv_pool(tp) / (tp * per_worker_bytes) * .96,
    )


def test_cost_routing_analytic_fallback_rejects_previously_overcounted_capacity():
    cost = model()
    costs = CMLFQRoutingCosts(rollout_model=cost)
    physical_capacity = int((2 * 16e9 - 28e9 - 1e9) / (2 * 81920) * .96)
    too_long = physical_capacity * 3 // 2
    assert math.isinf(costs.decode_seconds(2, "long", too_long - 100, 100))
    remaining, context = 1000, 2000
    per_worker_bytes = 14e9 + math.ceil(context + remaining / 2) * 81920
    expected = remaining * (per_worker_bytes / (2e12 * .85) + 80e-5)
    assert costs.decode_seconds(2, "short", remaining, context) == pytest.approx(expected)


def test_bandwidth_profile_changes_only_memory_time_not_tp_comm():
    slow, fast = model(mem_bw=1e12), model(mem_bw=2e12)
    overhead = 2 * slow.n_layers * slow.tp_comm_overhead
    for tp in (2, 4):
        assert slow.compute_decode_step_time(1000, tp) - overhead == pytest.approx(
            2 * (fast.compute_decode_step_time(1000, tp) - overhead),
        )

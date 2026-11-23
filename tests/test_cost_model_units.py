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


def test_finite_requests_do_not_use_unused_capacity_to_amortize_decode():
    from RL_Framework.infra.cost_model.model import RequestInfo

    cost = model(mem_capacity=80e9)
    prompt, generated = 1024, 128
    expected_prefill = cost.compute_prefill_step_time(1, 1024, 2)
    expected_decode = generated * cost.compute_decode_step_time(prompt + generated // 2, 2)
    assert cost.compute_instance_makespan([RequestInfo(prompt, generated)], 2) == pytest.approx(
        expected_prefill + expected_decode)


def test_shared_calibration_reaches_grp_and_routing_operator_models():
    from RL_Framework.config import AsyncRLConfig
    from RL_Framework.infra.cost_model.model import CostModel, RequestInfo
    from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel
    config = AsyncRLConfig(model_path="/tmp/model")
    config.profiling.calibration = {"rollout": {"2": {"prefill_scale": 3, "decode_scale": 2}}}
    baseline = CostModel()
    routed = CostModel(profiling=config.profiling)
    grp = HybridSimulatorCostModel(config)
    assert routed.rollout_model.compute_decode_step_time(1024, 2) == pytest.approx(
        2 * baseline.rollout_model.compute_decode_step_time(1024, 2))
    request = [RequestInfo(1024,128)]
    assert grp.evaluate_rollout_instance(2,request)[0] == pytest.approx(
        routed.evaluate_rollout_instance(2,request)[0])


def test_elastic_training_uses_complete_groups_and_does_not_halve_core_twice():
    from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig
    costs = CostModel(recompute_logprobs=False)
    shape = TrainParallelConfig(tp=2,dp=1,b_micro=1)
    counts = []
    times = []
    for hybrids in (0,1,2):
        seconds, details = costs.evaluate_elastic_training(shape,16,4,hybrids,[1024]*16)
        counts.append(details['sample_counts'])
        times.append(seconds)
    assert counts == [[16],[8,8],[8,4,4]]
    assert times[1] == pytest.approx(times[2])
    assert times[0] > times[1]


def test_elastic_training_counts_both_existing_core_dp_replicas():
    from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig

    costs = CostModel(recompute_logprobs=False)
    core = TrainParallelConfig(tp=2, dp=2, b_micro=1)
    lengths = [1024] * 16
    current, current_details = costs.evaluate_elastic_training(core, 16, 4, 0, lengths)
    lent, lent_details = costs.evaluate_elastic_training(core, 16, 4, 1, lengths)
    assert current_details["sample_counts"] == [8, 8]
    assert lent_details["sample_counts"] == [8, 4, 4]
    # The third participant leaves the slowest local partition at two groups.
    assert lent == pytest.approx(current)


def test_elastic_training_joins_each_replica_after_its_local_anchor_and_backprop():
    from RL_Framework.config import ModelArchConfig, ProfilingConfig
    from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig

    profile = ProfilingConfig(
        alpha_mlp_fwd=0., beta_mlp_fwd=0., alpha_mlp_bwd=100., beta_mlp_bwd=0.,
        alpha_attn_fwd=1., beta_attn_fwd=0., gamma_attn_fwd=0.,
        alpha_attn_bwd=0., beta_attn_bwd=0., gamma_attn_bwd=0.,
        calibration={"training": {"hybrid_gradient_s": 5., "optimizer_s": 2.}},
    )
    costs = CostModel(model_arch=ModelArchConfig(n_layers=1), profiling=profile)
    seconds, details = costs.evaluate_elastic_training(
        TrainParallelConfig(tp=1, b_micro=1), 8, 4, 1,
        [10, 1, 1, 1, 4, 4, 4, 4],
    )
    # Local anchor=sum(L^2); local FW/BW=2*sum(L^2)+100*sum(L).
    # Core: 103+1506=1609; Hybrid: 64+1728=1792; exposed costs: 5+2.
    assert details["sample_counts"] == [4, 4]
    assert seconds == pytest.approx(1799.)


def test_evaluate_training_shards_anchor_across_dp_and_ep():
    from RL_Framework.config import ModelArchConfig
    from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig

    costs = CostModel(model_arch=ModelArchConfig(n_layers=1, is_moe=True, num_experts=4, num_activated_experts=2))
    lengths = [1024] * 16
    _, d_dp1 = costs.evaluate_training(TrainParallelConfig(tp=2, dp=1, ep=1, b_micro=1), 16, lengths)
    _, d_dp2 = costs.evaluate_training(TrainParallelConfig(tp=2, dp=2, ep=1, b_micro=1), 16, lengths)
    _, d_dp4 = costs.evaluate_training(TrainParallelConfig(tp=2, dp=4, ep=1, b_micro=1), 16, lengths)
    _, d_ep2 = costs.evaluate_training(TrainParallelConfig(tp=2, dp=1, ep=2, b_micro=1), 16, lengths)

    assert d_dp2["anchor_s"] == pytest.approx(d_dp1["anchor_s"] / 2.0)
    assert d_dp4["anchor_s"] == pytest.approx(d_dp1["anchor_s"] / 4.0)
    expected_ep2_anchor = 8 * costs.training_model.compute_stage_fwd_time(
        1024, TrainParallelConfig(tp=2, dp=1, ep=2, b_micro=1)
    )
    assert d_ep2["anchor_s"] == pytest.approx(expected_ep2_anchor)


def test_evaluate_training_matches_zero_hybrid_elastic_critical_path():
    from RL_Framework.config import ModelArchConfig, ProfilingConfig
    from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig

    profile = ProfilingConfig(
        alpha_mlp_fwd=0., beta_mlp_fwd=0., alpha_mlp_bwd=100., beta_mlp_bwd=0.,
        alpha_attn_fwd=1., beta_attn_fwd=0., gamma_attn_fwd=0.,
        alpha_attn_bwd=0., beta_attn_bwd=0., gamma_attn_bwd=0.,
        calibration={"training": {"compute_scale": 1.5, "anchor_scale": 2.0, "optimizer_s": 3.0}},
    )
    costs = CostModel(model_arch=ModelArchConfig(n_layers=1), profiling=profile)
    # zero_level=0 isolates replica sharding from ZeRO-1 DP background contention
    core = TrainParallelConfig(tp=1, dp=2, b_micro=1, zero_level=0)
    lengths = [10, 1, 1, 1, 4, 4, 4, 4]
    total_train, details = costs.evaluate_training(core, 8, lengths)
    # Use hybrids=0 with core_replicas=2 -> same 2-replica partition without hybrid_gradient_s
    total_elastic, elastic_details = costs.evaluate_elastic_training(core, 8, 4, 0, lengths)

    # Replica 0: anchor=(100+3)*2=206, compute=1506*1.5=2259 -> 2465
    # Replica 1: anchor=64*2=128, compute=1728*1.5=2592 -> 2720 (critical!)
    assert details["replica_anchor_times"] == [pytest.approx(206.0), pytest.approx(128.0)]
    assert details["anchor_s"] == pytest.approx(128.0)
    assert total_train == pytest.approx(total_elastic + details["t_dp_exposed"] * 1.5)
    assert elastic_details["sample_counts"] == [4, 4]


def test_evaluate_training_handles_indivisible_batch_and_disabled_recompute():
    from RL_Framework.config import ModelArchConfig, ProfilingConfig
    from RL_Framework.infra.cost_model.model import CostModel, TrainParallelConfig

    profile = ProfilingConfig(
        calibration={"training": {"compute_scale": 2.0, "anchor_scale": 3.0, "optimizer_s": 1.5}},
    )
    costs_on = CostModel(model_arch=ModelArchConfig(n_layers=1), profiling=profile, recompute_logprobs=True)
    cfg = TrainParallelConfig(tp=2, dp=4, b_micro=1)
    _, details_on = costs_on.evaluate_training(cfg, 10, [512] * 10)
    per_sample_fwd = costs_on.training_model.compute_stage_fwd_time(
        512, TrainParallelConfig(tp=2, dp=1, b_micro=1)
    ) * 3.0
    assert details_on["replica_anchor_times"] == [
        pytest.approx(3 * per_sample_fwd),
        pytest.approx(3 * per_sample_fwd),
        pytest.approx(2 * per_sample_fwd),
        pytest.approx(2 * per_sample_fwd),
    ]
    assert details_on["anchor_s"] == pytest.approx(3 * per_sample_fwd)

    costs_off = CostModel(model_arch=ModelArchConfig(n_layers=1), profiling=profile, recompute_logprobs=False)
    raw_seconds, _ = costs_off.training_model.compute_iteration_time(cfg, 10, [512] * 10)
    total_off, details_off = costs_off.evaluate_training(cfg, 10, [512] * 10)
    assert details_off["anchor_s"] == 0.0
    assert total_off == pytest.approx(raw_seconds * 2.0 + 1.5)

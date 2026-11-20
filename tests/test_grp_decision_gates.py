"""EHP demand cannot authorize a rejected full resource repartition."""

from unittest.mock import Mock

import pytest

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    CoreRuntimeCapabilities,
    GlobalResourcePlan,
    GlobalResourcePlanner,
    PlannerDecision,
)
from RL_Framework.infra.cost_model.model import RolloutClusterConfig, TrainParallelConfig
from RL_Framework.infra.cost_model.optimizer import OptimizationResult


def planner_fixture(monkeypatch, reconfiguration_cost, *, training_bound=False):
    for variable in ("GRP_FORCE_RUNTIME_RECONFIGURE", "GRP_FORCE_ROLLOUT_TP_LIST", "GRP_FORCE_TRAIN_GPUS"):
        monkeypatch.delenv(variable, raising=False)
    config = AsyncRLConfig(
        model_path="not-loaded", train_gpus=2, rollout_gpus=2,
        train_tp_size=1, train_dp_size=2, micro_batch_size=1,
        batch_size=24, n_total_gpus=4,
    )
    cfg = config.global_resource_planner
    cfg.enabled = True
    cfg.train_backend = cfg.rollout_backend = "analytic"
    cfg.plan_interval = 1
    cfg.warmup_steps = 0
    cfg.min_history_size = 1
    cfg.reconfiguration_cost_s = reconfiguration_cost
    cfg.elastic_hybrid_planning_enabled = True
    cfg.elastic_hybrid_min_rollout_gpus = 1
    config.heterogeneous_rollout.enabled = True
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id="service-a", tp=1, gpus=[0]),
        HeterogeneousInstanceConfig(instance_id="service-b", tp=1, gpus=[1]),
    ]
    config.profiling.calibration = {
        "training": {"compute_scale": 1000 if training_bound else 0.0001, "anchor_scale": 0},
        "rollout": {"1": {
            "prefill_scale": 0.0001 if training_bound else 1000,
            "decode_scale": 0.0001 if training_bound else 1000,
        }},
        "horizon_updates": 4,
    }
    planner = GlobalResourcePlanner.from_config(config)
    train = TrainParallelConfig(tp=1, dp=2, b_micro=1)
    current = GlobalResourcePlan(
        train_config=train, rollout_config=RolloutClusterConfig([1, 1]),
        t_train=100, t_rollout=90, t_global=100,
        n_total_gpus=4, max_concurrent_rollouts=16,
    )
    planner._build_current_plan = Mock(return_value=current)
    # Freeze the optimizer result so these tests exercise the real decision
    # gates and EHP policy independently of the search's selected candidate.
    planner.optimizer.optimize = Mock(return_value=OptimizationResult(
        train_config=train, rollout_config=RolloutClusterConfig([2]),
        t_train=80, t_rollout=70, t_global=80,
    ))
    return config, planner, current


@pytest.mark.parametrize("resource_allowed,cost", [(False, 1e9), (True, 0)])
@pytest.mark.parametrize("elastic_adjust", [False, True])
def test_full_repartition_and_ehp_have_independent_gates(
    monkeypatch, resource_allowed, cost, elastic_adjust,
):
    config, planner, current = planner_fixture(monkeypatch, cost, training_bound=elastic_adjust)
    metrics = planner.observe_runtime(
        step=1, dispatcher_metrics={},
        step_stats={"remaining_updates": 4},
    )
    decision = planner.plan_if_needed(
        1, config, batch=[{"input_len": 128, "output_len": 4096}],
        runtime_metrics=metrics,
    )
    assert decision.resource_should_reconfigure is resource_allowed
    assert decision.elastic_hybrid_signal.should_adjust is elastic_adjust
    assert decision.should_reconfigure is (resource_allowed or elastic_adjust)
    assert decision.candidate_plan.rollout_tp_list == [2]
    assert decision.current_plan.rollout_tp_list == [1, 1]
    assert decision.candidate_plan.expected_gain_s == 20 - cost
    assert planner.active_plan is (decision.candidate_plan if resource_allowed else current)
    wire = decision.to_dict()
    assert wire["resource_should_reconfigure"] is resource_allowed
    assert wire["elastic_should_adjust"] is elastic_adjust
    assert wire["elastic_hybrid_signal"]["desired_workers"] == (1 if elastic_adjust else 0)
    assert len(wire["elastic_hybrid_signal"]["loan_instance_ids"]) == int(elastic_adjust)
    assert config.train_dp_size == 2
    assert config.rollout_gpus == 2


def test_release_also_leaves_rejected_full_candidate_inactive(monkeypatch):
    config, planner, current = planner_fixture(monkeypatch, 1e9)
    metrics = planner.observe_runtime(
        step=1, dispatcher_metrics={},
        step_stats={
            "remaining_updates": 4,
            "active_hybrid_replicas": 1,
            "loan_instance_ids": ["service-a"],
        },
    )
    decision = planner.plan_if_needed(
        1, config, batch=[{"input_len": 128, "output_len": 4096}], runtime_metrics=metrics,
    )
    assert decision.should_reconfigure
    assert decision.resource_should_reconfigure is False
    assert decision.elastic_hybrid_signal.action == "release"
    assert planner.active_plan is current


def test_explicit_rollout_repartition_reaches_the_full_gate_with_live_elastic_core(monkeypatch):
    config, planner, current = planner_fixture(monkeypatch, 1e9)
    planner.runtime_capabilities = CoreRuntimeCapabilities(
        world_size=2, tp=1, pp=1, cp=1, ep=1, dp=2,
    )
    monkeypatch.setenv("GRP_FORCE_ROLLOUT_TP_LIST", "2")
    decision = planner.plan_if_needed(
        1, config, batch=[{"input_len": 128, "output_len": 4096}],
    )
    assert decision.current_plan is current
    assert decision.candidate_plan.rollout_tp_list == [2]
    assert decision.resource_should_reconfigure


def test_skipped_real_decision_and_legacy_manual_decision_are_distinguishable(monkeypatch):
    config, planner, _ = planner_fixture(monkeypatch, 0)
    planner.warmup_steps = 2
    decision = planner.plan_if_needed(1, config)
    assert decision.resource_should_reconfigure is False
    assert not decision.should_reconfigure
    assert decision.to_dict()["elastic_should_adjust"] is False
    manual = PlannerDecision(step=1, should_reconfigure=True, reason="legacy")
    assert manual.resource_should_reconfigure is None
    assert manual.to_dict()["resource_should_reconfigure"] is None

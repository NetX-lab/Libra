"""Live Core ranks constrain executable GRP layouts while EHP stays elastic."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    CoreRuntimeCapabilities, ElasticHybridSignal, GlobalResourcePlan,
    GlobalResourcePlanner, PlannerDecision, RuntimeResourceMetrics,
)
from RL_Framework.infra.cost_model.model import RolloutClusterConfig, TrainParallelConfig
from RL_Framework.infra.cost_model.optimizer import OptimizationResult
from RL_Framework.infra.cost_model.preflight_planner import PreflightPlanner
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor


ROWS = [dict(input_len=1024, output_len=2048)] * 16


def config():
    result = AsyncRLConfig(model_path='not-loaded')
    result.n_total_gpus, result.train_gpus, result.rollout_gpus = 12, 4, 8
    result.train_tp_size, result.train_dp_size = 4, 1
    result.batch_size, result.micro_batch_size, result.n_samples = 16, 1, 4
    result.heterogeneous_rollout.enabled = True
    result.heterogeneous_rollout.instances = [HeterogeneousInstanceConfig(
        instance_id=f'rollout-{index}', tp=2, gpus=[index % 2 * 2, index % 2 * 2 + 1],
        host=f'rollout-{index // 2}', port=8000 + index) for index in range(4)]
    planner = result.global_resource_planner
    planner.enabled = True
    planner.initial_allocation_strategy = 'grp'
    planner.allowed_train_tp, planner.allowed_train_pp = [1, 2, 4], [1]
    planner.allowed_train_ep, planner.allowed_rollout_tp = [1], [1, 2, 4]
    planner.micro_batch_sizes = [1]
    planner.allocation_granularity_gpus = 4
    planner.min_train_gpus = planner.min_rollout_gpus = 4
    planner.elastic_hybrid_planning_enabled = True
    planner.runtime_reconfigure_training = True
    planner.runtime_training_pool_only = True
    planner.runtime_training_pool_plan_only = False
    planner.runtime_manage_rollout_processes = planner.hybrid_worker_launch_enabled = True
    planner.elastic_hybrid_require_planner_signal = True
    planner.elastic_hybrid_min_rollout_gpus = 4
    planner.reconfiguration_cost_s = planner.min_gain_ratio = 0
    planner.warmup_steps = 0
    planner.min_history_size = planner.plan_interval = 1
    return result


def engine(world=4, tp=4):
    return NS(world_size=world, train_tp_size=tp, train_pp_size=1,
        train_cp_size=1, train_ep_size=1, train_dp_size=world // tp,
        get_parallel_state=lambda: dict(train_tp=tp, train_pp=1, train_cp=1,
                                       train_ep=1, train_dp=world // tp),
        reconfigure_parallelism=Mock())


def result(train_gpus=8, tp=4):
    return OptimizationResult(train_config=TrainParallelConfig(tp=tp, pp=1,
        dp=train_gpus // tp, b_micro=1),
        rollout_config=RolloutClusterConfig(tp_list=[4] if train_gpus == 8 else [2, 2, 2, 2]),
        t_train=.1, t_rollout=.1, t_global=.1)


def as_plan(value):
    return GlobalResourcePlan(train_config=value.train_config, rollout_config=value.rollout_config,
        t_train=value.t_train, t_rollout=value.t_rollout, t_global=value.t_global,
        n_total_gpus=12, max_concurrent_rollouts=16)


def test_runtime_replaces_unlaunchable_core_resize_but_preserves_hybrid_demand():
    cfg = config()
    planner = GlobalResourcePlanner.from_config(cfg)
    planner.runtime_capabilities = CoreRuntimeCapabilities.from_engine(engine())
    planner.optimizer.optimize = Mock(side_effect=[result(8), result(4)])
    metrics = RuntimeResourceMetrics(step=5, train_time_s=1, rollout_time_s=10)
    decision = planner.plan_if_needed(step=5, config=cfg, batch=ROWS, runtime_metrics=metrics)
    assert decision.candidate_plan.train_gpus == 4
    assert decision.candidate_plan.train_config.tp == 4
    assert decision.candidate_plan.train_config.dp == 1
    assert decision.candidate_plan.rollout_gpus == 8
    assert decision.elastic_hybrid_signal.desired_replicas == 1
    assert decision.elastic_hybrid_signal.reason == 'candidate_requests_more_training_capacity'
    assert not decision.resource_should_reconfigure
    assert decision.should_reconfigure  # Real EHP membership still changes.
    first, second = planner.optimizer.optimize.call_args_list
    assert first.kwargs['fixed_train_gpus'] == 0
    assert second.kwargs['fixed_train_gpus'] == 4
    assert second.kwargs['allowed_train_tp'] == [4]
    assert second.kwargs['allowed_train_ep'] == second.kwargs['allowed_train_pp'] == [1]
    assert cfg.global_resource_planner.fixed_train_gpus == 0  # No configuration workaround.
    proof = decision.candidate_plan.metadata['runtime_candidate_filter']
    assert proof['reason'] == 'physical_core_rank_set_changed'
    assert proof['unconstrained_candidate']['train']['n_gpus'] == 8
    assert proof['elastic_demand_preserved']


def test_even_a_forced_resize_uses_ehp_instead_of_inventing_core_ranks(monkeypatch):
    cfg = config()
    planner = GlobalResourcePlanner.from_config(cfg)
    planner.runtime_capabilities = CoreRuntimeCapabilities.from_engine(engine())
    monkeypatch.setenv('GRP_FORCE_TRAIN_GPUS', '8')
    monkeypatch.setenv('GRP_FORCE_ROLLOUT_TP_LIST', '4')
    decision = planner.plan_if_needed(step=5, config=cfg, batch=ROWS,
        runtime_metrics=RuntimeResourceMetrics(step=5, train_time_s=1, rollout_time_s=10))
    assert decision.candidate_plan.train_gpus == 4
    assert not decision.resource_should_reconfigure
    assert decision.should_reconfigure and decision.elastic_hybrid_signal.desired_replicas == 1
    assert decision.reason == 'elastic_hybrid_adjustment'


@pytest.mark.parametrize('initial_gpus', [4, 8])
def test_startup_grp_retains_freedom_to_choose_the_initial_physical_world(initial_gpus):
    planner = PreflightPlanner(config())
    assert planner.planner.runtime_capabilities is None
    planner.planner.optimizer.optimize = Mock(return_value=result(initial_gpus))
    selected = planner.run(ROWS)
    assert selected.applied and selected.planned_config.train_gpus == initial_gpus
    assert selected.planned_config.rollout_gpus == 12 - initial_gpus
    assert planner.planner.optimizer.optimize.call_args.kwargs['fixed_train_gpus'] == 0


@pytest.mark.parametrize('axis', ['pp', 'cp', 'ep'])
def test_live_core_rebuild_rejects_unsupported_model_axes(axis):
    candidate = as_plan(result(4))
    setattr(candidate.train_config, axis, 2)
    candidate.train_config.dp = 1
    candidate.train_config.tp = 2
    capability = CoreRuntimeCapabilities.from_engine(engine())
    assert capability.incompatibility(candidate) == f'core_{axis}_rebuild_unsupported'


def test_fixed_core_keeps_rollout_repartition_and_actual_ehp_enabled():
    cfg = config()
    planner = GlobalResourcePlanner.from_config(cfg)
    planner.runtime_capabilities = CoreRuntimeCapabilities.from_engine(engine())
    candidate = result(4)
    candidate.rollout_config = RolloutClusterConfig(tp_list=[4, 4])
    planner.optimizer.optimize = Mock(return_value=candidate)
    decision = planner.plan_if_needed(step=5, config=cfg, batch=ROWS)
    assert decision.resource_should_reconfigure
    assert decision.candidate_plan.train_config.tp == 4
    assert decision.candidate_plan.train_config.dp == 1
    assert decision.candidate_plan.rollout_tp_list == [4, 4]
    executor = RuntimeElasticExecutor(config=cfg, planner=planner, train_engine=engine())
    prepared = executor.prepare_transaction(decision)
    assert prepared.should_execute and prepared.resource_reconfiguration
    assert cfg.global_resource_planner.runtime_reconfigure_training
    assert cfg.global_resource_planner.runtime_training_pool_only
    assert not cfg.global_resource_planner.runtime_training_pool_plan_only
    assert cfg.global_resource_planner.hybrid_worker_launch_enabled


def test_fixed_core_rejects_tp_dp_change_even_if_adapter_exposes_old_rebuild_hook():
    capability = CoreRuntimeCapabilities.from_engine(engine())
    assert not capability.rebuild_tp_dp
    assert capability.incompatibility(as_plan(result(4, tp=2))) == "core_tp_dp_rebuild_unsupported"


def test_executor_rejects_stale_resize_before_any_generation_or_core_side_effect():
    cfg, backend = config(), engine()
    dispatcher = NS(pause=Mock(), wait_until_idle=Mock(), resume=Mock())
    planner = GlobalResourcePlanner.from_config(cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner, train_engine=backend, dispatcher=dispatcher)
    executor._adopt_existing_rollout_processes = Mock()
    decision = PlannerDecision(step=5, should_reconfigure=True, reason='stale_async_plan',
        candidate_plan=as_plan(result(8)), resource_should_reconfigure=True)
    prepared = executor.prepare_transaction(decision)
    assert not prepared.should_execute and not prepared.drain_required
    assert prepared.reason == 'runtime_candidate_rejected:physical_core_rank_set_changed'
    assert prepared.coord_id == ''
    executor._adopt_existing_rollout_processes.assert_not_called()
    dispatcher.pause.assert_not_called()
    dispatcher.wait_until_idle.assert_not_called()
    backend.reconfigure_parallelism.assert_not_called()


def test_executor_preserves_valid_ehp_signal_from_a_stale_unlaunchable_core_candidate():
    cfg = config()
    current = as_plan(result(4))
    signal = ElasticHybridSignal(step=5, desired_workers=1, current_workers=0,
        max_workers=1, action='join', reason='training_bottleneck',
        train_rollout_ratio=2, rollout_pressure=0, ttl_steps=20, replica_size_gpus=4)
    decision = PlannerDecision(step=5, should_reconfigure=True, reason='reconfigure',
        current_plan=current, candidate_plan=as_plan(result(8)),
        elastic_hybrid_signal=signal, resource_should_reconfigure=True)
    executor = RuntimeElasticExecutor(config=cfg, planner=GlobalResourcePlanner.from_config(cfg), train_engine=engine())
    prepared = executor.prepare_transaction(decision)
    assert prepared.should_execute and not prepared.resource_reconfiguration
    assert prepared.candidate_plan is current
    assert prepared.effective_planner_signal is signal


def test_training_backend_without_rebuild_keeps_its_actual_tp_dp():
    backend = engine()
    backend.reconfigure_parallelism = None
    capability = CoreRuntimeCapabilities.from_engine(backend)
    assert capability.incompatibility(as_plan(result(4))) == ''
    assert capability.incompatibility(as_plan(result(4, tp=2))) == 'core_tp_dp_rebuild_unsupported'


def test_native_trainer_installs_live_capabilities_without_a_config_freeze():
    from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = config()
    planner = trainer.config.global_resource_planner
    planner.runtime_coordinate_reconfiguration_ranks = False
    planner.runtime_async_planning = False
    trainer.is_main_process = True
    trainer.train_engine = engine()
    trainer.rollout_engine = trainer.dispatcher = None
    trainer._runtime_boundary_protocol = trainer._elastic_gradient_server = None
    trainer._elastic_membership_run_id = ''
    trainer._is_rollout_source = lambda: True
    trainer._init_global_resource_planner()
    assert trainer.global_resource_planner.runtime_capabilities.world_size == 4
    assert trainer.config.global_resource_planner.fixed_train_gpus == 0

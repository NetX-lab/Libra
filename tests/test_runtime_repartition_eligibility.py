"""Candidate TP layouts are checked before mutating runtime ownership."""

from copy import deepcopy

import pytest

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlan,
    GlobalResourcePlanner,
    PlannerDecision,
)
from RL_Framework.infra.cost_model.model import RolloutClusterConfig, TrainParallelConfig
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor


@pytest.mark.parametrize('initial,target,should_execute', [
    ([1, 1, 1], [1, 2], True),
    ([1, 2], [1, 1, 1], True),
    ([1, 1, 1], [1, 1, 1], False),
    ([1, 2], [1, 2], False),
])
def test_prepare_recognizes_candidate_layout_without_changing_live_resources(
        monkeypatch, initial, target, should_execute):
    config = AsyncRLConfig(model_path='/not-loaded', train_gpus=1, rollout_gpus=3,
        train_tp_size=1, train_dp_size=1, n_total_gpus=4)
    config.global_resource_planner.runtime_manage_rollout_processes = True
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f'seed{index}', host='shared',
            tp=1, gpus=[index + 1], port=8000 + index) for index in range(3)
    ]
    planner = GlobalResourcePlanner.from_config(config)

    def plan(degrees):
        return GlobalResourcePlan(train_config=TrainParallelConfig(tp=1, dp=1, b_micro=1),
            rollout_config=RolloutClusterConfig(tp_list=degrees),
            t_train=1., t_rollout=1., t_global=1., n_total_gpus=4,
            max_concurrent_rollouts=config.max_concurrent_rollouts)

    planner.apply_plan_to_config(plan(initial), config)
    decision = PlannerDecision(step=2, should_reconfigure=True, reason='test_repartition',
        current_plan=plan(initial), candidate_plan=plan(target), resource_should_reconfigure=True)
    executor = RuntimeElasticExecutor(config=config, planner=planner)
    executor._process_meta = {meta.instance_id: meta
        for meta in executor._desired_rollout_processes(decision.current_plan)}
    before_config = deepcopy(config.to_dict())
    before_owners = deepcopy(executor._process_meta)

    def forbidden(*args, **kwargs):
        pytest.fail('Eligibility must not launch, stop, wait for or reconfigure a process')

    for name in ('_start_one_rollout_process', '_stop_rollout_process',
                 '_reconfigure_rollout_engine', '_wait_rollout_process_start_ready'):
        monkeypatch.setattr(executor, name, forbidden)
    prepared = executor.prepare_transaction(decision)

    assert prepared.should_execute is should_execute
    assert prepared.resource_reconfiguration is True
    assert prepared.candidate_plan is decision.candidate_plan
    assert prepared.result.applied is False and not prepared.result.actions
    assert config.to_dict() == before_config
    assert executor._process_meta == before_owners and not executor._processes
    if not should_execute:
        assert prepared.reason == 'runtime_plan_already_applied'

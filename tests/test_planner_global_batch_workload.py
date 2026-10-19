"""Historical sample count must not change the cost of one fixed global batch."""

import pytest

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner
from RL_Framework.infra.cost_model.model import RequestInfo, representative_global_batch


ROWS = [
    dict(input_len=1024, output_len=length)
    for length in ([128] * 4 + [1024] * 4 + [4096] * 4 + [8192] * 4)
]


def planner():
    config = AsyncRLConfig(model_path="not-loaded")
    config.n_total_gpus = 12
    config.train_gpus, config.rollout_gpus = 4, 8
    config.train_tp_size, config.train_dp_size = 4, 1
    config.batch_size, config.micro_batch_size, config.n_samples = 16, 1, 4
    config.heterogeneous_rollout.enabled = True
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f"rollout-{index}", tp=4)
        for index in range(2)
    ]
    controls = config.global_resource_planner
    controls.enabled = True
    controls.initial_allocation_strategy = "grp"
    controls.fixed_train_gpus = 0
    controls.allowed_train_tp, controls.allowed_train_pp = [1, 2, 4], [1]
    controls.allowed_train_ep, controls.allowed_rollout_tp = [1], [1, 2, 4]
    controls.micro_batch_sizes = [1]
    controls.allocation_granularity_gpus = 4
    controls.min_train_gpus = controls.min_rollout_gpus = 4
    controls.warmup_steps = 0
    controls.plan_interval = controls.min_history_size = 1
    controls.reconfiguration_cost_s = controls.min_gain_ratio = 0
    return config, GlobalResourcePlanner.from_config(config)


def costs(plan):
    return (plan.t_train, plan.t_rollout, plan.t_global)


def layout(plan):
    return (plan.train_config, plan.rollout_tp_list)


def test_repeating_the_same_history_does_not_inflate_current_assignment_cost():
    config, evaluator = planner()
    sample = evaluator._extract_requests([dict(input_len=1024, output_len=2048)] * 16)
    original = evaluator._build_current_plan(config, sample)
    for repeats in (4, 16):
        repeated = evaluator._build_current_plan(config, sample * repeats)
        assert costs(repeated) == pytest.approx(costs(original))
        assert layout(repeated) == layout(original)


def test_real_candidate_search_is_invariant_to_duplicate_history():
    decisions = []
    for repeats in (1, 4, 16):
        config, evaluator = planner()
        decisions.append(evaluator.plan_if_needed(step=5, config=config, batch=ROWS * repeats))
    for compared in decisions[1:]:
        assert costs(compared.current_plan) == pytest.approx(costs(decisions[0].current_plan))
        assert costs(compared.candidate_plan) == pytest.approx(costs(decisions[0].candidate_plan))
        assert layout(compared.candidate_plan) == layout(decisions[0].candidate_plan)
        assert compared.should_reconfigure == decisions[0].should_reconfigure
    assert [decision.num_requests for decision in decisions] == [16, 64, 256]
    assert [decision.candidate_plan.metadata["workload"]["representative_requests"]
            for decision in decisions] == [16, 16, 16]


def test_direct_optimizer_call_cannot_price_all_historical_requests_as_one_step():
    _, evaluator = planner()
    requests = evaluator._extract_requests(ROWS)
    kwargs = dict(n_total_gpus=12, B_global=16, allowed_train_tp=[1, 2, 4],
        allowed_train_pp=[1], allowed_train_ep=[1], micro_batch_sizes=[1],
        allocation_granularity_gpus=4, min_train_gpus=4, min_rollout_gpus=4,
        dp_batch_group_size=4)
    base = evaluator.optimizer.optimize(requests=requests, **kwargs)
    repeated = evaluator.optimizer.optimize(requests=requests * 16, **kwargs)
    assert repeated.train_config == base.train_config
    assert repeated.rollout_config == base.rollout_config
    assert (repeated.t_train, repeated.t_rollout, repeated.t_global) == pytest.approx(
        (base.t_train, base.t_rollout, base.t_global))


def test_forced_rollout_and_training_candidates_use_the_same_global_batch(monkeypatch):
    config, evaluator = planner()
    monkeypatch.setenv("GRP_FORCE_TRAIN_GPUS", "8")
    monkeypatch.setenv("GRP_FORCE_ROLLOUT_TP_LIST", "4")
    requests, training = [], []
    original_rollout = evaluator.evaluator.evaluate_rollout
    original_training = evaluator.evaluator.evaluate_training

    def rollout(cluster, rows):
        requests.append(len(rows))
        return original_rollout(cluster, rows)

    def train(topology, batch_size, lengths):
        training.append((batch_size, len(lengths)))
        return original_training(topology, batch_size, lengths)

    monkeypatch.setattr(evaluator.evaluator, "evaluate_rollout", rollout)
    monkeypatch.setattr(evaluator.evaluator, "evaluate_training", train)
    decision = evaluator.plan_if_needed(step=5, config=config, batch=ROWS * 16)
    assert decision.candidate_plan.train_gpus == 8
    assert requests and set(requests) == {16}
    assert training and set(training) == {(16, 16)}


def test_representative_lengths_preserve_oom_tail_and_do_not_modify_history():
    requests = [RequestInfo(prompt_length=1, gen_length=index) for index in range(1, 65)]
    original = list(requests)
    selected = representative_global_batch(requests, 16)
    assert len(selected) == 16 and selected[0] is requests[0] and selected[-1] is requests[-1]
    assert requests == original
    assert representative_global_batch(requests, 1) == [requests[-1]]
    assert representative_global_batch([], 16) == []
    with pytest.raises(ValueError, match="positive"):
        representative_global_batch(requests, 0)


def test_short_history_is_a_distribution_for_a_full_global_batch():
    requests = [RequestInfo(prompt_length=1024, gen_length=2048)]
    assert len(representative_global_batch(requests, 16)) == 16
    config, evaluator = planner()
    assert costs(evaluator._build_current_plan(config, requests)) == pytest.approx(
        costs(evaluator._build_current_plan(config, requests * 16)))

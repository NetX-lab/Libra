"""Finite legal-group costs for elastic paths, including overlapping loading."""
import pytest
from RL_Framework.infra.cost_model import global_resource_planner as planner


def test_background_load_overlaps_core_work_without_a_terminal_restore():
    reservations = [dict(ready=True, deadline_revision=1) for _ in range(8)]
    seconds = planner.bounded_update_time(train_s=5, core_train_s=10, rollout_batch_s=1,
        groups_per_batch=4, horizon=4, max_age=1, revision=0,
        reservations=reservations, publish_s=1, background_ready_s=15)
    assert seconds == pytest.approx(34)


@pytest.mark.parametrize('active, want', [(0, 26), (1, 24)])
def test_publication_cost_switches_after_the_first_hybrid_update(active, want):
    # Initial publication follows the current training regime. A new Hybrid
    # activates after the first Core update; its first publication still uses
    # the Core cost, then later Hybrid updates use the measured Hybrid cost.
    seconds = planner.bounded_update_time(train_s=1, core_train_s=1 if active else 10,
        rollout_batch_s=100, groups_per_batch=1, horizon=3, max_age=3, revision=0,
        reservations=[dict(ready=True, deadline_revision=3)]*3,
        publish_s=2, hybrid_publish_s=7, active_hybrid_replicas=active,
        pending_activation=not active, activation_s=3)
    assert seconds == pytest.approx(want)


def test_next_deadline_keeps_a_position_in_the_current_update():
    reservations = [dict(deadline_revision=2, remaining_s=seconds) for seconds in (7, 9)]
    reservations.append(dict(deadline_revision=3, ready=True))
    assert planner.bounded_update_time(train_s=1, rollout_batch_s=20,
        groups_per_batch=1, horizon=1, max_age=2, revision=1,
        reservations=reservations, max_concurrent_rollouts=2) == pytest.approx(8)


def elastic_fixture():
    from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
    config = AsyncRLConfig(model_path='/tmp/model', batch_size=16, n_samples=4,
        train_gpus=2,train_tp_size=2,train_dp_size=1,rollout_gpus=6,n_total_gpus=8)
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id='tp2',tp=2,gpus=[2,3]),
        HeterogeneousInstanceConfig(instance_id='tp4',tp=4,gpus=[0,1,2,3],host='other')]
    config.global_resource_planner.elastic_hybrid_planning_enabled = True
    config.global_resource_planner.elastic_hybrid_min_rollout_gpus = 2
    config.profiling.calibration = {'training': {'compute_scale':1000},
        'rollout': {str(tp): {'prefill_scale': .0001,'decode_scale': .0001} for tp in (2,4)},
        'horizon_updates': 10,'error_margin_s':0,
        'lifecycle':dict(borrow_stop_s=.1,load_s=.1,activate_s=.1,return_frontend_s=.1,publish_s=.1)}
    evaluator = planner.GlobalResourcePlanner.from_config(config)
    evaluator.observe_batch([dict(input_len=1024,output_len=128)]*16)
    current = evaluator._build_current_plan(config,evaluator._history)
    return config,evaluator,current


def test_inventory_and_busy_ratios_do_not_choose_an_unsustainable_loan():
    from RL_Framework.config import HeterogeneousInstanceConfig
    config,evaluator,current = elastic_fixture()
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f'tp2-{index}',tp=2,gpus=[0,1],host=f'node-{index}')
        for index in range(3)]
    config.profiling.calibration['training']['compute_scale'] = .0001
    config.profiling.calibration['rollout'] = {str(tp): {'prefill_scale': 1000,'decode_scale': 1000}
        for tp in (2,4)}
    evaluator = planner.GlobalResourcePlanner.from_config(config)
    evaluator.observe_batch([dict(input_len=1024,output_len=128)]*16)
    metrics = planner.RuntimeResourceMetrics(train_time_s=1000,rollout_time_s=.001,
        raw={'step_stats': {'remaining_updates':100},'dispatcher':{
            'group_reservations':[dict(ready=True,deadline_revision=1)]*4}})
    signal = evaluator._build_elastic_hybrid_signal(step=0,config=config,current=current,
        candidate=current,runtime_metrics=metrics)
    assert signal.desired_workers == 0
    assert signal.reason == 'calibrated_hold'


def test_error_margin_keeps_current_intent_and_horizon_ends_at_remaining_updates():
    config,evaluator,current = elastic_fixture()
    metrics = planner.RuntimeResourceMetrics(raw={'step_stats': {'remaining_updates':3}})
    signal = evaluator._build_elastic_hybrid_signal(step=0,config=config,current=current,
        candidate=current,runtime_metrics=metrics)
    assert signal.desired_workers > 0
    assert signal.horizon_updates == 3
    config.profiling.calibration['error_margin_s'] = 1e9
    signal = evaluator._build_elastic_hybrid_signal(step=0,config=config,current=current,
        candidate=current,runtime_metrics=metrics)
    assert signal.desired_workers == 0


@pytest.mark.parametrize('width', [2, 4])
def test_new_candidates_do_not_borrow_services_without_an_extra_replica(width, monkeypatch):
    from RL_Framework.config import HeterogeneousInstanceConfig
    config, evaluator, _ = elastic_fixture()
    config.train_tp_size = width
    config.train_gpus = width
    config.n_total_gpus = 4 * width
    config.rollout_gpus = 3 * width
    config.global_resource_planner.elastic_hybrid_min_rollout_gpus = width
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f'{host}-{gpu}', tp=1, gpus=[gpu], host=host)
        for host in ('a', 'b', 'c') for gpu in range(width)]
    current = evaluator._build_current_plan(config, evaluator._history)
    forecasts = []
    monkeypatch.setattr(planner, 'bounded_update_time', lambda **args: forecasts.append(args) or 1.)
    evaluator._build_elastic_hybrid_signal(step=0, config=config, current=current,
        candidate=current, runtime_metrics=planner.RuntimeResourceMetrics(
            raw={'step_stats': {'remaining_updates': 8}}))
    # Keep, any one complete host, or any two complete hosts. Borrowing a
    # partial extra host cannot create another complete replica.
    assert len(forecasts) == 7


def test_live_elastic_planning_does_not_run_the_global_layout_search(monkeypatch):
    config,evaluator,_ = elastic_fixture()
    evaluator.runtime_capabilities = planner.CoreRuntimeCapabilities(
        world_size=2,tp=2,ep=1,pp=1,dp=1,cp=1)
    evaluator.warmup_steps = 0
    evaluator.plan_interval = 1
    evaluator.min_history_size = 1
    def invalid_search(**kwargs):
        raise AssertionError('global layout search is invalid on the live Core')
    monkeypatch.setattr(evaluator.optimizer,'optimize',invalid_search)
    decision = evaluator.plan_if_needed(step=1,config=config)
    assert decision.resource_should_reconfigure is False
    assert decision.candidate_plan is decision.current_plan


def test_post_recheck_reuses_static_supply_with_fresh_reservations(monkeypatch):
    config, evaluator, _ = elastic_fixture()
    evaluator.runtime_capabilities = planner.CoreRuntimeCapabilities(
        world_size=2, tp=2, ep=1, pp=1, dp=1, cp=1)
    evaluator.warmup_steps = 0
    evaluator.plan_interval = 1
    evaluator.min_history_size = 1
    evaluator.replan_cooldown_steps = 0
    layouts = []
    evaluate_supply = evaluator._fixed_rollout_supply

    def track_supply(tp_list, requests):
        layouts.append(tuple(sorted(tp_list)))
        return evaluate_supply(tp_list, requests)

    monkeypatch.setattr(evaluator, '_fixed_rollout_supply', track_supply)
    windows = []
    forecast = planner.bounded_update_time

    def track_forecast(**inputs):
        windows.append(inputs['reservations'])
        return forecast(**inputs)

    monkeypatch.setattr(planner, 'bounded_update_time', track_forecast)
    decision = evaluator.plan_if_needed(step=1, config=config)
    assert sorted(layouts) == [(2,), (2, 4), (4,)]
    assert windows and all(not reservations for reservations in windows)

    reservations = [dict(deadline_revision=2, ready=True)]
    evaluator.recheck_elastic_signal(step=1, config=config,
        runtime_metrics=planner.RuntimeResourceMetrics(raw={
            'dispatcher': {'optimizer_revision': 2, 'group_reservations': reservations}}),
        decision=decision)
    assert sorted(layouts) == [(2,), (2, 4), (4,)]
    assert windows[-1] == reservations

    decision = evaluator.plan_if_needed(step=2, config=config)
    assert sorted(layouts) == [(2,), (2,), (2, 4), (2, 4), (4,), (4,)]

    evaluator.replace_history([dict(input_len=1024, output_len=512)] * 16)
    evaluator.recheck_elastic_signal(step=2, config=config, decision=decision)
    assert sorted(layouts) == [(2,), (2,), (2,), (2, 4), (2, 4), (2, 4), (4,), (4,), (4,)]


def test_natural_post_defers_a_legacy_all_lent_layout_to_step_start_recovery():
    config, evaluator, _ = elastic_fixture()
    runtime_metrics = planner.RuntimeResourceMetrics(raw={
        'step_stats': dict(remaining_updates=3, loan_instance_ids=['tp2', 'tp4'],
                           active_hybrid_replicas=3),
        'dispatcher': {'optimizer_revision': 2,
            'group_reservations': [dict(ready=True, deadline_revision=2)] * 4},
    })
    assert evaluator.recheck_elastic_signal(step=1, config=config, runtime_metrics=runtime_metrics) is None


def test_request_profile_refresh_keeps_the_actual_full_training_sequence_lengths():
    from types import SimpleNamespace
    config,evaluator,_ = elastic_fixture()
    count = evaluator.observe_training_batch([
        dict(input_ids=SimpleNamespace(shape=(1,4096))),
        dict(input_ids=SimpleNamespace(shape=(1,8192)))])
    assert count == 2
    evaluator.replace_history([dict(input_len=128,output_len=64)]*16)
    lengths = evaluator._training_workload_lengths(evaluator._history, 16)
    assert lengths == [4096]*8 + [8192]*8


def test_six_turn_trace_supply_counts_six_generations_but_keeps_single_call_cost():
    from types import SimpleNamespace
    config,evaluator,current = elastic_fixture()
    metrics = planner.RuntimeResourceMetrics(raw={'step_stats': {'remaining_updates':3}})
    single = evaluator._complete_trace_rollout_seconds(config,[2,4],evaluator._history)
    operator = evaluator.evaluator.analytic.rollout_model.compute_decode_step_time(1024,2)
    evaluator.observe_training_batch([dict(input_ids=SimpleNamespace(shape=(1,8192)),n_turns=6)]*16)
    multi = evaluator._complete_trace_rollout_seconds(config,[2,4],evaluator._history)
    assert multi == pytest.approx(6 * single)
    assert evaluator.evaluator.analytic.rollout_model.compute_decode_step_time(1024,2) == operator


def test_initial_search_prices_full_sequences_and_complete_multi_turn_supply():
    from RL_Framework.infra.cost_model.model import RequestInfo
    _,evaluator,_ = elastic_fixture()
    requests = [RequestInfo(128,64)]*16
    result = evaluator.optimizer.optimize(n_total_gpus=8,requests=requests,B_global=16,
        fixed_train_gpus=2,allowed_train_tp=[2],allowed_train_ep=[1],allowed_train_pp=[1],
        micro_batch_sizes=[1],training_sequence_lengths=[6144]*16,rollout_calls_per_trace=6)
    expected_training = evaluator.evaluator.evaluate_training(result.train_config,16,[6144]*16)[0]
    single_calls = evaluator.optimizer.evaluate_fixed_rollout_config(result.rollout_config,requests)[0]
    assert result.t_train == pytest.approx(expected_training)
    assert result.t_rollout == pytest.approx(6 * single_calls)


def test_publication_stops_at_the_current_partial_call_and_does_not_charge_tool_work_as_generation():
    # One second of a 2s call runs during borrow. Publication may finish that
    # one second, but cannot start the next call or consume the 3s tool tail.
    seconds = planner.bounded_update_time(train_s=1, rollout_batch_s=4,
        group_tail_s=3, rollout_call_s=2, groups_per_batch=1,
        horizon=1, max_age=0, revision=0,
        reservations=[dict(ready=False, remaining_s=7, deadline_revision=0,
                           members_running=1)],
        initial_s=1, publish_s=7, publish_inflight_progress_s=5)
    assert seconds == pytest.approx(14)


@pytest.mark.parametrize('concurrency,want', [(2, 86), (8, 30)])
def test_execution_slots_do_not_expand_to_the_reservation_window(concurrency, want):
    # Four groups of two traces are reserved. Each unfinished trace needs 20s.
    # Two execution slots take 80s to finish all four groups; eight take 20s.
    reservations = [dict(ready=False, deadline_revision=1, members_total=2,
                         members_ready=0, members_running=0) for _ in range(4)]
    seconds = planner.bounded_update_time(train_s=4, rollout_batch_s=20,
        groups_per_batch=2, horizon=2, max_age=1, revision=0,
        reservations=reservations, publish_s=1, group_size=2,
        max_concurrent_rollouts=concurrency)
    assert seconds == pytest.approx(want)


def test_elapsed_loading_time_does_not_authorize_activation_at_the_current_post():
    # A cold-load estimate of zero does not supply the missing actual READY.
    seconds = planner.bounded_update_time(train_s=5, core_train_s=10,
        rollout_batch_s=1, groups_per_batch=1, horizon=1, max_age=0, revision=0,
        reservations=[dict(ready=True, deadline_revision=0)], publish_s=1,
        pending_activation=True, activation_s=7, background_ready_s=0,
        activation_ready_replicas=0)
    assert seconds == pytest.approx(11)


def test_history_uses_recorded_tool_positions_for_generate_contexts():
    _, evaluator, _ = elastic_fixture()
    requests = evaluator._extract_requests([dict(input_len=100, output_len=300,
        total_output_tokens=600, sequence_length=700, n_turns=3,
        tool_returns=[dict(turn=0, token_position=200), dict(turn=1, token_position=500)])])
    # Actual generation prompts have lengths 100, 300, 600; model output is
    # 300/3=100, while the other 300 context tokens are tool output.
    assert [(r.prompt_length, r.gen_length, r.training_length) for r in requests] == [(333,100,700)]


def test_legacy_history_without_turn_count_retains_recorded_workload():
    _, evaluator, _ = elastic_fixture()
    rows = [{
        "input_len": 100,
        "output_len": 300,
        "total_output_tokens": 600,
        "tool_returns": [{"turn": 0, "token_position": 250, "payload_tokens": 150}],
        "profile_source": "runtime_online",
    }]
    assert evaluator.replace_history(rows) == 1
    request = evaluator._history[0]
    assert (request.prompt_length, request.gen_length) == (100, 300)
    assert evaluator._rollout_calls_per_trace(evaluator.evaluator.config) == 1


def test_rollout_shortfall_releases_an_inflight_join():
    config, _, _ = elastic_fixture()
    config.profiling.calibration["training"]["compute_scale"] = 0.0001
    config.profiling.calibration["rollout"]["4"] = {"prefill_scale": 1000, "decode_scale": 1000}
    evaluator = planner.GlobalResourcePlanner.from_config(config)
    evaluator.observe_batch([{"input_len": 1024, "output_len": 128}] * 16)
    metrics = planner.RuntimeResourceMetrics(raw={"step_stats": {
        "remaining_updates": 4,
        "loan_instance_ids": ["tp2"],
        "joining_hybrid_replicas": 1,
        "joining_remaining_s": 1000,
    }})
    signal = evaluator.recheck_elastic_signal(0, config, runtime_metrics=metrics)
    assert signal.action == "release"
    assert signal.current_loan_instance_ids == ["tp2"]
    assert signal.loan_instance_ids == []
    assert signal.joining_workers == 1


def test_same_host_small_services_can_form_a_beneficial_hybrid():
    from RL_Framework.config import HeterogeneousInstanceConfig
    config, _, _ = elastic_fixture()
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id='small-a', tp=1, gpus=[2], host='core'),
        HeterogeneousInstanceConfig(instance_id='small-b', tp=1, gpus=[3], host='core'),
        HeterogeneousInstanceConfig(instance_id='large', tp=4, gpus=[0,1,2,3], host='rollout')]
    config.profiling.calibration['rollout'] = {
        '1':dict(prefill_scale=1000,decode_scale=1000),
        '4':dict(prefill_scale=.0001,decode_scale=.0001)}
    evaluator = planner.GlobalResourcePlanner.from_config(config)
    evaluator.observe_batch([dict(input_len=1024,output_len=128)]*16)
    current = evaluator._build_current_plan(config, evaluator._history)
    metrics = planner.RuntimeResourceMetrics(raw={'step_stats':{'remaining_updates':100},
        'dispatcher':{'group_reservations':[dict(ready=True,deadline_revision=1)]*4}})
    signal = evaluator._build_elastic_hybrid_signal(step=0,config=config,current=current,
        candidate=current,runtime_metrics=metrics)
    assert signal.loan_instance_ids == ['small-a','small-b']
    assert signal.desired_replicas == 1


def test_target_generate_load_changes_borrow_cost_without_counting_trace_calls(monkeypatch):
    from RL_Framework.config import HeterogeneousInstanceConfig
    config, evaluator, _ = elastic_fixture()
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=f'tp2-{index}', tp=2,
            gpus=[0, 1], host=f'node-{index}') for index in range(3)]
    config.global_resource_planner.elastic_hybrid_min_rollout_gpus = 4
    config.profiling.calibration.update(calls_per_trace=6)
    config.profiling.calibration['lifecycle'] = dict(borrow_stop_s=2,
        borrow_snapshot_s=7, borrow_coordination_s=3, load_s=0,
        activate_s=0, return_frontend_s=10000, publish_s=0)
    evaluator = planner.GlobalResourcePlanner.from_config(config)
    evaluator.observe_batch([dict(input_len=1024, output_len=128)] * 16)
    # A cohort of n current calls needs n*3s on this representative operator.
    # Six calls per complete trace must not multiply this target drain.
    monkeypatch.setattr(evaluator.evaluator, 'evaluate_rollout_instance',
        lambda tp, requests: (3. * len(requests), {}))
    current = evaluator._build_current_plan(config, evaluator._history)
    stats = dict(remaining_updates=2)
    metrics = planner.RuntimeResourceMetrics(raw={'step_stats': stats, 'dispatcher': {
        'optimizer_revision': 1,
        'group_reservations': [dict(ready=True, deadline_revision=2)] * 8}})
    empty = evaluator._build_elastic_hybrid_signal(step=0, config=config,
        current=current, candidate=current, runtime_metrics=metrics)
    stats['generate_reservations_by_instance'] = {f'tp2-{index}': 2 for index in range(3)}
    busy = evaluator._build_elastic_hybrid_signal(step=0, config=config,
        current=current, candidate=current, runtime_metrics=metrics)
    assert empty.loan_instance_ids == busy.loan_instance_ids == ['tp2-0']
    assert empty.predicted_savings_s - busy.predicted_savings_s == pytest.approx(6)
    # Keeping an active loan pays none of its already incurred start cost.
    stats.update(loan_instance_ids=['tp2-0'], active_hybrid_replicas=1)
    held = evaluator._build_elastic_hybrid_signal(step=0, config=config,
        current=current, candidate=current, runtime_metrics=metrics)
    assert held.action == 'hold'


@pytest.mark.parametrize('active, ready, expected_ready_times', [
    (1, 0, None),
    (0, 1, [0., 26.]),
    (0, 0, [5., 26.]),
])
def test_expansion_prices_only_added_service_and_preserves_existing_replicas(
    monkeypatch, active, ready, expected_ready_times,
):
    from RL_Framework.config import HeterogeneousInstanceConfig

    config, evaluator, _ = elastic_fixture()
    config.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(instance_id=name, tp=2, gpus=[0, 1], host=name)
        for name in ('a', 'b', 'c')]
    current = evaluator._build_current_plan(config, evaluator._history)
    config.profiling.calibration['lifecycle'] = dict(borrow_stop_s=2,
        borrow_snapshot_s=7, borrow_coordination_s=3, return_frontend_s=11,
        load_s=8, activate_s=0, publish_s=0)
    monkeypatch.setattr(evaluator.evaluator, 'evaluate_rollout_instance',
        lambda tp, requests: (3. * len(requests), {}))
    forecasts = []
    forecast = planner.bounded_update_time

    def track_forecast(**inputs):
        forecasts.append(inputs)
        return forecast(**inputs)

    monkeypatch.setattr(planner, 'bounded_update_time', track_forecast)
    evaluator._build_elastic_hybrid_signal(step=0, config=config,
        current=current, candidate=current, runtime_metrics=planner.RuntimeResourceMetrics(
            raw={'step_stats': dict(remaining_updates=3, loan_instance_ids=['a'],
                active_hybrid_replicas=active, joining_hybrid_replicas=1 - active,
                joining_remaining_s=5., activation_ready_replicas=ready,
                generate_reservations_by_instance={'a': 9, 'b': 2})}))
    keep, expand_ab = forecasts[1], forecasts[4]
    assert expand_ab['initial_s'] == 18  # B drains 6s and stops in 2s; capture + coordination 10s.
    assert expand_ab['background_ready_s'] == 26
    assert expand_ab['active_hybrid_replicas'] == active
    assert expand_ab['activation_ready_replicas'] == ready
    assert expand_ab['core_train_s'] == keep['core_train_s']
    assert len(expand_ab['activation_train_s']) == 2 - active
    assert expand_ab['activation_ready_times_s'] == expected_ready_times

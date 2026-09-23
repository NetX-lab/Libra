"""CPU-only boundary protocol and real trainer/executor control-flow tests."""

import copy
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.elastic import runtime_boundary as boundary_module
from RL_Framework.infra.elastic.runtime_boundary import RuntimeBoundaryProtocol, read_record, write_record
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Planner:
    warmup_steps = 100
    history_size = 8
    min_history_size = 1

    def __init__(self):
        self.applied = 0

    def observe_batch(self, batch):
        return len(batch)

    def observe_runtime(self, **kwargs):
        return SimpleNamespace(to_dict=lambda: {})

    def apply_plan_to_config(self, plan, config):
        self.applied += 1


class Decision:
    def __init__(self, reconfigure=False):
        self.should_reconfigure = reconfigure
        self.reason = "reconfigure" if reconfigure else "same_plan"
        self.num_requests = 8
        self.elastic_hybrid_signal = None
        self.candidate_plan = SimpleNamespace(
            train_config=SimpleNamespace(tp=1, pp=1, dp=2), train_gpus=2,
            rollout_tp_list=[2], t_global=1.0, expected_gain_s=1.0,
            to_dict=lambda: {"rollout": {"tp_list": [2]}},
        )

    def to_dict(self):
        return {"reason": self.reason}


def protocol(path, rank, run_id="test-run", timeout=0.1):
    result = RuntimeBoundaryProtocol(path, rank, 2, timeout)
    result.run_id = run_id
    return result


@pytest.fixture
def trainers(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("RL_TRAIN_PHASE_TRACE", raising=False)
    config = AsyncRLConfig(
        model_path="/unused", train_gpus=2, rollout_gpus=2, n_total_gpus=4,
        train_tp_size=1, train_dp_size=2, batch_size=2, n_samples=1, micro_batch_size=1,
        log_dir=str(tmp_path),
    )
    cfg = config.global_resource_planner
    cfg.enabled = True
    cfg.runtime_drain_timeout_s = 2.0
    cfg.runtime_length_profile_enabled = False
    cfg.runtime_coordinate_batch_source_only = False
    leader = AsyncRLTrainer(config)
    peer = AsyncRLTrainer(copy.deepcopy(config))
    peer.rank = 1
    peer.is_main_process = False
    directory = leader._runtime_reconfiguration_coord_dir()
    for trainer in (leader, peer):
        trainer._runtime_boundary_protocol = protocol(directory, trainer.rank, timeout=2.0)
        trainer._runtime_length_profile_path = lambda: ""
        trainer._rebind_rollout_engine_from_config = lambda **kwargs: None
        trainer.applied_states = []
        trainer._apply_runtime_reconfiguration_state = trainer.applied_states.append
    leader.global_resource_planner = Planner()
    executor = RuntimeElasticExecutor(config=config, planner=leader.global_resource_planner)
    executor.runtime_run_id = "test-run"
    leader.runtime_elastic_executor = executor
    return leader, peer, executor


def complete(leader, decision, step=0):
    future = Future()
    future.set_result(decision)
    leader._grp_future = future
    leader._grp_future_step = step
    return future


def test_pre_noop_does_not_consume_completed_future(trainers, monkeypatch):
    leader, peer, _ = trainers
    future = complete(leader, Decision(True))
    clock = Clock()
    monkeypatch.setattr(boundary_module, "time", clock)
    leader._runtime_reconfiguration_boundary(0, "pre")
    peer._runtime_reconfiguration_boundary(0, "pre")
    assert leader._grp_future is future
    assert not clock.sleeps


@pytest.mark.parametrize("mode", ["no_planner", "warmup", "dynamic_disabled", "completed_noop"])
def test_post_noop_without_request_polling(trainers, monkeypatch, mode):
    leader, peer, executor = trainers
    clock = Clock()
    monkeypatch.setattr(boundary_module, "time", clock)
    if mode == "no_planner":
        leader.global_resource_planner = None
    elif mode == "dynamic_disabled":
        leader.config.global_resource_planner.runtime_dynamic_reconfiguration_enabled = False
        complete(leader, Decision(True))
    elif mode == "completed_noop":
        complete(leader, Decision())
    leader._run_global_resource_planner_at_boundary(0, [], {})
    peer._runtime_reconfiguration_boundary(0, "post")
    assert peer._runtime_boundary_protocol.wait(0, "post")["decision"] == "no_reconfig"
    assert not clock.sleeps
    assert not (executor._coordination_dir() / "request.json").exists()


def test_post_is_explicitly_closed_when_planner_path_raises(trainers, monkeypatch):
    leader, peer, _ = trainers

    def fail(*args, **kwargs):
        raise RuntimeError("planner path failed")

    monkeypatch.setattr(leader, "_run_global_resource_planner", fail)
    with pytest.raises(RuntimeError, match="planner path failed"):
        leader._run_global_resource_planner_at_boundary(0, [], {})
    assert peer._runtime_reconfiguration_boundary(0, "post") is None
    assert peer._runtime_boundary_protocol.wait(0, "post")["decision"] == "no_reconfig"


def test_async_future_survives_boundaries_and_is_consumed_once(trainers, monkeypatch):
    leader, peer, _ = trainers
    future = Future()
    leader._grp_future = future
    leader._grp_future_step = 0
    calls = []
    original = leader._apply_global_resource_planner_decision
    def apply(step, decision, stats):
        calls.append(step)
        return original(step, decision, stats)
    monkeypatch.setattr(leader, "_apply_global_resource_planner_decision", apply)
    clock = Clock()
    monkeypatch.setattr(boundary_module, "time", clock)
    for step in range(4):
        leader._runtime_reconfiguration_boundary(step, "pre")
        peer._runtime_reconfiguration_boundary(step, "pre")
        leader._run_global_resource_planner_at_boundary(step, [], {})
        peer._runtime_reconfiguration_boundary(step, "post")
        assert leader._grp_future is future
        assert not future.cancelled()
    future.set_result(Decision())
    leader._run_global_resource_planner_at_boundary(4, [], {})
    peer._runtime_reconfiguration_boundary(4, "post")
    leader._run_global_resource_planner_at_boundary(5, [], {})
    assert leader._grp_future is None
    assert calls == [0]
    assert not clock.sleeps


def test_draining_request_precedes_decision_and_ready(trainers, monkeypatch):
    leader, peer, executor = trainers
    complete(leader, Decision(True))
    root = executor._coordination_dir()
    seen = []
    original = leader._runtime_boundary_protocol.publish
    def publish(step, point, **kwargs):
        request = read_record(root / "request.json")
        assert request["coord_id"] == kwargs["coord_id"]
        assert not (root / "ready/rank_1.json").exists()
        seen.append(request["coord_id"])
        return original(step, point, **kwargs)
    monkeypatch.setattr(leader._runtime_boundary_protocol, "publish", publish)
    original_reconcile = executor._reconcile_pending_hybrid_joins

    def reconcile(result=None):
        if seen:
            ready = read_record(root / "ready/rank_1.json")
            assert ready.get("coord_id") == seen[0]
        return original_reconcile(result)

    monkeypatch.setattr(executor, "_reconcile_pending_hybrid_joins", reconcile)
    with ThreadPoolExecutor(max_workers=1) as pool:
        following = pool.submit(peer._runtime_reconfiguration_boundary, 0, "post")
        leader._run_global_resource_planner_at_boundary(0, [], {})
        following.result(timeout=3)
    assert len(seen) == 1
    for name in ["request.json", "ready/rank_1.json", "applied.json"]:
        state = read_record(root / name)
        assert state["coord_id"] == seen[0]
        assert state["run_id"] == "test-run"
    assert peer.applied_states[0]["coord_id"] == seen[0]
    assert executor.planner.applied == 1


def test_nondraining_decision_precedes_reset_and_follow_does_not_wait(trainers):
    leader, peer, executor = trainers
    leader.config.global_resource_planner.runtime_drain_before_reconfigure = False
    leader.config.global_resource_planner.runtime_cluster_swap_enabled = True
    # Exercise the real trainer pre-reset path without cluster/GPU operations.
    executor._cluster_swap_enabled = lambda strategy: False
    root = executor._coordination_dir()
    events = []
    class Dispatcher:
        def pause(self):
            decision = peer._runtime_boundary_protocol.wait(0, "post")
            assert decision["decision"] == "reconfig"
            assert decision["drain_required"] is False
            assert not (root / "applied.json").exists()
            peer._runtime_reconfiguration_boundary(0, "post")
            events.append("peer_returned_before_pause")
        def wait_until_idle(self, **kwargs):
            events.append("drain_local")
        def reset_after_reconfigure(self):
            events.append("reset")
        def resume(self):
            events.append("resume")
    leader.dispatcher = Dispatcher()
    complete(leader, Decision(True))
    leader._run_global_resource_planner_at_boundary(0, [], {})
    assert events[0] == "peer_returned_before_pause"
    assert not (root / "request.json").exists()
    assert not (root / "ready").exists()
    assert not peer.applied_states
    decision = peer._runtime_boundary_protocol.wait(0, "post")
    peer._follow_applied_runtime_reconfiguration_if_available(root)
    assert peer.applied_states[0]["coord_id"] == decision["coord_id"]


def test_nondraining_transactions_have_fresh_ids(trainers):
    leader, peer, executor = trainers
    leader.config.global_resource_planner.runtime_drain_before_reconfigure = False
    ids = []
    for step in range(2):
        complete(leader, Decision(True), step)
        leader._run_global_resource_planner_at_boundary(step, [], {})
        peer._runtime_reconfiguration_boundary(step, "post")
        ids.append(peer._runtime_boundary_protocol.wait(step, "post")["coord_id"])
    assert ids[0] != ids[1]
    assert read_record(executor._coordination_dir() / "applied.json")["coord_id"] == ids[1]


def test_follower_arrives_before_decision(tmp_path, monkeypatch):
    leader = protocol(tmp_path, 0)
    peer = protocol(tmp_path, 1)
    clock = Clock()
    def sleep(seconds):
        clock.now += seconds
        leader.publish(0, "post")
    clock.sleep = sleep
    monkeypatch.setattr(boundary_module, "time", clock)
    assert peer.wait(0, "post")["decision"] == "no_reconfig"
    assert clock.now == 0.05


@pytest.mark.parametrize("stale", ["step", "run", "point", "missing"])
def test_stale_or_missing_decision_is_protocol_timeout(tmp_path, monkeypatch, stale):
    peer = protocol(tmp_path, 1)
    value = {"run_id": "test-run", "step": 1, "point": "post", "decision": "no_reconfig"}
    if stale != "missing":
        value[{"step": "step", "run": "run_id", "point": "point"}[stale]] = {
            "step": 0, "run": "old-run", "point": "pre",
        }[stale]
        write_record(peer.decision_path(1, "post"), value)
    clock = Clock()
    monkeypatch.setattr(boundary_module, "time", clock)
    with pytest.raises(TimeoutError, match="boundary decision missing"):
        peer.wait(1, "post")
    assert clock.now < 1


def test_closed_boundary_cannot_be_rewritten(tmp_path):
    leader = protocol(tmp_path, 0)
    leader.publish(0, "post")
    with pytest.raises(RuntimeError, match="already closed"):
        leader.publish(0, "post", coord_id="late", drain_required=True)


def test_session_reuse_rejects_old_membership(tmp_path):
    def join_run():
        peers = [RuntimeBoundaryProtocol(tmp_path, rank, 8, 3) for rank in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            # Followers may encounter the previous run's announcement first.
            tasks = [pool.submit(peer.join) for peer in reversed(peers)]
            for task in tasks:
                task.result(timeout=4)
        ids = {peer.run_id for peer in peers}
        assert len(ids) == 1
        return ids.pop()
    assert join_run() != join_run()


def test_stale_session_without_live_leader_cannot_join(tmp_path, monkeypatch):
    write_record(tmp_path / "session/leader.json", {"run_id": "old", "world_size": 2})
    write_record(tmp_path / "session/old/members.json", {"0": "old0", "1": "old1"})
    clock = Clock()
    monkeypatch.setattr(boundary_module, "time", clock)
    with pytest.raises(TimeoutError, match="session handshake"):
        RuntimeBoundaryProtocol(tmp_path, 1, 2, 0.1).join()


def test_request_publication_failure_propagates_without_ready_wait(trainers, monkeypatch):
    leader, _, executor = trainers
    complete(leader, Decision(True))
    def fail(*args, **kwargs):
        raise OSError("boundary publication failed")
    monkeypatch.setattr(leader._runtime_boundary_protocol, "publish", fail)
    with pytest.raises(OSError, match="boundary publication failed"):
        leader._run_global_resource_planner_at_boundary(0, [], {})
    assert read_record(executor._coordination_dir() / "aborted.json")["coord_id"] == executor._active_runtime_coord_id
    assert not (executor._coordination_dir() / "ready/rank_1.json").exists()


def test_cluster_swap_coordinates_peers_once(trainers, monkeypatch):
    _, _, executor = trainers
    cfg = executor.config.global_resource_planner
    cfg.runtime_cluster_swap_enabled = True
    cfg.runtime_manage_rollout_processes = True
    calls = []
    executor.dispatcher = SimpleNamespace(pause=lambda: None, resume=lambda: None)
    monkeypatch.setattr(executor, "_adopt_existing_rollout_processes", lambda: None)
    monkeypatch.setattr(executor, "_runtime_plan_already_applied", lambda *a, **k: False)
    monkeypatch.setattr(executor, "_cluster_swap_rollout_training_pools", lambda *a: None)
    monkeypatch.setattr(executor, "_reconfigure_rollout_processes", lambda *a, **k: ([], []))
    original = executor._coordinate_peer_rank_drain
    def coordinate(*args):
        calls.append(executor._current_coordination_id(Decision(True).candidate_plan))
        return original(*args)
    def acknowledge(coord_id):
        root = executor._coordination_dir()
        assert read_record(root / "request.json")["coord_id"] == coord_id
        import os
        write_record(root / "ready/rank_1.json", {
            "coord_id": coord_id, "run_id": "test-run",
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
        })
    monkeypatch.setattr(executor, "_coordinate_peer_rank_drain", coordinate)
    result = executor.execute(Decision(True), request_published=acknowledge)
    assert len(calls) == 1
    assert result.actions.count("peer_reconfig_request") == 1


def test_old_applied_state_is_not_consumed_by_new_run(trainers):
    _, peer, executor = trainers
    root = executor._coordination_dir()
    write_record(root / "applied.json", {"run_id": "old-run", "coord_id": "old", "instances": []})
    peer._follow_applied_runtime_reconfiguration_if_available(root)
    assert not peer.applied_states


def test_draining_abort_matches_the_announced_transaction(trainers, monkeypatch):
    leader, peer, executor = trainers
    complete(leader, Decision(True))
    executor.rollout_engine = SimpleNamespace(reconfigure_from_plan=lambda *a: None)
    def fail(*args, **kwargs):
        raise RuntimeError("simulated rollout reconfiguration failure")
    monkeypatch.setattr(executor, "_reconfigure_rollout_engine", fail)
    with ThreadPoolExecutor(max_workers=1) as pool:
        following = pool.submit(peer._runtime_reconfiguration_boundary, 0, "post")
        with pytest.raises(RuntimeError, match="simulated rollout"):
            leader._run_global_resource_planner_at_boundary(0, [], {})
        with pytest.raises(RuntimeError, match="aborted on rank0"):
            following.result(timeout=3)
    boundary = peer._runtime_boundary_protocol.wait(0, "post")
    aborted = read_record(executor._coordination_dir() / "aborted.json")
    assert aborted["coord_id"] == boundary["coord_id"]
    assert aborted["run_id"] == boundary["run_id"]


def test_old_terminal_states_do_not_complete_current_request(trainers, monkeypatch):
    _, peer, executor = trainers
    root = executor._coordination_dir()
    import os
    current = {"run_id": "test-run", "coord_id": "current",
               "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", ""))}
    write_record(root / "request.json", current)
    write_record(root / "applied.json", {**current, "coord_id": "older"})
    write_record(root / "aborted.json", {**current, "coord_id": "older", "error": "old error"})
    clock = Clock()
    from RL_Framework.trainer import async_rl_trainer as trainer_module
    # Fake both clocks used by the existing follower, without affecting other modules.
    clock.time = clock.monotonic
    monkeypatch.setattr(trainer_module, "time", clock)
    peer.config.global_resource_planner.runtime_drain_timeout_s = 0.1
    with pytest.raises(TimeoutError, match="to apply"):
        peer._follow_runtime_reconfiguration_if_requested(expected_coord_id="current")
    assert not peer.applied_states


def test_background_and_boundary_do_not_apply_same_request_twice(trainers):
    leader, peer, executor = trainers
    complete(leader, Decision(True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        # The request callback also lets the background poller see the request.
        original = leader._publish_runtime_boundary_transaction
        background = []
        def publish(coord_id, drain_required):
            original(coord_id, drain_required)
            background.append(pool.submit(peer._follow_runtime_reconfiguration_if_requested))
        leader._publish_runtime_boundary_transaction = publish
        following = pool.submit(peer._runtime_reconfiguration_boundary, 0, "post")
        leader._run_global_resource_planner_at_boundary(0, [], {})
        following.result(timeout=3)
        for task in background:
            task.result(timeout=3)
    assert len(peer.applied_states) == 1
    # A later non-draining transaction must not re-enter the old drain request.
    leader.config.global_resource_planner.runtime_drain_before_reconfigure = False
    peer.config.global_resource_planner.runtime_drain_before_reconfigure = False
    leader._publish_runtime_boundary_transaction = original
    complete(leader, Decision(True), 1)
    leader._run_global_resource_planner_at_boundary(1, [], {})
    peer._runtime_reconfiguration_boundary(1, "post")
    peer._follow_runtime_reconfiguration_if_requested()
    assert len(peer.applied_states) == 2


def test_request_mismatch_is_protocol_error(trainers):
    leader, peer, executor = trainers
    leader._runtime_boundary_protocol.publish(0, "post", coord_id="expected", drain_required=True)
    write_record(executor._coordination_dir() / "request.json", {
        "run_id": "test-run", "coord_id": "wrong",
    })
    with pytest.raises(RuntimeError, match="mismatched drain request"):
        peer._runtime_reconfiguration_boundary(0, "post")


def test_late_future_reconfigures_only_at_later_post(trainers):
    leader, peer, executor = trainers
    leader.config.global_resource_planner.runtime_drain_before_reconfigure = False
    future = Future()
    leader._grp_future = future
    leader._grp_future_step = 0
    leader._run_global_resource_planner_at_boundary(0, [], {})
    future.set_result(Decision(True))
    leader._runtime_reconfiguration_boundary(1, "pre")
    assert leader._grp_future is future
    leader._run_global_resource_planner_at_boundary(1, [], {})
    assert peer._runtime_boundary_protocol.wait(0, "post")["decision"] == "no_reconfig"
    assert peer._runtime_boundary_protocol.wait(1, "pre")["decision"] == "no_reconfig"
    assert peer._runtime_boundary_protocol.wait(1, "post")["decision"] == "reconfig"
    assert leader._grp_future is None
    assert executor.planner.applied == 1

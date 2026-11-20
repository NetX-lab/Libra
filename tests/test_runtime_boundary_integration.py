"""CPU-only integration coverage for boundary, membership, and shutdown fixes."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from types import SimpleNamespace

import pytest
import torch

from RL_Framework.config import AsyncRLConfig
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.infra.cost_model.global_resource_planner import PlannerDecision
from RL_Framework.infra.execution.async_runner import AsyncTaskRunner
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import CausalPrefixTree
from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor
from RL_Framework.infra.sync.staleness import StalenessManager
from RL_Framework.infra.elastic.hybrid_pool import (
    ElasticHybridPool,
    InterReplicaGradientDomain,
    JoinCancelledError,
)
from RL_Framework.infra.elastic.runtime_boundary import (
    RuntimeBoundaryProtocol,
    read_record,
    write_record,
)
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor
from RL_Framework.infra.elastic import runtime_executor as runtime_module
from RL_Framework.infra.sync import nccl_weight_sync
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


@pytest.fixture
def membership_boundary(tmp_path, monkeypatch):
    """Keep transaction, dispatcher and CMLFQ real; replace GPU work."""
    config = make_config(tmp_path)
    cfg = config.global_resource_planner
    cfg.runtime_reconfigure_training = True
    cfg.runtime_training_pool_plan_only = False
    cfg.hybrid_worker_launch_enabled = True
    cfg.runtime_coordinate_reconfiguration_ranks = False
    config.heterogeneous_rollout.instances = []
    sched = config.heterogeneous_rollout.scheduling
    sched.scheduler_type = "cmlfq_cost"
    sched.cmlfq_buckets = {"short": {"tp_degrees": [1], "max_tokens": 5000}}
    sched.cmlfq_tree_path = ""
    trainer = AsyncRLTrainer(config)
    trainer._use_heterogeneous = True
    trainer.train_engine = SimpleNamespace(get_version=lambda: 0)
    trainer.rollout_engine = HeterogeneousRolloutEngine.from_config(config)
    trainer.staleness_manager = StalenessManager(trainer.train_engine, 1, 1, 1)
    runner = AsyncTaskRunner()
    trainer.dispatcher = BatchTaskDispatcher(
        runner, trainer.staleness_manager, trainer._create_rollout_task,
    )
    executor = RuntimeElasticExecutor(
        config=config, planner=SimpleNamespace(), train_engine=trainer.train_engine,
        rollout_engine=trainer.rollout_engine, dispatcher=trainer.dispatcher,
    )
    executor.trainer = trainer
    trainer.runtime_elastic_executor = executor
    plan = SimpleNamespace(
        train_config=SimpleNamespace(tp=1, ep=1, pp=1, dp=1),
        train_gpus=1, rollout_tp_list=[], t_global=1.0, expected_gain_s=0.0,
        to_dict=lambda: {},
    )
    decision = PlannerDecision(
        step=1, should_reconfigure=True, reason="membership", current_plan=plan,
        candidate_plan=plan, resource_should_reconfigure=False,
    )
    monkeypatch.setattr(executor, "_ensure_elastic_pool", lambda: None)
    monkeypatch.setattr(executor, "_desired_hybrid_workers", lambda *args: 0)
    monkeypatch.setattr(executor, "_reconfigure_training_pool", lambda *args: [])
    from RL_Framework.infra.elastic import rollout_lending
    monkeypatch.setattr(rollout_lending, "return_rollout", lambda owner: [])
    monkeypatch.setattr(executor, "_publish_current_rollout_weights",
                        lambda: pytest.fail("membership must defer publication to the common step boundary"))
    drains = []
    wait = trainer.dispatcher.wait_until_idle

    def drain(**kwargs):
        drains.append(1)
        return wait(**kwargs)

    monkeypatch.setattr(trainer.dispatcher, "wait_until_idle", drain)
    yield SimpleNamespace(
        trainer=trainer, executor=executor, decision=decision, runner=runner, drains=drains,
    )
    trainer.dispatcher.destroy()
    executor.close()


def test_membership_keeps_live_scheduler_groups_and_survivor_work(
    membership_boundary, monkeypatch,
):
    h = membership_boundary
    trainer = h.trainer
    engine, scheduler = trainer.rollout_engine, trainer.rollout_engine.scheduler
    pending_group = [{"grpo_group_id": "accepted", "grpo_rollout_index": 0}]
    trainer._pending_grpo_groups = {"accepted": pending_group}
    entered, finish = threading.Event(), threading.Event()

    def resource_action(*args):
        entered.set()
        assert finish.wait(3), "test did not finish the local resource action"
        return []

    monkeypatch.setattr(h.executor, "_reconfigure_training_pool", resource_action)
    monkeypatch.setattr(trainer, "_rebind_rollout_engine_from_config",
                        lambda **kwargs: pytest.fail("membership replaced the live rollout client"))
    with ThreadPoolExecutor(max_workers=1) as threads:
        applying = threads.submit(trainer._apply_global_resource_planner_decision, 1, h.decision, {})
        try:
            assert entered.wait(3)
            assert not trainer.dispatcher.is_paused()
            scheduler.register_instance(0, "short", 1)
            event = {"tool_type": "search", "output": "ok", "payload_tokens": 2}
            state = DefaultToolStateExtractor().extract(event)
            request = engine.begin_cmlfq_request("survivor-trace", 100)
            engine.route_cmlfq_tool_return(request, event, 100)
            engine.finish_cmlfq_request(request, 1000)
        finally:
            finish.set()
        applying.result(timeout=3)
    assert trainer.rollout_engine is engine
    assert trainer.rollout_engine.scheduler is scheduler
    assert trainer._pending_grpo_groups["accepted"] is pending_group
    assert scheduler.prefix_tree.lookup("survivor-trace", [state]) is not None
    assert scheduler._tree_update_count == 1
    assert h.drains == []


@pytest.mark.parametrize("direct", [False, True], ids=["trainer", "executor"])
@pytest.mark.parametrize("already_paused", [False, True], ids=["open", "parent-paused"])
def test_membership_boundary_preserves_pause_owner_without_draining(
    membership_boundary, direct, already_paused,
):
    h = membership_boundary
    if already_paused:
        h.trainer.dispatcher.pause()
    prepared = h.executor.prepare_transaction(h.decision)
    assert not prepared.drain_required
    assert not prepared.should_pause_dispatcher
    if direct:
        assert h.executor.execute(h.decision).applied
    else:
        h.trainer._apply_global_resource_planner_decision(1, h.decision, {})
    assert h.trainer.dispatcher.is_paused() is already_paused
    assert h.drains == []


@pytest.mark.parametrize("already_paused", [False, True], ids=["open", "parent-paused"])
def test_membership_activation_failure_preserves_survivor_and_pause_owner(
    membership_boundary, monkeypatch, already_paused,
):
    h = membership_boundary
    engine = h.trainer.rollout_engine
    if already_paused:
        h.trainer.dispatcher.pause()
    def fail_activation():
        raise RuntimeError("state installation failed")
    monkeypatch.setattr(h.executor, "_finish_pending_activations", fail_activation)
    with pytest.raises(RuntimeError, match="state installation failed"):
        h.trainer._apply_global_resource_planner_decision(1, h.decision, {})
    assert h.trainer.rollout_engine is engine
    assert h.trainer.dispatcher.is_paused() is already_paused
    assert h.drains == []


@pytest.mark.parametrize("noop", ["maintenance", "disabled"])
def test_nonexecuting_membership_boundary_does_not_pause_or_drain(membership_boundary, noop):
    h = membership_boundary
    if noop == "maintenance":
        h.decision.should_reconfigure = False
    else:
        h.trainer.config.global_resource_planner.runtime_dynamic_reconfiguration_enabled = False
    h.trainer._apply_global_resource_planner_decision(1, h.decision, {})
    assert not h.trainer.dispatcher.is_paused()
    assert h.drains == []


class NoReconfigurationDecision:
    should_reconfigure = False
    candidate_plan = None
    current_plan = None
    reason = "same_plan"
    num_requests = 1

    def __init__(self, signal=None):
        self.elastic_hybrid_signal = signal

    def to_dict(self):
        return {"reason": self.reason}


class CloseRecorder:
    def __init__(self, callback=None):
        self.closed = False
        self.callback = callback

    def close(self):
        if self.callback is not None:
            self.callback()
        self.closed = True


class TrainEngineCloseRecorder:
    def __init__(self, callback=None):
        self.closed = False
        self.callback = callback

    def close_elastic_communication_domain(self):
        if self.callback is not None:
            self.callback()
        self.closed = True


def make_config(tmp_path):
    config = AsyncRLConfig(
        model_path="/unused",
        train_gpus=1,
        rollout_gpus=1,
        n_total_gpus=2,
        train_tp_size=1,
        train_dp_size=1,
        batch_size=1,
        n_samples=1,
        micro_batch_size=1,
        log_dir=str(tmp_path),
    )
    planner_cfg = config.global_resource_planner
    planner_cfg.enabled = True
    planner_cfg.runtime_drain_timeout_s = 2.0
    planner_cfg.runtime_coordinate_batch_source_only = False
    planner_cfg.hybrid_worker_task_dir = str(tmp_path / "tasks")
    return config


def make_protocol_pair(config, run_id="integration-run"):
    directory = AsyncRLTrainer(config)._runtime_reconfiguration_coord_dir()
    leader = RuntimeBoundaryProtocol(directory, 0, 2, 0.2)
    follower = RuntimeBoundaryProtocol(directory, 1, 2, 0.2)
    leader.run_id = follower.run_id = run_id
    return leader, follower


def make_cleanup_trainers(tmp_path):
    config = make_config(tmp_path)
    leader = AsyncRLTrainer(config)
    follower = AsyncRLTrainer(config)
    follower.rank = 1
    follower.is_main_process = False
    leader_protocol, follower_protocol = make_protocol_pair(config)
    leader._runtime_boundary_protocol = leader_protocol
    follower._runtime_boundary_protocol = follower_protocol
    leader.train_engine = TrainEngineCloseRecorder()
    follower.train_engine = TrainEngineCloseRecorder()
    return leader, follower


def test_release_only_post_is_visible_at_next_pre_before_gradient_freeze(tmp_path):
    config = make_config(tmp_path)
    pool = ElasticHybridPool(
        core_train_workers=["dp0"],
        core_rollout_workers=["rollout0"],
        zero_sync_steps=0,
    )
    executor = RuntimeElasticExecutor(
        config=config,
        planner=SimpleNamespace(latest_runtime_metrics=SimpleNamespace(step=1)),
        elastic_pool=pool,
        membership_run_id="membership-run",
    )
    leader = AsyncRLTrainer(config)
    follower = AsyncRLTrainer(config)
    follower.rank = 1
    follower.is_main_process = False
    leader_protocol, follower_protocol = make_protocol_pair(config)
    leader._runtime_boundary_protocol = leader_protocol
    follower._runtime_boundary_protocol = follower_protocol
    leader.runtime_elastic_executor = executor
    leader._runtime_post_boundary = 0

    engine = object.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = InterReplicaGradientDomain(core_replica_ids=["dp0"])
    engine.get_data_parallel_rank = lambda: 0
    engine._pending_hybrid_gradients = []
    engine._hybrid_gradient_condition = threading.Condition()
    engine._elastic_active_gradient_timeout_s = 0.0
    engine._elastic_step_hybrid_workers = ()
    engine.optimizer_revision = 0
    engine.current_version = 0
    engine._elastic_seen_payloads = set()
    engine._elastic_batch_partition = None
    engine._elastic_gradient_update_callback = None
    engine._model_main_gradients = lambda: [torch.zeros(1)]
    follower.train_engine = engine
    follower._elastic_membership_run_id = "membership-run"
    follower._elastic_membership_epochs = {}

    try:
        pool.join_replica(
            "replica0", ["rollout0"], "dp0",
            replica_activation_barrier=lambda *args: {"acknowledged": True, "state_version": 0,
                                                      "optimizer_revision": 0},
        ).result(timeout=2)
        executor.hybrid_runtime_state()
        assert follower._prepare_elastic_training_step(0) == ["replica0"]
        membership_path = tmp_path / "tasks/membership/replica0.json"

        original_publish = leader_protocol.publish

        def publish(step, point, **kwargs):
            # NO_RECONFIG closes POST before release-only maintenance starts.
            if point == "post":
                assert json.loads(membership_path.read_text())["role"] == "hybrid_training"
            return original_publish(step, point, **kwargs)

        leader_protocol.publish = publish
        leader._apply_global_resource_planner_decision(
            0,
            NoReconfigurationDecision({"step": 1, "desired_replicas": 0}),
            {},
        )
        follower._runtime_reconfiguration_boundary(0, "post")
        assert json.loads(membership_path.read_text())["role"] == "hybrid_rollout"

        # The leader's next PRE is the gate after maintenance/publication.
        executor.hybrid_runtime_state()
        leader_protocol.publish(1, "pre")
        follower._runtime_reconfiguration_boundary(1, "pre")
        assert follower._prepare_elastic_training_step(1) == []
        engine._apply_elastic_inter_replica_gradients()
    finally:
        executor.close()


@pytest.mark.parametrize("decision", ["noop", "no_drain"])
def test_final_boundary_waits_for_leader_close_before_follower_teardown(tmp_path, decision):
    leader, follower = make_cleanup_trainers(tmp_path)
    done_path = leader._runtime_boundary_protocol.runtime_done_path()
    leader.runtime_elastic_executor = CloseRecorder()
    follower._elastic_gradient_server = CloseRecorder(
        lambda: pytest.fail("follower service closed before runtime_done")
        if not read_record(done_path).get("status") == "success"
        else None
    )

    if decision == "noop":
        leader._runtime_boundary_protocol.publish(0, "post")
    else:
        leader._runtime_boundary_protocol.publish(
            0,
            "post",
            coord_id="non-draining",
            drain_required=False,
        )
    follower._runtime_reconfiguration_boundary(0, "post")

    with ThreadPoolExecutor(max_workers=1) as threads:
        waiting = threads.submit(follower._cleanup)
        with pytest.raises(FutureTimeout):
            waiting.result(timeout=0.05)
        leader._cleanup()
        waiting.result(timeout=2)

    assert read_record(done_path)["status"] == "success"
    assert follower.train_engine.closed
    assert follower._elastic_gradient_server is None


def test_cleanup_releases_cached_rollout_weight_communicators(tmp_path):
    """Catch interpreter-shutdown aborts from lingering TCPStore owners."""
    leader, _follower = make_cleanup_trainers(tmp_path)
    cache_key = ("127.0.0.1", 29620, 2, 0, "cuda:0")
    nccl_weight_sync._COMMUNICATOR_CACHE[cache_key] = SimpleNamespace(comm=None, disabled=False)

    leader._cleanup()

    assert nccl_weight_sync._COMMUNICATOR_CACHE == {}


def test_lending_transaction_drains_peer_source_before_reporting_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    leader, follower = make_cleanup_trainers(tmp_path)
    config = leader.config
    config.global_resource_planner.runtime_reconfigure_training = True
    config.global_resource_planner.runtime_training_pool_plan_only = False
    config.global_resource_planner.hybrid_worker_launch_enabled = True
    plan = SimpleNamespace(train_gpus=1, to_dict=lambda: {})
    signal = {"desired_replicas": 1, "loan_instance_ids": ["borrowed"]}
    executor = RuntimeElasticExecutor(config=config, planner=None)
    executor.runtime_run_id = "integration-run"
    prepared = executor.prepare_transaction(PlannerDecision(
        step=0, should_reconfigure=True, reason="borrow", current_plan=plan,
        candidate_plan=plan, resource_should_reconfigure=False, elastic_hybrid_signal=signal,
    ))
    executor.publish_peer_rank_drain_request(prepared)
    leader._runtime_boundary_protocol.publish(0, "post", coord_id=prepared.coord_id, drain_required=True)
    root = leader._runtime_reconfiguration_coord_dir()
    ran = threading.Event()
    release = threading.Event()
    follower.dispatcher = BatchTaskDispatcher(AsyncTaskRunner(), SimpleNamespace(), lambda task: None)
    drain = follower.dispatcher.wait_until_idle

    def wait_until_idle(*, timeout):
        ran.set()
        assert release.wait(2)
        drain(timeout=timeout)

    follower.dispatcher.wait_until_idle = wait_until_idle
    with ThreadPoolExecutor(max_workers=2) as threads:
        following = threads.submit(follower._runtime_reconfiguration_boundary, 0, "post")
        try:
            assert ran.wait(1)
            assert follower.dispatcher.is_paused()
            assert not (root / "ready/rank_1.json").exists()
        finally:
            release.set()
            write_record(root / "applied.json", {
                "run_id": "integration-run", "coord_id": prepared.coord_id, "job_id": "",
            })
        following.result(timeout=2)
    assert not follower.dispatcher.is_paused()


def test_final_draining_transaction_reaches_terminal_then_done(tmp_path):
    leader, follower = make_cleanup_trainers(tmp_path)
    root = leader._runtime_reconfiguration_coord_dir()
    coord_id = "draining-final"
    write_record(root / "request.json", {
        "run_id": "integration-run",
        "coord_id": coord_id,
        "job_id": "",
    })
    leader._runtime_boundary_protocol.publish(
        0,
        "post",
        coord_id=coord_id,
        drain_required=True,
    )
    follower._apply_runtime_reconfiguration_state = lambda state: None
    follower._reset_rollout_pipeline_after_reconfigure = lambda: None
    leader.runtime_elastic_executor = CloseRecorder()

    with ThreadPoolExecutor(max_workers=2) as threads:
        following = threads.submit(follower._runtime_reconfiguration_boundary, 0, "post")
        ready = root / "ready/rank_1.json"
        for _ in range(100):
            if read_record(ready).get("coord_id") == coord_id:
                break
            threading.Event().wait(0.01)
        else:
            pytest.fail("follower did not publish ready")
        write_record(root / "applied.json", {
            "run_id": "integration-run",
            "coord_id": coord_id,
            "job_id": "",
            "instances": [],
            "training": {"enabled": False},
        })
        following.result(timeout=2)

        waiting = threads.submit(follower._cleanup)
        with pytest.raises(FutureTimeout):
            waiting.result(timeout=0.05)
        leader._cleanup()
        waiting.result(timeout=2)

    assert read_record(leader._runtime_boundary_protocol.runtime_done_path())["status"] == "success"


def test_close_failure_publishes_failed_runtime_done(tmp_path):
    leader, follower = make_cleanup_trainers(tmp_path)

    def fail_close():
        raise RuntimeError("simulated close failure")

    leader.runtime_elastic_executor = CloseRecorder(fail_close)
    with pytest.raises(RuntimeError, match="simulated close failure"):
        leader._cleanup()
    record = read_record(leader._runtime_boundary_protocol.runtime_done_path())
    assert record["status"] == "failed"
    with pytest.raises(RuntimeError, match="shutdown failed on rank0"):
        follower._cleanup()


@pytest.mark.parametrize("failure_type", [RuntimeError, TimeoutError])
def test_final_hybrid_task_failure_still_closes_executor_and_notifies_followers(tmp_path, failure_type):
    leader, follower = make_cleanup_trainers(tmp_path)
    failure = failure_type("last Hybrid task failed")
    executor = CloseRecorder()

    def wait_for_tasks():
        raise failure

    executor.wait_for_hybrid_training_tasks = wait_for_tasks
    leader.runtime_elastic_executor = executor
    with pytest.raises(failure_type) as caught:
        leader._cleanup(wait_for_hybrid_tasks=True)
    assert caught.value is failure
    assert executor.closed
    assert read_record(leader._runtime_boundary_protocol.runtime_done_path())["status"] == "failed"
    with pytest.raises(RuntimeError, match="last Hybrid task failed"):
        follower._cleanup()


def test_late_hybrid_activation_is_fenced_before_runtime_done(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    pool = ElasticHybridPool(
        core_train_workers=["dp0"],
        core_rollout_workers=["rollout0"],
        zero_sync_steps=0,
    )
    executor = RuntimeElasticExecutor(
        config=config,
        planner=SimpleNamespace(),
        elastic_pool=pool,
    )
    entered = threading.Event()
    resume = threading.Event()
    launches = []

    def build(**kwargs):
        entered.set()
        assert resume.wait(2)
        return "fake worker"

    monkeypatch.setattr(executor, "_build_hybrid_worker_command", build)
    monkeypatch.setattr(executor, "_cluster_swap_training_slot", lambda worker: {"host": "", "gpus": [0]})
    monkeypatch.setattr(executor, "_wait_hybrid_worker_ready", lambda worker: None)
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *a, **k: launches.append(a))

    leader, follower = make_cleanup_trainers(tmp_path)
    leader.runtime_elastic_executor = executor
    handle = pool.join_training(
        "rollout0",
        "dp0",
        activation_barrier=executor._activate_hybrid_worker,
    )
    try:
        assert entered.wait(2)
        leader._cleanup()
        follower._cleanup()
    finally:
        resume.set()
    with pytest.raises(JoinCancelledError):
        handle.result(timeout=2)
    assert launches == []
    assert read_record(leader._runtime_boundary_protocol.runtime_done_path())["status"] == "success"

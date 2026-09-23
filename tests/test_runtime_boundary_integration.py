"""CPU-only integration coverage for boundary, membership, and shutdown fixes."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
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
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class NoReconfigurationDecision:
    should_reconfigure = False
    candidate_plan = None
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
    follower.train_engine = engine
    follower._elastic_membership_run_id = "membership-run"
    follower._elastic_membership_epochs = {}

    try:
        pool.join_replica("replica0", ["rollout0"], "dp0").result(timeout=2)
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
        with pytest.raises(TimeoutError):
            waiting.result(timeout=0.05)
        leader._cleanup()
        waiting.result(timeout=2)

    assert read_record(done_path)["status"] == "success"
    assert follower.train_engine.closed
    assert follower._elastic_gradient_server is None


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
        with pytest.raises(TimeoutError):
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

"""CPU-only publication -> peer refresh -> training-source regression tests."""

import json
import threading
import time
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
from RL_Framework.infra.elastic.runtime_executor import (
    RuntimeElasticExecutor,
    RuntimeReconfigurationResult,
)
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
from RL_Framework.trainer import async_rl_trainer as trainer_module


@pytest.fixture
def make_harness(tmp_path):
    pools = []

    def make(run_id="run-a"):
        config = AsyncRLConfig(
            model_path="/unused", train_gpus=1, rollout_gpus=2, n_total_gpus=3,
            train_tp_size=1, train_dp_size=1, batch_size=1, n_samples=1,
            micro_batch_size=1, log_dir=str(tmp_path),
        )
        cfg = config.global_resource_planner
        cfg.enabled = True
        cfg.hybrid_worker_task_dir = str(tmp_path / "tasks")
        cfg.elastic_hybrid_min_rollout_gpus = 0
        pool = ElasticHybridPool(
            core_train_workers=["dp0"], core_rollout_workers=["rollout0", "rollout1"],
            zero_sync_steps=0,
        )
        pools.append(pool)
        executor = RuntimeElasticExecutor(
            config=config, elastic_pool=pool, membership_run_id=run_id,
            planner=SimpleNamespace(latest_runtime_metrics=SimpleNamespace(step=2)),
        )
        engine = object.__new__(MegatronCoreTrainEngine)
        engine.elastic_gradient_domain = InterReplicaGradientDomain(core_replica_ids=["dp0"])
        engine.get_data_parallel_rank = lambda: 0
        engine._pending_hybrid_gradients = []
        engine._hybrid_gradient_condition = threading.Condition()
        engine._elastic_active_gradient_timeout_s = 0.0
        engine._elastic_step_hybrid_workers = ()
        peer = object.__new__(AsyncRLTrainer)
        peer.config = config
        peer.train_engine = engine
        peer._elastic_membership_run_id = run_id
        peer._elastic_membership_epochs = {}
        directory = tmp_path / "tasks" / "membership"
        return SimpleNamespace(
            config=config, pool=pool, executor=executor, peer=peer, engine=engine,
            directory=directory,
        )

    yield make
    for pool in pools:
        pool.close()


def activate(harness, replica="replica0", worker="rollout0"):
    harness.pool.join_replica(replica, [worker], "dp0").result(timeout=3)
    harness.executor.hybrid_runtime_state()
    harness.peer._prepare_elastic_training_step(0)


def record(harness, replica="replica0"):
    return json.loads((harness.directory / f"{replica}.json").read_text())


def release(harness):
    # Release-only maintenance: no PlannerDecision or runtime transaction.
    harness.executor.accept_planner_signal({"step": 2, "desired_replicas": 0})


def test_m1_release_publishes_exit_without_a_runtime_transaction(make_harness):
    h = make_harness()
    activate(h)
    previous = record(h)
    assert h.engine._elastic_step_hybrid_workers == ("replica0",)

    release(h)
    # Do not call hybrid_runtime_state: release itself must publish the exit.
    exited = record(h)
    assert h.pool.gradient_domain.active_hybrid_ids() == ()
    assert exited["role"] == "hybrid_rollout"
    assert exited["membership_epoch"] > previous["membership_epoch"]
    assert exited["run_id"] == "run-a"
    h.peer._refresh_nonblocking_elastic_membership()
    assert h.engine.elastic_gradient_domain.active_hybrid_ids() == ()
    assert not (h.executor._coordination_dir() / "request.json").exists()


def test_m2_stale_training_record_cannot_undo_consumed_exit(make_harness):
    h = make_harness()
    activate(h)
    stale = record(h)
    release(h)
    h.peer._prepare_elastic_training_step(1)
    exit_epoch = h.peer._elastic_membership_epochs["replica0"]

    (h.directory / "replica0.json").write_text(json.dumps(stale))
    assert h.peer._prepare_elastic_training_step(2) == []
    assert h.peer._elastic_membership_epochs["replica0"] == exit_epoch
    h.executor.hybrid_runtime_state()
    assert record(h)["role"] == "hybrid_rollout"


def test_m3_pre_refresh_precedes_freezing_gradient_sources(make_harness, monkeypatch):
    h = make_harness()
    activate(h)
    release(h)
    frozen = []
    original = h.engine.set_elastic_training_step

    def set_step(step, sources):
        assert h.engine.elastic_gradient_domain.active_hybrid_ids() == ()
        frozen.append((step, tuple(sources)))
        original(step, sources)

    monkeypatch.setattr(h.engine, "set_elastic_training_step", set_step)
    assert h.peer._prepare_elastic_training_step(1) == []
    assert frozen == [(1, ())]


def test_m4_original_probe_no_longer_waits_for_released_gradient(make_harness):
    h = make_harness()
    activate(h)
    # The real wait path would reject the still-active source without a payload.
    with pytest.raises(TimeoutError, match="missing=.*replica0"):
        h.engine._apply_elastic_inter_replica_gradients()
    release(h)
    assert h.peer._prepare_elastic_training_step(1) == []
    h.engine._apply_elastic_inter_replica_gradients()


@pytest.mark.parametrize("replica", [True, False], ids=["replica", "singleton"])
@pytest.mark.parametrize("rejoin", [False, True], ids=["released", "new-generation"])
def test_m5_cancelled_join_cannot_republish_active_membership(make_harness, replica, rejoin):
    h = make_harness()
    entered, resume = threading.Event(), threading.Event()

    def activating(*args):
        entered.set()
        assert resume.wait(3), "test did not release activation"

    worker = "replica0" if replica else "rollout0"
    if replica:
        handle = h.pool.join_replica(
            worker, ["rollout0"], "dp0", replica_activation_barrier=activating,
        )
    else:
        handle = h.pool.join_training("rollout0", "dp0", activation_barrier=activating)
    try:
        assert entered.wait(3)
        h.executor._pending_hybrid_joins[worker] = handle
        h.executor._pending_hybrid_join_started_at[worker] = time.time()
        h.executor.hybrid_runtime_state()
        release(h)
        assert h.pool.snapshot()["rollout0"].transition_generation == handle.generation + 1
        exit_record = record(h, worker)
        if rejoin:
            if replica:
                newer = h.pool.join_replica(worker, ["rollout0"], "dp0")
            else:
                newer = h.pool.join_training(worker, "dp0")
            newer.result(timeout=3)
            h.executor.hybrid_runtime_state()
            expected_record = record(h, worker)
            assert expected_record["membership_epoch"] > exit_record["membership_epoch"]
        else:
            expected_record = exit_record
        resume.set()
        with pytest.raises(JoinCancelledError):
            handle.result(timeout=3)
        h.executor.hybrid_runtime_state()
        assert record(h, worker) == expected_record
        assert h.pool.gradient_domain.attached_hybrid_ids() == ((worker,) if rejoin else ())
        assert h.peer._prepare_elastic_training_step(1) == ([worker] if rejoin else [])
    finally:
        resume.set()
        try:
            handle.result(timeout=3)
        except JoinCancelledError:
            pass


@pytest.mark.parametrize("replica", [True, False], ids=["replica", "singleton"])
def test_activation_and_release_commit_membership_and_role_atomically(make_harness, monkeypatch, replica):
    h = make_harness()
    marked, resume = threading.Event(), threading.Event()
    mark_active = h.pool.gradient_domain.mark_active

    def pause_after_domain_activation(worker):
        mark_active(worker)
        marked.set()
        assert resume.wait(3)

    monkeypatch.setattr(h.pool.gradient_domain, "mark_active", pause_after_domain_activation)
    worker = "replica0" if replica else "rollout0"
    if replica:
        handle = h.pool.join_replica(worker, ["rollout0"], "dp0")
    else:
        handle = h.pool.join_training(worker, "dp0")
    h.executor._pending_hybrid_joins[worker] = handle
    h.executor._pending_hybrid_join_started_at[worker] = time.time()
    with ThreadPoolExecutor(max_workers=1) as workers:
        try:
            assert marked.wait(3)
            releasing = workers.submit(release, h)
            # Cancellation must not split domain activation from the matching
            # worker-role update. With the old gap it completed in this window.
            try:
                releasing.result(timeout=0.1)
            except TimeoutError:
                pass
        finally:
            resume.set()
        handle.result(timeout=3)
        releasing.result(timeout=3)
    assert h.pool.snapshot()["rollout0"].role.value == "hybrid_rollout"
    assert h.pool.gradient_domain.active_hybrid_ids() == ()
    assert h.peer._prepare_elastic_training_step(1) == []


def test_m6_repeated_release_is_idempotent(make_harness):
    h = make_harness()
    activate(h)
    release(h)
    first = record(h)
    release(h)
    h.executor.hybrid_runtime_state()
    assert record(h) == first
    assert h.peer._prepare_elastic_training_step(1) == []


@pytest.mark.parametrize("legacy", [False, True], ids=["old-run", "unversioned-run"])
def test_m7_old_run_files_do_not_pollute_reused_directory(make_harness, legacy):
    old = make_harness("old-run")
    activate(old)
    if legacy:
        state = record(old)
        state.pop("run_id")
        (old.directory / "replica0.json").write_text(json.dumps(state))
    new = make_harness("new-run")
    assert new.peer._prepare_elastic_training_step(0) == []
    activate(new)
    assert new.engine._elastic_step_hybrid_workers == ("replica0",)
    assert record(new)["run_id"] == "new-run"


def test_m8_publication_failure_is_fatal(make_harness, monkeypatch):
    h = make_harness()
    activate(h)

    def fail(*args):
        raise OSError("simulated shared filesystem failure")

    monkeypatch.setattr(h.executor, "_write_json_atomic", fail)
    with pytest.raises(RuntimeError, match="failed to publish EHP membership for replica0"):
        release(h)
    assert h.pool.gradient_domain.active_hybrid_ids() == ()
    # The error cannot be swallowed: peers would otherwise retain this state.
    assert record(h)["role"] == "hybrid_training"


def test_m9_active_replica_retains_its_membership_epoch(make_harness):
    h = make_harness()
    activate(h)
    original = record(h)
    h.executor.accept_planner_signal({"step": 2, "desired_replicas": 1})
    h.executor.hybrid_runtime_state()
    assert h.peer._prepare_elastic_training_step(1) == ["replica0"]
    assert record(h) == original
    assert h.engine.elastic_gradient_domain.membership_state()["membership_epochs"]["replica0"] == original["membership_epoch"]


def test_m10_partial_release_preserves_other_replica(make_harness, monkeypatch):
    h = make_harness()
    activate(h)
    activate(h, "replica1", "rollout1")
    preserved = record(h, "replica1")
    h.executor._active_elastic_signal = {"step": 2, "desired_replicas": 1}
    monkeypatch.setattr(h.executor, "_training_reconfiguration_targets",
                        lambda *args: (["dp0"], "dp0", ["rollout0", "rollout1"]))
    h.executor._reconfigure_training_pool(
        1, SimpleNamespace(train_gpus=2), RuntimeReconfigurationResult(False, "test"),
    )
    assert record(h)["role"] == "hybrid_rollout"
    assert record(h, "replica1") == preserved
    assert h.peer._prepare_elastic_training_step(1) == ["replica1"]


def test_release_serializes_with_delayed_membership_publication(make_harness, monkeypatch):
    h = make_harness()
    activate(h)
    publishing, detached, resume = threading.Event(), threading.Event(), threading.Event()
    write = h.executor._write_json_atomic
    detach = h.pool.release_replica_to_rollout

    def delayed_write(path, payload):
        if payload["role"] == "hybrid_training":
            publishing.set()
            assert resume.wait(3)
        write(path, payload)

    def release_replica(replica_id):
        detach(replica_id)
        detached.set()

    monkeypatch.setattr(h.executor, "_write_json_atomic", delayed_write)
    monkeypatch.setattr(h.pool, "release_replica_to_rollout", release_replica)
    with ThreadPoolExecutor(max_workers=2) as workers:
        try:
            observer = workers.submit(h.executor._publish_hybrid_membership)
            assert publishing.wait(3)
            releasing = workers.submit(release, h)
            assert detached.wait(3)
        finally:
            resume.set()
        observer.result(timeout=3)
        releasing.result(timeout=3)
    assert record(h)["role"] == "hybrid_rollout"
    assert h.peer._prepare_elastic_training_step(1) == []


def test_setup_reuses_existing_barrier_for_fresh_membership_session(make_harness, monkeypatch):
    leader = make_harness().peer
    peer = make_harness().peer
    barrier = threading.Barrier(2, timeout=3)
    barrier_calls = []

    class Server:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return SimpleNamespace(to_dict=lambda: {"host": "fake", "port": 1})

    def synchronize():
        barrier_calls.append(1)
        barrier.wait()

    monkeypatch.setattr(trainer_module, "ElasticGradientServer", Server)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "barrier", synchronize)
    for rank, trainer in enumerate((leader, peer)):
        trainer.rank = rank
        trainer.is_main_process = rank == 0
        trainer.runtime_elastic_executor = None
        trainer.config.global_resource_planner.hybrid_worker_launch_enabled = True
        engine = trainer.train_engine
        engine.configure_elastic_training = lambda *a, _engine=engine, **kw: _engine.elastic_gradient_domain
        engine.get_elastic_core_replica_ids = lambda: ["dp0"]
        engine.get_elastic_lane_state = lambda: {}

    identities = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        for _ in range(2):
            tasks = [workers.submit(t._init_nonblocking_elastic_core) for t in (leader, peer)]
            for task in tasks:
                task.result(timeout=4)
            assert leader._elastic_membership_run_id == peer._elastic_membership_run_id
            identities.append(leader._elastic_membership_run_id)
    assert identities[0] != identities[1]
    assert len(barrier_calls) == 4  # One existing setup barrier per rank per run.

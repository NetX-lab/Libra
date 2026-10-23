"""Cold startup shares the join budget; live state catch-up keeps its ACK budget."""

import json
import threading
from types import SimpleNamespace

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.elastic import hybrid_pool as pool_module
from RL_Framework.infra.elastic import runtime_executor as runtime_module
from RL_Framework.infra.elastic.hybrid_pool import ElasticHybridPool
from RL_Framework.infra.elastic.runtime_executor import (
    RuntimeElasticExecutor,
    RuntimeReconfigurationResult,
)


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    timing = SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now)
    monkeypatch.setattr(runtime_module, "time", timing)
    monkeypatch.setattr(pool_module, "time", timing)
    config = AsyncRLConfig(model_path="/not-loaded", train_gpus=1, rollout_gpus=1,
        n_total_gpus=2, train_tp_size=1, train_dp_size=1)
    cfg = config.global_resource_planner
    cfg.hybrid_worker_task_dir = str(tmp_path)
    cfg.hybrid_worker_launch_enabled = True
    cfg.hybrid_worker_remote_control_enabled = True
    cfg.elastic_hybrid_join_timeout_s = 180
    cfg.hybrid_worker_ready_timeout_s = 60
    pool = ElasticHybridPool(core_train_workers=["dp0"],
        core_rollout_workers=["rollout0"], snapshot_fetcher=lambda *args: 0)
    executor = RuntimeElasticExecutor(config=config, planner=SimpleNamespace(), elastic_pool=pool)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(json.dumps({"format": "libra-ehp-lanes-v1",
        "optimizer_revision": 0, "snapshot_id": "capture0",
        "lanes": {"dp0": {"0": "ehp://source/lane0"}}}))
    executor._snapshot_manifests[0] = str(manifest)
    monkeypatch.setattr(executor, "_capture_core_snapshot", lambda: str(manifest))
    monkeypatch.setattr(executor, "_build_hybrid_worker_command", lambda **kw: "not executed")
    monkeypatch.setattr(executor, "_cluster_swap_training_slot",
        lambda *args: {"host": "allocated-node", "gpus": [1]})
    monkeypatch.setattr(executor, "_desired_hybrid_workers", lambda *args: 1)
    monkeypatch.setattr(executor, "_training_reconfiguration_targets",
        lambda *args: (["dp0"], "dp0", ["rollout0"]))
    # The loader's initial installation is the subject here, before the real
    # activation queue hands control back to a Core training boundary.
    monkeypatch.setattr(executor, "_queue_activation", lambda identity, receipt: receipt)
    ready_at = 70.0
    identity = SimpleNamespace(worker="ehp_replica_rollout0")

    def advance(_seconds):
        clock.now += 1
        if clock.now >= ready_at:
            receipt = dict(snapshot=str(manifest), rank=0, pid=123, acknowledged=True,
                snapshot_id="capture0", optimizer_revision=0, state_version=0,
                lane_snapshot_id="lane0", target_core_id="dp0", source_uri="ehp://source/lane0")
            (tmp_path / f"{identity.worker}.rank_0.ready").write_text(
                "".join(f"{key}={value}\n" for key, value in receipt.items()))
        return False

    executor._closing = SimpleNamespace(is_set=lambda: False, wait=advance)
    yield SimpleNamespace(executor=executor, pool=pool, clock=clock,
        manifest=manifest, identity=identity)
    pool.close()


@pytest.mark.parametrize("replica", [True, False], ids=["replica", "singleton"])
def test_initial_installation_can_be_ready_after_seventy_seconds(bootstrap, replica):
    h = bootstrap
    if replica:
        h.executor._reconfigure_training_pool(1, SimpleNamespace(),
            RuntimeReconfigurationResult(False, "test"))
        handle = h.executor._pending_hybrid_joins["ehp_replica_rollout0"]
    else:
        h.identity.worker = "rollout0"
        handle = h.pool.join_training("rollout0", "dp0", timeout_s=180,
            activation_barrier=h.executor._activate_hybrid_worker)
    handle.result(timeout=3)
    assert h.clock.now == 70
    assert h.pool.snapshot()["rollout0"].role.value == "hybrid_training"


def test_live_state_catchup_still_times_out_after_sixty_seconds(bootstrap):
    h = bootstrap
    with pytest.raises(TimeoutError, match="after 60s"):
        h.executor._wait_hybrid_worker_ready(h.identity.worker, state_version=0,
            replica_world_size=1, snapshot_path=str(h.manifest), target_core_id="dp0")
    assert h.clock.now == 60


@pytest.mark.parametrize("preparation_seconds", [0, 20])
def test_initial_installation_does_not_extend_the_overall_join_deadline(
        bootstrap, monkeypatch, preparation_seconds):
    h = bootstrap
    h.executor.config.global_resource_planner.elastic_hybrid_join_timeout_s = 65

    def prepare(**kwargs):
        h.clock.now += preparation_seconds
        return "not executed"

    monkeypatch.setattr(h.executor, "_build_hybrid_worker_command", prepare)
    h.executor._reconfigure_training_pool(1, SimpleNamespace(),
        RuntimeReconfigurationResult(False, "test"))
    handle = h.executor._pending_hybrid_joins["ehp_replica_rollout0"]
    with pytest.raises(TimeoutError):
        handle.result(timeout=3)
    # The pool checks its original deadline on every poll; preparation cannot
    # restart it. The clock advances in one-second increments in this fixture.
    assert h.clock.now <= 66


@pytest.fixture
def queued_activation(bootstrap, monkeypatch):
    h = bootstrap
    executor = h.executor
    monkeypatch.setattr(executor, "_queue_activation",
        RuntimeElasticExecutor._queue_activation.__get__(executor))
    monkeypatch.setattr(executor, "_publish_hybrid_membership", lambda: None)
    monkeypatch.setattr(executor, "_stop_hybrid_worker_process", lambda *args: None)
    monkeypatch.setattr(executor, "_stop_managed_rollout_processes", lambda: None)
    from RL_Framework.infra.elastic import core_reconfiguration
    monkeypatch.setattr(core_reconfiguration, "dispatch_core_command", lambda *args, **kwargs: [])
    closing = threading.Event()
    cold_wait = executor._closing.wait
    state = SimpleNamespace(catching_up=False, ack_at=0, capture_seconds=0)
    latest = h.manifest.with_name("snapshot14.json")
    latest.write_text(json.dumps({"format": "libra-ehp-lanes-v1",
        "optimizer_revision": 14, "snapshot_id": "capture14",
        "lanes": {"dp0": {"0": "ehp://source/lane14"}}}))

    def acknowledge():
        receipt = dict(snapshot=str(latest), rank=0, pid=123, acknowledged=True,
            snapshot_id="capture14", optimizer_revision=14, state_version=14,
            lane_snapshot_id="lane14", target_core_id="dp0", source_uri="ehp://source/lane14")
        (latest.parent / f"{h.identity.worker}.rank_0.ready").write_text(
            "".join(f"{key}={value}\n" for key, value in receipt.items()))

    def advance(seconds):
        if not state.catching_up:
            return cold_wait(seconds)
        h.clock.now += 1
        if h.clock.now >= state.ack_at:
            acknowledge()
        return closing.is_set()

    def capture():
        h.clock.now += state.capture_seconds
        return str(latest)

    executor._closing = SimpleNamespace(is_set=closing.is_set, set=closing.set, wait=advance)
    executor._reconfigure_training_pool(1, SimpleNamespace(),
        RuntimeReconfigurationResult(False, "test"))
    handle = executor._pending_hybrid_joins[h.identity.worker]
    for _ in range(200):
        if executor.has_pending_activations():
            break
        threading.Event().wait(.01)
    assert executor.has_pending_activations()
    assert h.clock.now == 70
    assert not handle.done()
    # Submission occurred at t=0, before the fake clock's cold-start polls.
    executor._pending_hybrid_join_started_at[h.identity.worker] = 0
    executor.train_engine = SimpleNamespace(optimizer_revision=14)
    monkeypatch.setattr(executor, "_capture_core_snapshot", capture)
    state.catching_up = True
    h.handle, h.catchup, h.acknowledge = handle, state, acknowledge
    try:
        yield h
    finally:
        closing.set()
        with executor._pending_activation_lock:
            for request in executor._pending_activations.values():
                request["error"] = "test cleanup"
                request["event"].set()


@pytest.mark.parametrize("boundary_at,capture_seconds", [(140, 0), (70, 90)],
    ids=["ack-at-190", "cold70-capture90-install50"])
def test_activation_queue_can_finish_after_cold_deadline(
        queued_activation, boundary_at, capture_seconds):
    h = queued_activation
    h.clock.now = boundary_at
    h.catchup.capture_seconds = capture_seconds
    h.catchup.ack_at = boundary_at + capture_seconds + 50
    h.executor._finish_pending_activations()
    workers = h.handle.result(timeout=3)
    assert h.clock.now == h.catchup.ack_at
    assert all(worker.join_state.value == "active" and worker.state_version == 14
        for worker in workers)
    assert h.pool.snapshot()["rollout0"].state_version == 14


def test_reconcile_does_not_cancel_a_join_waiting_in_activation_queue(queued_activation):
    h = queued_activation
    h.clock.now = 190
    h.executor._reconcile_pending_hybrid_joins()
    assert h.executor._pending_hybrid_joins[h.identity.worker] is h.handle
    assert not h.handle.done()
    assert h.pool.snapshot()["rollout0"].join_state.value == "activating"


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_queued_activation_late_ack_cannot_activate_cancelled_join(queued_activation, action):
    h = queued_activation
    if action == "cancel":
        h.handle.cancel()
    else:
        h.executor.close()
    h.clock.now = 90
    h.catchup.ack_at = 90
    h.acknowledge()
    with pytest.raises(RuntimeError):
        h.executor._finish_pending_activations()
    with pytest.raises(RuntimeError):
        h.handle.result(timeout=3)
    worker = h.pool.snapshot()["rollout0"]
    assert worker.join_state.value == "cancelled"
    assert worker.role.value != "hybrid_training"

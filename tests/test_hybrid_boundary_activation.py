"""Ready joins wait for Core boundaries; installation keeps its ACK deadline."""
from concurrent.futures import TimeoutError as FutureTimeout
import json
import queue
import threading
from types import SimpleNamespace as NS

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.elastic import hybrid_pool as pool_module
from RL_Framework.infra.elastic import runtime_executor as runtime_module
from RL_Framework.infra.elastic.hybrid_pool import ElasticHybridPool
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor


@pytest.fixture
def boundary_join(tmp_path, monkeypatch):
    clock = NS(now=0.0)
    timing = NS(monotonic=lambda: clock.now, time=lambda: clock.now)
    monkeypatch.setattr(runtime_module, "time", timing)
    monkeypatch.setattr(pool_module, "time", timing)
    config = AsyncRLConfig(model_path="/not-loaded", train_gpus=1, rollout_gpus=1,
        n_total_gpus=2, train_tp_size=1, train_dp_size=1)
    cfg = config.global_resource_planner
    cfg.elastic_hybrid_join_timeout_s = 180
    cfg.hybrid_worker_ready_timeout_s = 60
    cfg.hybrid_worker_task_dir = str(tmp_path)
    pool = ElasticHybridPool(core_train_workers=["dp0"],
        core_rollout_workers=["rollout0"], snapshot_fetcher=lambda *_: 0)
    runtime = RuntimeElasticExecutor(config=config, planner=NS(), elastic_pool=pool,
        train_engine=NS(optimizer_revision=14))
    checkpoints = queue.Queue()
    requests = []

    class ControlledEvent:
        """Expose each real queue poll without relying on thread scheduling."""
        def __init__(self):
            self.condition = threading.Condition()
            self.ready = False
            self.polls = 0
            requests.append(self)

        def set(self):
            with self.condition:
                self.ready = True
                self.condition.notify_all()

        def wait(self, _timeout):
            with self.condition:
                if not self.ready:
                    checkpoints.put("waiting")
                    assert self.condition.wait_for(
                        lambda: self.ready or self.polls, timeout=3), "test did not release the queue poll"
                if self.polls:
                    self.polls -= 1
                return self.ready

        def poll(self):
            with self.condition:
                self.polls += 1
                self.condition.notify_all()

    # Runtime locks and pool futures retain real threading primitives. Only
    # the request Event created inside _queue_activation is controlled.
    monkeypatch.setattr(runtime_module, "threading", NS(Event=ControlledEvent))
    closing = threading.Event()
    state = NS(capture_seconds=0, install_seconds=0, checkpoint=None)
    manifest = tmp_path / "state14.json"
    manifest.write_text(json.dumps({"format": "libra-ehp-lanes-v1",
        "snapshot_id": "capture14", "optimizer_revision": 14,
        "lanes": {"dp0": {"0": "ehp://source/lane14"}}}))

    def advance(seconds):
        clock.now += seconds
        requests[0].poll()
        state.checkpoint = checkpoints.get(timeout=3)

    def acknowledge():
        receipt = dict(snapshot=str(manifest), rank=0, pid=123, acknowledged=True,
            snapshot_id="capture14", optimizer_revision=14, state_version=14,
            lane_snapshot_id="lane14", target_core_id="dp0", source_uri="ehp://source/lane14")
        (tmp_path / "replica.rank_0.ready").write_text(
            "".join(f"{key}={value}\n" for key, value in receipt.items()))

    def capture():
        if state.capture_seconds:
            advance(state.capture_seconds)
        return str(manifest)

    def install(_seconds):
        advance(state.install_seconds)
        acknowledge()
        return closing.is_set()

    runtime._closing = NS(is_set=closing.is_set, set=closing.set, wait=install)
    monkeypatch.setattr(runtime, "_capture_core_snapshot", capture)
    monkeypatch.setattr(runtime, "_publish_hybrid_membership", lambda: None)
    monkeypatch.setattr(runtime, "_stop_managed_rollout_processes", lambda: None)

    def activate(replica_id, members, target_core_id, version):
        return runtime._queue_activation(replica_id,
            dict(acknowledged=True, state_version=version, optimizer_revision=version))

    handle = pool.join_replica("replica", ["rollout0"], "dp0",
        replica_activation_barrier=activate, timeout_s=180)
    handle.future.add_done_callback(lambda future: checkpoints.put("finished"))
    runtime._pending_hybrid_joins["replica"] = handle
    runtime._pending_hybrid_join_started_at["replica"] = 0
    assert checkpoints.get(timeout=3) == "waiting"
    assert runtime.has_pending_activations() and not handle.done()

    def finish():
        if not state.install_seconds:
            acknowledge()
        runtime._finish_pending_activations()
        return handle.result(timeout=3)

    h = NS(runtime=runtime, pool=pool, clock=clock, state=state,
        advance=advance, finish=finish, handle=handle)
    try:
        yield h
    finally:
        closing.set()
        for event in requests:
            event.set()
        pool.close()
        try:
            handle.result(timeout=3)
        except FutureTimeout:
            if not handle.done():
                raise
        except RuntimeError:
            pass


def test_ready_join_waits_past_180_seconds_until_a_core_boundary(boundary_join):
    h = boundary_join
    h.advance(181)
    workers = h.finish()
    assert h.clock.now == 181
    assert all(worker.join_state.value == "active" and worker.state_version == 14
        for worker in workers)


def test_capture_and_ack_can_cross_the_old_queue_deadline(boundary_join):
    h = boundary_join
    h.advance(50)
    h.state.capture_seconds = 96
    h.state.install_seconds = 50
    workers = h.finish()
    assert h.clock.now == 196
    assert all(worker.join_state.value == "active" and worker.state_version == 14
        for worker in workers)
    assert h.pool.snapshot()["rollout0"].state_version == 14


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_cancel_and_cleanup_still_wake_ready_join(boundary_join, action):
    h = boundary_join
    if action == "cancel":
        h.runtime._release_all_hybrid_workers(reason="test cancellation")
    else:
        h.runtime.close()
        h.advance(1)  # the production Event's next 50ms poll observes closing
    with pytest.raises(RuntimeError, match="cancelled"):
        h.handle.result(timeout=3)
    worker = h.pool.snapshot()["rollout0"]
    assert worker.join_state.value == "cancelled"
    assert worker.role.value != "hybrid_training"


@pytest.mark.parametrize("cold,seconds", [(True, 180), (False, 60)])
def test_worker_installation_ack_still_has_its_existing_deadline(
        tmp_path, monkeypatch, cold, seconds):
    clock = NS(now=0.0)
    monkeypatch.setattr(runtime_module, "time", NS(monotonic=lambda: clock.now))
    config = AsyncRLConfig(model_path="/not-loaded", train_gpus=1, rollout_gpus=1,
        n_total_gpus=2, train_tp_size=1, train_dp_size=1)
    config.global_resource_planner.hybrid_worker_task_dir = str(tmp_path)
    config.global_resource_planner.hybrid_worker_ready_timeout_s = 60
    runtime = RuntimeElasticExecutor(config=config, planner=NS())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "libra-ehp-lanes-v1",
        "snapshot_id": "state", "optimizer_revision": 2,
        "lanes": {"dp0": {"0": "ehp://source/lane0"}}}))

    def advance(_seconds):
        clock.now += 1
        return False

    runtime._closing = NS(is_set=lambda: False, wait=advance)
    with pytest.raises(TimeoutError, match=f"after {seconds}s"):
        runtime._wait_hybrid_worker_ready("replica", state_version=2,
            replica_world_size=1, snapshot_path=str(manifest), target_core_id="dp0",
            **({"timeout_s": 180} if cold else {}))
    assert clock.now == seconds

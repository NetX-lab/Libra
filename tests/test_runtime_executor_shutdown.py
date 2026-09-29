"""Deterministic CPU tests; no actual worker, subprocess, GPU or network."""

import json
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.elastic.hybrid_pool import ElasticHybridPool, JoinCancelledError
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor, RuntimeReconfigurationResult
from RL_Framework.infra.elastic import runtime_executor as runtime_module


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.terminations = 0
        self.kills = 0
        self.fail_stop = False

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.fail_stop:
            raise OSError("simulated stop failure")
        self.terminations += 1
        self.returncode = -15

    def kill(self):
        self.kills += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def harness(tmp_path, monkeypatch):
    instances = []
    processes = []

    def popen(*args, **kwargs):
        proc = FakeProcess(100 + len(processes))
        processes.append(proc)
        return proc

    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)

    def make(workers=3, background=3):
        cfg = AsyncRLConfig(
            model_path="/unused", train_gpus=1, rollout_gpus=workers,
            n_total_gpus=workers + 1, train_tp_size=1, train_dp_size=1,
            batch_size=1, n_samples=1, micro_batch_size=1, log_dir=str(tmp_path),
        )
        cfg.global_resource_planner.hybrid_worker_task_dir = str(tmp_path)
        pool = ElasticHybridPool(
            core_train_workers=["dp0"],
            core_rollout_workers=[f"rollout{i}" for i in range(workers)],
            zero_sync_steps=0, max_background_workers=background,
        )
        executor = RuntimeElasticExecutor(config=cfg, planner=SimpleNamespace(), elastic_pool=pool)
        monkeypatch.setattr(executor, "_build_hybrid_worker_command", lambda **kw: "fake worker")
        monkeypatch.setattr(executor, "_cluster_swap_training_slot", lambda wid: {"host": "fake-host", "gpus": [0]})
        # Most tests isolate launch ordering. A separate test keeps the real poll.
        monkeypatch.setattr(executor, "_wait_hybrid_worker_ready", lambda wid: None)
        h = SimpleNamespace(executor=executor, pool=pool, processes=processes, popen=popen, path=tmp_path)
        instances.append(h)
        return h

    yield make
    for h in instances:
        for proc in processes:
            proc.fail_stop = False
        h.executor.close()


def join(h, *, replica=False, worker="rollout0", callback=None):
    callback = callback or h.executor._activate_hybrid_worker
    if replica:
        return h.pool.join_replica("replica_" + worker, [worker], "dp0", activation_barrier=callback)
    return h.pool.join_training(worker, "dp0", activation_barrier=callback)


def assert_cancelled(handle):
    with pytest.raises((JoinCancelledError, CancelledError)):
        handle.result(timeout=3)


@pytest.mark.parametrize("replica", [False, True], ids=["singleton", "replica"])
@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_s1_close_fences_callback_paused_before_popen(harness, monkeypatch, replica, remote):
    h = harness()
    h.executor.config.global_resource_planner.hybrid_worker_remote_control_enabled = remote
    entered, resume = threading.Event(), threading.Event()

    def prepare(**kwargs):
        entered.set()
        assert resume.wait(3)
        return "fake worker"

    monkeypatch.setattr(h.executor, "_build_hybrid_worker_command", prepare)
    handle = join(h, replica=replica)
    try:
        assert entered.wait(3)
        h.executor.close()
        count_at_close = len(h.processes)
        pool_at_close = h.pool.snapshot()
    finally:
        resume.set()
    assert_cancelled(handle)
    assert h.pool.snapshot() == pool_at_close
    assert count_at_close == len(h.processes) == 0
    assert h.executor._hybrid_worker_processes == {}
    assert h.executor._hybrid_worker_meta == {}
    assert not list(h.path.glob(".launch_*.json"))


def test_s2_registered_process_stopped_before_close_returns(harness):
    h = harness()
    join(h).result(timeout=3)
    proc = h.processes[0]
    h.executor.close()
    assert proc.poll() == -15
    assert proc.terminations == 1
    assert h.executor._hybrid_worker_processes == {}
    assert h.executor._hybrid_worker_meta == {}


def test_s3_popen_to_registration_is_atomic_with_close(harness, monkeypatch):
    h = harness()
    created, resume, closing = threading.Event(), threading.Event(), threading.Event()

    def popen(*args, **kwargs):
        proc = h.popen(*args, **kwargs)
        created.set()
        assert resume.wait(3)
        return proc

    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    handle = join(h)

    def close():
        closing.set()
        h.executor.close()

    with ThreadPoolExecutor(max_workers=1) as runner:
        try:
            assert created.wait(3)
            finished = runner.submit(close)
            assert closing.wait(3)
            with pytest.raises(TimeoutError):
                finished.result(timeout=0.05)
        finally:
            resume.set()
        finished.result(timeout=3)
    try:
        handle.result(timeout=3)
    except JoinCancelledError:
        pass  # Either final activation or close may acquire the pool lock first.
    assert len(h.processes) == 1
    assert h.processes[0].poll() is not None
    assert h.executor._hybrid_worker_processes == {}


def test_s4_close_cancels_queued_join(harness):
    h = harness(background=1)
    entered, resume = threading.Event(), threading.Event()

    def snapshot(*args):
        entered.set()
        assert resume.wait(3)
        return 0

    h.pool.snapshot_fetcher = snapshot
    first = join(h)
    try:
        assert entered.wait(3)
        queued = join(h, worker="rollout1")
        h.executor.close()
        assert queued.future.cancelled()
    finally:
        resume.set()
    assert_cancelled(first)
    assert_cancelled(queued)
    assert not h.processes


def test_s5_mixed_activations_all_reach_safe_terminal_state(harness):
    h = harness(background=1)
    join(h).result(timeout=3)
    entered, resume = threading.Event(), threading.Event()

    def paused(*args):
        entered.set()
        assert resume.wait(3)
        h.executor._activate_hybrid_worker(*args)

    running = join(h, worker="rollout1", callback=paused)
    try:
        assert entered.wait(3)
        queued = join(h, worker="rollout2")
        h.executor.close()
    finally:
        resume.set()
    assert_cancelled(running)
    assert_cancelled(queued)
    assert len(h.processes) == 1
    assert h.processes[0].poll() is not None
    assert not h.executor._hybrid_worker_processes


def test_s6_close_is_idempotent_including_services(harness):
    h = harness()
    join(h).result(timeout=3)
    calls = []
    h.executor.gradient_server = SimpleNamespace(close=lambda: calls.append("closed"))
    h.executor.close()
    h.executor.close()
    assert calls == ["closed"]
    assert h.processes[0].terminations == 1


def test_s7_new_joins_rejected_after_close(harness):
    h = harness()
    h.executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        join(h)
    with pytest.raises(RuntimeError, match="closed"):
        join(h, replica=True)
    with pytest.raises(JoinCancelledError, match="closing"):
        h.executor._ensure_elastic_pool()
    with pytest.raises(JoinCancelledError, match="closing"):
        h.executor._ensure_gradient_server()


@pytest.mark.parametrize("replica", [False, True], ids=["singleton", "replica"])
def test_s8_cancelled_generation_cannot_launch_after_rejoin(harness, monkeypatch, replica):
    h = harness()
    entered, resume = threading.Event(), threading.Event()

    def prepare(**kwargs):
        entered.set()
        assert resume.wait(3)
        return "fake worker"

    monkeypatch.setattr(h.executor, "_build_hybrid_worker_command", prepare)
    old = join(h, replica=replica)
    try:
        assert entered.wait(3)
        old.cancel()
        newer = join(h, replica=replica, callback=lambda *args: None)
        newer.result(timeout=3)
    finally:
        resume.set()
    assert_cancelled(old)
    assert not h.processes
    assert not h.executor._hybrid_worker_processes


def test_s9_remote_start_is_fenced_and_committed_start_gets_stop(harness):
    h = harness()
    h.executor.config.global_resource_planner.hybrid_worker_remote_control_enabled = True
    args = dict(worker_id="rollout0", target_core_id="dp0", snapshot_path="/fake", command="fake", slot={"host": "fake-host", "gpus": [0]})
    h.executor._request_remote_hybrid_worker(**args)
    assert (h.path / ".launch_rollout0.json").exists()
    h.executor.close()
    assert not (h.path / ".launch_rollout0.json").exists()
    assert json.loads((h.path / ".stop_rollout0.json").read_text())["worker_id"] == "rollout0"
    with pytest.raises(JoinCancelledError):
        h.executor._request_remote_hybrid_worker(**args)
    assert not (h.path / ".launch_rollout0.json").exists()
    assert not h.executor._hybrid_worker_meta


def test_s10_callback_failure_during_close_cannot_skip_other_workers(harness):
    h = harness()
    join(h).result(timeout=3)
    entered, resume = threading.Event(), threading.Event()

    def fail(*args):
        entered.set()
        assert resume.wait(3)
        raise ValueError("callback failed")

    handle = join(h, worker="rollout1", callback=fail)
    try:
        assert entered.wait(3)
        h.executor.close()
        state = h.pool.snapshot()
    finally:
        resume.set()
    with pytest.raises(ValueError, match="callback failed"):
        handle.result(timeout=3)
    assert state == h.pool.snapshot()
    assert h.processes[0].poll() is not None
    assert not h.executor._hybrid_worker_processes


def test_readiness_poll_is_cancelled_without_waiting_for_timeout(harness, monkeypatch):
    h = harness()
    entered = threading.Event()
    h.executor.config.global_resource_planner.hybrid_worker_ready_timeout_s = 3600

    def wait(worker):
        entered.set()
        RuntimeElasticExecutor._wait_hybrid_worker_ready(h.executor, worker)

    monkeypatch.setattr(h.executor, "_wait_hybrid_worker_ready", wait)
    handle = join(h)
    assert entered.wait(3)
    h.executor.close()
    assert_cancelled(handle)
    assert h.processes[0].poll() is not None


def test_replica_launch_callback_and_alias_cleanup(harness, monkeypatch):
    h = harness()
    h.executor.config.global_resource_planner.hybrid_worker_launch_enabled = True
    monkeypatch.setattr(h.executor, "_desired_hybrid_workers", lambda *args: 1)
    monkeypatch.setattr(h.executor, "_training_reconfiguration_targets", lambda *args: (["dp0"], "dp0", ["rollout0"]))
    h.executor._reconfigure_training_pool(1, SimpleNamespace(train_gpus=2), RuntimeReconfigurationResult(False, "test"))
    handle = next(iter(h.executor._pending_hybrid_joins.values()))
    handle.result(timeout=3)
    assert len(h.executor._hybrid_worker_processes) == 2
    h.executor.close()
    assert h.processes[0].terminations == 1
    assert not h.executor._hybrid_worker_processes
    assert not h.executor._hybrid_worker_meta


def test_replica_group_callback_preparation_can_finish_after_close(harness, monkeypatch):
    h = harness()
    h.executor.config.global_resource_planner.hybrid_worker_launch_enabled = True
    entered, resume = threading.Event(), threading.Event()

    def prepare(**kwargs):
        entered.set()
        assert resume.wait(3)
        return "fake worker"

    monkeypatch.setattr(h.executor, "_build_hybrid_worker_command", prepare)
    monkeypatch.setattr(h.executor, "_desired_hybrid_workers", lambda *args: 1)
    monkeypatch.setattr(h.executor, "_training_reconfiguration_targets", lambda *args: (["dp0"], "dp0", ["rollout0"]))
    h.executor._reconfigure_training_pool(1, SimpleNamespace(train_gpus=2), RuntimeReconfigurationResult(False, "test"))
    handle = next(iter(h.executor._pending_hybrid_joins.values()))
    try:
        assert entered.wait(3)
        h.executor.close()
    finally:
        resume.set()
    assert_cancelled(handle)
    assert not h.processes
    assert not h.executor._pending_hybrid_joins
    assert not h.executor._hybrid_worker_processes


def test_remote_stop_publication_failure_is_not_silent(harness, monkeypatch):
    h = harness()
    h.executor.config.global_resource_planner.hybrid_worker_remote_control_enabled = True
    h.executor._request_remote_hybrid_worker(worker_id="rollout0", target_core_id="dp0", snapshot_path="/fake", command="fake", slot={"host": "fake-host"})
    write = h.executor._write_json_atomic

    def fail(*args):
        raise OSError("stop publication failed")

    monkeypatch.setattr(h.executor, "_write_json_atomic", fail)
    with pytest.raises(RuntimeError, match="stop publication failed"):
        h.executor.close()
    assert "rollout0" in h.executor._hybrid_worker_meta
    monkeypatch.setattr(h.executor, "_write_json_atomic", write)
    h.executor.close()
    assert not h.executor._hybrid_worker_meta


def test_close_failure_keeps_ownership_and_cleans_other_workers(harness):
    h = harness()
    join(h).result(timeout=3)
    join(h, worker="rollout1").result(timeout=3)
    h.processes[0].fail_stop = True
    with pytest.raises(RuntimeError, match="shutdown cleanup failed"):
        h.executor.close()
    assert h.executor._closing.is_set()
    assert not h.executor._closed
    assert h.processes[1].poll() is not None
    assert "rollout0" in h.executor._hybrid_worker_processes
    count = len(h.processes)
    with pytest.raises(JoinCancelledError, match="closing"):
        h.executor._launch_prewarmed_hybrid_worker(
            worker_id="rollout2", target_core_id="dp0", command="fake", snapshot_path="/fake",
        )
    assert len(h.processes) == count
    h.processes[0].fail_stop = False
    h.executor.close()
    assert h.executor._closed
    assert not h.executor._hybrid_worker_processes


def test_prewarm_and_legacy_launch_are_fenced(harness):
    h = harness()
    handle = join(h, callback=lambda *args: None)
    handle.result(timeout=3)
    h.executor.close()
    with pytest.raises(JoinCancelledError):
        h.executor._launch_prewarmed_hybrid_worker(worker_id="rollout0", target_core_id="dp0", command="fake", snapshot_path="/fake")
    assert h.executor._launch_hybrid_worker_after_join(handle) is None
    assert not h.processes


def test_restart_committed_before_close_is_also_reaped(harness, monkeypatch):
    h = harness()
    h.executor._launch_prewarmed_hybrid_worker(
        worker_id="rollout0", target_core_id="dp0", command="fake", snapshot_path="/old",
    )
    entered, resume = threading.Event(), threading.Event()

    def wait(worker):
        entered.set()
        assert resume.wait(3)

    monkeypatch.setattr(h.executor, "_wait_hybrid_worker_ready", wait)
    handle = join(h)
    try:
        assert entered.wait(3)
        assert len(h.processes) == 2
        h.executor.close()
    finally:
        resume.set()
    assert_cancelled(handle)
    assert all(proc.poll() is not None for proc in h.processes)
    assert not h.executor._hybrid_worker_processes


def test_legacy_completed_join_generation_is_checked_at_launch(harness):
    h = harness()
    old = join(h, callback=lambda *args: None)
    old.result(timeout=3)
    old.cancel()
    join(h, callback=lambda *args: None).result(timeout=3)
    assert h.executor._launch_hybrid_worker_after_join(old) is None
    assert not h.processes


def test_close_from_activation_does_not_join_its_own_thread(harness):
    h = harness()
    handle = join(h, callback=lambda *args: h.executor.close())
    assert_cancelled(handle)
    assert h.executor._closed
    assert not h.processes

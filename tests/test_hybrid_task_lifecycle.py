"""Hybrid completion receipts fence normal return; exit fences cancellation."""
import threading
from types import SimpleNamespace

import pytest

from RL_Framework.infra.elastic.runtime_executor import ManagedHybridWorkerProcess, RuntimeElasticExecutor


def _executor(tmp_path):
    executor = RuntimeElasticExecutor.__new__(RuntimeElasticExecutor)
    executor.config = SimpleNamespace(global_resource_planner=SimpleNamespace(
        hybrid_worker_task_dir=str(tmp_path), hybrid_worker_remote_control_dir=str(tmp_path),
        hybrid_worker_remote_control_enabled=True, hybrid_update_timeout_s=.3,
        hybrid_worker_stop_timeout_s=.1))
    executor._lifecycle_lock = threading.RLock()
    executor._hybrid_worker_processes = {}
    canonical = ManagedHybridWorkerProcess('h', 'dp0', 'worker', 123, 'snapshot', host='node')
    member = ManagedHybridWorkerProcess('member', 'dp0', 'worker', -1, 'snapshot', host='node', replica_id='h')
    executor._hybrid_worker_meta = {'h': canonical, 'member': member}
    return executor


def test_normal_return_waits_for_actual_task_completion_before_detach(tmp_path):
    executor = _executor(tmp_path)
    executor._snapshot_cleanup_requested = False
    executor._pending_activation_lock = threading.RLock()
    executor._pending_activations = {}
    executor._pending_hybrid_join_lock = threading.RLock()
    executor._pending_hybrid_joins = {}
    executor._pending_hybrid_join_started_at = {}
    executor._retired_hybrid_joins = []
    executor._publish_hybrid_membership = lambda: None
    worker = SimpleNamespace(worker_id='member', role='hybrid_training')
    detached = []
    executor.elastic_pool = SimpleNamespace(replica_snapshot=lambda: {'h': (worker,)},
        release_replica_to_rollout=lambda replica: detached.append(replica))
    executor._stop_snapshot_loader = lambda *_: None
    task = tmp_path / 'h.step_14.pt'
    task.write_text('already consumed by Core, still applying on Hybrid')
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (executor._release_all_hybrid_workers(reason='planner'), finished.set()))
    thread.start()
    try:
        assert not finished.wait(.08)
        assert not detached
        task.rename(str(task) + '.done')
        assert finished.wait(2)
        assert detached == ['h']
    finally:
        task.rename(str(task) + '.done') if task.exists() else None
        thread.join(2)


def test_failed_or_missing_completion_is_not_success(tmp_path):
    executor = _executor(tmp_path)
    executor.config.global_resource_planner.hybrid_update_timeout_s = .05
    task = tmp_path / 'h.step_14.pt'; task.touch()
    with pytest.raises(TimeoutError, match='completion'):
        executor.wait_for_hybrid_training_tasks('member')
    assert task.exists() and not (tmp_path / 'h.step_14.pt.done').exists()


def test_worker_failure_while_waiting_propagates_without_completion(tmp_path):
    executor = _executor(tmp_path)
    task = tmp_path / 'h.step_14.pt'; task.touch()
    worker = threading.Timer(.02, lambda: task.rename(str(task) + '.failed'))
    worker.start()
    try:
        with pytest.raises(RuntimeError, match='failed before completion'):
            executor.wait_for_hybrid_training_tasks('member')
    finally:
        worker.join(2)
    assert not (tmp_path / 'h.step_14.pt.done').exists()


def test_confirmed_exit_retires_only_old_owner_tasks_without_faking_done(tmp_path, monkeypatch):
    from RL_Framework.infra.elastic import rollout_lending
    executor = _executor(tmp_path)
    task = tmp_path / 'h.step_14.pt'; task.write_text('membership_epoch=7;state_version=14')
    other = tmp_path / 'other.step_14.pt'; other.touch()
    (tmp_path / 'h.step_13.pt.done').touch()
    observed = []
    def acknowledge(executor, worker, request_id=None):
        observed.append(worker)
        assert task.exists() and 'h' in executor._hybrid_worker_meta
        return dict(worker_id=worker, request_id=request_id, exited=True)
    monkeypatch.setattr(rollout_lending, 'wait_hybrid_stopped', acknowledge)
    executor._stop_hybrid_worker_process('member')
    assert observed == ['h']
    assert not task.exists() and not (tmp_path / 'h.step_14.pt.done').exists()
    assert (tmp_path / 'h.step_14.pt.cancelled').read_text() == 'membership_epoch=7;state_version=14'
    assert other.exists() and (tmp_path / 'h.step_13.pt.done').exists()
    assert not executor._hybrid_worker_meta


def test_failed_stop_retains_pending_task_for_owned_retry(tmp_path, monkeypatch):
    from RL_Framework.infra.elastic import rollout_lending
    executor = _executor(tmp_path)
    task = tmp_path / 'h.step_14.pt'; task.touch()
    def fail(*args, **kwargs):
        raise TimeoutError('old process has not exited')
    monkeypatch.setattr(rollout_lending, 'wait_hybrid_stopped', fail)
    with pytest.raises(TimeoutError):
        executor._stop_hybrid_worker_process('member')
    assert task.exists() and executor._hybrid_worker_meta
    assert not (tmp_path / 'h.step_14.pt.cancelled').exists()


def test_local_exit_also_retires_unfinished_task(tmp_path, monkeypatch):
    from RL_Framework.infra.elastic import process_lifecycle
    executor = _executor(tmp_path)
    proc = object()
    executor._hybrid_worker_processes = {'h': proc, 'member': proc}
    task = tmp_path / 'h.step_14.pt'; task.touch()
    exited = []
    def stop(owned):
        assert owned is proc and task.exists()
        exited.append(True)
    monkeypatch.setattr(process_lifecycle, 'stop_local_worker', stop)
    executor._stop_hybrid_worker_process('member')
    assert exited and not task.exists()
    assert (tmp_path / 'h.step_14.pt.cancelled').exists()
    assert not executor._hybrid_worker_processes and not executor._hybrid_worker_meta

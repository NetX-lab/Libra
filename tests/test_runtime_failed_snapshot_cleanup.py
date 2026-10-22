"""Snapshot ownership survives join failure until a common Core boundary."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest
import torch

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.elastic.core_reconfiguration import execute_core_command
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor
from RL_Framework.infra.elastic.state_transfer import LiveStateServer


class SnapshotEngine:
    rank = 0
    world_size = 1
    optimizer_revision = 1

    def __init__(self):
        self.server = LiveStateServer(host='127.0.0.1', advertised_host='127.0.0.1')
        self.urls, self.released = {}, []

    def capture_elastic_state_snapshot(self, *args):
        revision = self.optimizer_revision
        if revision not in self.urls:
            self.urls[revision] = self.server.publish({'model': torch.zeros(32)})
        return revision

    def get_elastic_state_snapshot_path(self, revision):
        return self.urls[revision]

    def get_elastic_lane_state(self):
        return dict(data_parallel_rank=0, tensor_parallel_rank=0,
                    context_parallel_rank=0, pipeline_parallel_rank=0)

    def release_elastic_state_snapshot(self, revision):
        uri = self.urls.pop(revision, None)
        if uri:
            self.server.release(uri)
        self.released.append(revision)


@pytest.fixture
def runtime(tmp_path):
    config = AsyncRLConfig(model_path='/not-loaded', train_gpus=1, rollout_gpus=1,
        n_total_gpus=2, train_tp_size=1, train_dp_size=1, log_dir=str(tmp_path))
    engine = SnapshotEngine()
    executor = RuntimeElasticExecutor(config=config, planner=NS(), train_engine=engine)
    executor._active_runtime_coord_id = 'snapshot-test'
    yield executor, engine
    engine.server.close()


def failed_join():
    future = Future()
    future.set_exception(TimeoutError('initial installation failed'))
    return NS(member_worker_ids=('rollout0',), done=future.done, result=future.result)


def boundary(executor, monkeypatch, *, execute=True):
    prepared = NS(coord_id='common-boundary', should_execute=execute, candidate_plan=None,
        decision=None, result=NS(applied=True), effective_planner_signal=None)
    monkeypatch.setattr(executor, 'accept_planner_signal', lambda *args, **kwargs: None)
    return executor.execute_prepared(prepared)


def fail_snapshot(executor):
    path = executor._capture_core_snapshot()
    executor._pending_hybrid_joins['replica'] = failed_join()
    executor._reconcile_pending_hybrid_joins()
    return path


def test_failed_join_releases_only_at_next_common_boundary(runtime, monkeypatch):
    executor, engine = runtime
    path = fail_snapshot(executor)
    assert engine.released == []  # metrics/background reconciliation is not collective
    boundary(executor, monkeypatch)
    assert engine.released == [1]
    assert not executor._snapshot_manifests
    assert not engine.server._snapshots
    from pathlib import Path
    assert not Path(path).exists()


@pytest.mark.parametrize('busy', ['join', 'activation', 'cancelled_loader', 'stop'])
def test_release_waits_for_every_consumer(runtime, monkeypatch, busy):
    executor, engine = runtime
    fail_snapshot(executor)
    if busy == 'join':
        executor._pending_hybrid_joins['other'] = NS(done=lambda: False)
    elif busy == 'activation':
        executor._pending_activations['other'] = {}
    elif busy == 'cancelled_loader':
        executor._retired_hybrid_joins.append(NS(done=lambda: False))
    else:
        executor._snapshot_stop_pending.add('rollout0')
        monkeypatch.setattr(executor, '_stop_hybrid_worker_process',
            lambda *args: (_ for _ in ()).throw(RuntimeError('exit not acknowledged')))
    if busy == 'stop':
        with pytest.raises(RuntimeError, match='exit not acknowledged'):
            boundary(executor, monkeypatch)
    else:
        boundary(executor, monkeypatch)
    assert engine.released == []
    assert executor._snapshot_manifests


def test_no_transaction_cannot_issue_release(runtime, monkeypatch):
    executor, engine = runtime
    fail_snapshot(executor)
    boundary(executor, monkeypatch, execute=False)
    assert engine.released == []


def test_collective_release_failure_retains_manifest_for_retry(runtime, monkeypatch):
    executor, engine = runtime
    path = fail_snapshot(executor)
    import RL_Framework.infra.elastic.core_reconfiguration as commands
    original = commands.dispatch_core_command
    monkeypatch.setattr(commands, 'dispatch_core_command',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('rank1 receipt missing')))
    with pytest.raises(RuntimeError, match='rank1 receipt missing'):
        boundary(executor, monkeypatch)
    assert executor._snapshot_manifests == {1: path}
    monkeypatch.setattr(commands, 'dispatch_core_command', original)
    boundary(executor, monkeypatch)
    assert not executor._snapshot_manifests


def test_repeated_failed_revisions_do_not_accumulate(runtime, monkeypatch):
    executor, engine = runtime
    for revision in (1, 2, 3):
        engine.optimizer_revision = revision
        fail_snapshot(executor)
        boundary(executor, monkeypatch)
        assert not engine.server._snapshots
    assert engine.released == [1, 2, 3]


def test_every_core_rank_releases_before_boundary_returns(runtime, monkeypatch):
    import threading
    from RL_Framework.infra.elastic.runtime_boundary import read_record
    executor, engine = runtime
    fail_snapshot(executor)
    peer = SnapshotEngine()
    peer.rank = 1
    peer.capture_elastic_state_snapshot()
    engine.world_size = 2
    errors = []
    finished = threading.Event()

    def follow():
        try:
            while not finished.wait(.005):
                command = read_record(executor._coordination_dir() / 'core_command.json')
                if command.get('operation') == 'release':
                    execute_core_command(peer, command, executor._coordination_dir())
                    return
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=follow)
    thread.start()
    try:
        boundary(executor, monkeypatch)
        assert not errors
        assert engine.released == peer.released == [1]
        assert not executor._snapshot_manifests
    finally:
        finished.set()
        thread.join(timeout=2)
        peer.server.close()


def test_pending_cleanup_requests_membership_boundary_without_planner_resize(runtime):
    executor, _engine = runtime
    fail_snapshot(executor)
    assert executor.has_pending_boundary_work()
    current = NS()
    decision = NS(should_reconfigure=False, current_plan=current, candidate_plan=None,
        resource_should_reconfigure=False, reason='hold', elastic_hybrid_signal=None)
    prepared = executor.prepare_transaction(decision)
    assert prepared.should_execute
    assert not prepared.resource_reconfiguration
    assert prepared.candidate_plan is current
    assert prepared.coord_id


def test_cancelled_loader_remains_owned_until_future_finishes(runtime, monkeypatch):
    import threading
    executor, engine = runtime
    executor._capture_core_snapshot()
    future = Future()
    cancelled = []
    handle = NS(member_worker_ids=('rollout0',), done=future.done,
        cancel=lambda: cancelled.append(True))
    executor._pending_hybrid_joins['replica'] = handle
    event = threading.Event()
    request = {'event': event}
    executor._pending_activations['replica'] = request
    executor._release_all_hybrid_workers(reason='pressure')
    assert cancelled and event.is_set() and request['error']
    assert not executor._pending_hybrid_joins
    boundary(executor, monkeypatch)
    assert not engine.released
    future.set_exception(RuntimeError('cancelled generation'))
    boundary(executor, monkeypatch)
    assert engine.released == [1]


def test_partial_cancel_stop_failure_tracks_all_removed_loaders(runtime, monkeypatch):
    executor, engine = runtime
    executor._capture_core_snapshot()
    futures = [Future(), Future()]
    cancelled = []
    for index, future in enumerate(futures):
        handle = NS(member_worker_ids=(f'rollout{index}',), done=future.done,
            cancel=lambda index=index: cancelled.append(index))
        executor._pending_hybrid_joins[str(index)] = handle
    monkeypatch.setattr(executor, '_stop_hybrid_worker_process',
        lambda *args: (_ for _ in ()).throw(RuntimeError('stop failed')))
    with pytest.raises(RuntimeError, match='stop failed'):
        executor._release_all_hybrid_workers(reason='pressure')
    assert cancelled == [0, 1]
    assert len(executor._retired_hybrid_joins) == 2
    assert not engine.released
    assert executor._snapshot_stop_pending == {'rollout0'}


def test_cold_loader_does_not_request_repeated_cleanup_boundaries(runtime):
    executor, _engine = runtime
    executor._capture_core_snapshot()
    future = Future()
    executor._pending_hybrid_joins['replica'] = NS(done=future.done)
    assert not executor.has_pending_boundary_work()
    future.set_exception(TimeoutError('cold load failed'))
    assert executor.has_pending_boundary_work()


def test_cancelled_fetch_requests_cleanup_only_after_it_stops(runtime):
    executor, _engine = runtime
    executor._capture_core_snapshot()
    future = Future()
    executor._retired_hybrid_joins.append(NS(done=future.done))
    assert not executor.has_pending_boundary_work()
    future.set_result(None)
    assert executor.has_pending_boundary_work()

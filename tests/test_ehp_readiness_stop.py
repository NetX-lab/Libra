from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import threading
import time

import pytest

from RL_Framework.infra.elastic.readiness import clear_ready_receipts, read_ready_receipts, snapshot_contract
from RL_Framework.infra.elastic.runtime_executor import (
    ManagedHybridWorkerProcess, ManagedRolloutProcess, RuntimeElasticExecutor,
)


def _executor(tmp_path, **options):
    engine = RuntimeElasticExecutor.__new__(RuntimeElasticExecutor)
    cfg = dict(hybrid_worker_task_dir=str(tmp_path), hybrid_worker_remote_control_dir=str(tmp_path),
               hybrid_worker_ready_timeout_s=.15, hybrid_worker_stop_timeout_s=.15,
               hybrid_worker_remote_control_enabled=False, vllm_stop_timeout_s=.1,
               vllm_stop_command_template='stop {instance_id}')
    cfg.update(options)
    engine.config = SimpleNamespace(global_resource_planner=SimpleNamespace(**cfg))
    engine._hybrid_worker_meta = {}
    engine._hybrid_worker_processes = {}
    engine._closing = threading.Event()
    engine._lifecycle_lock = threading.RLock()
    engine._hybrid_resource_guard = lambda *args, **kwargs: nullcontext()
    return engine


def _manifest(tmp_path):
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps({'format': 'libra-ehp-lanes-v1', 'optimizer_revision': 9,
                                'snapshot_id': 'actual-boundary', 'lanes': {'dp1': {'0': 'ehp://a', '1': 'ehp://b'}}}))
    return str(path)


def _ready(tmp_path, snapshot, lane_rank, **overrides):
    fields = dict(snapshot=snapshot, snapshot_id='actual-boundary', lane_snapshot_id=f'lane-{lane_rank}',
                  acknowledged='True', state_version=9, optimizer_revision=9, target_core_id='dp1',
                  rank=lane_rank, pid=123 + lane_rank, source_uri='ehp://a' if lane_rank == 0 else 'ehp://b')
    fields.update(overrides)
    (tmp_path / f'h.rank_{lane_rank}.ready').write_text(''.join(f'{key}={value}\n' for key, value in fields.items()))


def test_requires_every_model_lane_and_rejects_generic_virtual_ready(tmp_path):
    snapshot = _manifest(tmp_path)
    engine = _executor(tmp_path)
    (tmp_path / 'h.ready').write_text('ready')
    _ready(tmp_path, snapshot, 0)
    with pytest.raises(TimeoutError, match='all 2 lanes'):
        engine._wait_hybrid_worker_ready('h', state_version=9, replica_world_size=2,
                                         snapshot_path=snapshot, target_core_id='dp1')
    _ready(tmp_path, snapshot, 1)
    receipt = engine._wait_hybrid_worker_ready('h', state_version=9, replica_world_size=2,
                                              snapshot_path=snapshot, target_core_id='dp1')
    assert receipt['acknowledged'] and len(receipt['rank_receipts']) == 2


@pytest.mark.parametrize('change', [dict(snapshot_id='previous-boundary'), dict(optimizer_revision=8),
                                     dict(state_version=8), dict(rank=5), dict(acknowledged='False'),
                                     dict(target_core_id='dp0'), dict(lane_snapshot_id=''),
                                     dict(source_uri='ehp://wrong-lane'), dict(pid=123)])
def test_stale_or_wrong_lane_receipt_cannot_release_activation(tmp_path, change):
    snapshot = _manifest(tmp_path)
    contract = snapshot_contract(snapshot, state_version=9, target_core_id='dp1')
    _ready(tmp_path, snapshot, 0)
    _ready(tmp_path, snapshot, 1, **change)
    assert read_ready_receipts(tmp_path, 'h', 2, contract) is None


def test_timeout_zero_waits_for_actual_replacement_ack(tmp_path):
    snapshot = _manifest(tmp_path)
    engine = _executor(tmp_path, hybrid_worker_ready_timeout_s=0)
    _ready(tmp_path, snapshot, 0, snapshot_id='old')
    result = []
    worker = threading.Thread(target=lambda: result.append(engine._wait_hybrid_worker_ready(
        'h', state_version=9, replica_world_size=1, snapshot_path=snapshot, target_core_id='dp1')))
    worker.start()
    time.sleep(.06)
    assert worker.is_alive() and not result
    _ready(tmp_path, snapshot, 0)
    worker.join(2)
    assert not worker.is_alive()
    assert result[0]['optimizer_revision'] == 9


def test_clear_ready_before_restart_removes_same_snapshot_files(tmp_path):
    snapshot = _manifest(tmp_path)
    _ready(tmp_path, snapshot, 0)
    (tmp_path / 'h.ready').write_text('old')
    clear_ready_receipts(tmp_path, 'h')
    assert not list(tmp_path.glob('*.ready'))


def test_singleton_activation_returns_boundary_approved_receipt(tmp_path):
    engine = _executor(tmp_path)
    snapshot = _manifest(tmp_path)
    engine._snapshot_path_for_version = lambda version: snapshot
    engine._cluster_swap_training_slot = lambda worker: {}
    engine._is_hybrid_worker_running = lambda worker: True
    engine._hybrid_worker_meta['h'] = ManagedHybridWorkerProcess('h', 'dp1', 'existing', 1, snapshot, replica_id='h')
    initial = {'acknowledged': True, 'optimizer_revision': 9, 'state_version': 9}
    approved = dict(initial, optimizer_revision=10, state_version=10)
    engine._wait_hybrid_worker_ready = lambda *args, **kwargs: initial
    queued = []
    engine._queue_activation = lambda replica, receipt: queued.append((replica, receipt)) or approved
    result = engine._activate_hybrid_worker('h', 'dp1', 9)
    assert result is approved and queued == [('h', initial)]


def test_remote_stop_retains_ownership_until_matching_exit_ack_and_ignores_stale_done(tmp_path):
    engine = _executor(tmp_path, hybrid_worker_remote_control_enabled=True, hybrid_worker_stop_timeout_s=0)
    meta = ManagedHybridWorkerProcess('h', 'dp1', 'remote', 1234, 'state', host='node')
    engine._hybrid_worker_meta = {'h': meta, 'member': meta}
    (tmp_path / '.stop_h.done').write_text(json.dumps({'request_id': 'stale', 'worker_id': 'h', 'exited': True}))
    observed = []
    def controller():
        deadline = time.monotonic() + 2
        while not (tmp_path / '.stop_h.json').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        request = json.loads((tmp_path / '.stop_h.json').read_text())
        observed.append(request)
        assert engine._hybrid_worker_meta['h'] is meta
        time.sleep(.05)
        (tmp_path / '.stop_h.done').write_text(json.dumps(dict(request, exited=True)))
    thread = threading.Thread(target=controller)
    thread.start()
    engine._stop_hybrid_worker_process('member')
    thread.join(2)
    assert observed[0]['request_id'] != 'stale'
    assert not engine._hybrid_worker_meta


def test_stop_timeout_keeps_ownership_and_retry_uses_same_request(tmp_path):
    engine = _executor(tmp_path, hybrid_worker_remote_control_enabled=True)
    meta = ManagedHybridWorkerProcess('h', 'dp1', 'remote', 1234, 'state', host='node')
    engine._hybrid_worker_meta['h'] = meta
    with pytest.raises(TimeoutError):
        engine._stop_hybrid_worker_process('h')
    assert engine._hybrid_worker_meta['h'] is meta
    request = json.loads((tmp_path / '.stop_h.json').read_text())
    (tmp_path / '.stop_h.done').write_text(json.dumps(dict(request, exited=True)))
    engine._stop_hybrid_worker_process('h')
    assert meta.stop_request_id == request['request_id']
    assert not engine._hybrid_worker_meta


@pytest.mark.parametrize('timeout', [False, True])
def test_external_rollout_stop_failure_propagates(tmp_path, monkeypatch, timeout):
    engine = _executor(tmp_path, vllm_stop_timeout_s=0)
    meta = ManagedRolloutProcess('engine', 'command', 1, [0], 8000, log_path=str(tmp_path / 'stop.log'))
    def run(*args, **kwargs):
        assert kwargs['timeout'] is None
        if timeout:
            raise subprocess.TimeoutExpired('stop', 1)
        return SimpleNamespace(returncode=17)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(TimeoutError if timeout else RuntimeError):
        engine._run_external_stop(meta)


def test_local_stop_waits_for_launcher_and_child_processes():
    import sys
    from RL_Framework.infra.elastic.process_lifecycle import _live_group_members, stop_local_worker
    script = "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); print(child.pid,flush=True); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        child_pid = int(proc.stdout.readline())
        assert child_pid in _live_group_members(proc.pid)
        stop_local_worker(proc, timeout=1)
        assert proc.poll() is not None
        assert _live_group_members(proc.pid) == []
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_hybrid_factory_partitions_core_dp2_into_one_complete_model_replica(monkeypatch):
    import torch
    from RL_Framework.config import AsyncRLConfig
    from RL_Framework.engine import train_factory
    from RL_Framework.scripts.elastic_hybrid_worker import _initialize_megatron_worker
    config = SimpleNamespace(train_backend='megatron_core', max_seq_length=32768,
                             train_tp_size=2, train_pp_size=1, train_cp_size=1, train_ep_size=1,
                             train_dp_size=2, train_gpus=4, micro_batch_size=1, learning_rate=1e-6,
                             global_resource_planner=SimpleNamespace(hybrid_lockstep_gradient_sync=True))
    observed = []
    engine = SimpleNamespace(initialize=lambda **kwargs: None, world_size=2,
                             get_parallel_state=lambda: dict(train_tp=2, train_pp=1, train_cp=1, train_ep=1, train_dp=1))
    def factory(cfg):
        observed.append((cfg.train_tp_size, cfg.train_dp_size, cfg.train_gpus, cfg.learning_rate))
        return engine
    monkeypatch.setattr(AsyncRLConfig, 'from_yaml', lambda path: config)
    monkeypatch.setattr(train_factory, 'create_train_engine', factory)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setenv('WORLD_SIZE', '2')
    result = _initialize_megatron_worker(SimpleNamespace(config='current-core-config.yaml', initialize_only=True))
    assert result is engine
    assert observed == [(2, 1, 2, 1e-6)]
    monkeypatch.setenv('WORLD_SIZE', '4')
    with pytest.raises(ValueError, match='one complete model replica'):
        _initialize_megatron_worker(SimpleNamespace(config='current-core-config.yaml', initialize_only=True))


def test_remote_member_alias_stops_actual_replica_controller_key(tmp_path, monkeypatch):
    from RL_Framework.infra.elastic import rollout_lending
    engine = _executor(tmp_path, hybrid_worker_remote_control_enabled=True)
    canonical = ManagedHybridWorkerProcess('replica0', 'dp1', 'remote group', 1234, 'state', host='node')
    engine._hybrid_worker_meta['replica0'] = canonical
    for rank in range(2):
        worker = f'rollout{rank}'
        engine._hybrid_worker_meta[worker] = ManagedHybridWorkerProcess(
            worker, 'dp1', 'remote group', -1, 'state', host='node', replica_id='replica0', replica_rank=rank,
            replica_world_size=2,
        )
    stopped = []
    def acknowledge(executor, worker, request_id=None):
        stopped.append(worker)
        assert len(executor._hybrid_worker_meta) == 3
        return {'worker_id': worker, 'request_id': request_id, 'exited': True}
    monkeypatch.setattr(rollout_lending, 'wait_hybrid_stopped', acknowledge)
    engine._stop_hybrid_worker_process('rollout1')
    assert stopped == ['replica0']
    assert (tmp_path / '.stop_replica0.json').exists()
    assert not (tmp_path / '.stop_rollout1.json').exists()
    assert not engine._hybrid_worker_meta

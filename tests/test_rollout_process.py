"""Placement uses the existing Slurm allocation and precise owned process IDs."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.elastic import rollout_process as placement
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor, ManagedRolloutProcess


def test_default_full_launcher_keeps_serving_worker_kv_and_exact_gpu_contract(monkeypatch):
    monkeypatch.delenv('VLLM_SERVER_MODULE', raising=False)
    monkeypatch.delenv('LIBRA_KV_STORAGE_PATH', raising=False)
    monkeypatch.setenv('LIBRA_CAPTURE_BEHAVIOR', '1')
    monkeypatch.setenv('RANK', '2')
    monkeypatch.setenv('MASTER_ADDR', 'training-only')
    monkeypatch.setenv('TORCHELASTIC_WORKER_GROUP', 'training')
    executor = object.__new__(RuntimeElasticExecutor)
    executor.config = AsyncRLConfig(model_path='not-loaded')
    executor.config.rollout_weight_sync_control_dir = '/tmp/libra-test-control'
    executor.config.heterogeneous_rollout.scheduling.cmlfq_kv_backend = 'cpu_offload'
    instance = NS(instance_id='current', tp=2, host='rollout-node', gpus=[2, 3])
    command = executor._build_vllm_command(instance, 9000)
    args = shlex.split(command)
    assert args[:2] == ['env', 'CUDA_VISIBLE_DEVICES=2,3']
    assert 'RL_Framework.scripts.restartable_vllm_server' in args
    assert 'RL_Framework.scripts.vllm_hot_reload_api_server' in args
    assert args[args.index('--worker-extension-cls') + 1] == 'RL_Framework.vllm_hot_reload.InplaceReloadWorkerExtension'
    assert args[args.index('--instance-id') + 1] == 'current'
    connector = json.loads(args[args.index('--kv-transfer-config') + 1])
    assert connector['kv_connector'] == 'LibraCPUOffloadConnector'
    assert connector['kv_connector_extra_config']['storage_path'] == '/tmp/libra-test-control/kv-store'
    meta = ManagedRolloutProcess('current', command, 0, [2, 3], 9000, host='rollout-node', tp=2)
    env = executor._rollout_launch_env(meta)
    assert env['CUDA_VISIBLE_DEVICES'] == env['PLANNED_CUDA_VISIBLE_DEVICES'] == '2,3'
    assert env['LIBRA_CAPTURE_BEHAVIOR'] == '1' and env['LIBRA_EXPECTED_TP_SIZE'] == '2'
    assert env['LIBRA_INSTANCE_ID'] == 'current'
    assert env['LIBRA_KV_STORAGE_PATH'] == connector['kv_connector_extra_config']['storage_path']
    assert 'RANK' not in env and 'MASTER_ADDR' not in env and 'TORCHELASTIC_WORKER_GROUP' not in env


def test_remote_command_targets_current_allocation_and_preserves_gpu_ids(monkeypatch):
    def query(argv, **kwargs):
        assert argv == ['scontrol', 'show', 'hostnames', 'node[1-2]']
        return NS(stdout='node1\nnode2\n')
    monkeypatch.setattr(placement.subprocess, 'run', query)
    env = dict(SLURM_JOB_ID='123', SLURM_JOB_NODELIST='node[1-2]', SLURMD_NODENAME='node1')
    command = placement.slurm_command('exec python serve.py', 'node2', [2, 3], env, step_name='owned')
    args = shlex.split(command)
    assert args[args.index('--jobid') + 1] == '123'
    assert args[args.index('--nodelist') + 1] == 'node2'
    assert '--overlap' in args and not any(arg.startswith('--gpus') for arg in args)
    assert 'CUDA_VISIBLE_DEVICES=2,3' in args and 'PLANNED_CUDA_VISIBLE_DEVICES=2,3' in args
    assert args[-3:] == ['bash', '-c', 'exec python serve.py']
    with pytest.raises(RuntimeError, match='not one unambiguous node'):
        placement.slurm_command('serve', 'outside-job', [0], env)


def test_remote_host_without_allocation_is_rejected_and_local_stays_local(monkeypatch):
    with pytest.raises(RuntimeError, match='current Slurm allocation'):
        placement.placement_command('serve', 'remote-node', [0], {})
    assert placement.placement_command('serve', '127.0.0.1', [0], {}) == 'serve'
    assert placement.is_local_host('node1.cluster', dict(SLURMD_NODENAME='node1'))
    assert not placement.is_local_host('node2', dict(SLURMD_NODENAME='node1'))


def test_launch_remote_tracks_ownership_and_uses_a_dedicated_local_session(tmp_path, monkeypatch):
    monkeypatch.setattr(placement, 'allocation_nodes', lambda env: ['node1', 'node2'])
    seen = []
    def popen(command, **kwargs):
        seen.append((command, kwargs))
        return NS(pid=123)
    monkeypatch.setattr(placement.subprocess, 'Popen', popen)
    meta = NS(instance_id='current', host='node2', gpus=[0, 1], command='serve')
    proc = placement.launch_rollout_process(meta, dict(SLURM_JOB_ID='77', SLURMD_NODENAME='node1'), control_dir=tmp_path)
    assert seen[0][1]['start_new_session'] is True
    assert 'RL_Framework.infra.elastic.rollout_process' in seen[0][0]
    assert proc._libra_rollout_owner['job_id'] == '77'
    assert Path(proc._libra_rollout_owner['state_path']).parent == tmp_path / 'process-owners'


def test_remote_stop_cancels_only_receipted_step_and_zero_waits_for_verified_exit(tmp_path, monkeypatch):
    path = tmp_path / 'owned.json'
    path.write_text(json.dumps(dict(token='token', job_id='77', step_id='9', status='running')))
    owner = dict(remote=True, token='token', job_id='77', state_path=str(path), step_name='owned')
    proc = NS(_libra_rollout_owner=owner)
    proc.poll = lambda: 0
    stopped, commands = [], []
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    monkeypatch.setattr(placement, 'stop_owned_session', lambda proc, timeout: stopped.append(timeout))
    def run(argv, **kwargs):
        commands.append(argv)
        if argv[0] == 'scancel':
            assert argv == ['scancel', '--signal=TERM', '77.9']
            path.write_text(json.dumps(dict(token='token', job_id='77', step_id='9', status='exited', owned_processes_exited=True)))
        return NS(stdout='')
    monkeypatch.setattr(placement.subprocess, 'run', run)
    placement.stop_rollout_process(proc, timeout=0)
    assert stopped == [0] and commands == [['scancel', '--signal=TERM', '77.9']]
    path.write_text(json.dumps(dict(token='other-owner', job_id='77', step_id='9')))
    with pytest.raises(RuntimeError, match='does not match'):
        placement.stop_rollout_process(proc, timeout=0)
    assert commands == [['scancel', '--signal=TERM', '77.9']]


def test_owned_stop_covers_nested_descendant_sessions(tmp_path):
    child_pid = tmp_path / 'child.pid'
    script = ('import subprocess,sys,time,pathlib; '
              'p=subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],start_new_session=True); '
              f'pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)')
    proc = subprocess.Popen([sys.executable, '-c', script], start_new_session=True)
    try:
        deadline = time.monotonic() + 3
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert child_pid.exists()
        pid = int(child_pid.read_text())
        placement.stop_owned_session(proc, timeout=.05)
        assert proc.poll() is not None
        record = placement._process_record(pid)
        assert record is None or record['state'] == 'Z'
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('change', ['wrong-job', 'wrong-instance', 'wrong-port', 'wrong-role', 'wrong-control'])
def test_adopted_identity_mismatch_never_signals_any_pid(monkeypatch, change):
    monkeypatch.setenv('SLURM_JOB_ID', '777')
    contract = dict(pid=120, host='127.0.0.1', instance_id='wanted', port=9000, job_id='777', control_dir='/fyp/run/control')
    argv = [sys.executable, '-m', 'RL_Framework.scripts.restartable_vllm_server',
            '--instance-id', 'wanted', '--health-url', 'http://127.0.0.1:9000/health',
            '--control-dir', '/fyp/run/control']
    environment = dict(SLURM_JOB_ID='777')
    if change == 'wrong-job':
        environment['SLURM_JOB_ID'] = 'other-job'
    elif change == 'wrong-instance':
        argv[argv.index('--instance-id')+1] = 'unrelated'
    elif change == 'wrong-port':
        argv[argv.index('--health-url')+1] = 'http://127.0.0.1:9100/health'
    elif change == 'wrong-role':
        argv[2] = 'unrelated.application'
    elif change == 'wrong-control':
        argv[-1] = '/fyp/another-run/control'
    monkeypatch.setattr(placement, '_process_record', lambda pid: dict(pid=pid, state='S', start='1', parent=1, group=1))
    monkeypatch.setattr(placement, '_process_command', lambda pid: (argv, environment))
    monkeypatch.setattr(placement.os, 'kill', lambda *args: pytest.fail('unverified process must never be signalled'))
    with pytest.raises(RuntimeError):
        placement._stop_adopted_on_host(contract, timeout=0)


def test_remote_adopted_pid_is_verified_and_stopped_on_target_allocation_node(monkeypatch):
    monkeypatch.setattr(placement, 'allocation_nodes', lambda env: ['train', 'rollout'])
    monkeypatch.setattr(placement, '_verify_adopted_identity', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('not local launcher')))
    monkeypatch.setattr(placement.os, 'kill', lambda *args: pytest.fail('remote PID must never be signalled locally'))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == 'srun'
        assert argv[argv.index('--nodelist')+1] == 'rollout'
        assert kwargs['timeout'] is None
        command = shlex.split(argv[-1])
        contract = json.loads(command[command.index('--stop-adopted')+1])
        assert contract['pid'] == 999 and contract['job_id'] == '777'
        assert not contract.get('discover_service')
        return NS(stdout=json.dumps(dict(instance_id='wanted', port=9000, job_id='777', owned_processes_exited=True)))
    monkeypatch.setattr(placement.subprocess, 'run', run)
    meta = NS(pid=999, host='rollout', instance_id='wanted', port=9000, gpus=[0, 1])
    receipt = placement.stop_adopted_rollout(meta, dict(SLURM_JOB_ID='777', SLURMD_NODENAME='train'), timeout=0)
    assert receipt['owned_processes_exited'] and len(calls) == 1


def test_verified_local_srun_registry_pid_authorizes_unique_remote_service_lookup(monkeypatch):
    monkeypatch.setattr(placement, 'allocation_nodes', lambda env: ['train', 'rollout'])
    record = dict(pid=111, start='birth', parent=1, state='S', group=1)
    monkeypatch.setattr(placement, '_verify_adopted_identity', lambda pid, contract, launcher=False: record)
    monkeypatch.setattr(placement, '_still_owned', lambda record: False)
    def run(argv, **kwargs):
        command = shlex.split(argv[-1])
        contract = json.loads(command[command.index('--stop-adopted')+1])
        assert contract['discover_service'] is True
        return NS(stdout=json.dumps(dict(instance_id='wanted', port=9000, job_id='777', owned_processes_exited=True)))
    monkeypatch.setattr(placement.subprocess, 'run', run)
    meta = NS(pid=111, host='rollout', instance_id='wanted', port=9000, gpus=[0])
    assert placement.stop_adopted_rollout(meta, dict(SLURM_JOB_ID='777', SLURMD_NODENAME='train'))['owned_processes_exited']


def test_local_adopted_service_stops_only_verified_tree_and_preserves_other_instance(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '777')
    script = tmp_path / 'restartable_vllm_server.py'
    script.write_text('import time; time.sleep(60)')
    control = str(tmp_path / 'control')
    commands = [[sys.executable, str(script), '--instance-id', name, '--health-url',
                 f'http://127.0.0.1:{port}/health', '--control-dir', control]
                for name, port in [('wanted', 9000), ('unrelated', 9100)]]
    processes = [subprocess.Popen(command) for command in commands]
    try:
        time.sleep(.05)
        meta = NS(pid=processes[0].pid, host='127.0.0.1', instance_id='wanted', port=9000)
        receipt = placement.stop_adopted_rollout(meta, dict(os.environ), timeout=.05, control_dir=control)
        assert receipt['owned_processes_exited'] and receipt['pid'] == processes[0].pid
        processes[0].wait(timeout=1)
        assert processes[1].poll() is None
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=1)


def test_remote_adopted_exit_without_matching_owner_receipt_is_rejected(monkeypatch):
    monkeypatch.setattr(placement, 'allocation_nodes', lambda env: ['train', 'rollout'])
    monkeypatch.setattr(placement, '_verify_adopted_identity', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(placement.subprocess, 'run', lambda *args, **kwargs: NS(stdout=json.dumps(
        dict(instance_id='another', port=9000, job_id='777', owned_processes_exited=True))))
    meta = NS(pid=999, host='rollout', instance_id='wanted', port=9000, gpus=[0])
    with pytest.raises(RuntimeError, match='differs from the requested owner'):
        placement.stop_adopted_rollout(meta, dict(SLURM_JOB_ID='777', SLURMD_NODENAME='train'))

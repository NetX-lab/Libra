"""Executor never lends a rollout GPU before its owned process has exited."""
from types import SimpleNamespace as NS

import pytest

from RL_Framework.infra.elastic import rollout_process
from RL_Framework.infra.elastic.runtime_executor import ManagedRolloutProcess, RuntimeElasticExecutor


def executor(tmp_path):
    value = object.__new__(RuntimeElasticExecutor)
    value.config = NS(rollout_weight_sync_control_dir=str(tmp_path),
        global_resource_planner=NS(vllm_stop_timeout_s=2., vllm_stop_command_template=''))
    value._processes, value._process_meta = {}, {}
    value._rollout_launch_env = lambda meta: {'SLURM_JOB_ID': '123'}
    return value


def test_start_uses_node_placement_and_stop_retains_ownership_on_failure(tmp_path, monkeypatch):
    value = executor(tmp_path)
    meta = ManagedRolloutProcess('rollout', 'service-command', 0, [2, 3], 8000, host='node2', tp=2)
    proc = NS(pid=7)
    launches = []
    monkeypatch.setattr(rollout_process, 'launch_rollout_process',
        lambda meta, env, **kw: launches.append((meta.host, env, kw)) or proc)
    started = value._start_one_rollout_process(meta)
    assert launches[0][0] == 'node2' and launches[0][2]['control_dir'] == str(tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError('service has not exited')
    monkeypatch.setattr(rollout_process, 'stop_rollout_process', fail)
    with pytest.raises(TimeoutError):
        value._stop_rollout_process('rollout')
    assert value._processes['rollout'] is proc and value._process_meta['rollout'] is started
    monkeypatch.setattr(rollout_process, 'stop_rollout_process', lambda *args, **kwargs: None)
    assert value._stop_rollout_process('rollout') == [started]
    assert value._process_meta == value._processes == {}
    assert value._stop_rollout_process('rollout') == []


def test_adopted_stop_passes_remote_identity_without_local_signal(tmp_path, monkeypatch):
    value = executor(tmp_path)
    meta = ManagedRolloutProcess('initial', 'service', 321, [0, 1], 8001,
                                 host='node2', tp=2, adopted=True)
    value._process_meta['initial'] = meta
    stopped = []
    monkeypatch.setattr(rollout_process, 'stop_adopted_rollout',
        lambda meta, **kw: stopped.append((meta, kw)))
    assert value._stop_rollout_process('initial') == [meta]
    assert stopped[0][0].host == 'node2' and stopped[0][1]['env']['SLURM_JOB_ID'] == '123'


def test_unknown_process_cannot_be_reported_stopped(tmp_path):
    value = executor(tmp_path)
    meta = ManagedRolloutProcess('unknown', 'service', -1, [0, 1], 8000, adopted=True)
    value._process_meta['unknown'] = meta
    with pytest.raises(RuntimeError, match='No process ownership'):
        value._stop_rollout_process('unknown')
    assert value._process_meta['unknown'] is meta

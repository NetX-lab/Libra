"""Release occupied rollout GPUs before replacement model initialization."""
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from RL_Framework.infra.elastic.runtime_executor import ManagedRolloutProcess, RuntimeElasticExecutor

def process(name, gpus, port, *, host='node-a', pid=11):
    return ManagedRolloutProcess(name, 'unused', pid, list(gpus), port, host, len(gpus))



class Executor(RuntimeElasticExecutor):
    def __init__(self, current, desired, *, fail_stop=None):
        config = NS(global_resource_planner=NS(runtime_drain_before_reconfigure=True),
            heterogeneous_rollout=NS(vllm_base_port=8000, instances=[
                NS(instance_id=meta.instance_id, port=meta.port) for meta in desired]))
        super().__init__(config=config, planner=None)
        self.desired = desired
        self.events = []
        self.fail_stop = fail_stop
        self._process_meta = {meta.instance_id: meta for meta in current}
        self._processes = {meta.instance_id: object() for meta in current}

    def _desired_rollout_processes(self, plan):
        return self.desired

    def _command_for_instance_id(self, instance_id, port):
        return f'start {instance_id} {port}'

    def _start_one_rollout_process(self, meta):
        self.events.append(('start', meta.instance_id))
        old = self._process_meta.get(meta.instance_id)
        if old is not None and not self._same_rollout_process(old, meta):
            self._blue_green_old_processes = getattr(self, '_blue_green_old_processes', []) + [
                (old, self._processes.get(meta.instance_id))]
        started = replace(meta, pid=100 + len(self.events))
        self._process_meta[meta.instance_id] = started
        self._processes[meta.instance_id] = object()
        return started

    def _wait_rollout_process_start_ready(self, meta):
        self.events.append(('ready', meta.instance_id))

    def _stop_process_meta(self, meta, proc):
        self.events.append(('stop', meta.instance_id))
        if meta.instance_id == self.fail_stop:
            raise RuntimeError('old owner did not exit')
        return [meta]

    def _reconfigure_rollout_engine(self, plan, *, wait_ready):
        self.events.append(('cutover', wait_ready))

    def _sleep_before_rollout_stop(self):
        self.events.append(('grace-before',))

    def _sleep_after_rollout_stop(self, stopped):
        if stopped:
            self.events.append(('grace-after', tuple(meta.instance_id for meta in stopped)))



def test_merge_releases_both_old_tp1_owners_before_new_tp2_and_preserves_survivor():
    keep = process('keep', [1], 8000)
    executor = Executor([keep, process('old-b', [2], 8001), process('old-c', [3], 8002)],
        [replace(keep), process('merged', [2, 3], 8001)])
    stopped, started = executor._blue_green_reconfigure_rollout_processes(None)
    events = executor.events
    assert events.index(('stop', 'old-b')) < events.index(('start', 'merged'))
    assert events.index(('stop', 'old-c')) < events.index(('start', 'merged'))
    assert events.index(('grace-after', ('old-b', 'old-c'))) < events.index(('start', 'merged'))
    assert executor._process_meta['keep'] is keep and ('stop', 'keep') not in events
    assert {meta.instance_id for meta in stopped} == {'old-b', 'old-c'}
    assert [meta.instance_id for meta in started] == ['merged']
    assert started[0].port == 8003



def test_split_releases_tp2_once_before_either_tp1_starts():
    keep = process('keep', [1], 8000)
    executor = Executor([keep, process('merged', [2, 3], 8003)],
        [replace(keep), process('new-b', [2], 8001), process('new-c', [3], 8002)])
    stopped, started = executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events.count(('stop', 'merged')) == 1
    for name in ('new-b', 'new-c'):
        assert executor.events.index(('stop', 'merged')) < executor.events.index(('start', name))
    assert [meta.instance_id for meta in stopped] == ['merged']
    assert [meta.instance_id for meta in started] == ['new-b', 'new-c']
    assert executor._process_meta['keep'] is keep



def test_nonoverlapping_gpu_keeps_start_ready_cutover_before_old_stop():
    executor = Executor([process('old', [2], 8000)], [process('new', [3], 8001)])
    executor._blue_green_reconfigure_rollout_processes(None)
    events = executor.events
    assert events.index(('start', 'new')) < events.index(('ready', 'new'))
    assert events.index(('ready', 'new')) < events.index(('cutover', True))
    assert events.index(('cutover', True)) < events.index(('stop', 'old'))
    assert all(event[0] != 'grace-after' for event in events)



def test_overlapping_stop_failure_preserves_owner_and_prevents_any_launch():
    old = process('old', [2], 8000)
    executor = Executor([old], [process('new', [2], 8001)], fail_stop='old')
    owned_process = executor._processes['old']
    with pytest.raises(RuntimeError, match='old owner did not exit'):
        executor._blue_green_reconfigure_rollout_processes(None)
    assert executor._process_meta['old'] is old
    assert executor._processes['old'] is owned_process
    assert executor.events == [('stop', 'old')]



def test_same_id_replacement_stops_old_once_and_retains_new_process():
    old = process('same', [2], 8000)
    executor = Executor([old], [process('same', [2, 3], 8000)])
    stopped, started = executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events.index(('stop', 'same')) < executor.events.index(('start', 'same'))
    assert executor.events.count(('stop', 'same')) == 1
    assert stopped == [old] and executor._process_meta['same'] is started[0]
    assert started[0].pid != old.pid and started[0].port == 8001

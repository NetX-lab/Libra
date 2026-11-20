"""Blue-green restarts must release physical GPU owners before overlapping launches."""

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


def test_1911493_tp2_split_keeps_distinct_publication_endpoints_and_survivors():
    survivors = [process('grp_tp2_0', [2, 3], 8000, host='node-a'),
                 process('grp_tp2_1', [0, 1], 8001, host='node-b')]
    old = process('grp_tp2_2', [2, 3], 8002, host='node-b')
    desired = [*(replace(meta) for meta in survivors),
        process('grp_tp1_2', [2], 8002, host='node-b'),
        process('grp_tp1_3', [3], 8003, host='node-b')]
    executor = Executor([*survivors, old], desired)
    executor.config.heterogeneous_rollout.instances = [
        NS(instance_id=meta.instance_id, host=meta.host, tp=meta.tp, port=meta.port)
        for meta in desired]

    stopped, started = executor._blue_green_reconfigure_rollout_processes(None)

    targets = [dict(instance_id=instance.instance_id, endpoint=f'{instance.host}:{instance.port}',
                    tp_size=instance.tp) for instance in executor.config.heterogeneous_rollout.instances]
    assert len(targets) == 4
    assert len({meta.port for meta in started}) == 2
    assert {meta.port for meta in started}.isdisjoint({8000, 8001, 8002})
    assert stopped == [old]
    for survivor in survivors:
        assert executor._process_meta[survivor.instance_id] is survivor
        assert ('stop', survivor.instance_id) not in executor.events


@pytest.mark.parametrize('old_host,new_host,overlap', [
    ('node-a', 'node-b', False), ('node-a.cluster', 'node-a', False),
    ('127.0.0.1', 'localhost', True), ('10.1.1.1', '10.2.2.2', False),
])
def test_gpu_ownership_is_scoped_to_physical_host(old_host, new_host, overlap):
    executor = Executor([process('old', [2], 8000, host=old_host)],
        [process('new', [2], 8001, host=new_host)])
    executor._blue_green_reconfigure_rollout_processes(None)
    stop = executor.events.index(('stop', 'old'))
    start = executor.events.index(('start', 'new'))
    assert (stop < start) is overlap


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


def test_missing_stop_confirmation_prevents_overlapping_launch():
    executor = Executor([process('old', [2], 8000)], [process('new', [2], 8001)])
    executor._stop_rollout_process = lambda _identity: []
    with pytest.raises(RuntimeError, match='confirm.*exit'):
        executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events == [] and 'old' in executor._process_meta


def test_overlapping_live_pool_requires_configured_drain_before_any_stop_or_start():
    old = process('old', [2], 8000)
    executor = Executor([old], [process('new', [2], 8001)])
    executor.config.global_resource_planner.runtime_drain_before_reconfigure = False
    with pytest.raises(RuntimeError, match='drain'):
        executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events == [] and executor._process_meta['old'] is old


def test_new_target_cannot_borrow_unchanged_survivor_gpu():
    healthy = process('healthy', [1], 8000)
    executor = Executor([healthy], [replace(healthy), process('new', [1], 8001)])
    with pytest.raises(ValueError, match='unchanged'):
        executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events == [] and executor._process_meta['healthy'] is healthy


def test_overlapping_desired_layout_rejected_before_stopping_current_owner():
    old = process('old', [2], 8000)
    executor = Executor([old], [process('new-a', [2], 8001), process('new-b', [2], 8002)])
    with pytest.raises(ValueError, match='overlap'):
        executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events == [] and executor._process_meta['old'] is old


def test_same_id_replacement_stops_old_once_and_retains_new_process():
    old = process('same', [2], 8000)
    executor = Executor([old], [process('same', [2, 3], 8000)])
    stopped, started = executor._blue_green_reconfigure_rollout_processes(None)
    assert executor.events.index(('stop', 'same')) < executor.events.index(('start', 'same'))
    assert executor.events.count(('stop', 'same')) == 1
    assert stopped == [old] and executor._process_meta['same'] is started[0]
    assert started[0].pid != old.pid and started[0].port == 8001

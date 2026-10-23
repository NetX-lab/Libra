"""Routine rollout publication preserves the current Hybrid GPU loan."""
import io
import json
from types import SimpleNamespace as NS

import pytest
import torch

from RL_Framework.infra.sync import nccl_weight_sync, publication_session
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


@pytest.fixture
def publication(tmp_path, monkeypatch):
    instances = [NS(instance_id=name, host=name, port=8100 + index, tp=index + 1)
        for index, name in enumerate(('a', 'b', 'c'))]
    endpoints = {inst.instance_id: f'http://{inst.host}:{inst.port}' for inst in instances}
    calls, requests, status_urls, exported = [], [], [], []
    members = ('hybrid-b',)
    client = NS(scheduler=NS(_instances=[NS(instance_id=inst.instance_id, is_ready=True)
        for inst in instances]))
    config = NS(weight_sync_mode='nccl', heterogeneous_rollout=NS(instances=instances),
        rollout_weight_sync_control_dir=str(tmp_path), rollout_nccl_host='core',
        rollout_nccl_port=29620, rollout_nccl_chunk_mb=1, rollout_nccl_rate_limit_gbps=0,
        rollout_weight_sync_timeout_s=2, rollout_weight_sync_poll_interval_s=.001,
        enable_rollout_tracing=False)
    dispatcher = NS(paused=False)
    dispatcher.pause = lambda: setattr(dispatcher, 'paused', True)
    dispatcher.resume = lambda: setattr(dispatcher, 'paused', False)
    dispatcher.cancel_queued = lambda: dict(dispatcher_queue=0, runner_queue=0)
    dispatcher.get_runtime_metrics = lambda: dict(staleness_running=0)
    dispatcher.wait_until_idle = lambda **kwargs: None

    def weights():
        exported.append(True)
        yield 'weight', torch.ones(2)

    trainer = NS(config=config, is_main_process=True, control=None, dispatcher=dispatcher,
        rollout_engine=client, _use_heterogeneous=True,
        _rollout_unavailable_ids={'b'},
        runtime_elastic_executor=NS(_lent_rollout={'engines': [instances[1]]}),
        train_engine=NS(train_backend='megatron_core', optimizer_revision=7,
            _elastic_step_hybrid_workers=members, stream_rollout_weights=weights,
            set_version=lambda version: calls.append(('version', version))),
        _restore_lent_rollouts_for_publication=lambda: calls.append(('restore',)),
        _write_json_atomic=lambda path, value: path.write_text(json.dumps(value)),
        _record_rollout_weight_version=lambda **kwargs: None,
        _record_weight_sync_phase=lambda phase: None,
        _trace_train_phase=lambda *args: None,
        _wait_for_all_ranks_before_weight_sync=lambda step: None)

    def rebind(**kwargs):
        calls.append(('rebind',))
        trainer.rollout_engine = NS(rebound=True)

    trainer._rebind_rollout_engine_from_config = rebind
    for name in ('_nccl_rollout_topology', '_sync_weights_nccl', '_sync_weights',
            '_reopen_restored_rollouts', '_resume_rollout_pipeline'):
        setattr(trainer, name, getattr(AsyncRLTrainer, name).__get__(trainer))
    h = NS(trainer=trainer, instances=instances, endpoints=endpoints, client=client,
        calls=calls, requests=requests, status_urls=status_urls, exported=exported,
        expected=['a', 'c'], directory=tmp_path, members=members)

    def status(url, timeout):
        status_urls.append(url)
        identity = next(identity for identity, endpoint in endpoints.items()
            if url == endpoint + '/publication/status')
        return io.StringIO(json.dumps(dict(server_instance_id='service-' + identity)))

    def send(weights, spec, **kwargs):
        tensors = list(weights)
        request = json.loads((tmp_path / 'reload_request.json').read_text())
        requests.append(request)
        expected_offsets, offset = {}, 0
        for inst in instances:
            if inst.instance_id in h.expected:
                expected_offsets[inst.instance_id] = offset
                offset += inst.tp
        assert request['nccl_rank_offsets'] == expected_offsets
        assert spec.world_size == request['nccl_world_size'] == 1 + offset
        assert spec.session_id == request['nccl_session_id']
        for name, tensor in tensors:
            kwargs['on_tensor'](name, tensor.numel() * tensor.element_size())
        for inst in instances:
            identity = inst.instance_id
            if identity not in expected_offsets:
                continue
            workers = [dict(rank=rank, nccl_rank=expected_offsets[identity] + rank + 1,
                received_tensors=len(tensors), receive_complete=True, loaded_parameters=1,
                service_generation=1) for rank in range(inst.tp)]
            record = dict(version=request['version'], snapshot_id=request['snapshot_id'],
                optimizer_revision=request['optimizer_revision'], server_response=dict(
                    service_instance_id='service-' + identity, service_generation=1,
                    prefix_cache_cleared=True, workers=workers))
            (tmp_path / f'ack_{identity}_{request["version"]}.json').write_text(json.dumps(record))

    monkeypatch.setattr(publication_session.urllib.request, 'urlopen', status)
    monkeypatch.setattr(nccl_weight_sync, 'send_weights', send)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    return h


def test_periodic_publication_keeps_hybrid_members_and_existing_rollout_client(publication):
    h = publication
    # Exercise the real periodic + ordinary publication lifecycle separately
    # from the sender's receiver selection, covered below.
    h.trainer._sync_weights_nccl = lambda version: h.calls.append(('publish', version))
    AsyncRLTrainer._periodic_weight_sync(h.trainer, 7)
    assert ('publish', 8) in h.calls
    assert ('restore',) not in h.calls
    assert ('rebind',) not in h.calls
    assert h.trainer.rollout_engine is h.client
    assert h.trainer.train_engine._elastic_step_hybrid_workers == h.members
    assert h.trainer._rollout_unavailable_ids == {'b'}
    assert not h.trainer.dispatcher.paused


@pytest.mark.parametrize('borrowed,restored', [('a', False), ('b', False), ('b', True)],
    ids=['first-lent', 'middle-lent', 'middle-explicitly-returned'])
def test_native_publication_targets_only_live_or_explicitly_returned_receivers(
        publication, borrowed, restored):
    h = publication
    h.trainer._rollout_unavailable_ids = {borrowed}
    if restored:
        h.trainer._restored_rollouts_pending_publication = {borrowed}
    h.expected = [inst.instance_id for inst in h.instances
        if inst.instance_id != borrowed or restored]
    h.trainer._sync_weights(8)
    assert len(h.requests) == 1
    assert h.status_urls == [h.endpoints[identity] + '/publication/status' for identity in h.expected]
    assert [(row[0], row[1], row[2]) for row in h.trainer._nccl_session_members] == [
        (inst.instance_id, h.endpoints[inst.instance_id], inst.tp)
        for inst in h.instances if inst.instance_id in h.expected]
    assert ('restore',) not in h.calls
    assert h.trainer.rollout_engine is h.client
    assert h.trainer.train_engine._elastic_step_hybrid_workers == h.members


@pytest.mark.parametrize('borrowed', ['a', 'b'], ids=['first-lent', 'middle-lent'])
def test_publication_session_resolves_filtered_receivers_by_instance_identity(publication, borrowed):
    h = publication
    offsets, offset = {}, 0
    for inst in h.instances:
        if inst.instance_id != borrowed:
            offsets[inst.instance_id] = offset
            offset += inst.tp
    publication_session.prepare_nccl_session(h.trainer, offsets)
    assert h.status_urls == [h.endpoints[identity] + '/publication/status' for identity in offsets]
    assert [(row[0], row[1], row[2], row[3]) for row in h.trainer._nccl_session_members] == [
        (inst.instance_id, h.endpoints[inst.instance_id], inst.tp, offsets[inst.instance_id])
        for inst in h.instances if inst.instance_id in offsets]


def test_publication_with_every_receiver_lent_keeps_admission_paused_without_sender_only_group(publication):
    h = publication
    h.trainer._rollout_unavailable_ids = set(h.endpoints)
    h.expected = []
    AsyncRLTrainer._periodic_weight_sync(h.trainer, 7)
    assert not h.requests and not h.status_urls and not h.exported
    assert not (h.directory / 'reload_request.json').exists()
    assert h.trainer.dispatcher.paused
    assert h.trainer.rollout_engine is h.client
    assert h.trainer.train_engine._elastic_step_hybrid_workers == h.members
    assert ('restore',) not in h.calls


def test_extension_rollback_publishes_only_explicit_receivers():
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = NS(heterogeneous_rollout=NS(instances=[
        NS(instance_id='old-loan', tp=1), NS(instance_id='returned', tp=2),
        NS(instance_id='still-serving', tp=1)]))
    trainer.rollout_engine = None
    trainer._use_heterogeneous = True
    trainer._rollout_unavailable_ids = {'old-loan', 'returned'}
    trainer._restored_rollouts_pending_publication = {'returned'}
    trainer._publication_target_ids = {'returned'}
    assert trainer._nccl_rollout_topology() == (3, {'returned': 0})
    assert trainer._current_rollout_instance_ids() == ['returned']
    assert trainer._rollout_unavailable_ids == {'old-loan', 'returned'}

"""The native sender reuses only the current physical receiver membership."""
import io
import json
from types import SimpleNamespace as NS

import pytest
import torch

from RL_Framework.infra.sync import nccl_weight_sync, publication_session
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


@pytest.mark.parametrize('endpoints,environment,expected', [
    ('', '', ['http://127.0.0.1:8000', 'http://127.0.0.1:8001', 'http://127.0.0.1:8002']),
    ('', 'remote-a:8100,remote-b:8200', ['http://remote-a:8100', 'http://remote-b:8200']),
    ('explicit:8300', 'ignored:8400', ['http://explicit:8300']),
])
def test_session_reuses_the_existing_homogeneous_endpoint_setup(
        monkeypatch, endpoints, environment, expected):
    config = NS(rollout_backend='vllm', model_path='/model', vllm_endpoints=endpoints,
        vllm_host='127.0.0.1', vllm_port=8000, vllm_num_instances=0,
        rollout_gpus=3, vllm_tp_size=1)
    owner = NS(config=config, is_main_process=False)
    monkeypatch.setenv('VLLM_ENDPOINTS', environment)
    AsyncRLTrainer._setup_homogeneous_engine(owner)
    assert [engine.base_url for engine in owner.rollout_engine.engines] == expected
    observed = []

    def status(url, timeout):
        observed.append(url)
        return io.StringIO(json.dumps(dict(server_instance_id=url)))

    monkeypatch.setattr(publication_session.urllib.request, 'urlopen', status)
    offsets = {f'instance_{index}': index for index in range(owner.rollout_engine.num_instances)}
    first = publication_session.prepare_nccl_session(owner, offsets)
    assert publication_session.prepare_nccl_session(owner, offsets) == first
    assert observed == [endpoint + '/publication/status' for endpoint in expected] * 2


def test_native_publication_reuses_membership_and_sends_a_fresh_same_revision_transfer(tmp_path, monkeypatch):
    states = dict(a='service-a', b='service-b')
    config = NS(heterogeneous_rollout=NS(instances=[
        NS(instance_id=name, host=name, port=8000, tp=1) for name in states]),
        rollout_weight_sync_control_dir=str(tmp_path), rollout_nccl_host='core',
        rollout_nccl_port=29620, rollout_nccl_chunk_mb=1, rollout_nccl_rate_limit_gbps=0,
        rollout_weight_sync_timeout_s=2, rollout_weight_sync_poll_interval_s=.001)
    trainer = NS(config=config, is_main_process=True, control=None,
        train_engine=NS(train_backend='megatron_core', optimizer_revision=7,
            stream_rollout_weights=lambda: iter([('weight', torch.ones(2))])),
        _nccl_rollout_topology=lambda: (3, {'a': 0, 'b': 1}),
        _write_json_atomic=lambda path, value: path.write_text(json.dumps(value)),
        _record_rollout_weight_version=lambda **kwargs: None,
        _record_weight_sync_phase=lambda phase: None)

    def status(url, timeout):
        host = url.split('://', 1)[1].split(':', 1)[0]
        return io.StringIO(json.dumps(dict(server_instance_id=states[host])))

    monkeypatch.setattr(publication_session.urllib.request, 'urlopen', status)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    requests, generations = [], {}

    def send(weights, spec, **kwargs):
        tensors = list(weights)
        for name, tensor in tensors:
            kwargs['on_tensor'](name, tensor.numel() * tensor.element_size())
        request = json.loads((tmp_path / 'reload_request.json').read_text())
        assert spec.session_id == request['nccl_session_id']
        assert request['optimizer_revision'] == trainer.train_engine.optimizer_revision
        requests.append(request)
        for identity, offset in request['nccl_rank_offsets'].items():
            service = states[identity]
            generations[service] = generations.get(service, 0) + 1
            worker = dict(rank=0, nccl_rank=offset+1, received_tensors=len(tensors),
                receive_complete=True, loaded_parameters=1, service_generation=generations[service])
            record = dict(version=request['version'], snapshot_id=request['snapshot_id'],
                optimizer_revision=request['optimizer_revision'], server_response=dict(
                    service_instance_id=service, service_generation=generations[service],
                    prefix_cache_cleared=True, workers=[worker]))
            (tmp_path / f'ack_{identity}_{request["version"]}.json').write_text(json.dumps(record))

    monkeypatch.setattr(nccl_weight_sync, 'send_weights', send)
    AsyncRLTrainer._sync_weights_nccl(trainer, 1)
    AsyncRLTrainer._sync_weights_nccl(trainer, 1)
    assert requests[0]['snapshot_id'] != requests[1]['snapshot_id']
    assert requests[0]['nccl_session_id'] == requests[1]['nccl_session_id']
    states['a'] = 'returned-service-a'
    AsyncRLTrainer._sync_weights_nccl(trainer, 1)
    assert requests[2]['nccl_session_id'] != requests[1]['nccl_session_id']

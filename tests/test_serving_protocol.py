"""Shared serving correctness across arbitrary TP layouts, reload and KV leases."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest
import torch

from RL_Framework.scripts.serving_protocol import ServingState, GenerationMiddleware, fence_kv_payload
from RL_Framework.vllm_hot_reload import InplaceReloadWorkerExtension


def workers(tp, generation=1, tensors=7, offset=0):
    return [dict(rank=rank, nccl_rank=1+offset+rank, service_generation=generation,
                 received_tensors=tensors, loaded_parameters=tensors, receive_complete=True) for rank in range(tp)]


@pytest.mark.parametrize('tp', [1, 2, 4])
def test_initial_and_all_tp_publication_identity(tp):
    async def run():
        state = ServingState(tp, '/models/test')
        assert state.status()['server_generation_verified']
        assert state.optimizer_revision == state.server_generation == 0
        await state.begin_reload(0)
        state.install(snapshot_id='revision100', optimizer_revision=100, workers=workers(tp, offset=4), rank_offset=4)
        await state.reopen()
        status = state.status()
        assert status['server_generation'] == 1 and status['optimizer_revision'] == 100
        assert status['workers'] == workers(tp, offset=4)
        assert status['server_generation_verified']
    asyncio.run(run())


@pytest.mark.parametrize('mutate', [
    lambda rows: rows.pop(),
    lambda rows: rows[1].update(rank=0),
    lambda rows: rows[1].update(service_generation=0),
    lambda rows: rows[1].update(receive_complete=False),
    lambda rows: rows[1].update(received_tensors=6),
    lambda rows: rows[1].update(nccl_rank=20),
])
def test_incomplete_stale_or_inconsistent_tp_receipts_cannot_advance(mutate):
    async def run():
        state = ServingState(2)
        await state.begin_reload(0)
        rows = workers(2)
        mutate(rows)
        with pytest.raises(RuntimeError):
            state.install(snapshot_id='new', optimizer_revision=1, workers=rows)
        await state.fail('invalid receipt')
        assert not state.status()['server_generation_verified']
        assert state.server_generation == 0
        with pytest.raises(RuntimeError):
            await state.reopen()
        with pytest.raises(RuntimeError):
            await state.enter()
    asyncio.run(run())


def test_reload_drains_inflight_requests_and_kv_continuations_before_install():
    async def run():
        state = ServingState(2)
        lease = await state.acquire_kv_lease()
        assert await state.enter() == 0
        installing = asyncio.create_task(state.begin_reload(0))
        await asyncio.sleep(0)
        assert state.closed and not installing.done()
        with pytest.raises(RuntimeError):
            await state.acquire_kv_lease()
        waiting = asyncio.create_task(state.enter())
        await asyncio.sleep(0)
        assert not waiting.done()
        assert await state.enter(lease['lease_id']) == 0  # Existing KV continuation progresses.
        await state.leave()
        await state.leave()
        assert not installing.done()
        await state.release_kv_lease(lease['lease_id'])
        await installing
        state.install(snapshot_id='new', optimizer_revision=1, workers=workers(2))
        await state.reopen()
        assert await waiting == 1
        await state.leave()
        assert state.active_requests == 0
    asyncio.run(run())


def test_positive_drain_timeout_reopens_before_any_mutation():
    async def run():
        state = ServingState(1)
        await state.enter()
        with pytest.raises(TimeoutError):
            await state.begin_reload(.001)
        assert not state.closed and state.failed is None and state.server_generation == 0
        await state.leave()
    asyncio.run(run())


def test_server_owns_target_kv_identity_and_strips_stale_import():
    state = ServingState(2, '/models/test')
    payload = dict(kv_transfer_params=dict(libra_offload_key='save', libra_load_key='old',
        libra_kv_identity=dict(snapshot_id='old', optimizer_revision=-1), libra_kv_source_identity={'old': True}))
    fenced = fence_kv_payload(payload, state)['kv_transfer_params']
    assert 'libra_load_key' not in fenced and 'libra_kv_source_identity' not in fenced
    assert fenced['libra_kv_identity']['server_instance_id'] == state.server_instance_id
    assert fenced['libra_offload_key'] == 'save'


def test_generation_headers_match_actual_completion_and_body_identity_is_authoritative():
    async def run():
        state, sent = ServingState(1), []
        received_payload = []
        async def app(scope, receive, send):
            received_payload.append(json.loads((await receive())['body']))
            await send(dict(type='http.response.start', status=200, headers=[(b'content-type', b'application/json')]))
            await send(dict(type='http.response.body', body=b'{"choices":[]}'))
        middleware = GenerationMiddleware(app, state)
        payload = dict(prompt=[1], kv_transfer_params=dict(libra_offload_key='save', libra_load_key='old'))
        async def receive():
            return dict(type='http.request', body=json.dumps(payload).encode(), more_body=False)
        async def send(message):
            sent.append(message)
        scope = dict(type='http', method='POST', path='/v1/completions',
                     headers=[(b'x-libra-request-id', b'actual-request')])
        await middleware(scope, receive, send)
        headers = dict(sent[0]['headers'])
        assert headers[b'x-libra-request-id'] == b'actual-request'
        assert headers[b'x-libra-service-generation'] == b'0'
        assert headers[b'x-libra-service-generation-verified'] == b'1'
        assert headers[b'x-libra-reload-overlap'] == b'0'
        assert state.active_requests == 0
        params = received_payload[0]['kv_transfer_params']
        assert 'libra_load_key' not in params
        assert params['libra_kv_identity']['server_instance_id'] == state.server_instance_id
    asyncio.run(run())


@pytest.mark.parametrize('consume_all', [True, False])
def test_worker_attests_actual_source_tensor_count_and_done_marker(monkeypatch, consume_all):
    from RL_Framework.infra.sync import nccl_weight_sync
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def source(spec):
        for i in range(3):
            yield f'weight{i}', torch.ones(1)
    monkeypatch.setattr(nccl_weight_sync, 'receive_weights', source)
    worker = InplaceReloadWorkerExtension()
    worker.rank = 1
    def load(weights):
        if consume_all:
            return {name for name, tensor in weights}
        return {next(weights)[0]}
    worker.model_runner = NS(model=NS(load_weights=load))
    if not consume_all:
        with pytest.raises(RuntimeError, match='done marker'):
            worker.reload_weights_nccl('host', 1, 3, 0)
        assert getattr(worker, '_libra_service_generation', 0) == 0
    else:
        result = worker.reload_weights_nccl('host', 1, 3, 0, expected_tensor_count=3)
        assert result['receive_complete'] and result['received_tensors'] == 3
        assert result['service_generation'] == 1 and result['nccl_rank'] == 2


def test_supervisor_echoes_snapshot_and_revision_into_physical_ack(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/restartable_vllm_server.py'
    spec = importlib.util.spec_from_file_location('serving_supervisor_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = dict(mode='nccl', version=100, snapshot_id='final', optimizer_revision=100,
        topology_epoch=8, nccl_host='train', nccl_port=29620, nccl_world_size=5,
        nccl_rank_offsets={'current': 0}, nccl_chunk_bytes=1024)
    (tmp_path / 'reload_request.json').write_text(json.dumps(request))
    args = NS(instance_id='current', control_dir=str(tmp_path), health_url='http://service/health',
        initial_model='/models/test', ready_timeout=0, reload_strategy='parallel', reload_method='inplace',
        poll_interval=.01, command=['server'])
    monkeypatch.setattr(module, 'parse_args', lambda: args)
    child = NS(poll=lambda: None)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: child)
    monkeypatch.setattr(module, 'wait_until_ready', lambda *args: None)
    monkeypatch.setattr(module, 'stop_process', lambda *args: None)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    forwarded = []
    def reload(url, payload, timeout):
        forwarded.append(payload)
        return dict(workers=workers(4), service_instance_id='actual', service_generation=1)
    monkeypatch.setattr(module, 'reload_nccl', reload)
    calls = 0
    def sleep(seconds):
        nonlocal calls
        calls += 1
        if calls == 1:
            request['snapshot_id'] = 'same-params-new-topology'
            request['topology_epoch'] = 9
            (tmp_path / 'reload_request.json').write_text(json.dumps(request))
        else:
            raise KeyboardInterrupt()
    monkeypatch.setattr(module.time, 'sleep', sleep)
    assert module.main() == 130
    assert [payload['snapshot_id'] for payload in forwarded] == ['final', 'same-params-new-topology']
    ack = json.loads((tmp_path / 'ack_current_100.json').read_text())
    assert ack['snapshot_id'] == 'same-params-new-topology' and ack['optimizer_revision'] == 100
    assert ack['topology_epoch'] == 9 and ack['server_response']['service_instance_id'] == 'actual'


def load_serving_app(monkeypatch, tp=4, invalid_ack=False):
    """Import the actual FastAPI routes with only the unavailable vLLM runtime stubbed."""
    import argparse
    import types
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    modules = {}
    for name in ('vllm', 'vllm.entrypoints', 'vllm.entrypoints.openai',
                 'vllm.entrypoints.openai.api_server', 'vllm.entrypoints.openai.cli_args',
                 'vllm.utils', 'vllm.v1', 'vllm.v1.engine', 'vllm.v1.engine.core'):
        modules[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules['vllm'].__version__ = '0.9.2'
    modules['vllm.entrypoints.openai'].api_server = modules['vllm.entrypoints.openai.api_server']
    modules['vllm.entrypoints.openai.cli_args'].make_arg_parser = lambda parser: parser
    modules['vllm.entrypoints.openai.cli_args'].validate_parsed_serve_args = lambda args: None
    modules['vllm.utils'].FlexibleArgumentParser = argparse.ArgumentParser
    modules['vllm.v1.engine.core'].EngineCore = type('Core', (), {})
    monkeypatch.setitem(sys.modules, 'uvloop', NS(run=lambda coroutine: None))
    def build(args):
        app = FastAPI()
        @app.post('/v1/completions')
        async def completion(request: Request):
            return JSONResponse(dict(choices=[dict(text='ok')], received=await request.json()))
        return app
    async def utility(name):
        assert name == 'libra_reset_prefix_cache'
        return True
    async def rpc(name, timeout, args=()):
        assert timeout is None or timeout == 120
        if name == 'reload_weights_nccl':
            rows = workers(tp, offset=args[3])
            return rows[:-1] if invalid_ack else rows
        raise AssertionError(name)
    client = NS(collective_rpc=rpc, engine_core=NS(call_utility_async=utility))
    api = modules['vllm.entrypoints.openai.api_server']
    api.build_app = build
    api.engine_client = lambda request: request.app.state.client
    path = Path(__file__).resolve().parents[1] / 'scripts/vllm_hot_reload_api_server.py'
    spec = importlib.util.spec_from_file_location('libra_serving_routes_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    app = module.build_app_with_reload_routes(NS(tensor_parallel_size=tp, model='/models/test'))
    app.state.client = client
    return app


def test_real_http_routes_fence_reload_by_kv_lease_and_attest_physical_workers(monkeypatch):
    import httpx
    app = load_serving_app(monkeypatch)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://service') as client:
            initial = (await client.get('/publication/status')).json()
            assert initial['server_generation_verified'] and initial['tp_size'] == 4
            assert (await client.get('/libra/kv/identity')).json()['snapshot_id'] == initial['snapshot_id']
            lease = (await client.post('/libra/kv/lease', json={})).json()
            publishing = asyncio.create_task(client.post('/reload_weights_nccl', json=dict(
                host='training', port=29620, world_size=5, rank_offset=0,
                snapshot_id='final100', optimizer_revision=100, timeout_seconds=0)))
            await asyncio.sleep(0)
            assert not publishing.done()
            # Holding an existing lease allows its continuation, even as the
            # publisher has closed new admission and waits for that same lease.
            continuation = await client.post('/v1/completions', headers={
                'X-Libra-KV-Lease': lease['lease_id'], 'X-Libra-Request-Id': 'old-request'},
                json=dict(prompt=[1]))
            assert continuation.headers['x-libra-service-generation'] == '0'
            await client.delete('/libra/kv/lease/' + lease['lease_id'])
            reply = await publishing
            assert reply.status_code == 200
            assert reply.json()['prefix_cache_cleared']
            assert len(reply.json()['workers']) == 4
            live = (await client.get('/publication/status')).json()
            assert live['optimizer_revision'] == 100 and live['snapshot_id'] == 'final100'
            assert live['server_generation'] == 1 and live['server_generation_verified']
            assert live['workers'] == reply.json()['workers']
            response = await client.post('/v1/completions', headers={'X-Libra-Request-Id': 'new-request'},
                json=dict(prompt=[1], kv_transfer_params=dict(libra_offload_key='next', libra_load_key='old',
                    libra_kv_identity=lease)))
            assert response.headers['x-libra-service-generation-verified'] == '1'
            assert response.headers['x-libra-service-generation'] == '1'
            assert 'libra_load_key' not in response.json()['received']['kv_transfer_params']
    asyncio.run(run())


def test_real_http_partial_tp_failure_quarantines_service(monkeypatch):
    import httpx
    app = load_serving_app(monkeypatch, invalid_ack=True)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://service') as client:
            response = await client.post('/reload_weights_nccl', json=dict(host='training', port=29620,
                world_size=5, rank_offset=0, snapshot_id='broken', optimizer_revision=1, timeout_seconds=0))
            assert response.status_code == 500
            state = (await client.get('/publication/status')).json()
            assert state['failed'] and not state['server_generation_verified']
            assert state['server_generation'] == 0
            assert (await client.post('/v1/completions', json=dict(prompt=[1]))).status_code == 409
    asyncio.run(run())

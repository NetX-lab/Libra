"""Shared serving identity, request/KV leases and global reload receipts.

Generation and reload share one boundary so cached KV remains tied to the
weights and service incarnation that produced it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid


class ServingState:
    def __init__(self, tp_size, model_path=''):
        if int(tp_size) < 1:
            raise ValueError('Expected positive physical TP worker count')
        self.tp_size = int(tp_size)
        self.server_instance_id = uuid.uuid4().hex
        self.server_generation = 0
        self.snapshot_id = 'initial:' + hashlib.sha256(str(model_path).encode()).hexdigest()
        self.optimizer_revision = 0
        self.workers = []
        self.active_requests = 0
        self.kv_leases = set()
        self.closed = False
        self.failed = None
        self.condition = asyncio.Condition()

    def status(self):
        return dict(server_instance_id=self.server_instance_id, server_generation=self.server_generation,
            snapshot_id=self.snapshot_id, optimizer_revision=self.optimizer_revision,
            tp_size=self.tp_size, workers=self.workers, active_requests=self.active_requests,
            kv_leases=len(self.kv_leases), closed=self.closed, failed=self.failed,
            server_generation_verified=self.failed is None and not self.closed,
            server_monotonic=time.monotonic())

    async def acquire_kv_lease(self, lease_id=None):
        async with self.condition:
            if self.closed or self.failed:
                raise RuntimeError('Rollout service is unavailable for a new KV lease')
            lease_id = lease_id or uuid.uuid4().hex
            self.kv_leases.add(lease_id)
            return dict(self.status(), lease_id=lease_id)

    async def release_kv_lease(self, lease_id):
        async with self.condition:
            self.kv_leases.discard(lease_id)
            self.condition.notify_all()

    async def enter(self, lease_id=None):
        async with self.condition:
            if lease_id is not None and lease_id not in self.kv_leases:
                raise RuntimeError('Unknown or expired KV lease')
            await self.condition.wait_for(lambda: not self.closed or lease_id in self.kv_leases or self.failed)
            if self.failed:
                raise RuntimeError('Rollout service is quarantined: ' + self.failed)
            self.active_requests += 1
            return self.server_generation

    async def leave(self):
        async with self.condition:
            self.active_requests -= 1
            if self.active_requests < 0:
                raise RuntimeError('Unbalanced serving request lifetime')
            self.condition.notify_all()

    async def begin_reload(self, timeout):
        async with self.condition:
            if self.closed or self.failed:
                raise RuntimeError('Rollout service is unavailable for reload')
            self.closed = True
            wait = self.condition.wait_for(lambda: self.active_requests == 0 and not self.kv_leases)
            try:
                if timeout > 0:
                    await asyncio.wait_for(wait, timeout)
                else:
                    await wait
            except BaseException:
                # No mutation occurred before the drain; retry is safe.
                self.closed = False
                self.condition.notify_all()
                raise

    def install(self, *, snapshot_id, optimizer_revision, workers, rank_offset=0, expected_tensor_count=None):
        if not self.closed:
            raise RuntimeError('Reload has no exclusive drained service ownership')
        ranks = [worker.get('rank') for worker in workers]
        if (len(workers) != self.tp_size or sorted(ranks) != list(range(self.tp_size))
                or any(worker.get('service_generation') != self.server_generation + 1 for worker in workers)):
            raise RuntimeError('All TP workers must attest the next service generation')
        if snapshot_id is not None:
            counts = {worker.get('received_tensors') for worker in workers}
            count = expected_tensor_count
            if count is None and len(counts) == 1:
                count = next(iter(counts))
            if (not snapshot_id or type(optimizer_revision) is not int or optimizer_revision < self.optimizer_revision
                    or type(count) is not int or count < 1
                    or sorted(worker.get('nccl_rank', -1) for worker in workers) != list(range(1+rank_offset, 1+rank_offset+self.tp_size))
                    or any(worker.get('receive_complete') is not True or worker.get('received_tensors') != count
                           or worker.get('loaded_parameters', 0) < 1 for worker in workers)):
                raise RuntimeError('Missing complete immutable-snapshot tensor receipt on a TP worker')
        self.server_generation += 1
        self.snapshot_id, self.optimizer_revision = snapshot_id, optimizer_revision if snapshot_id else 0
        self.workers = [dict(worker) for worker in workers]

    async def reopen(self):
        async with self.condition:
            if self.failed:
                raise RuntimeError('A failed service cannot be reopened')
            self.closed = False
            self.condition.notify_all()

    async def fail(self, error):
        async with self.condition:
            self.failed = str(error)
            self.closed = True
            self.condition.notify_all()


def generation_headers(state, request_id, generation, status, streaming=False):
    verified = bool(state.failed is None and generation == state.server_generation
                    and 200 <= status < 300 and not streaming)
    values = {
        'x-libra-request-id': request_id,
        'x-libra-service-instance': state.server_instance_id,
        'x-libra-service-generation': str(state.server_generation),
        'x-libra-service-generation-start': str(generation),
        'x-libra-service-generation-end': str(state.server_generation),
        'x-libra-service-generation-verified': '1' if verified else '0',
        'x-libra-reload-overlap': '0' if generation == state.server_generation else '1',
    }
    return [(key.encode(), value.encode()) for key, value in values.items()]


def fence_kv_payload(payload, state):
    """The service owns target weight identity; stale imports must recompute."""
    from RL_Framework.infra.scheduling.kv_identity import same_snapshot, snapshot_identity
    params = payload.get('kv_transfer_params') if isinstance(payload, dict) else None
    if isinstance(params, dict) and ('libra_offload_key' in params or 'libra_load_key' in params):
        identity = snapshot_identity(state.status())
        if not same_snapshot(params.get('libra_kv_identity'), identity):
            params.pop('libra_load_key', None)
            params.pop('libra_kv_source_identity', None)
        params['libra_kv_identity'] = identity
    return payload


class GenerationMiddleware:
    """Admit request lifetimes through the same boundary used by weight reload."""
    def __init__(self, app, state):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        if (scope['type'] != 'http' or scope.get('method') != 'POST'
                or scope.get('path') not in {'/v1/completions', '/v1/chat/completions'}):
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        lease = headers.get(b'x-libra-kv-lease')
        request_id = headers.get(b'x-libra-request-id', uuid.uuid4().hex.encode()).decode()
        try:
            generation = await self.state.enter(lease.decode() if lease else None)
        except RuntimeError as exc:
            body = json.dumps(dict(detail=str(exc))).encode()
            await send(dict(type='http.response.start', status=409, headers=[(b'content-type', b'application/json')]))
            return await send(dict(type='http.response.body', body=body))
        start, bodies, streaming = None, [], False
        async def observe(message):
            nonlocal start, streaming
            if message['type'] == 'http.response.start':
                start = dict(message)
                streaming = b'text/event-stream' in dict(message.get('headers', [])).get(b'content-type', b'')
                if streaming:
                    start['headers'] = list(start.get('headers', [])) + generation_headers(
                        self.state, request_id, generation, start['status'], streaming=True)
                    await send(start)
                return
            if message['type'] != 'http.response.body' or streaming:
                return await send(message)
            bodies.append(message)
            if not message.get('more_body', False):
                start['headers'] = list(start.get('headers', [])) + generation_headers(
                    self.state, request_id, generation, start['status'])
                await send(start)
                for part in bodies:
                    await send(part)
        try:
            if receive is None:
                return await self.app(scope, receive, observe)
            chunks = []
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                chunks.append(message.get('body', b''))
                if not message.get('more_body', False):
                    break
            body = b''.join(chunks)
            try:
                payload = json.loads(body)
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get('kv_transfer_params'), dict):
                body = json.dumps(fence_kv_payload(payload, self.state)).encode()
                scope = dict(scope, headers=[(key, value) for key, value in scope.get('headers', [])
                                             if key.lower() != b'content-length']
                             + [(b'content-length', str(len(body)).encode())])
            consumed = False
            async def replay():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return dict(type='http.request', body=body, more_body=False)
                return await receive()
            await self.app(scope, replay, observe)
        finally:
            await self.state.leave()

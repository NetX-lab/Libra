"""NCCL membership sessions follow physical services, independently of weights."""
from __future__ import annotations

import copy
import json
import os
import urllib.request
import uuid


def publication_targets(config, *, instance_ids=None):
    """Resolve current physical endpoints using the same order as the NCCL ranks."""
    hetero = getattr(config, 'heterogeneous_rollout', None)
    instances = getattr(hetero, 'instances', []) if hetero else []
    if instances:
        targets = []
        for index, instance in enumerate(instances):
            host = instance.host or hetero.vllm_host
            host = '127.0.0.1' if host == '0.0.0.0' else host
            port = int(instance.port or hetero.vllm_base_port + index)
            targets.append(dict(instance_id=instance.instance_id or f'instance_{index}',
                endpoint=f'{host}:{port}', tp_size=int(instance.tp)))
    else:
        endpoint_string = getattr(config, 'vllm_endpoints', '') or os.environ.get('VLLM_ENDPOINTS', '')
        if endpoint_string:
            endpoints = [item.strip() for item in endpoint_string.split(',') if item.strip()]
        else:
            count = config.vllm_num_instances
            if count <= 0:
                count = max(1, config.rollout_gpus // max(1, config.vllm_tp_size))
            endpoints = [f'{config.vllm_host}:{config.vllm_port + index}' for index in range(count)]
        targets = [dict(instance_id=f'instance_{index}', endpoint=endpoint,
                        tp_size=int(config.vllm_tp_size))
                   for index, endpoint in enumerate(endpoints)]
    ids = [target['instance_id'] for target in targets]
    endpoints = [target['endpoint'] for target in targets]
    if (not targets or len(set(ids)) != len(ids) or len(set(endpoints)) != len(endpoints)
            or any(target['tp_size'] < 1 for target in targets)):
        raise ValueError('Expected distinct physical rollout services with positive TP sizes')
    if instance_ids is None:
        return targets
    selected = list(instance_ids)
    if len(set(selected)) != len(selected) or not set(selected).issubset(ids):
        raise ValueError('Publication subset contains duplicate or unknown physical services')
    by_id = {target['instance_id']: target for target in targets}
    return [by_id[name] for name in selected]


def topology_identity(config, targets=None):
    targets = publication_targets(config) if targets is None else targets
    return dict(target_ids=[target['instance_id'] for target in targets], targets=targets)


def publication_inventory(owner):
    """Resolve eligible services from current inventory and actual loan ownership."""
    inventory = publication_targets(owner.config)
    names = {target['instance_id'] for target in inventory}
    unavailable = set(getattr(owner, '_rollout_unavailable_ids', ()))
    restored = set(getattr(owner, '_restored_rollouts_pending_publication', ()))
    borrowed = unavailable - restored
    executor = getattr(owner, 'runtime_elastic_executor', None)
    loan = getattr(executor, '_lent_rollout', None)
    if executor is not None:
        owned = {meta.instance_id for meta in loan['engines']} if loan else set()
        if owned != borrowed:
            raise RuntimeError('Rollout availability differs from actual Hybrid loan ownership')
        borrowed = owned
    if not (unavailable | restored | borrowed).issubset(names):
        raise RuntimeError('Rollout ownership names services outside the committed inventory')
    targets = [target for target in inventory if target['instance_id'] not in borrowed]
    return dict(inventory_targets=inventory, borrowed_target_ids=sorted(borrowed), targets=targets)


def live_publication_state(endpoint):
    endpoint = endpoint.rstrip('/')
    url = endpoint if '://' in endpoint else 'http://' + endpoint
    with urllib.request.urlopen(url + '/publication/status', timeout=130) as response:
        return json.load(response)


def prepare_nccl_session(owner, targets, offsets, *, read_state=None):
    """Pin the physical UUIDs read before this publication on sender and receivers."""
    if list(offsets) != [target['instance_id'] for target in targets]:
        raise RuntimeError('NCCL ranks differ from the current physical services')
    read_state = read_state or live_publication_state
    previous, members = {}, []
    for target in targets:
        state = read_state(target['endpoint'])
        generation = state.get('server_generation')
        identity = state.get('server_instance_id')
        if (not identity or type(generation) is not int or generation < 0
                or state.get('server_generation_verified') is not True
                or state.get('tp_size') != target['tp_size'] or state.get('failed')):
            raise RuntimeError('Unverified current physical service: ' + target['instance_id'])
        previous[target['instance_id']] = (identity, generation)
        members.append((target['instance_id'], target['endpoint'], target['tp_size'],
                        offsets[target['instance_id']], identity))
    membership = tuple(members)
    session = getattr(owner, '_nccl_publication_session', None)
    if session is None or session[0] != membership:
        session = owner._nccl_publication_session = (membership, uuid.uuid4().hex)
    return dict(nccl_session_id=session[1],
        expected_service_instance_ids={key: value[0] for key, value in previous.items()}), previous


def verify_nccl_publication(request, targets, previous, *, read_ack):
    """Accept one completed installation ACK for each target in this request."""
    receipts = []
    for target in targets:
        name = target['instance_id']
        record = read_ack(name)
        reply = record.get('server_response') if isinstance(record, dict) else None
        if (not isinstance(reply, dict) or record.get('instance_id') != name
                or any(record.get(key) != request[key] for key in
                       ('version', 'snapshot_id', 'optimizer_revision'))
                or any(reply.get(key) != request[key] for key in ('snapshot_id', 'optimizer_revision'))
                or reply.get('service_instance_id') != previous[name][0]
                or reply.get('service_generation') != previous[name][1] + 1
                or reply.get('prefix_cache_cleared') is not True):
            raise RuntimeError('Invalid physical-engine ACK identity or installation response: ' + name)
        receipts.append(dict(version=request['version'], instance_id=name, endpoint=target['endpoint'],
            snapshot_id=request['snapshot_id'], optimizer_revision=request['optimizer_revision'],
            server_instance_id=reply['service_instance_id'], server_generation=reply['service_generation'],
            expected_tp_size=target['tp_size']))
    return receipts


def verify_live_publication_target(target, ack, *, read_state=None):
    """Check current installation once when reusing an earlier transaction."""
    read_state = read_state or live_publication_state
    state = read_state(target['endpoint'])
    if (ack.get('endpoint') != target['endpoint'] or ack.get('expected_tp_size') != target['tp_size']
            or not isinstance(state, dict) or state.get('server_generation_verified') is not True
            or state.get('closed') is not False or state.get('failed')
            or any(state.get(key) != ack.get(key) for key in
                   ('server_instance_id', 'server_generation', 'snapshot_id', 'optimizer_revision'))
            or state.get('tp_size') != target['tp_size']):
        raise RuntimeError('Live physical-engine state differs from installation ACK: ' + target['instance_id'])
    return copy.deepcopy(state)


def attest_nccl_publication(config, receipt, *, targets=None, read_state=None):
    """Recheck a verified transaction against the currently eligible services."""
    targets = publication_targets(config) if targets is None else targets
    topology = topology_identity(config, targets)
    if any(receipt.get(key) != value for key, value in topology.items()):
        raise RuntimeError('Publication receipt belongs to an old rollout topology')
    acks = receipt.get('acknowledgements', [])
    by_id = {ack['instance_id']: ack for ack in acks}
    if len(acks) != len(targets) or set(by_id) != set(topology['target_ids']):
        raise RuntimeError('Missing or duplicate physical-engine ACK coverage')
    states = []
    for target in targets:
        state = verify_live_publication_target(target, by_id[target['instance_id']], read_state=read_state)
        states.append(dict(state, instance_id=target['instance_id'], endpoint=target['endpoint']))
    return states

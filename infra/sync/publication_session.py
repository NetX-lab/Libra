"""NCCL membership sessions follow physical services, independently of weights."""
from __future__ import annotations

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


def publication_inventory(owner):
    """Select the configured services that are available for publication."""
    inventory = publication_targets(owner.config)
    names = {target['instance_id'] for target in inventory}
    unavailable = set(getattr(owner, '_rollout_unavailable_ids', ()))
    restored = set(getattr(owner, '_restored_rollouts_pending_publication', ()))
    borrowed = unavailable - restored
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

"""Keep global NCCL membership aligned with the live rollout processes."""
from __future__ import annotations

import json
import urllib.request
import uuid


def prepare_nccl_session(owner, offsets):
    """Reuse the group while the current receivers keep their physical UUIDs."""
    config = owner.config
    hetero = getattr(config, 'heterogeneous_rollout', None)
    instances = getattr(hetero, 'instances', ()) if hetero else ()
    if instances:
        receivers = {}
        for index, instance in enumerate(instances):
            host = instance.host or hetero.vllm_host
            host = '127.0.0.1' if host == '0.0.0.0' else host
            port = int(instance.port or hetero.vllm_base_port + index)
            identity = str(getattr(instance, 'instance_id', f'instance_{index}'))
            receivers[identity] = (f'http://{host}:{port}', int(instance.tp))
    else:
        # The existing homogeneous setup already resolves VLLM_ENDPOINTS and
        # host/base-port/count defaults. Use those actual clients unchanged.
        receivers = {f'instance_{index}': (engine.base_url, int(config.vllm_tp_size))
                     for index, engine in enumerate(owner.rollout_engine.engines)}
    members = []
    for instance_id, offset in offsets.items():
        endpoint, tp = receivers[instance_id]
        with urllib.request.urlopen(endpoint.rstrip('/') + '/publication/status', timeout=130) as response:
            state = json.load(response)
        members.append((instance_id, endpoint, tp, offset, state['server_instance_id']))
    membership = tuple(members)
    if getattr(owner, '_nccl_session_members', None) != membership:
        owner._nccl_session_id = uuid.uuid4().hex
        owner._nccl_session_members = membership
    return dict(nccl_session_id=owner._nccl_session_id)

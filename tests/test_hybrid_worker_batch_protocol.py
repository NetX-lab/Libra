"""The worker forwards fixed-batch counts through its actual task loop."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


def _worker():
    path = Path(__file__).parents[1] / 'scripts' / 'elastic_hybrid_worker.py'
    spec = importlib.util.spec_from_file_location('hybrid_batch_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_file_retains_local_and_global_sample_counts(tmp_path):
    worker = _worker()
    path = tmp_path / 'payload.pt'
    torch.save(dict(tensors=[torch.ones(1)], sample_count=4, global_sample_count=12), path)
    args = SimpleNamespace(replica_id='h', replica_rank=0, replica_world_size=1)
    payload = worker._load_payload_file(path, worker_id='h', target_core_id='dp0', args=args)
    assert payload.sample_count == 4 and payload.global_sample_count == 12


def test_training_task_passes_global_count_to_engine_and_gradient_client(tmp_path, monkeypatch):
    worker = _worker()
    snapshot = tmp_path / 'snapshot.pt'
    snapshot.touch()
    args = SimpleNamespace(snapshot_path=str(snapshot), initialize_only=False,
        worker_mode='megatron_core', gradient_host='localhost', gradient_port=1,
        authkey='', gradient_transport='tcp', payload_file='', idle=False,
        task_dir=str(tmp_path), worker_id='h', replica_id='h', target_core_id='dp0')
    received, sent = [], []

    def compute(trajectories, **metadata):
        received.append(metadata)
        return GradientPayload('h', 'dp0', (torch.ones(1),),
            step=metadata['step'], state_version=metadata['state_version'],
            membership_epoch=metadata['membership_epoch'], sample_count=len(trajectories),
            global_sample_count=metadata['global_sample_count'])

    engine = SimpleNamespace(recompute_logprobs=lambda _: None,
        compute_elastic_gradient_payload=compute)
    monkeypatch.setattr(worker, 'parse_args', lambda: args)
    monkeypatch.setattr(worker, '_initialize_megatron_worker', lambda _: engine)
    monkeypatch.setattr(worker, '_endpoint_for_lane', lambda *_: {})
    monkeypatch.setattr(worker, '_gradient_client', lambda *_:
        SimpleNamespace(send=lambda payload, **_: sent.append(payload)))
    monkeypatch.setattr(worker.torch.distributed, 'is_initialized', lambda: False)
    monkeypatch.setenv('RANK', '0')
    torch.save(dict(trajectories=[{}] * 4, step=3, state_version=2,
        membership_epoch=7, sample_count=4, global_sample_count=12), tmp_path / 'h.0.pt')
    torch.save(dict(shutdown=True), tmp_path / 'h.1.pt')
    worker.main()
    assert received[0]['global_sample_count'] == 12
    assert received[0]['state_version'] == 2
    assert len(sent) == 1 and sent[0].sample_count == 4 and sent[0].global_sample_count == 12
    assert (tmp_path / 'h.0.pt.done').exists()

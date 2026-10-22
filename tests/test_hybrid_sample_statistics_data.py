"""Hybrid loss statistics are sample means carried by the existing transports."""
import shutil

import pytest

from hybrid_statistics_helpers import compute, rows
from RL_Framework.infra.elastic.gradient_ipc import ElasticGradientClient, ElasticGradientServer
from RL_Framework.infra.elastic.native_rdma import NativeRDMAGradientClient, NativeRDMAGradientServer


@pytest.mark.parametrize('zero_mask',[False,True])
def test_payload_statistics_match_single_sample_means(monkeypatch,zero_mask):
    samples=rows(8,4,zero_mask=zero_mask)
    worker,payload=compute(monkeypatch,samples)
    per_sample=[]
    for row in samples:
        batch=worker._pad_micro_batch([row])
        per_sample.append(worker._loss(batch,worker._forward(batch))[1])
    reference=worker._merge_stats(per_sample)
    for key in ('loss','pg_loss','kl','reward_sum','reward_mean','n_samples'):
        assert payload.stats[key]==pytest.approx(reference[key],abs=1e-6)
    assert payload.stats['n_samples']==4 and payload.stats['n_updates']==2


def test_tcp_transports_statistics_from_real_hybrid_compute(monkeypatch):
    _,payload=compute(monkeypatch,rows(8,4))
    assert hasattr(payload,'stats'), 'Hybrid compute discarded the loss statistics'
    received=[]
    server=ElasticGradientServer(on_payload=received.append)
    endpoint=server.start()
    try:
        ElasticGradientClient(endpoint).send(payload)
        assert received[0].stats==payload.stats
    finally:
        server.close()


def test_native_metadata_transports_statistics_without_requiring_rdma_hardware(tmp_path,monkeypatch):
    _,payload=compute(monkeypatch,rows(8,4))
    assert hasattr(payload,'stats'), 'Hybrid compute discarded the loss statistics'
    received=[]
    server=NativeRDMAGradientServer(on_payload=received.append,work_dir=tmp_path/'server')
    staged=tmp_path/'wire.pt'
    client=NativeRDMAGradientClient(server.endpoint)
    client.transfer.send_file_once=lambda input_path,**kw:shutil.copyfile(input_path,staged)
    client.send(payload)
    def transfer(output_path,**kw):
        shutil.copyfile(staged,output_path)
        server._stop.set()
    server.transfer.receive_file_once=transfer
    server._serve()
    assert server.last_error is None
    assert received[0].stats==payload.stats

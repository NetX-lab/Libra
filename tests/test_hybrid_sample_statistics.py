"""Core reports the full batch only after the activation protocol accepts gradients."""
from dataclasses import replace

import pytest

from hybrid_statistics_helpers import compute, engine, rows


@pytest.mark.parametrize('core_count,zero_mask',[(8,False),(8,True),(0,False)])
def test_real_payload_and_grpo_report_all_samples_without_counting_extra_core_updates(
        monkeypatch,core_count,zero_mask):
    core_rows=rows(0,core_count)
    hybrid_count=4 if core_count else 12
    hybrid_rows=rows(8,hybrid_count,zero_mask=zero_mask)
    _,payload=compute(monkeypatch,hybrid_rows)
    core,steps=engine(monkeypatch)
    core.set_elastic_batch_partition(core_count,12,{'hybrid':hybrid_count})
    core.enqueue_hybrid_gradient_payload(payload)
    reference=core._merge_stats([core._loss(batch,core._forward(batch))[1]
        for batch in core._iter_micro_batches(core_rows+hybrid_rows)])
    result=core.grpo_update(core_rows)
    assert result['n_samples']==12
    assert result['reward_sum']==reference['reward_sum']
    for key in ('loss','pg_loss','kl','reward_mean'):
        assert result[key]==pytest.approx(reference[key],abs=1e-6)
    assert result['n_updates']==(core_count+2)//3
    if core_count:
        assert result['grad_norm']==7.
    assert len(steps)==1 and core.optimizer_revision==3


def test_only_this_reduction_matching_lane_and_target_statistics_are_used(monkeypatch):
    _,payload=compute(monkeypatch,rows(8,4),lane=1,width=2)
    core,_=engine(monkeypatch,lane=1,width=2)
    core.set_elastic_batch_partition(8,12,{'hybrid':4})
    core.enqueue_hybrid_gradient_payload(payload)
    assert hasattr(payload,'stats'), 'Hybrid compute discarded the loss statistics'
    ignored=replace(payload,replica_rank=0,stats=dict(payload.stats,n_samples=999.))
    stale=replace(payload,replica_id='old',step=2,stats=dict(payload.stats,n_samples=888.))
    prior=(stale.replica_id,stale.replica_rank,stale.step,stale.state_version,stale.membership_epoch)
    core.elastic_gradient_domain._consumed_payloads.add(prior)
    core._pending_hybrid_gradients.extend([ignored,stale])
    result=core.grpo_update(rows(0,8))
    assert result['n_samples']==12 and result['n_updates']==3

"""Fixed-batch gradient weighting, delivery identity and CPU reduction."""
from dataclasses import replace
import threading
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.infra.elastic.hybrid_pool import GradientPayload, InterReplicaGradientDomain, ReplicaRole
from RL_Framework.infra.elastic.gradient_ipc import ElasticGradientClient, ElasticGradientServer

def _domain():
    domain = InterReplicaGradientDomain(core_replica_ids=['dp0', 'dp1'])
    domain.request_join('hybrid', 'dp0', membership_epoch=7)
    domain.mark_active('hybrid', membership_epoch=7)
    return domain



def _payload(**changes):
    return replace(GradientPayload('hybrid', 'dp0', (torch.tensor([9.]),),
                                  step=3, state_version=2, membership_epoch=7,
                                  sample_count=2, global_sample_count=8), **changes)



def test_uneven_partitions_match_full_batch_gradient_and_adam_update():
    # Core means 1 (four samples) and 5 (two); Hybrid mean 9 (two).
    result = _domain().reduce_core_gradients(
        core_gradients={'dp0': (torch.tensor([1.]),), 'dp1': (torch.tensor([5.]),)},
        core_sample_counts={'dp0': 4, 'dp1': 2}, global_sample_count=8,
        hybrid_payloads=[_payload()], step=3, state_version=2,
    )
    expected = torch.tensor([(4 + 10 + 18) / 8])
    for gradients in result.values():
        torch.testing.assert_close(gradients[0], expected)
    full = torch.nn.Parameter(torch.tensor([2.]))
    split = torch.nn.Parameter(full.detach().clone())
    reference = torch.optim.Adam([full], lr=.1)
    elastic = torch.optim.Adam([split], lr=.1)
    full.grad, split.grad = expected.clone(), result['dp0'][0].clone()
    reference.step()
    elastic.step()
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(reference.state[full]['exp_avg'], elastic.state[split]['exp_avg'])









def _gloo_worker(rank, rendezvous, output, in_place=False):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    isolated = dist.new_group([0, 1], backend='gloo')
    domain = _domain()
    domain.bind_hybrid_process_group(isolated)
    result = domain.reduce_core_gradients(
        core_gradients={f'dp{rank}': (torch.tensor([1. if rank == 0 else 5.]),)},
        core_sample_counts={f'dp{rank}': 4 if rank == 0 else 2},
        global_sample_count=8, hybrid_payloads=[_payload()] if rank == 0 else [],
        step=3, state_version=2,
        in_place=in_place,
    )
    output[rank] = float(result[f'dp{rank}'][0].item())
    dist.destroy_process_group()



@pytest.mark.parametrize('in_place', [False, True])
def test_real_gloo_lane_without_hybrid_still_participates(tmp_path, in_place):
    context = mp.get_context('spawn')
    with context.Manager() as manager:
        results = manager.dict()
        processes = [context.Process(target=_gloo_worker, args=(rank, str(tmp_path / 'rendezvous'), results, in_place)) for rank in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        try:
            assert [process.exitcode for process in processes] == [0, 0]
            assert dict(results) == {0: 4., 1: 4.}
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()















def test_tcp_preserves_batch_weighting_metadata():
    received = []
    server = ElasticGradientServer(on_payload=received.append)
    endpoint = server.start()
    try:
        ElasticGradientClient(endpoint).send(_payload())
        assert received[0].sample_count == 2
        assert received[0].global_sample_count == 8
        assert received[0].state_version == 2
    finally:
        server.close()






def test_empty_core_lane_enters_elastic_reduce_without_pending_or_expected_hybrids():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = _domain()
    engine._pending_hybrid_gradients = []
    engine._elastic_step_hybrid_workers = ()
    engine._elastic_active_gradient_timeout_s = 0
    engine._hybrid_gradient_condition = threading.Condition()
    engine._elastic_batch_partition = (0, 8, {'hybrid': 4})
    engine._elastic_training_step = 3
    engine._elastic_expected_state_version = 2
    engine._elastic_gradient_update_callback = None
    grad = torch.zeros(1)
    engine._model_main_gradients = lambda: [grad]
    engine.get_data_parallel_rank = lambda: 1
    entered = []
    def reduce(**kwargs):
        entered.append(kwargs)
        return {'dp1': (torch.tensor([7.]),)}
    engine.elastic_gradient_domain.reduce_core_gradients = reduce
    engine._apply_elastic_inter_replica_gradients()
    assert len(entered) == 1
    assert entered[0]['in_place'] is True
    assert entered[0]['core_gradients']['dp1'][0].data_ptr() == grad.data_ptr()
    torch.testing.assert_close(grad, torch.tensor([7.]))



def test_native_dp_without_hybrid_domain_keeps_loss_scale_one():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = None
    engine.get_data_parallel_world_size = lambda: 2
    engine.set_elastic_batch_partition(8, 16, {})
    assert engine._elastic_core_loss_scale() == 1.



class _GradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.]))
        self.weight.main_grad = torch.zeros(1)
        self.weight.register_hook(self._accumulate)

    def _accumulate(self, gradient):
        self.weight.main_grad.add_(gradient)
        return gradient

    def zero_grad_buffer(self):
        self.weight.main_grad.zero_()



class _AdamAdapter:
    def __init__(self, model):
        self.model = model
        self.inner = torch.optim.Adam(model.parameters(), lr=.01)

    def zero_grad(self):
        self.inner.zero_grad()

    def scale_loss(self, loss):
        return loss

    def step(self):
        self.model.weight.grad = self.model.weight.main_grad.clone()
        self.inner.step()
        return True, self.model.weight.main_grad.norm(), 0



def _gloo_engine_worker(rank, rendezvous, output):
    import sys
    import types
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    isolated = dist.new_group([0, 1], backend='gloo')
    def finalize(models):
        for model in models:
            dist.all_reduce(model.weight.main_grad)
            model.weight.main_grad.div_(2)
    sys.modules['megatron.core.distributed'] = types.SimpleNamespace(finalize_model_grads=finalize)
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    model = _GradientModel()
    engine.model = [model]
    engine.optimizer = _AdamAdapter(model)
    engine.optimizer_revision = 0
    engine.current_version = 0
    engine.elastic_gradient_domain = _domain()
    engine.elastic_gradient_domain.bind_hybrid_process_group(isolated)
    engine.get_data_parallel_world_size = lambda: 2
    engine.get_data_parallel_rank = lambda: rank
    engine._hybrid_gradient_condition = threading.Condition()
    engine._elastic_active_gradient_timeout_s = 10
    engine._elastic_gradient_update_callback = None
    engine._elastic_seen_payloads = set()
    engine._pending_hybrid_gradients = []
    engine.set_elastic_training_step(0, ['hybrid'] if rank == 0 else [])
    engine.set_elastic_batch_partition(4 if rank == 0 else 0, 8, {'hybrid': 4})
    if rank == 0:
        engine.enqueue_hybrid_gradient_payload(_payload(tensors=(torch.tensor([-7.]),), step=0,
                                                       state_version=0, sample_count=4))
    engine._iter_micro_batches = lambda rows: ({'trajectories': rows[start:start + 3]} for start in range(0, len(rows), 3))
    engine._forward = lambda batch: (model.weight - torch.tensor(batch['trajectories'])).square().mean()
    engine._loss = lambda batch, loss: (loss, {'loss': float(loss.detach()), 'pg_loss': float(loss.detach()),
                                             'kl': 0., 'reward_sum': 0., 'n_samples': len(batch['trajectories']), 'n_updates': 1})
    engine._reduce_stats = lambda stats: stats
    engine.grpo_update([0., 1., 2., 3.] if rank == 0 else [])
    optimizer_state = engine.optimizer.inner.state[model.weight]
    # Simulate a real RS buffer: only this rank's shard contains finalized
    # gradients; the other shard still has stale local values.
    bucket_data = torch.tensor([1., 2., 999., 999.]) if rank == 0 else torch.tensor([-999., -999., 3., 4.])
    model.bucket_groups = [types.SimpleNamespace(
        intra_distributed_optimizer_instance_group=dist.group.WORLD,
        buckets=[types.SimpleNamespace(grad_data=bucket_data)],
    )]
    engine.use_distributed_optimizer = True
    engine._materialize_elastic_core_gradients()
    output[rank] = (float(model.weight.detach()), float(optimizer_state['exp_avg']),
                    float(optimizer_state['exp_avg_sq']), engine.optimizer_revision, bucket_data.tolist())
    dist.destroy_process_group()



def test_full_engine_gloo_prescale_finalize_elastic_sum_matches_full_batch_with_empty_core(tmp_path):
    context = mp.get_context('spawn')
    with context.Manager() as manager:
        results = manager.dict()
        processes = [context.Process(target=_gloo_engine_worker, args=(rank, str(tmp_path / 'rendezvous'), results)) for rank in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        try:
            assert [process.exitcode for process in processes] == [0, 0]
            expected = torch.nn.Parameter(torch.tensor([2.]))
            optimizer = torch.optim.Adam([expected], lr=.01)
            (expected - torch.arange(8)).square().mean().backward()
            optimizer.step()
            for result in results.values():
                assert result[0] == pytest.approx(float(expected))
                assert result[1] == pytest.approx(float(optimizer.state[expected]['exp_avg']))
                assert result[2] == pytest.approx(float(optimizer.state[expected]['exp_avg_sq']))
                assert result[3] == 1
                assert result[4] == [1., 2., 3., 4.]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()


@pytest.mark.parametrize('change,message', [
    ({'step': -1}, 'step'), ({'state_version': -1}, 'revision'),
    ({'membership_epoch': 6}, 'epoch'), ({'global_sample_count': 16}, 'batch size'),
    ({'zero_placeholder': True}, 'placeholder'),
])
def test_rejects_stale_or_incomplete_gradient_identity(change, message):
    with pytest.raises(ValueError, match=message):
        _domain().reduce_core_gradients(
            core_gradients={'dp0': (torch.ones(1),), 'dp1': (torch.ones(1),)},
            core_sample_counts={'dp0': 4, 'dp1': 2}, global_sample_count=8,
            hybrid_payloads=[_payload(**change)], step=3, state_version=2,
        )



def test_rejects_duplicate_payload_before_double_counting():
    domain = _domain()
    kwargs = dict(core_gradients={'dp0': (torch.ones(1),), 'dp1': (torch.ones(1),)},
                  hybrid_payloads=[_payload(), _payload()], step=3, state_version=2)
    with pytest.raises(ValueError, match='duplicate'):
        domain.reduce_core_gradients(**kwargs)



def test_engine_ingress_rejects_unknown_revision_and_duplicate_without_waking_waiters():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = _domain()
    engine.optimizer_revision = 2
    engine._elastic_seen_payloads = set()
    engine._pending_hybrid_gradients = []
    engine._hybrid_gradient_condition = threading.Condition()
    engine.set_elastic_training_step(3, ['hybrid'])
    engine.set_elastic_batch_partition(4, 8, {'hybrid': 2})
    for bad in (_payload(state_version=-1), _payload(step=-1), _payload(membership_epoch=6)):
        with pytest.raises(ValueError):
            engine.enqueue_hybrid_gradient_payload(bad)
    assert not engine._pending_hybrid_gradients
    engine.enqueue_hybrid_gradient_payload(_payload())
    with pytest.raises(ValueError, match='duplicate'):
        engine.enqueue_hybrid_gradient_payload(_payload())
    assert len(engine._pending_hybrid_gradients) == 1

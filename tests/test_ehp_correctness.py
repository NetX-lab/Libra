"""Numerical fixed-batch EHP updates and real state installation receipts."""
from dataclasses import replace
import threading
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.infra.elastic.hybrid_pool import (
    ElasticHybridPool, GradientPayload, InterReplicaGradientDomain, ReplicaRole,
)
from RL_Framework.infra.elastic.gradient_ipc import ElasticGradientClient, ElasticGradientServer
from RL_Framework.infra.elastic.state_transfer import LiveStateServer, receive_live_state


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


def _gloo_worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    isolated = dist.new_group([0, 1], backend='gloo')
    domain = _domain()
    domain.bind_hybrid_process_group(isolated)
    result = domain.reduce_core_gradients(
        core_gradients={f'dp{rank}': (torch.tensor([1. if rank == 0 else 5.]),)},
        core_sample_counts={f'dp{rank}': 4 if rank == 0 else 2},
        global_sample_count=8, hybrid_payloads=[_payload()] if rank == 0 else [],
        step=3, state_version=2,
    )
    output[rank] = float(result[f'dp{rank}'][0].item())
    dist.destroy_process_group()


def test_real_gloo_lane_without_hybrid_still_participates(tmp_path):
    context = mp.get_context('spawn')
    with context.Manager() as manager:
        results = manager.dict()
        processes = [context.Process(target=_gloo_worker, args=(rank, str(tmp_path / 'rendezvous'), results)) for rank in range(2)]
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


def _cpu_engine():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.model = [torch.nn.Linear(2, 1)]
    engine.optimizer = torch.optim.Adam(engine.model[0].parameters(), lr=.01)
    engine.optimizer_revision = 0
    engine.current_version = 0
    engine.get_parallel_state = lambda: {'train_tp': 1, 'train_dp': 1}
    return engine


def _adam_step(engine):
    engine.optimizer.zero_grad()
    engine.model[0](torch.tensor([[1., 2.]])).square().sum().backward()
    engine.optimizer.step()
    engine.optimizer_revision += 1


def test_live_snapshot_preserves_theta_adam_step_and_next_update_without_files(tmp_path):
    source, joined = _cpu_engine(), _cpu_engine()
    _adam_step(source)
    _adam_step(source)
    snapshot = source.capture_elastic_state()
    server = LiveStateServer(host='127.0.0.1', advertised_host='127.0.0.1')
    try:
        uri = server.publish(snapshot)
        receipt = joined.restore_elastic_state(receive_live_state(uri))
        assert receipt['acknowledged'] and receipt['optimizer_revision'] == 2
        _adam_step(source)
        _adam_step(joined)
        for expected, actual in zip(source.model[0].parameters(), joined.model[0].parameters()):
            torch.testing.assert_close(expected, actual)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(source.optimizer.state[expected][key], joined.optimizer.state[actual][key])
        server.release(uri)
        with pytest.raises(RuntimeError, match='expired'):
            receive_live_state(uri)
        assert list(tmp_path.iterdir()) == []
    finally:
        server.close()


def test_unacknowledged_snapshot_never_activates():
    pool = ElasticHybridPool(core_train_workers=['dp0'], hybrid_workers=['h'], snapshot_fetcher=lambda *_: 8)
    try:
        handle = pool.join_training('h', 'dp0', activation_barrier=lambda *_: None)
        with pytest.raises(RuntimeError, match='ACK'):
            handle.result(timeout=2)
        assert pool.snapshot()['h'].role == ReplicaRole.HYBRID_ROLLOUT
    finally:
        pool.close()


def test_actual_install_ack_can_catch_up_beyond_snapshot():
    installed = threading.Event()
    def activate(*_):
        installed.wait(2)
        return {'acknowledged': True, 'state_version': 11, 'optimizer_revision': 11}
    pool = ElasticHybridPool(core_train_workers=['dp0'], hybrid_workers=['h'], snapshot_fetcher=lambda *_: 8)
    try:
        handle = pool.join_training('h', 'dp0', activation_barrier=activate)
        time.sleep(.03)
        assert not handle.done()
        assert pool.gradient_domain.is_joining('h')
        installed.set()
        assert handle.result(timeout=2).state_version == 11
    finally:
        pool.close()


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


def test_lane_manifest_selects_matching_target_core_and_model_lane(tmp_path):
    import json
    source, wrong_lane, joined = _cpu_engine(), _cpu_engine(), _cpu_engine()
    _adam_step(source)
    _adam_step(wrong_lane)
    server = LiveStateServer(host='127.0.0.1', advertised_host='127.0.0.1')
    try:
        correct = server.publish(source.capture_elastic_state())
        wrong = server.publish(wrong_lane.capture_elastic_state())
        manifest = tmp_path / 'lanes.json'
        manifest.write_text(json.dumps({'format': 'libra-ehp-lanes-v1', 'optimizer_revision': 1,
                                        'snapshot_id': 'boundary-1',
                                        'lanes': {'dp0': {'0': wrong}, 'dp1': {'1': correct}}}))
        receipt = joined.load_elastic_state_snapshot(str(manifest), target_core_id='dp1', replica_rank=1)
        assert receipt['snapshot_id'] == 'boundary-1'
        assert receipt['target_core_id'] == 'dp1'
        for expected, actual in zip(source.model[0].parameters(), joined.model[0].parameters()):
            torch.testing.assert_close(expected, actual)
        manifest.write_text(json.dumps({'format': 'libra-ehp-lanes-v1', 'optimizer_revision': 2,
                                        'snapshot_id': 'boundary-2', 'lanes': {'dp1': {'1': correct}}}))
        with pytest.raises(ValueError, match='revision'):
            joined.load_elastic_state_snapshot(str(manifest), target_core_id='dp1', replica_rank=1)
    finally:
        server.close()


def test_chained_distributed_optimizer_snapshot_includes_actual_master_and_adam_tensors():
    from types import SimpleNamespace
    class DistributedChild:
        data_parallel_group_gloo = object()
        def __init__(self, value):
            self.state = {'param': torch.tensor([value]), 'exp_avg': torch.tensor([value + 1]),
                          'exp_avg_sq': torch.tensor([value + 2]), 'step': torch.tensor(9)}
        def get_parameter_state_dp_zero(self):
            return self.state
        def load_parameter_state_from_dp_zero(self, state):
            self.state = state
    source = SimpleNamespace(chained_optimizers=[DistributedChild(1.), DistributedChild(3.)])
    target = SimpleNamespace(chained_optimizers=[DistributedChild(0.), DistributedChild(0.)])
    payload = MegatronCoreTrainEngine._clone_cpu_state(MegatronCoreTrainEngine._capture_optimizer_parameter_state(source))
    source.chained_optimizers[0].state['param'].add_(100)
    MegatronCoreTrainEngine._restore_optimizer_parameter_state(target, payload)
    assert target.chained_optimizers[0].state['param'].item() == 1.
    assert target.chained_optimizers[1].state['exp_avg_sq'].item() == 5.
    assert target.chained_optimizers[1].state['step'].item() == 9


def test_receiving_dp_root_rejects_nonroot_source_missing_adam_and_accepts_canonical_dp0():
    from types import SimpleNamespace
    loaded = []
    target = SimpleNamespace(data_parallel_group_gloo=SimpleNamespace(rank=lambda: 0),
                             load_parameter_state_from_dp_zero=loaded.append)
    with pytest.raises(ValueError, match='master parameters and Adam'):
        MegatronCoreTrainEngine._restore_optimizer_parameter_state(target, {'kind': 'dp_zero', 'state': None})
    assert not loaded
    canonical = {'param': torch.tensor([2.]), 'exp_avg': torch.tensor([.2]), 'exp_avg_sq': torch.tensor([.02])}
    MegatronCoreTrainEngine._restore_optimizer_parameter_state(target, {'kind': 'dp_zero', 'state': canonical})
    assert loaded == [canonical]


def test_actual_mcore_dp_zero_restore_preserves_step_master_moments_and_removes_adapter(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = pytest.importorskip('megatron.core.optimizer.distrib_optimizer')
    monkeypatch.setattr(module, 'HAVE_APEX_OR_TE', False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    # The production initialization may already have installed its global
    # ordinary-checkpoint compatibility wrapper. Isolate the real native setter
    # for the negative reproduction, then restore that wrapper before the EHP
    # installation to verify that the scoped context can coexist with it.
    existing_setter = module.DistributedOptimizer._set_main_param_and_optimizer_states
    native_setter = existing_setter
    if getattr(native_setter, '_rl_framework_step_compat', False):
        closure = dict(zip(native_setter.__code__.co_freevars,
                           (cell.cell_contents for cell in native_setter.__closure__)))
        native_setter = closure['original_set_state']
    monkeypatch.setattr(module.DistributedOptimizer, '_set_main_param_and_optimizer_states', native_setter)
    assert not dist.is_initialized()
    dist.init_process_group('gloo', init_method=f'file://{tmp_path / "dp-zero-rendezvous"}', rank=0, world_size=1)
    try:
        def engine(values):
            master = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32), requires_grad=False)
            model = torch.nn.Parameter(master.detach().bfloat16().clone())
            inner = torch.optim.AdamW([{
                'params': [master], 'wd_mult': 1., 'lr_mult': 1.,
                'is_expert_parallel': False, 'is_decoupled_lr': False,
            }], lr=.01, betas=(.9, .95))
            optimizer = module.DistributedOptimizer.__new__(module.DistributedOptimizer)
            optimizer.optimizer = inner
            optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
            optimizer.config = SimpleNamespace(
                exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32,
                use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fp16=False,
            )
            optimizer.grad_scaler = None
            optimizer.data_parallel_group_gloo = dist.group.WORLD
            optimizer.model_param_group_index_map = {model: (0, 0)}
            dtype = (torch.bfloat16, torch.float32)
            optimizer.gbuf_ranges = [{dtype: [{'param_map': {
                model: {'gbuf_world': module.Range(0, 2), 'param': module.Range(0, 2),
                        'gbuf_local': module.Range(0, 2)},
            }}]}]
            optimizer.buffers = [SimpleNamespace(
                numel_unpadded=2, params=[model],
                buckets=[SimpleNamespace(grad_data=torch.zeros(2), numel_unpadded=2)],
            )]
            chunk = torch.nn.Module()
            chunk.register_parameter('weight', model)
            result = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
            result.model, result.optimizer = [chunk], optimizer
            result.optimizer_revision, result.current_version = 3, 0
            result.get_parallel_state = lambda: dict(train_tp=1, train_dp=1)
            return result, master, model

        source, source_master, source_model = engine([1., 2.])
        for _ in range(3):
            source_master.grad = torch.tensor([.1, -.2])
            source.optimizer.optimizer.step()
        with torch.no_grad():
            source_model.copy_(source_master)
        snapshot = source.capture_elastic_state()
        target, target_master, target_model = engine([-9., -9.])
        # Actual pinned DP-zero gather/scatter excludes scalar step from tensor
        # transfers, while its native setter iterates the destination step key.
        target.optimizer.load_state_dict(snapshot['optimizer'])
        with pytest.raises(KeyError, match='step'):
            target.optimizer.load_parameter_state_from_dp_zero(snapshot['distributed_optimizer']['state'])
        monkeypatch.setattr(module.DistributedOptimizer, '_set_main_param_and_optimizer_states', existing_setter)
        receipt = target.restore_elastic_state(snapshot)
        assert receipt['acknowledged'] and receipt['optimizer_revision'] == 3
        assert '_set_main_param_and_optimizer_states' not in vars(target.optimizer)
        assert module.DistributedOptimizer._set_main_param_and_optimizer_states is existing_setter
        torch.testing.assert_close(target_master, source_master, rtol=0, atol=0)
        torch.testing.assert_close(target_model, source_model, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(target.optimizer.optimizer.state[target_master][key],
                                       source.optimizer.optimizer.state[source_master][key], rtol=0, atol=0)
        for adapter, parameter in ((source, source_master), (target, target_master)):
            parameter.grad = torch.tensor([-.3, .4])
            adapter.optimizer.optimizer.step()
        torch.testing.assert_close(target_master, source_master, rtol=0, atol=0)
        incomplete = MegatronCoreTrainEngine._clone_cpu_state(snapshot)
        dtype = (torch.bfloat16, torch.float32)
        del incomplete['distributed_optimizer']['state'][0][dtype]['exp_avg']
        with pytest.raises(KeyError, match='exp_avg'):
            target.restore_elastic_state(incomplete)
        assert '_set_main_param_and_optimizer_states' not in vars(target.optimizer)
        assert module.DistributedOptimizer._set_main_param_and_optimizer_states is existing_setter
    finally:
        dist.destroy_process_group()

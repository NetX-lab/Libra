"""Native master/moment/step restoration at an elastic boundary."""
import pytest
import torch
import torch.distributed as dist
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


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

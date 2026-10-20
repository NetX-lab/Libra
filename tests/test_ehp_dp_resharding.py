"""Native MCore Adam resharding from two Core DP replicas to one Hybrid.

Each test isolates one model-parallel lane: real Gloo gathers the Core shards,
then a separately initialized DP1 group restores the native DP-zero payload.
The small optimizer fixture follows test_ehp_correctness.py and needs no CUDA.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


def _engine(module, rank, world, values, padded_extra=0):
    full = torch.tensor(values, dtype=torch.float32)
    padded_numel = len(values) + padded_extra
    width = padded_numel // world
    start, end = rank * width, min((rank + 1) * width, len(values))
    master = torch.nn.Parameter(full[start:end].clone(), requires_grad=False)
    model = torch.nn.Parameter(full.bfloat16().clone())
    inner = torch.optim.AdamW([{'params': [master], 'wd_mult': 1., 'lr_mult': 1.,
                               'is_expert_parallel': False, 'is_decoupled_lr': False}],
                              lr=.01, betas=(.9, .95))
    optimizer = module.DistributedOptimizer.__new__(module.DistributedOptimizer)
    optimizer.optimizer = inner
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.config = SimpleNamespace(exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fp16=False)
    optimizer.grad_scaler = None
    optimizer.data_parallel_group_gloo = dist.group.WORLD
    optimizer.model_param_group_index_map = {model: (0, 0)}
    dtype = (torch.bfloat16, torch.bfloat16)
    optimizer.gbuf_ranges = [{dtype: [{'param_map': {model: {
        'gbuf_world': module.Range(start, end), 'param': module.Range(start, end),
        'gbuf_local': module.Range(0, end - start)}}}]}]
    optimizer.buffers = [SimpleNamespace(numel_unpadded=len(values), params=[model],
        buckets=[SimpleNamespace(grad_data=torch.zeros(padded_numel), numel_unpadded=len(values))])]
    chunk = torch.nn.Module()
    chunk.register_parameter('weight', model)
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.model, engine.optimizer = [chunk], optimizer
    engine.optimizer_revision, engine.current_version = 3, 0
    engine.get_parallel_state = lambda: dict(train_tp=1, train_dp=world)
    return engine, master, model


def _capture_core_shards(rank, rendezvous, directory, padded_extra):
    from megatron.core.optimizer import distrib_optimizer as module
    module.HAVE_APEX_OR_TE = False
    torch.cuda.current_device = lambda: 'cpu'
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        engine, master, model = _engine(module, rank, 2, [1., 2., 3., 4.], padded_extra)
        gradient = torch.tensor([.1, -.2, .3, -.4])
        width = (4 + padded_extra) // 2
        for _ in range(3):
            master.grad = gradient[rank * width:rank * width + master.numel()].clone()
            engine.optimizer.optimizer.step()
        local = torch.zeros(width)
        local[:master.numel()].copy_(master.detach())
        shards = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(shards, local)
        with torch.no_grad():
            model.copy_(torch.cat(shards)[:4])
        snapshot = engine.capture_elastic_state()
        if rank == 0:
            torch.save(snapshot, Path(directory) / 'core-dp0.pt')
        else:
            # MCore returns empty per-bucket metadata on followers, not full
            # Adam/master tensors. Production canonicalizes all targets to dp0.
            state = snapshot['distributed_optimizer']['state']
            assert not state[0][(torch.bfloat16, torch.bfloat16)]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('source_padding', [0, 2])
def test_native_mcore_dp2_to_dp1_restores_full_master_moments_step_and_next_update(tmp_path, monkeypatch, source_padding):
    module = pytest.importorskip(
        'megatron.core.optimizer.distrib_optimizer',
        reason='native DP-zero resharding requires the pinned Megatron-Core source',
    )
    monkeypatch.setattr(module, 'HAVE_APEX_OR_TE', False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    mp.spawn(_capture_core_shards, args=(str(tmp_path / 'source-rendezvous'), str(tmp_path), source_padding), nprocs=2, join=True)
    snapshot = torch.load(tmp_path / 'core-dp0.pt', weights_only=False)
    dist.init_process_group('gloo', init_method=f'file://{tmp_path / "target-rendezvous"}',
                            rank=0, world_size=1, timeout=timedelta(seconds=30))
    try:
        target, master, model = _engine(module, 0, 1, [-9.] * 4)
        receipt = target.restore_elastic_state(snapshot)
        assert receipt['acknowledged'] and receipt['optimizer_revision'] == 3
        reference = torch.nn.Parameter(torch.tensor([1., 2., 3., 4.]))
        adam = torch.optim.AdamW([reference], lr=.01, betas=(.9, .95))
        for _ in range(3):
            reference.grad = torch.tensor([.1, -.2, .3, -.4])
            adam.step()
        torch.testing.assert_close(master, reference, rtol=0, atol=0)
        torch.testing.assert_close(model, reference.bfloat16(), rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(target.optimizer.optimizer.state[master][key],
                                       adam.state[reference][key], rtol=0, atol=0)
        master.grad = torch.tensor([-.7, .4, .2, -.1])
        reference.grad = master.grad.clone()
        target.optimizer.optimizer.step()
        adam.step()
        torch.testing.assert_close(master, reference, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(target.optimizer.optimizer.state[master][key],
                                       adam.state[reference][key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()

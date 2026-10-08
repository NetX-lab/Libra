"""Contract/regression checks; these do not claim V4 hardware validation."""
import sys
from types import SimpleNamespace, ModuleType

import pytest
import torch

from RL_Framework.config import AsyncRLConfig, ModelArchConfig
from RL_Framework.engine.mindspeed_train_engine import MindSpeedTrainEngine, MBridgeAdapter


def config(**extra):
    values = dict(model_path='/model/v4', train_backend='megatron_core',
                  megatron_model_provider='mindspeed', train_gpus=64, rollout_gpus=32,
                  train_tp_size=4, train_pp_size=2, train_ep_size=32,
                  expert_tensor_parallel_size=1, train_dp_size=8, batch_size=32,
                  model_arch=ModelArchConfig(num_experts=256, expert_intermediate_size=2048))
    values.update(extra)
    return AsyncRLConfig(**values)


def test_v4_topology_accepts_decoupled_expert_tp():
    c = config()
    assert (c.train_tp_size, c.train_pp_size, c.train_dp_size, c.train_ep_size) == (4, 2, 8, 32)


@pytest.mark.parametrize('extra', [dict(train_ep_size=64), dict(train_cp_size=2, train_dp_size=4),
                                 dict(megatron_use_precision_aware_optimizer=True)])
def test_v4_rejects_unsupported_topology(extra):
    with pytest.raises(ValueError):
        config(**extra)


def test_runtime_args_preserve_v4_and_override_training_settings():
    e = MindSpeedTrainEngine(model_path='/model/v4', train_tp_size=4, train_pp_size=2,
                            train_ep_size=32, global_batch_size=256, micro_batch_size=1)
    a = e.build_mindspeed_args(4096)
    assert a['num_layer_list'] == '21,22'
    assert a['spec'] == ['mindspeed_llm.tasks.models.spec.deepseek4_spec', 'layer_spec']
    assert a['seq_length'] == 4096 and a['global_batch_size'] == 256
    assert a['tokenizer_name_or_path'] == '/model/v4'
    assert a['enable_mhc'] and a['enable_dsa_indexer']
    assert a['use_precision_aware_optimizer'] is False


def engine():
    e = MindSpeedTrainEngine(model_path='/model/v4', train_tp_size=1, train_pp_size=2,
                            micro_batch_size=1, kl_coef=0.0)
    e._device = lambda: torch.device('cpu')
    e.tokenizer = SimpleNamespace(pad_token_id=0)
    return e


def trajectories():
    return [dict(input_ids=torch.tensor([[1, 2, 3, 4]]), logprobs=torch.zeros(1, 4),
                 loss_mask=torch.tensor([[0., 1., 1., 1.]]), rewards=torch.tensor([1.])) for _ in range(2)]


def test_pipeline_delegates_backward_and_uses_libra_loss(monkeypatch):
    e = engine()
    weight = torch.nn.Parameter(torch.ones(1, 4, 6))
    e.model = [lambda **kw: weight]
    calls = []
    mod = ModuleType('megatron.core.pipeline_parallel')
    def schedule(**kw):
        calls.append(kw)
        stats = []
        for _ in range(kw['num_microbatches']):
            output, fn = kw['forward_step_func'](kw['data_iterator'], kw['model'][0])
            loss, metrics = fn(output)
            (loss / kw['num_microbatches']).backward()
            stats.append(metrics)
        return stats
    mod.get_forward_backward_func = lambda: schedule
    monkeypatch.setitem(sys.modules, 'megatron.core.pipeline_parallel', mod)
    stats, _ = e._pipeline(trajectories())
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0
    assert len(stats) == 2 and calls[0]['seq_length'] == 4
    assert calls[0]['forward_only'] is False


def test_pipeline_rejects_inconsistent_communication_shapes(monkeypatch):
    mod = ModuleType('megatron.core.pipeline_parallel')
    mod.get_forward_backward_func = lambda: pytest.fail('must reject before collectives')
    monkeypatch.setitem(sys.modules, 'megatron.core.pipeline_parallel', mod)
    ts = trajectories()
    for key in ('input_ids', 'logprobs', 'loss_mask'):
        ts[1][key] = ts[1][key][:, :3]
    with pytest.raises(ValueError, match='Align trajectories'):
        engine()._pipeline(ts)


def test_data_distribution_handles_tp1_pp2(monkeypatch):
    import RL_Framework.engine.mindspeed_train_engine as module
    e = engine()
    e.is_batch_source = lambda: True
    e._parallel_state = lambda: SimpleNamespace(is_pipeline_first_stage=lambda: True,
                get_pipeline_model_parallel_first_rank=lambda: 0,
                get_pipeline_model_parallel_group=lambda: 'pp')
    calls = []
    monkeypatch.setattr(module.dist, 'broadcast_object_list', lambda payload, **kw: calls.append(kw))
    ts = trajectories()
    assert e.distribute_trajectories(ts) is ts
    assert calls == [dict(src=0, group='pp')]


def test_bridge_adapter_consumes_collective_export_on_each_rank():
    seen = []
    def export(model):
        seen.append(model)
        yield 'layer.weight', torch.tensor([2.])
    adapter = MBridgeAdapter(SimpleNamespace(export_weights=export), None)
    assert list(adapter.export_hf_weights(['stage'], cpu=True))[0][0] == 'layer.weight'
    assert seen == [['stage']]


def test_ehp_uses_complete_v4_dp_ep_replica():
    from RL_Framework.config import GlobalResourcePlannerConfig
    e = MindSpeedTrainEngine(model_path='/model/v4', train_tp_size=4,
                            train_pp_size=2, train_dp_size=8, train_ep_size=32,
                            expert_tensor_parallel_size=1, micro_batch_size=1)
    assert e.get_elastic_replica_size_gpus() == 64
    assert e.get_elastic_core_replica_ids() == ['model']
    assert e.get_elastic_local_core_id() == 'model'
    e.get_elastic_core_process_group = lambda: None
    domain = e.configure_elastic_training(['model'], decouple_communication_domains=False,
                                          replica_world_size=64)
    assert domain.replica_world_size == 64
    with pytest.raises(ValueError, match=r'complete TP\*PP\*DP replica'):
        e.configure_elastic_training(['model'], decouple_communication_domains=False,
                                      replica_world_size=8)


def test_mindspeed_config_allows_ep_sharded_ehp():
    from RL_Framework.config import GlobalResourcePlannerConfig
    planner = GlobalResourcePlannerConfig(elastic_hybrid_planning_enabled=True)
    assert config(global_resource_planner=planner).global_resource_planner.elastic_hybrid_planning_enabled


def test_mindspeed_rejects_cuda_only_weight_transport():
    with pytest.raises(ValueError, match="HCCL transport"):
        config(weight_sync_mode="nccl", rollout_weight_sync_mode="nccl")


def test_alignment_pads_across_microbatches_without_training_on_padding():
    e = engine()
    data = trajectories()
    for t in data:
        t['attention_mask'] = torch.ones_like(t['input_ids'])
    for key in ('input_ids', 'logprobs', 'loss_mask', 'attention_mask'):
        data[1][key] = data[1][key][:, :3]
    result = e.align_distributed_trajectories(data)
    assert all(t['input_ids'].shape == (1,4) for t in result)
    assert result[1]['loss_mask'][0,-1] == 0
    assert result[1]['attention_mask'][0,-1] == 0


def test_export_must_not_drop_trainable_weights():
    original = {'weight_map': {'layers.0.weight': 'a', 'mtp.0.weight': 'b'}}
    with pytest.raises(RuntimeError, match='non-MTP'):
        MBridgeAdapter._missing_frozen_names(original, [])
    assert MBridgeAdapter._missing_frozen_names(original, ['layers.0.weight']) == ['mtp.0.weight']


def test_disk_export_keeps_frozen_mtp_and_original_config(tmp_path, monkeypatch):
    import json
    import RL_Framework.engine.mindspeed_train_engine as module
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    (source/'frozen.safetensors').write_bytes(b'fixture')
    original = dict(weight_map={'layers.0.weight':'train.safetensors', 'mtp.0.weight':'frozen.safetensors'},
                    metadata={'total_size':16})
    (source/'model.safetensors.index.json').write_text(json.dumps(original))
    (source/'config.json').write_text('{"model_type":"deepseek_v4"}')
    def save(model, path, memory_efficient):
        (target/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layers.0.weight':'updated.safetensors'}}))
    adapter = MBridgeAdapter(SimpleNamespace(save_weights=save),
                             SimpleNamespace(save_pretrained=lambda p: None), source)
    monkeypatch.setattr(module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(module.dist, 'broadcast_object_list', lambda *a, **kw: None)
    adapter.save_hf_pretrained([], target)
    result = json.loads((target/'model.safetensors.index.json').read_text())
    assert result['weight_map']['layers.0.weight'] == 'updated.safetensors'
    assert (target/result['weight_map']['mtp.0.weight']).resolve() == source/'frozen.safetensors'
    assert json.loads((target/'config.json').read_text())['model_type'] == 'deepseek_v4'

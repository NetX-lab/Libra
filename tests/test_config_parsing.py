from RL_Framework.config import HardwareConfig


def test_from_dict_parses_numeric_strings_and_ignores_unknown_keys():
    config = HardwareConfig.from_dict(
        {
            "gpus_per_node": "8",
            "mem_capacity": "80500000000.5",
            "tp_comm_overhead": "5e-6",
            "unknown_key": "ignored",
        }
    )

    assert config.gpus_per_node == 8
    assert isinstance(config.gpus_per_node, int)
    assert config.mem_capacity == 80500000000.5
    assert isinstance(config.mem_capacity, float)
    assert config.tp_comm_overhead == 5e-6
    assert isinstance(config.tp_comm_overhead, float)
    assert not hasattr(config, "unknown_key")


def test_decoupled_configuration_round_trips_to_megatron_engine(tmp_path, monkeypatch):
    from RL_Framework.config import AsyncRLConfig
    from RL_Framework.engine.train_factory import create_train_engine

    monkeypatch.chdir(tmp_path)
    config = AsyncRLConfig.from_dict(dict(
        model_path='unused', grpo_objective='decoupled',
        weight_sync_mode='nccl', rollout_weight_sync_mode='nccl',
    ))

    loaded = AsyncRLConfig.from_dict(config.to_dict())
    assert create_train_engine(loaded).grpo_objective == 'decoupled'
    assert AsyncRLConfig(model_path='unused').grpo_objective == 'legacy'

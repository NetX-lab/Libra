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

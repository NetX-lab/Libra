"""Support code for Test hetero config."""

import sys
import os
import tempfile
import textwrap


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import types
_pkg = types.ModuleType("RL_Framework")
_pkg.__path__ = [_project_root]
_pkg.__package__ = "RL_Framework"
sys.modules.setdefault("RL_Framework", _pkg)

from RL_Framework.config import (
    AsyncRLConfig,
    HeterogeneousRolloutConfig,
    HeterogeneousInstanceConfig,
    SchedulingConfig,
    _from_dict,
)


# =====================================================================

# =====================================================================

_pass_count = 0
_fail_count = 0


def check(name: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  [PASS] {name}")
    else:
        _fail_count += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =====================================================================

# =====================================================================

def test_instance_config_defaults():
    """Test instance config defaults."""
    section("1.1 HeterogeneousInstanceConfig default value")
    cfg = HeterogeneousInstanceConfig()
    check("Default instance_id is empty", cfg.instance_id == "")
    check("Default tp=1", cfg.tp == 1)
    check("Default gpus is an empty list", cfg.gpus == [])
    check("Default description is empty", cfg.description == "")


def test_instance_config_from_dict():
    """Test instance config from dict."""
    section("1.2 HeterogeneousInstanceConfig from_dict")
    d = {"instance_id": "tp2_inst0", "tp": 2, "gpus": [4, 5], "description": "TP=2"}
    cfg = HeterogeneousInstanceConfig.from_dict(d)
    check("instance_id is correct", cfg.instance_id == "tp2_inst0")
    check("tp=2", cfg.tp == 2)
    check("gpus=[4,5]", cfg.gpus == [4, 5])
    check("description", cfg.description == "TP=2")


def test_instance_config_from_dict_extra_keys():
    """Test instance config from dict extra keys."""
    section("1.3 HeterogeneousInstanceConfig extra keys")
    d = {"tp": 4, "gpus": [0, 1, 2, 3], "unknown_key": "should_be_ignored"}
    cfg = HeterogeneousInstanceConfig.from_dict(d)
    check("tp=4", cfg.tp == 4)
    check("gpus is correct", cfg.gpus == [0, 1, 2, 3])
    check("No unknown_key attribute", not hasattr(cfg, "unknown_key"))


# =====================================================================

# =====================================================================

def test_scheduling_config_defaults():
    """Test scheduling config defaults."""
    section("2.1 SchedulingConfig default value")
    cfg = SchedulingConfig()
    check("Default scheduler_type='length_aware'", cfg.scheduler_type == "length_aware")
    check("Default load_balance='least_connections'", cfg.load_balance_strategy == "least_connections")
    check("Default max_queue_length=100", cfg.max_queue_length == 100)
    check("Default enable_fallback=True", cfg.enable_fallback is True)
    check("length_thresholds contains short/medium/long",
          all(k in cfg.length_thresholds for k in ["short", "medium", "long"]))
    check("routing_rules contains four categories",
          len(cfg.routing_rules) == 4)


def test_scheduling_config_la_mlfq_fields():
    """Test scheduling config la mlfq fields."""
    section("2.2 SchedulingConfig LA-MLFQ specific fields")
    cfg = SchedulingConfig()
    check("la_mlfq_buckets has short/long",
          "short" in cfg.la_mlfq_buckets and "long" in cfg.la_mlfq_buckets)
    check("la_mlfq_migration_threshold=3000", cfg.la_mlfq_migration_threshold == 3000)
    check("la_mlfq_scout_timeout=30.0", cfg.la_mlfq_scout_timeout == 30.0)
    check("la_mlfq_history_ttl=5", cfg.la_mlfq_history_ttl == 5)


    short_bucket = cfg.la_mlfq_buckets["short"]
    check("Short bucket has tp_degrees", "tp_degrees" in short_bucket)
    check("Short bucket has max_tokens", "max_tokens" in short_bucket)


def test_scheduling_config_from_dict():
    """Test scheduling config from dict."""
    section("2.3 SchedulingConfig from_dict")
    d = {
        "scheduler_type": "la_mlfq",
        "load_balance_strategy": "round_robin",
        "max_queue_length": 64,
        "la_mlfq_migration_threshold": 5000,
        "la_mlfq_scout_timeout": 60.0,
        "length_thresholds": {"short": 1024, "medium": 2048, "long": 4096},
    }
    cfg = SchedulingConfig.from_dict(d)
    check("scheduler_type='la_mlfq'", cfg.scheduler_type == "la_mlfq")
    check("load_balance='round_robin'", cfg.load_balance_strategy == "round_robin")
    check("max_queue_length=64", cfg.max_queue_length == 64)
    check("la_mlfq_migration_threshold=5000", cfg.la_mlfq_migration_threshold == 5000)
    check("la_mlfq_scout_timeout=60.0", cfg.la_mlfq_scout_timeout == 60.0)
    check("length_thresholds.short=1024", cfg.length_thresholds["short"] == 1024)


# =====================================================================

# =====================================================================

def test_hetero_config_defaults():
    """Test hetero config defaults."""
    section("3.1 HeterogeneousRolloutConfig default value")
    cfg = HeterogeneousRolloutConfig()
    check("Default enabled=False", cfg.enabled is False)
    check("Default instances is empty", len(cfg.instances) == 0)
    check("Default total_gpus=0", cfg.total_gpus == 0)
    check("Default vllm_base_port=8000", cfg.vllm_base_port == 8000)
    check("scheduling is a SchedulingConfig", isinstance(cfg.scheduling, SchedulingConfig))


def test_hetero_config_from_dict():
    """Test hetero config from dict."""
    section("3.2 HeterogeneousRolloutConfig from_dict")
    d = {
        "enabled": True,
        "total_gpus": 6,
        "available_gpus": [2, 3, 4, 5, 6, 7],
        "vllm_host": "127.0.0.1",
        "vllm_base_port": 8000,
        "gpu_memory_utilization": 0.90,
        "instances": [
            {"instance_id": "tp1_inst0", "tp": 1, "gpus": [2]},
            {"instance_id": "tp1_inst1", "tp": 1, "gpus": [3]},
            {"instance_id": "tp2_inst0", "tp": 2, "gpus": [4, 5]},
            {"instance_id": "tp2_inst1", "tp": 2, "gpus": [6, 7]},
        ],
        "scheduling": {
            "scheduler_type": "la_mlfq",
            "length_thresholds": {"short": 1024, "medium": 2048, "long": 4096},
            "routing_rules": {"short": [1], "medium": [1, 2], "long": [2], "extra_long": [2]},
            "load_balance_strategy": "least_connections",
            "max_queue_length": 64,
        },
    }
    cfg = HeterogeneousRolloutConfig.from_dict(d)

    check("enabled=True", cfg.enabled is True)
    check("total_gpus=6", cfg.total_gpus == 6)
    check("Four instances", len(cfg.instances) == 4)
    check("n_instances=4", cfg.n_instances == 4)
    check("tp_list=[1,1,2,2]", cfg.tp_list == [1, 1, 2, 2])


    check("Instance 0 is a HeterogeneousInstanceConfig",
          isinstance(cfg.instances[0], HeterogeneousInstanceConfig))
    check("Instance 0 tp=1", cfg.instances[0].tp == 1)
    check("Instance 2 tp=2", cfg.instances[2].tp == 2)
    check("Instance 2 gpus=[4,5]", cfg.instances[2].gpus == [4, 5])


    check("scheduling.scheduler_type='la_mlfq'", cfg.scheduling.scheduler_type == "la_mlfq")
    check("scheduling.max_queue_length=64", cfg.scheduling.max_queue_length == 64)
    check("scheduling.length_thresholds.short=1024",
          cfg.scheduling.length_thresholds["short"] == 1024)


def test_hetero_config_auto_compute():
    """Test hetero config auto compute."""
    section("3.3 HeterogeneousRolloutConfig automatic calculation")
    d = {
        "enabled": True,
        "instances": [
            {"tp": 1, "gpus": [0]},
            {"tp": 2, "gpus": [1, 2]},
            {"tp": 4, "gpus": [3, 4, 5, 6]},
        ],
    }
    cfg = HeterogeneousRolloutConfig.from_dict(d)

    check("Automatically computes total_gpus=1+2+4=7", cfg.total_gpus == 7)
    check("tp_list=[1,2,4]", cfg.tp_list == [1, 2, 4])


    check("Instance 0 automatic ID", cfg.instances[0].instance_id == "hetero_tp1_0")
    check("Instance 1 automatic ID", cfg.instances[1].instance_id == "hetero_tp2_1")
    check("Instance 2 automatic ID", cfg.instances[2].instance_id == "hetero_tp4_2")


# =====================================================================

# =====================================================================

def test_async_rl_config_training_cluster():
    """Test async rl config training cluster."""
    section("4.1 AsyncRLConfig training cluster configuration")
    cfg = AsyncRLConfig(
        model_path="/path/to/user/test_model",
        train_gpus=4,
        rollout_gpus=4,
        tp_size=2,
        vllm_tp_size=2,
        learning_rate=1e-6,
        batch_size=32,
        total_steps=100,
    )
    check("model_path is correct", cfg.model_path == "/path/to/user/test_model")
    check("train_gpus=4", cfg.train_gpus == 4)
    check("rollout_gpus=4", cfg.rollout_gpus == 4)
    check("tp_size=2", cfg.tp_size == 2)
    check("vllm_tp_size=2", cfg.vllm_tp_size == 2)
    check("learning_rate=1e-6", cfg.learning_rate == 1e-6)
    check("batch_size=32", cfg.batch_size == 32)
    check("total_steps=100", cfg.total_steps == 100)
    check("train_backend defaults to fsdp", cfg.train_backend == "fsdp")
    check("train_tp_size inherits tp_size", cfg.train_tp_size == 2)
    check("train_pp_size defaults to 1", cfg.train_pp_size == 1)
    check("train_dp_size is inferred as 2", cfg.train_dp_size == 2)
    check("tokenizer_path defaults to model_path",
          cfg.tokenizer_path == cfg.model_path)


def test_async_rl_config_with_heterogeneous():
    """Test async rl config with heterogeneous."""
    section("4.2 AsyncRLConfig + heterogeneous rollout integration")
    hetero = HeterogeneousRolloutConfig.from_dict({
        "enabled": True,
        "instances": [
            {"tp": 1, "gpus": [2]},
            {"tp": 1, "gpus": [3]},
            {"tp": 2, "gpus": [4, 5]},
        ],
        "scheduling": {
            "scheduler_type": "la_mlfq",
            "la_mlfq_migration_threshold": 5000,
        },
    })
    cfg = AsyncRLConfig(
        model_path="/path/to/user/test_model",
        train_gpus=2,
        rollout_gpus=4,
        heterogeneous_rollout=hetero,
    )
    check("heterogeneous_rollout.enabled=True", cfg.heterogeneous_rollout.enabled is True)
    check("Three heterogeneous instances", cfg.heterogeneous_rollout.n_instances == 3)
    check("scheduler_type='la_mlfq'",
          cfg.heterogeneous_rollout.scheduling.scheduler_type == "la_mlfq")
    check("la_mlfq_migration_threshold=5000",
          cfg.heterogeneous_rollout.scheduling.la_mlfq_migration_threshold == 5000)


def test_async_rl_config_multi_node():
    """Test async rl config multi node."""
    section("4.3 AsyncRLConfig multi-node configuration")
    cfg = AsyncRLConfig(
        model_path="/path/to/user/test_model",
        train_gpus=16,
        rollout_gpus=16,
        batch_size=64,
        num_nodes=4,
        gpus_per_node_override=8,
        master_addr="192.0.2.50",
        master_port=29500,
    )
    check("num_nodes=4", cfg.num_nodes == 4)
    check("gpus_per_node_override=8", cfg.gpus_per_node_override == 8)
    check("master_addr='192.0.2.50'", cfg.master_addr == "192.0.2.50")
    check("master_port=29500", cfg.master_port == 29500)
    check("train_gpus=16", cfg.train_gpus == 16)
    check("rollout_gpus=16", cfg.rollout_gpus == 16)


def test_async_rl_config_megatron3d_validation():
    """Test async rl config megatron3d validation."""
    section("4.4 AsyncRLConfig Megatron3D configuration validation")

    cfg = AsyncRLConfig(
        model_path="/path/to/user/test_model",
        train_gpus=4,
        rollout_gpus=4,
        train_backend="megatron3d",
        train_tp_size=2,
        train_pp_size=2,
        train_dp_size=1,
        weight_sync_mode="disk",
        batch_size=32,
        micro_batch_size=4,
    )
    check("train_backend=megatron3d", cfg.train_backend == "megatron3d")
    check("train_tp_size=2", cfg.train_tp_size == 2)
    check("train_pp_size=2", cfg.train_pp_size == 2)
    check("train_dp_size=1", cfg.train_dp_size == 1)

    invalid_raised = False
    try:
        AsyncRLConfig(
            model_path="/path/to/user/test_model",
            train_gpus=4,
            rollout_gpus=4,
            train_backend="megatron3d",
            train_tp_size=2,
            train_pp_size=2,
            train_dp_size=1,
            weight_sync_mode="nccl",
            batch_size=32,
            micro_batch_size=4,
        )
    except ValueError:
        invalid_raised = True
    check("Invalid weight_sync_mode is rejected", invalid_raised)


# =====================================================================

# =====================================================================

def test_yaml_hetero_config_load():
    """Test yaml hetero config load."""
    section("5.1 Loading a heterogeneous YAML configuration")
    yaml_content = textwrap.dedent("""\
        model_path: "/path/to/user/test_model"
        train_gpus: 2
        rollout_gpus: 6

        heterogeneous_rollout:
          enabled: true
          total_gpus: 6
          available_gpus: [2, 3, 4, 5, 6, 7]
          vllm_host: "127.0.0.1"
          vllm_base_port: 8000
          instances:
            - instance_id: "tp1_0"
              tp: 1
              gpus: [2]
            - instance_id: "tp1_1"
              tp: 1
              gpus: [3]
            - instance_id: "tp2_0"
              tp: 2
              gpus: [4, 5]
            - instance_id: "tp2_1"
              tp: 2
              gpus: [6, 7]
          scheduling:
            scheduler_type: "la_mlfq"
            length_thresholds:
              short: 1024
              medium: 2048
              long: 4096
            routing_rules:
              short: [1]
              medium: [1, 2]
              long: [2]
              extra_long: [2]
            load_balance_strategy: "least_connections"
            max_queue_length: 64
            enable_fallback: true
            la_mlfq_migration_threshold: 4000
            la_mlfq_scout_timeout: 45.0
            la_mlfq_history_ttl: 3
    """)

    try:
        import yaml
    except ImportError:
        print("  [SKIP] yaml is not installed; skipping YAML loading test")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        with open(tmp_path) as f:
            raw = yaml.safe_load(f)


        hetero_dict = raw.pop("heterogeneous_rollout", {})
        hetero_cfg = HeterogeneousRolloutConfig.from_dict(hetero_dict)

        check("YAML: enabled=True", hetero_cfg.enabled is True)
        check("YAML: four instances", hetero_cfg.n_instances == 4)
        check("YAML: tp_list=[1,1,2,2]", hetero_cfg.tp_list == [1, 1, 2, 2])
        check("YAML: scheduler_type='la_mlfq'",
              hetero_cfg.scheduling.scheduler_type == "la_mlfq")
        check("YAML: max_queue_length=64", hetero_cfg.scheduling.max_queue_length == 64)
        check("YAML: la_mlfq_migration_threshold=4000",
              hetero_cfg.scheduling.la_mlfq_migration_threshold == 4000)
        check("YAML: la_mlfq_scout_timeout=45.0",
              hetero_cfg.scheduling.la_mlfq_scout_timeout == 45.0)
        check("YAML: la_mlfq_history_ttl=3",
              hetero_cfg.scheduling.la_mlfq_history_ttl == 3)
        check("YAML: length_thresholds.short=1024",
              hetero_cfg.scheduling.length_thresholds["short"] == 1024)
        check("YAML: routing_rules.long=[2]",
              hetero_cfg.scheduling.routing_rules["long"] == [2])

    finally:
        os.unlink(tmp_path)


def test_scheduler_type_variations():
    """Test scheduler type variations."""
    section("5.2 scheduler_type three variants")
    for stype in ["length_aware", "la_mlfq", "load_balance"]:
        d = {"scheduler_type": stype}
        cfg = SchedulingConfig.from_dict(d)
        check(f"scheduler_type='{stype}'", cfg.scheduler_type == stype)


# =====================================================================

# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("  Heterogeneous rollout and training cluster configuration unit tests")
    print("=" * 60)

    # 1. HeterogeneousInstanceConfig
    test_instance_config_defaults()
    test_instance_config_from_dict()
    test_instance_config_from_dict_extra_keys()

    # 2. SchedulingConfig
    test_scheduling_config_defaults()
    test_scheduling_config_la_mlfq_fields()
    test_scheduling_config_from_dict()

    # 3. HeterogeneousRolloutConfig
    test_hetero_config_defaults()
    test_hetero_config_from_dict()
    test_hetero_config_auto_compute()

    # 4. AsyncRLConfig
    test_async_rl_config_training_cluster()
    test_async_rl_config_with_heterogeneous()
    test_async_rl_config_multi_node()
    test_async_rl_config_megatron3d_validation()


    test_yaml_hetero_config_load()
    test_scheduler_type_variations()


    print(f"\n{'='*60}")
    total = _pass_count + _fail_count
    print(f"  Result: {_pass_count}/{total} passed, {_fail_count} failed")
    print(f"{'='*60}")

    if _fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

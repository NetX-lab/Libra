"""Support code for Config."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, fields, field
from typing import Optional


_RESERVED_KEYS = {"base_config", "hardware_config", "model_arch_config"}


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _from_dict(cls, d: dict):
    """From dict."""
    if d is None:
        return cls()
    valid_keys = {f.name for f in fields(cls)}
    filtered = {}
    for k, v in d.items():
        if k not in valid_keys:
            continue

        if isinstance(v, str):
            raw_value = v
            try:
                numeric_value = float(raw_value)
                if (
                    numeric_value.is_integer()
                    and "." not in raw_value
                    and "e" not in raw_value.lower()
                ):
                    v = int(numeric_value)
                else:
                    v = numeric_value
            except (ValueError, OverflowError):
                pass
        filtered[k] = v
    return cls(**filtered)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_raw_dict(path: str) -> dict:
    """Load raw dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported configuration file format: '{ext}', use .yaml / .yml / .json"
        )


def _resolve_config_dict(raw: dict, source_dir: str, _seen: set | None = None) -> dict:
    """Resolve config dict."""
    if _seen is None:
        _seen = set()

    merged: dict = {}


    base_path = raw.get("base_config")
    if base_path is not None:
        base_path = _resolve_path(base_path, source_dir)
        if base_path in _seen:
            raise ValueError(f"Detected a cyclic configuration reference: {base_path}")
        _seen.add(base_path)
        base_raw = _load_raw_dict(base_path)
        base_dir = os.path.dirname(os.path.abspath(base_path))
        merged = _resolve_config_dict(base_raw, base_dir, _seen)


    hw_file = raw.get("hardware_config")
    if hw_file is not None:
        hw_path = _resolve_path(hw_file, source_dir)
        hw_dict = _load_raw_dict(hw_path)

        existing_hw = merged.get("hardware", {})
        merged["hardware"] = _deep_merge(existing_hw, hw_dict)


    ma_file = raw.get("model_arch_config")
    if ma_file is not None:
        ma_path = _resolve_path(ma_file, source_dir)
        ma_dict = _load_raw_dict(ma_path)
        existing_ma = merged.get("model_arch", {})
        merged["model_arch"] = _deep_merge(existing_ma, ma_dict)


    current = {k: v for k, v in raw.items() if k not in _RESERVED_KEYS}
    merged = _deep_merge(merged, current)

    return merged


def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve path."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Hardware config implementation."""
    flops_peak: float = 312e12
    mem_bw: float = 2.0e12
    mem_capacity: float = 80e9
    bw_intra_node: float = 400e9
    bw_inter_node: float = 200e9
    gpus_per_node: int = 8
    tp_comm_overhead: float = 5e-6
    latency_inter_node: float = 5e-6

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareConfig":
        """Build an instance from a dictionary."""
        return _from_dict(cls, d)

    @classmethod
    def from_file(cls, path: str) -> "HardwareConfig":
        """Load an instance from a file."""
        d = _load_raw_dict(path)
        return cls.from_dict(d)


@dataclass
class ModelArchConfig:
    """Model arch config implementation."""
    num_params: float = 3e9
    d_model: int = 2048
    n_layers: int = 36
    n_heads: int = 16
    n_kv_heads: int = 2
    vocab_size: int = 151936
    intermediate_size: int = 11008
    dtype_bytes: int = 2
    head_dim: int = 0


    is_moe: bool = False
    active_num_params: float = 0.0
    num_experts: int = 1
    num_shared_experts: int = 0
    num_activated_experts: int = 1
    expert_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0

    def __post_init__(self):
        if self.head_dim <= 0:
            self.head_dim = self.d_model // max(self.n_heads, 1)
        if self.expert_intermediate_size <= 0:
            self.expert_intermediate_size = self.intermediate_size

    @property
    def activated_expert_ratio(self) -> float:
        """Activated expert ratio."""
        if not self.is_moe or self.num_experts <= 1:
            return 1.0
        return min(
            1.0,
            self.num_activated_experts / max(self.num_experts, 1),
        )

    @property
    def effective_num_params(self) -> float:
        """Effective num params."""
        if not self.is_moe:
            return self.num_params
        if self.active_num_params > 0:
            return self.active_num_params
        return self.num_params * self.activated_expert_ratio

    @classmethod
    def from_dict(cls, d: dict) -> "ModelArchConfig":
        """Build an instance from a dictionary."""
        return _from_dict(cls, d)

    @classmethod
    def from_file(cls, path: str) -> "ModelArchConfig":
        """Load an instance from a file."""
        d = _load_raw_dict(path)
        return cls.from_dict(d)


@dataclass
class ProfilingConfig:
    """Profiling config implementation."""

    alpha_mlp_fwd: float = 3.5e-8     # ~35ns/token
    beta_mlp_fwd: float = 1.0e-5

    alpha_mlp_bwd: float = 7.0e-8
    beta_mlp_bwd: float = 2.0e-5

    alpha_attn_fwd: float = 1.5e-12
    beta_attn_fwd: float = 2.5e-8
    gamma_attn_fwd: float = 8.0e-6

    alpha_attn_bwd: float = 3.0e-12
    beta_attn_bwd: float = 5.0e-8
    gamma_attn_bwd: float = 1.5e-5

    rho_compute_bound: float = 0.08
    rho_memory_bound: float = 0.45
    theta_slowdown: float = 1.0

    prefill_mfu: float = 0.55

    decode_bw_util: float = 0.85

    prefill_chunk_size: int = 1024

    kv_frag_rate: float = 0.04        # ~4%

    act_workspace_bytes: float = 1.0e9

    train_mem_frag_rate: float = 0.08   # 8%

    train_workspace_bytes: float = 0.5e9  # 500MB

    @classmethod
    def from_dict(cls, d: dict) -> "ProfilingConfig":
        """Build an instance from a dictionary."""
        return _from_dict(cls, d)

    @classmethod
    def from_file(cls, path: str) -> "ProfilingConfig":
        """Load an instance from a file."""
        d = _load_raw_dict(path)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class HeterogeneousInstanceConfig:
    """Heterogeneous instance config implementation."""
    instance_id: str = ""
    tp: int = 1
    gpus: list[int] = field(default_factory=list)
    description: str = ""
    host: str = ""
    port: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "HeterogeneousInstanceConfig":
        return _from_dict(cls, d)


@dataclass
class SchedulingConfig:
    """Scheduling config implementation."""

    scheduler_type: str = "length_aware"  # "length_aware" / "la_mlfq" / "cmlfq" / "load_balance"



    length_thresholds: dict = field(default_factory=lambda: {
        "short": 5000,
        "medium": 10000,
        "long": 15000,
    })

    routing_rules: dict = field(default_factory=lambda: {
        "short": [1, 2],
        "medium": [2, 4],
        "long": [4, 8],
        "extra_long": [8, 4],
    })

    load_balance_strategy: str = "least_connections"

    max_queue_length: int = 100

    enable_fallback: bool = True

    adaptive_routing: bool = True

    request_timeout: float = 600.0



    la_mlfq_buckets: dict = field(default_factory=lambda: {
        "short": {"tp_degrees": [1, 2], "max_tokens": 5000},
        "long": {"tp_degrees": [4, 8], "max_tokens": 50000},
    })

    la_mlfq_migration_threshold: int = 3000

    la_mlfq_scout_timeout: float = 30.0

    la_mlfq_history_ttl: int = 5



    cmlfq_buckets: dict = field(default_factory=lambda: {
        "short": {"tp_degrees": [1, 2], "max_tokens": 5000},
        "long": {"tp_degrees": [4, 8], "max_tokens": 50000},
    })

    cmlfq_rebuild_interval: int = 50

    cmlfq_migration_profile_path: str = ""

    cmlfq_tree_path: str = ""

    cmlfq_tree_persist_interval: int = 1

    cmlfq_shared_load_dir: str = ""

    cmlfq_shared_load_ttl_s: float = 60.0

    cmlfq_shared_load_heartbeat_s: float = 10.0

    cmlfq_payload_small_threshold: int = 500
    cmlfq_payload_large_threshold: int = 5000

    @classmethod
    def from_dict(cls, d: dict) -> "SchedulingConfig":
        return _from_dict(cls, d)


@dataclass
class HeterogeneousRolloutConfig:
    """Heterogeneous rollout config implementation."""
    enabled: bool = False

    total_gpus: int = 0
    available_gpus: list[int] = field(default_factory=list)

    instances: list[HeterogeneousInstanceConfig] = field(default_factory=list)

    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)

    vllm_base_port: int = 8000
    vllm_host: str = "127.0.0.1"
    startup_timeout: int = 300
    max_model_len: int = 0
    gpu_memory_utilization: float = 0.90
    enable_thinking: bool = False

    def __post_init__(self):

        if self.total_gpus <= 0 and self.instances:
            self.total_gpus = sum(
                inst.tp if isinstance(inst, HeterogeneousInstanceConfig)
                else inst.get("tp", 1) if isinstance(inst, dict) else 1
                for inst in self.instances
            )

        for i, inst in enumerate(self.instances):
            if isinstance(inst, HeterogeneousInstanceConfig) and not inst.instance_id:
                inst.instance_id = f"hetero_tp{inst.tp}_{i}"

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def tp_list(self) -> list[int]:
        """Tp list."""
        return [
            inst.tp if isinstance(inst, HeterogeneousInstanceConfig)
            else inst.get("tp", 1) if isinstance(inst, dict) else 1
            for inst in self.instances
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "HeterogeneousRolloutConfig":
        """Build an instance from a dictionary."""
        if d is None:
            return cls()


        raw_instances = d.pop("instances", [])
        instances = []
        for inst_d in raw_instances:
            if isinstance(inst_d, dict):

                if "gpus" in inst_d and not isinstance(inst_d["gpus"], list):
                    inst_d["gpus"] = list(inst_d["gpus"])
                instances.append(HeterogeneousInstanceConfig.from_dict(inst_d))
            elif isinstance(inst_d, HeterogeneousInstanceConfig):
                instances.append(inst_d)


        sched_dict = d.pop("scheduling", None)
        scheduling = SchedulingConfig.from_dict(sched_dict) if sched_dict else SchedulingConfig()


        gpu_dict = d.pop("gpu", None)
        if gpu_dict and isinstance(gpu_dict, dict):
            if "total_gpus" in gpu_dict and "total_gpus" not in d:
                d["total_gpus"] = gpu_dict["total_gpus"]
            if "available_gpus" in gpu_dict and "available_gpus" not in d:
                d["available_gpus"] = gpu_dict["available_gpus"]


        obj = _from_dict(cls, d)
        obj.instances = instances
        obj.scheduling = scheduling
        obj.__post_init__()
        return obj


@dataclass
class GlobalResourcePlannerConfig:
    """Global resource planner config implementation."""
    enabled: bool = False




    train_backend: str = "analytic"
    rollout_backend: str = "analytic"
    plan_interval: int = 10
    warmup_steps: int = 1
    min_history_size: int = 8
    min_gain_ratio: float = 0.05
    reconfiguration_cost_s: float = 15.0
    max_history_size: int = 4096
    allowed_rollout_tp: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    rollout_node_tp_pattern: list[int] = field(default_factory=list)
    require_heterogeneous_rollout_tp: bool = False
    allowed_train_tp: list[int] = field(default_factory=list)
    allowed_train_pp: list[int] = field(default_factory=list)
    fixed_train_gpus: int = 0
    # Startup placement is a GRP decision.  ``fixed_train_gpus`` is retained
    # only for backwards-compatible, explicitly configured deployments and is
    # ignored when this strategy is ``grp``.
    initial_allocation_strategy: str = "grp"  # grp | configured
    # Set only by the preflight planner after it has applied a concrete GRP plan.
    # A live trainer refuses to start with a fixed seed split when GRP is required.
    initial_allocation_applied: bool = False
    allocation_granularity_gpus: int = 1
    min_train_gpus: int = 1
    min_rollout_gpus: int = 1
    micro_batch_sizes: list[int] = field(default_factory=list)
    startup_profile_enabled: bool = True
    startup_profile_sample_size: int = 64
    startup_profile_strategy: str = "spread"  # spread | random | first
    startup_profile_seed: int = 0
    startup_profile_samples_per_prompt: int = 1
    startup_profile_dataset_jsonl: str = ""
    startup_profile_history_jsonl: str = ""
    startup_profile_summary_json: str = ""
    startup_profile_reuse_existing: bool = True
    startup_profile_allow_synthetic_fallback: bool = False
    apply_to_runtime: bool = True
    verbose: bool = False
    runtime_length_profile_enabled: bool = True
    runtime_length_profile_jsonl: str = ""
    runtime_async_planning: bool = True
    runtime_max_pending_plans: int = 1
    runtime_dynamic_reconfiguration_enabled: bool = True
    runtime_online_replanning: bool = True
    runtime_replan_cooldown_steps: int = 1
    runtime_queue_pressure_threshold: float = 0.75
    runtime_active_rollout_pressure_threshold: float = 0.85
    runtime_rejected_rollout_delta_threshold: int = 8
    runtime_rollout_train_imbalance_threshold: float = 1.25


    # {input_json}, {output_json}, {trace_csv}, {output_dir}, {sailor_path}, {vidur_path}
    simulator_allow_fallback: bool = True
    simulator_timeout_s: float = 120.0
    simulator_cache_dir: str = "./logs/global_resource_planner/simulator_cache"
    sailor_path: str = "../sailor"
    sailor_python: str = "python"
    sailor_train_command: str = ""
    vidur_path: str = "../vidur"
    vidur_python: str = "python"
    vidur_rollout_command: str = ""
    vidur_model_name: str = "meta-llama/Meta-Llama-3-8B"
    vidur_device: str = "a100"
    vidur_qps: float = 1000000.0
    vidur_scheduler: str = "sarathi"
    vidur_batch_size_cap: int = 512
    vidur_chunk_size: int = 512
    vidur_time_limit_s: float = 0.0



    runtime_manage_rollout_processes: bool = False
    runtime_adopt_existing_rollout_processes: bool = True
    runtime_drain_before_reconfigure: bool = True
    runtime_drain_timeout_s: float = 3600.0
    runtime_rollout_reconfigure_strategy: str = "diff"  # diff | restart_all | blue_green | prewarm | cluster_swap
    runtime_cluster_swap_enabled: bool = False
    runtime_prewarm_no_spare_fallback_strategy: str = "restart_all"
    runtime_rollout_log_dir: str = ""
    runtime_rollout_manifest_path: str = ""
    runtime_rollout_pid_registry_path: str = ""
    runtime_wait_rollout_process_start_ready: bool = False
    runtime_post_rollout_stop_grace_s: float = 0.0
    runtime_reconfigure_training: bool = False
    runtime_training_pool_only: bool = True
    runtime_training_pool_target_gpus: int = 0
    runtime_training_pool_plan_only: bool = True
    runtime_batch_collection_timeout_s: float = 0.0
    runtime_batch_collection_max_retries: int = 0
    runtime_use_nccl_barrier_after_rollout: bool = False
    runtime_use_nccl_barrier_before_weight_sync: bool = False
    runtime_coordinate_reconfiguration_ranks: bool = True
    runtime_coordinate_batch_source_only: bool = True
    runtime_peer_request_wait_s: float = 45.0
    vllm_launch_command_template: str = ""
    vllm_stop_command_template: str = ""
    vllm_ready_timeout_s: float = 300.0
    vllm_stop_timeout_s: float = 30.0
    hybrid_worker_launch_enabled: bool = False
    hybrid_training_prewarm_enabled: bool = False
    hybrid_training_prewarm_count: int = 0
    hybrid_training_prewarm_worker_ids: list[str] = field(default_factory=list)
    hybrid_worker_python: str = "python"
    hybrid_worker_command_template: str = ""
    hybrid_worker_task_dir: str = "./logs/elastic_training_tasks"
    hybrid_worker_ready_timeout_s: float = 60.0
    hybrid_worker_remote_control_enabled: bool = False
    hybrid_worker_remote_control_dir: str = ""
    elastic_hybrid_planning_enabled: bool = False
    elastic_hybrid_require_planner_signal: bool = False
    # Deprecated and ignored. EHP capacity is always derived from the currently
    # available rollout GPUs and complete-replica width.
    elastic_hybrid_max_workers: int = 0
    elastic_hybrid_replica_size_gpus: int = 0
    elastic_hybrid_min_rollout_gpus: int = 0
    elastic_hybrid_borrow_train_rollout_ratio: float = 1.15
    elastic_hybrid_release_train_rollout_ratio: float = 0.90
    elastic_hybrid_max_rollout_pressure: float = 0.80
    elastic_hybrid_join_timeout_s: float = 180.0
    elastic_hybrid_signal_ttl_steps: int = 20
    elastic_hybrid_require_isolated_ccl: bool = False
    gradient_transport_backend: str = "tcp"  # tcp | native_rdma
    decouple_communication_domains: bool = True
    gradient_server_host: str = "127.0.0.1"
    gradient_server_public_host: str = ""
    gradient_server_port: int = 0
    gradient_server_authkey: str = "rl-framework-elastic"
    native_rdma_device: str = "mlx5_0"
    native_rdma_gid_index: int = 0
    native_rdma_ib_port: int = 1
    native_rdma_max_bytes: int = 67108864
    hybrid_worker_mode: str = "megatron_core"
    hybrid_worker_config_path: str = ""
    hybrid_worker_endpoint_dir: str = ""
    hybrid_lockstep_gradient_sync: bool = True
    hybrid_active_gradient_timeout_s: float = 300.0
    hybrid_update_timeout_s: float = 300.0

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalResourcePlannerConfig":
        return _from_dict(cls, d)


@dataclass
class AsyncRLConfig:
    """Async r l config implementation."""

    model_path: str
    tokenizer_path: str = ""

    rollout_backend: str = "vllm"  # vllm | mock


    train_gpus: int = 4
    rollout_gpus: int = 4
    rollout_gpu_ids: str = ""
    tp_size: int = 1
    vllm_tp_size: int = 1
    train_backend: str = "megatron_core"
    train_tp_size: int = 0
    train_pp_size: int = 1
    train_dp_size: int = 0
    train_cp_size: int = 1
    train_ep_size: int = 1
    expert_tensor_parallel_size: int = 1
    pipeline_schedule: str = "1f1b"
    virtual_pipeline_model_parallel_size: int = 0
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    megatron_grad_reduce_in_fp32: bool = False
    megatron_use_precision_aware_optimizer: bool = False
    megatron_optimizer_cpu_offload: bool = False
    megatron_optimizer_offload_fraction: float = 0.0
    megatron_optimizer_pin_cpu_grads: bool = True
    megatron_optimizer_pin_cpu_params: bool = True
    megatron_checkpoint_format: str = "torch_dist"
    megatron_fully_parallel_save: bool = True
    megatron_async_save: bool = False
    megatron_streaming_export: bool = True
    megatron_use_transformer_engine: bool = False
    megatron_use_cpu_initialization: bool = False
    megatron_recompute_num_layers: int = 1


    max_concurrent_rollouts: int = 64
    max_head_offpolicyness: int = 4
    queue_size: int = 256
    enable_rollout_tracing: bool = False
    sync_interval: int = 1


    recompute_logprobs: bool = True


    weight_sync_mode: str = "disk"
    sync_path: str = "./logs/async_rl_weights"
    rollout_weight_sync_mode: str = "none"  # none | restart | nccl
    rollout_weight_sync_control_dir: str = ""
    rollout_weight_sync_timeout_s: float = 1200.0
    rollout_weight_sync_export_only: bool = True
    rollout_weight_reload_method: str = "restart"  # restart | inplace
    rollout_weight_reload_strategy: str = "parallel"  # parallel | serial
    rollout_nccl_host: str = ""
    rollout_nccl_port: int = 29620
    rollout_nccl_chunk_mb: int = 256
    rollout_nccl_rate_limit_gbps: float = 0.0
    rollout_weight_sync_poll_interval_s: float = 0.05
    require_rollout_weight_sync: bool = False


    learning_rate: float = 1e-6
    ppo_epochs: int = 1
    batch_size: int = 32
    micro_batch_size: int = 4
    kl_coef: float = 0.001
    clip_epsilon: float = 0.2
    gamma: float = 1.0


    max_new_tokens: int = 1024
    # Workflow-specific limits.  Keeping these in the experiment config makes
    # multi-turn capability runs reproducible instead of depending on shell
    # environment variables.
    r2e_max_turns: int = 3
    r2e_max_prompt_tokens: int = 0
    r2e_stop_reward: float = 0.5
    n_samples: int = 4
    temperature: float = 1.0
    top_p: float = 1.0


    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    max_seq_length: int = 2048


    total_steps: int = 100
    eval_interval: int = 10
    save_interval: int = 10
    seed: int = 42


    ray_address: str = "auto"
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_num_instances: int = 0
    vllm_endpoints: str = ""


    num_nodes: int = 1
    gpus_per_node_override: int = 0
    train_gpus_per_node: int = 0
    rollout_gpus_per_node: int = 0
    master_addr: str = ""
    master_port: int = 29500


    log_dir: str = "./logs"
    wandb_project: str = "RL_Framework"
    wandb_run_name: str = "run1"


    keep_latest_checkpoints: int = 5


    enable_history_collection: bool = False
    history_output_dir: str = ""
    history_save_raw_lengths: bool = False
    history_flush_interval: int = 10
    history_experiment_name: str = ""


    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model_arch: ModelArchConfig = field(default_factory=ModelArchConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    n_total_gpus: int = 0


    heterogeneous_rollout: HeterogeneousRolloutConfig = field(
        default_factory=HeterogeneousRolloutConfig
    )
    global_resource_planner: GlobalResourcePlannerConfig = field(
        default_factory=GlobalResourcePlannerConfig
    )

    def __post_init__(self):
        """Post init."""
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_path

        if not 0.0 <= self.r2e_stop_reward <= 1.0:
            raise ValueError("r2e_stop_reward must be between 0 and 1")

        if self.train_tp_size <= 0:
            self.train_tp_size = max(1, self.tp_size)
        else:
            self.tp_size = self.train_tp_size

        if self.train_pp_size <= 0:
            raise ValueError("train_pp_size must be greater than zero")
        if self.train_dp_size <= 0:
            denom = self.train_tp_size * self.train_pp_size
            if self.train_backend == "megatron_core":
                denom *= self.train_cp_size
            if self.train_gpus % denom != 0:
                raise ValueError(
                    "train_gpus must be divisible by the configured model-parallel size: "
                    f"train_gpus={self.train_gpus}, model_parallel={denom}"
                )
            self.train_dp_size = self.train_gpus // denom
        if self.train_backend not in ["fsdp", "megatron3d", "megatron_core"]:
            raise ValueError(
                "train_backend must be 'fsdp', 'megatron3d', or 'megatron_core'"
            )


        if self.train_gpus <= 0:
            raise ValueError("train_gpus must be greater than zero")


        if self.weight_sync_mode not in ["disk", "nccl"]:
            raise ValueError("weight_sync_mode must be 'disk' or 'nccl'")
        if self.rollout_weight_sync_mode not in ["none", "restart", "nccl"]:
            raise ValueError(
                "rollout_weight_sync_mode must be 'none', 'restart', or 'nccl'"
            )
        if self.rollout_weight_reload_method not in ["restart", "inplace"]:
            raise ValueError(
                "rollout_weight_reload_method must be 'restart' or 'inplace'"
            )
        if self.rollout_weight_reload_strategy not in ["parallel", "serial"]:
            raise ValueError(
                "rollout_weight_reload_strategy must be 'parallel' or 'serial'"
            )
        if self.rollout_weight_sync_mode == "nccl":
            if self.train_backend != "megatron_core":
                raise ValueError(
                    "NCCL rollout sync requires train_backend='megatron_core'"
                )
            if self.weight_sync_mode != "nccl":
                raise ValueError(
                    "NCCL rollout sync requires weight_sync_mode='nccl'"
                )
            if self.rollout_weight_reload_strategy != "parallel":
                raise ValueError("NCCL rollout reload requires parallel strategy")
            if self.rollout_nccl_chunk_mb <= 0:
                raise ValueError("rollout_nccl_chunk_mb must be greater than zero")
            if self.rollout_nccl_rate_limit_gbps < 0:
                raise ValueError("rollout_nccl_rate_limit_gbps cannot be negative")
            if self.rollout_weight_sync_poll_interval_s <= 0:
                raise ValueError(
                    "rollout_weight_sync_poll_interval_s must be greater than zero"
                )
        if self.pipeline_schedule != "1f1b":
            raise ValueError("pipeline_schedule currently supports only '1f1b'")
        if self.virtual_pipeline_model_parallel_size != 0:
            raise ValueError(
                "virtual_pipeline_model_parallel_size currently supports only 0"
            )
        topology_size = self.train_tp_size * self.train_pp_size * self.train_dp_size
        if self.train_backend == "megatron_core":
            topology_size *= self.train_cp_size
        if topology_size != self.train_gpus:
            raise ValueError(
                "The training topology must match train_gpus: "
                f"topology_size={topology_size}, train_gpus={self.train_gpus}"
            )
        if self.batch_size % (self.train_dp_size * self.micro_batch_size) != 0:
            raise ValueError(
                "batch_size must be divisible by train_dp_size * micro_batch_size: "
                f"{self.batch_size} vs {self.train_dp_size}*{self.micro_batch_size}"
            )
        if self.train_backend == "megatron3d" and self.weight_sync_mode != "disk":
            raise ValueError(
                "The legacy Megatron backend supports only weight_sync_mode='disk'"
            )
        if self.train_backend == "megatron_core":
            if self.weight_sync_mode not in {"disk", "nccl"}:
                raise ValueError(
                    "Megatron-Core weight_sync_mode must be 'disk' or 'nccl'"
                )
            if self.weight_sync_mode == "nccl" and self.rollout_weight_sync_mode != "nccl":
                raise ValueError(
                    "weight_sync_mode='nccl' requires rollout_weight_sync_mode='nccl'"
                )
            if not 0.0 <= self.megatron_optimizer_offload_fraction <= 1.0:
                raise ValueError(
                    "megatron_optimizer_offload_fraction must be between 0 and 1"
                )
            if (
                self.megatron_optimizer_offload_fraction > 0.0
                and not self.megatron_optimizer_cpu_offload
            ):
                raise ValueError(
                    "megatron_optimizer_offload_fraction requires "
                    "megatron_optimizer_cpu_offload=true"
                )
            if self.train_pp_size != 1:
                raise ValueError(
                    "MegatronCoreTrainEngine currently requires train_pp_size=1"
                )
            if self.train_ep_size <= 0:
                raise ValueError("train_ep_size must be greater than zero")
            if self.expert_tensor_parallel_size <= 0:
                raise ValueError(
                    "expert_tensor_parallel_size must be greater than zero"
                )
            if self.train_tp_size % self.expert_tensor_parallel_size != 0:
                raise ValueError(
                    "expert_tensor_parallel_size must divide train_tp_size: "
                    f"{self.expert_tensor_parallel_size} vs {self.train_tp_size}"
                )
            if self.train_dp_size % self.train_ep_size != 0:
                raise ValueError(
                    "train_ep_size must divide train_dp_size: "
                    f"{self.train_ep_size} vs {self.train_dp_size}"
                )
            if self.model_arch.num_experts > 1:
                if self.model_arch.num_experts % self.train_ep_size != 0:
                    raise ValueError(
                        "num_experts must be divisible by train_ep_size: "
                        f"{self.model_arch.num_experts} vs {self.train_ep_size}"
                    )
                if (
                    self.model_arch.expert_intermediate_size
                    % self.expert_tensor_parallel_size
                    != 0
                ):
                    raise ValueError(
                        "expert_intermediate_size must be divisible by "
                        "expert_tensor_parallel_size"
                    )
            if (
                not self.megatron_use_transformer_engine
                and self.train_cp_size != 1
            ):
                raise ValueError(
                    "PyTorch SDPA attention requires train_cp_size=1"
                )
            if self.megatron_checkpoint_format != "torch_dist":
                raise ValueError(
                    "Megatron-Core integration currently requires "
                    "megatron_checkpoint_format='torch_dist'"
                )


        import os
        for runtime_dir in (self.sync_path, self.log_dir):
            try:
                os.makedirs(runtime_dir, exist_ok=True)
            except PermissionError:
                # Production configs may target shared paths that are mounted
                # only on compute nodes. Parsing and validation should remain
                # side-effect tolerant; runtime components create their own
                # directories when the target filesystem is available.
                pass


        if self.n_total_gpus <= 0:
            self.n_total_gpus = self.train_gpus + self.rollout_gpus


        if self.num_nodes > 1:
            if self.train_gpus_per_node <= 0:
                self.train_gpus_per_node = self.train_gpus // self.num_nodes
            if self.rollout_gpus_per_node <= 0:
                self.rollout_gpus_per_node = self.rollout_gpus // self.num_nodes


        if not self.master_addr:
            slurm_nodelist = os.environ.get("SLURM_NODELIST", "")
            if slurm_nodelist:
                import subprocess
                try:
                    result = subprocess.run(
                        ["scontrol", "show", "hostnames", slurm_nodelist],
                        capture_output=True, text=True, check=True
                    )
                    nodes = result.stdout.strip().split("\n")
                    self.master_addr = nodes[0] if nodes else "localhost"
                except Exception:
                    self.master_addr = "localhost"
            else:
                self.master_addr = "localhost"

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "AsyncRLConfig":
        """Build an instance from a dictionary."""

        hw_dict = d.pop("hardware", None)
        ma_dict = d.pop("model_arch", None)
        prof_dict = d.pop("profiling", None)
        hetero_dict = d.pop("heterogeneous_rollout", None)
        planner_dict = d.pop("global_resource_planner", None)


        _nested_keys = {
            "hardware",
            "model_arch",
            "profiling",
            "heterogeneous_rollout",
            "global_resource_planner",
        }
        valid_keys = {f.name for f in fields(cls) if f.name not in _nested_keys}
        filtered = {k: v for k, v in d.items() if k in valid_keys}


        filtered["hardware"] = HardwareConfig.from_dict(hw_dict) if hw_dict else HardwareConfig()
        filtered["model_arch"] = ModelArchConfig.from_dict(ma_dict) if ma_dict else ModelArchConfig()
        filtered["profiling"] = ProfilingConfig.from_dict(prof_dict) if prof_dict else ProfilingConfig()
        filtered["heterogeneous_rollout"] = (
            HeterogeneousRolloutConfig.from_dict(hetero_dict) if hetero_dict
            else HeterogeneousRolloutConfig()
        )
        filtered["global_resource_planner"] = (
            GlobalResourcePlannerConfig.from_dict(planner_dict) if planner_dict
            else GlobalResourcePlannerConfig()
        )

        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "AsyncRLConfig":
        """From yaml."""
        raw = _load_raw_dict(path)
        source_dir = os.path.dirname(os.path.abspath(path))
        resolved = _resolve_config_dict(raw, source_dir)
        return cls.from_dict(resolved)

    @classmethod
    def from_json(cls, path: str) -> "AsyncRLConfig":
        """From json."""
        raw = _load_raw_dict(path)
        source_dir = os.path.dirname(os.path.abspath(path))
        resolved = _resolve_config_dict(raw, source_dir)
        return cls.from_dict(resolved)

    @classmethod
    def from_file(cls, path: str) -> "AsyncRLConfig":
        """Load an instance from a file."""
        raw = _load_raw_dict(path)
        source_dir = os.path.dirname(os.path.abspath(path))
        resolved = _resolve_config_dict(raw, source_dir)
        return cls.from_dict(resolved)

    def to_dict(self) -> dict:
        """Serialize the object to a dictionary."""
        import dataclasses
        result = dataclasses.asdict(self)
        return result

    def to_yaml(self, path: str):
        """To yaml."""
        import yaml
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_json(self, path: str):
        """To json."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(config_path: str | None = None) -> AsyncRLConfig:
    """Load config."""

    path = config_path or os.environ.get("ASYNC_RL_CONFIG", None)

    if path:
        return AsyncRLConfig.from_file(path)


    raise ValueError(
        "No configuration file was specified. Provide one using one of the following methods:\n"
        "  1. function argument: load_config('configs/gsm8k.yaml')\n"
        "  2. command line: python train.py --config configs/gsm8k.yaml\n"
        "  3. environment variable: export ASYNC_RL_CONFIG=configs/gsm8k.yaml"
    )


def parse_args_and_load_config() -> AsyncRLConfig:
    """Parse args and load config."""
    import argparse
    parser = argparse.ArgumentParser(description="Asynchronous RL training")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file path (.yaml/.yml/.json). Defaults to ASYNC_RL_CONFIG.",
    )
    args, _ = parser.parse_known_args()
    return load_config(args.config)




@dataclass
class ResourceConfig:
    """Resource config implementation."""
    max_concurrent_requests: int = 32
    batch_size: int = 32  # rollout batch size
    gpu_allocation: int = 4
    target_rollout_time: float = 30.0


@dataclass
class GenerationConfig:
    """Generation config implementation."""
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = True
    stop_token_ids: list[int] = field(default_factory=list)

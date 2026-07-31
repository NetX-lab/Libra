# Configuration Reference

Libra loads YAML or JSON into the dataclasses in `config.py`. Unknown keys now
raise `ValueError`; they are not silently ignored. Paths below are exact YAML
paths.

## Loading and inheritance

Select a configuration in one of three ways:

```bash
python examples/search_r1_async_rl.py --config configs/search_r1.yaml
export ASYNC_RL_CONFIG=configs/search_r1.yaml
python -c "from RL_Framework import load_config; print(load_config('configs/search_r1.yaml'))"
```

Top-level composition keys are handled before dataclass validation:

| Key | Meaning |
| --- | --- |
| `base_config` | YAML/JSON file recursively merged as the base |
| `hardware_config` | File merged into the `hardware` section |
| `model_arch_config` | File merged into the `model_arch` section |

Relative paths are resolved relative to the file that declares them. Merging is
recursive, and the current file wins over its base. Cyclic `base_config`
references raise an error.

Minimal structure:

```yaml
base_config: base.yaml
hardware_config: hardware_config/a800_80g.yaml
model_arch_config: model_arch_config/qwen3-14b.yaml

model_path: Qwen/Qwen3-14B
train_backend: megatron_core
train_gpus: 8
rollout_gpus: 8

heterogeneous_rollout:
  enabled: true
  max_model_len: 32768
  instances:
    - instance_id: rollout_tp4
      tp: 4
      gpus: [0, 1, 2, 3]
  scheduling:
    scheduler_type: cmlfq

global_resource_planner:
  enabled: true
  runtime_dynamic_reconfiguration_enabled: true
```

## Top-level training and rollout options

| Option | Default | Meaning |
| --- | ---: | --- |
| `model_path` | required | Policy model or local checkpoint |
| `tokenizer_path` | `""` | Defaults to `model_path` |
| `train_backend` | `megatron_core` | `fsdp`, `megatron3d`, or `megatron_core` |
| `train_gpus` | `4` | GPUs in the fixed training topology |
| `rollout_gpus` | `4` | GPUs assigned to rollout at launch |
| `rollout_gpu_ids` | `""` | Launcher-facing rollout GPU list |
| `tp_size` | `1` | Legacy training TP alias |
| `vllm_tp_size` | `1` | Homogeneous vLLM TP size |
| `train_tp_size` | `0` | Training TP; `0` inherits `tp_size` |
| `train_pp_size` | `1` | Training pipeline parallel size |
| `train_dp_size` | `0` | Training DP; `0` derives it from topology |
| `train_cp_size` | `1` | Megatron-Core context parallel size |
| `train_ep_size` | `1` | Megatron-Core expert parallel size |
| `expert_tensor_parallel_size` | `1` | Expert tensor parallel size |
| `pipeline_schedule` | `1f1b` | Currently only `1f1b` is accepted |
| `virtual_pipeline_model_parallel_size` | `0` | Currently only `0` is accepted |
| `sequence_parallel` | `true` | Enable Megatron sequence parallelism |
| `use_distributed_optimizer` | `true` | Use the distributed optimizer |
| `max_concurrent_rollouts` | `64` | Maximum in-flight rollout tasks |
| `max_head_offpolicyness` | `4` | Maximum accepted policy-version lag |
| `queue_size` | `256` | Dispatcher queue capacity |
| `enable_rollout_tracing` | `false` | Record per-rollout traces |
| `sync_interval` | `1` | Training steps between sync attempts |
| `recompute_logprobs` | `true` | Recompute log probabilities before updates |

`heterogeneous_rollout.max_model_len`, not a top-level `max_model_len`, controls
the vLLM context length for heterogeneous rollout workers.

## Optimization, generation, data, and runtime

| Group | Exact top-level fields |
| --- | --- |
| Weight synchronization | `weight_sync_mode`, `sync_path`, `rollout_weight_sync_mode`, `rollout_weight_sync_control_dir`, `rollout_weight_sync_timeout_s`, `rollout_weight_sync_export_only`, `require_rollout_weight_sync` |
| RL optimization | `learning_rate`, `ppo_epochs`, `batch_size`, `micro_batch_size`, `kl_coef`, `clip_epsilon`, `gamma` |
| Generation | `max_new_tokens`, `n_samples`, `temperature`, `top_p` |
| Dataset | `dataset_name`, `dataset_split`, `max_seq_length` |
| Run control | `total_steps`, `eval_interval`, `save_interval`, `seed` |
| Homogeneous vLLM/Ray | `ray_address`, `vllm_host`, `vllm_port`, `vllm_num_instances`, `vllm_endpoints` |
| Distributed launch | `num_nodes`, `gpus_per_node_override`, `train_gpus_per_node`, `rollout_gpus_per_node`, `master_addr`, `master_port`, `n_total_gpus` |
| Logging/checkpoints | `log_dir`, `wandb_project`, `wandb_run_name`, `keep_latest_checkpoints` |
| Runtime history | `enable_history_collection`, `history_output_dir`, `history_save_raw_lengths`, `history_flush_interval`, `history_experiment_name` |

Defaults for these fields are centralized in `AsyncRLConfig` and the shared
`configs/base.yaml`.

## Megatron-Core options

All fields in this table are top-level:

| Option | Default | Notes |
| --- | ---: | --- |
| `megatron_grad_reduce_in_fp32` | `false` | Reduce gradients in FP32 |
| `megatron_use_precision_aware_optimizer` | `false` | Enable MCore precision-aware optimizer |
| `megatron_optimizer_cpu_offload` | `false` | Enable optimizer CPU offload |
| `megatron_optimizer_offload_fraction` | `0.0` | Must be in `[0,1]`; positive values require CPU offload |
| `megatron_optimizer_pin_cpu_grads` | `true` | Pin offloaded CPU gradients |
| `megatron_optimizer_pin_cpu_params` | `true` | Pin offloaded CPU parameters |
| `megatron_checkpoint_format` | `torch_dist` | MCore integration currently requires `torch_dist` |
| `megatron_fully_parallel_save` | `true` | Parallel distributed-checkpoint save |
| `megatron_async_save` | `false` | Asynchronous checkpoint save |
| `megatron_streaming_export` | `true` | Stream Hugging Face weight export |
| `megatron_use_transformer_engine` | `false` | Use Transformer Engine layers |
| `megatron_use_cpu_initialization` | `false` | Initialize model weights through CPU |
| `megatron_recompute_num_layers` | `1` | Layers per recompute block |

grouped-GEMM is an installed extension, not a configuration field. There is no
`megatron_grouped_gemm` YAML option; install and validate the extension as
described in [the Megatron-Core guide](megatron_core_backend.md).

## Heterogeneous rollout

All rollout fields live under `heterogeneous_rollout`:

| Field | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `false` | Enable heterogeneous vLLM instances |
| `total_gpus` | `0` | Derived from instances when zero |
| `available_gpus` | `[]` | Optional allowed device list |
| `instances` | `[]` | Rollout instance definitions |
| `vllm_base_port` | `8000` | Starting port for managed instances |
| `vllm_host` | `127.0.0.1` | Bind/connect host |
| `startup_timeout` | `300` | Startup timeout in seconds |
| `max_model_len` | `0` | vLLM context length; `0` delegates to model/vLLM |
| `gpu_memory_utilization` | `0.90` | vLLM memory-utilization target |
| `enable_thinking` | `false` | Enable model-specific thinking mode |

Each `heterogeneous_rollout.instances[]` accepts `instance_id`, `tp`, `gpus`,
`description`, `host`, and `port`.

Scheduling fields live under `heterogeneous_rollout.scheduling`:

| Group | Fields |
| --- | --- |
| Common | `scheduler_type`, `length_thresholds`, `routing_rules`, `load_balance_strategy`, `max_queue_length`, `enable_fallback`, `adaptive_routing`, `request_timeout` |
| LA-MLFQ | `la_mlfq_buckets`, `la_mlfq_migration_threshold`, `la_mlfq_scout_timeout`, `la_mlfq_history_ttl` |
| C-MLFQ | `cmlfq_buckets`, `cmlfq_rebuild_interval`, `cmlfq_migration_profile_path`, `cmlfq_tree_path`, `cmlfq_tree_persist_interval`, `cmlfq_shared_load_dir`, `cmlfq_shared_load_ttl_s`, `cmlfq_shared_load_heartbeat_s`, `cmlfq_payload_small_threshold`, `cmlfq_payload_large_threshold` |

`scheduler_type` accepts `length_aware`, `la_mlfq`, `cmlfq`, or
`load_balance`.

## Global Resource Planner

Every field below must be nested under `global_resource_planner`.

| Group | Fields |
| --- | --- |
| Planner core | `enabled`, `train_backend`, `rollout_backend`, `plan_interval`, `warmup_steps`, `min_history_size`, `min_gain_ratio`, `reconfiguration_cost_s`, `max_history_size`, `allowed_rollout_tp`, `require_heterogeneous_rollout_tp`, `allowed_train_tp`, `allowed_train_pp`, `fixed_train_gpus`, `micro_batch_sizes`, `apply_to_runtime`, `verbose` |
| Online triggers | `runtime_async_planning`, `runtime_max_pending_plans`, `runtime_dynamic_reconfiguration_enabled`, `runtime_online_replanning`, `runtime_replan_cooldown_steps`, `runtime_queue_pressure_threshold`, `runtime_active_rollout_pressure_threshold`, `runtime_rejected_rollout_delta_threshold`, `runtime_rollout_train_imbalance_threshold` |
| Simulators | `simulator_allow_fallback`, `simulator_timeout_s`, `simulator_cache_dir`, `sailor_path`, `sailor_python`, `sailor_train_command`, `vidur_path`, `vidur_python`, `vidur_rollout_command`, `vidur_model_name`, `vidur_device`, `vidur_qps`, `vidur_scheduler`, `vidur_batch_size_cap`, `vidur_chunk_size`, `vidur_time_limit_s` |
| Rollout lifecycle | `runtime_manage_rollout_processes`, `runtime_adopt_existing_rollout_processes`, `runtime_drain_before_reconfigure`, `runtime_drain_timeout_s`, `runtime_rollout_reconfigure_strategy`, `runtime_cluster_swap_enabled`, `runtime_prewarm_no_spare_fallback_strategy`, `runtime_rollout_log_dir`, `runtime_rollout_manifest_path`, `runtime_rollout_pid_registry_path`, `runtime_wait_rollout_process_start_ready`, `runtime_post_rollout_stop_grace_s` |
| Training/runtime coordination | `runtime_reconfigure_training`, `runtime_training_pool_only`, `runtime_training_pool_target_gpus`, `runtime_training_pool_plan_only`, `runtime_batch_collection_timeout_s`, `runtime_batch_collection_max_retries`, `runtime_use_nccl_barrier_after_rollout`, `runtime_use_nccl_barrier_before_weight_sync`, `runtime_coordinate_reconfiguration_ranks`, `runtime_coordinate_batch_source_only`, `runtime_peer_request_wait_s` |
| Managed vLLM commands | `vllm_launch_command_template`, `vllm_stop_command_template`, `vllm_ready_timeout_s`, `vllm_stop_timeout_s` |
| Hybrid workers | `hybrid_worker_launch_enabled`, `hybrid_training_prewarm_enabled`, `hybrid_training_prewarm_count`, `hybrid_training_prewarm_worker_ids`, `hybrid_worker_python`, `hybrid_worker_command_template`, `hybrid_worker_task_dir`, `hybrid_worker_ready_timeout_s` |
| Gradient transport | `gradient_transport_backend`, `decouple_communication_domains`, `gradient_server_host`, `gradient_server_public_host`, `gradient_server_port`, `gradient_server_authkey`, `native_rdma_device`, `native_rdma_gid_index`, `native_rdma_ib_port`, `native_rdma_max_bytes` |

`runtime_rollout_reconfigure_strategy` accepts `diff`, `restart_all`,
`blue_green`, `prewarm`, or `cluster_swap`. `gradient_transport_backend`
accepts `tcp` or `native_rdma`.

## Cost-model inputs

The `hardware`, `model_arch`, and `profiling` sections feed the analytic planner.
They can be written inline; hardware and model architecture can also come from
`hardware_config` and `model_arch_config`.

| Section | Complete field list |
| --- | --- |
| `hardware` | `flops_peak`, `mem_bw`, `mem_capacity`, `bw_intra_node`, `bw_inter_node`, `gpus_per_node`, `tp_comm_overhead`, `latency_inter_node` |
| `model_arch` | `num_params`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`, `intermediate_size`, `dtype_bytes`, `head_dim`, `is_moe`, `active_num_params`, `num_experts`, `num_shared_experts`, `num_activated_experts`, `expert_intermediate_size`, `shared_expert_intermediate_size` |
| `profiling` | `alpha_mlp_fwd`, `beta_mlp_fwd`, `alpha_mlp_bwd`, `beta_mlp_bwd`, `alpha_attn_fwd`, `beta_attn_fwd`, `gamma_attn_fwd`, `alpha_attn_bwd`, `beta_attn_bwd`, `gamma_attn_bwd`, `rho_compute_bound`, `rho_memory_bound`, `theta_slowdown`, `prefill_mfu`, `decode_bw_util`, `prefill_chunk_size`, `kv_frag_rate`, `act_workspace_bytes`, `train_mem_frag_rate`, `train_workspace_bytes` |

Use the supplied files under `configs/hardware_config` and
`configs/model_arch_config` as unit and schema examples.

## Validated constraints

Configuration construction enforces the following relationships:

- `train_gpus > 0`.
- `train_backend` is `fsdp`, `megatron3d`, or `megatron_core`.
- The derived TP/PP/DP topology equals `train_gpus`; Megatron-Core also includes
  CP in this calculation.
- `batch_size` is divisible by `train_dp_size * micro_batch_size`.
- Megatron backends require `weight_sync_mode: disk`.
- Megatron-Core currently requires `train_pp_size: 1`,
  `megatron_checkpoint_format: torch_dist`, and CP=1 when Transformer Engine is
  disabled.
- `expert_tensor_parallel_size` divides `train_tp_size`.
- `train_ep_size` divides `train_dp_size`; MoE expert counts and expert hidden
  size must also divide their configured parallel dimensions.
- Positive `megatron_optimizer_offload_fraction` requires
  `megatron_optimizer_cpu_offload: true`.

For recommended combinations and launcher environment-variable mappings, see
the [cluster manual](manual.md).

# Configuration Reference

Libra loads its primary configuration from YAML files under `configs/` into the
dataclasses defined in `config.py`. The tables below cover the options most
commonly changed for experiments and cluster deployments.

## Core Options

| Option | Meaning |
| --- | --- |
| `model_path` | Policy model path |
| `tokenizer_path` | Tokenizer path; defaults to `model_path` when empty |
| `train_backend` | Training backend; defaults to `megatron_core` |
| `train_gpus` | GPUs assigned to the core training pool |
| `rollout_gpus` | GPUs assigned to rollout at launch |
| `train_tp_size` | Tensor-parallel size for training |
| `train_pp_size` | Pipeline-parallel size for training |
| `train_cp_size` | Context-parallel size for Megatron-Core |
| `batch_size` | Global training batch size |
| `n_samples` | Rollouts per prompt group for GRPO |
| `max_model_len` | vLLM context length |
| `max_new_tokens` | Maximum generated tokens per rollout request |
| `max_concurrent_rollouts` | Maximum in-flight rollout tasks |
| `max_head_offpolicyness` | Maximum accepted policy-version lag |
| `recompute_logprobs` | Recompute log probabilities with the current model |
| `recompute_micro_batch_size` | Temporary batch size used only by recompute-logprobs; defaults to 1 to bound logits memory |
| `global_resource_planner.memory_budget_check_enabled` | Include recompute-logprobs temporary memory in GRP candidate budgets |
| `global_resource_planner.memory_budget_dry_run_enabled` | Run one complete runtime recompute probe before training |
| `global_resource_planner.memory_budget_safety_margin_bytes` | Reserve device memory outside the analytic estimate |
| `global_resource_planner.memory_budget_logits_dtype_bytes` | Dtype width used for the logits temporary estimate |
| `global_resource_planner.memory_budget_workspace_factor` | Multiplier for simultaneous logits/workspace temporaries |
| `sync_interval` | Training steps between weight sync attempts |
| `rollout_weight_sync_mode` | `none`, checkpoint-based `restart`, or official Ascend `hccl` refresh |
| `rollout_weight_reload_method` | For checkpoint refreshes, `restart` replaces vLLM or `inplace` keeps the resident server |
| `rollout_weight_reload_strategy` | `parallel` reloads workers concurrently; `serial` uses a per-node lock |
| `rollout_weight_sync_poll_interval_s` | Poll interval for rollout reload ACKs |
| `rollout_hccl_host` | Trainer address used by the stateless HCCL rendezvous |
| `rollout_hccl_port` | Trainer rendezvous port for the persistent HCCL communicator |
| `rollout_hccl_packed_buffer_mb` | Minimum packed transfer-buffer size; enlarged for the largest tensor |
| `rollout_hccl_num_buffers` | Number of official packed-transfer buffers |

## Megatron-Core Options

| Option | Meaning |
| --- | --- |
| `megatron_use_precision_aware_optimizer` | Keep optimizer-state precision aligned with mixed precision |
| `megatron_optimizer_cpu_offload` | Offload optimizer state to CPU |
| `megatron_optimizer_offload_fraction` | Fraction of optimizer state to offload |
| `megatron_use_cpu_initialization` | Build model weights through CPU initialization |
| `megatron_grouped_gemm` | Enable grouped GEMM when available |

## Runtime Planner and Elastic Options

| Option | Meaning |
| --- | --- |
| `global_resource_planner.runtime_dynamic_reconfiguration_enabled` | Master switch for runtime reconfiguration |
| `runtime_online_replanning` | Use online metrics in planner decisions |
| `runtime_manage_rollout_processes` | Let Libra start, stop, and adopt rollout processes |
| `runtime_rollout_reconfigure_strategy` | `diff`, `restart_all`, `blue_green`, `prewarm`, or `cluster_swap` |
| `runtime_cluster_swap_enabled` | Enable rollout/training pool exchange without spare GPUs |
| `runtime_reconfigure_training` | Enable training-side pool changes |
| `runtime_training_pool_plan_only` | Record training-pool changes without attaching workers |
| `decouple_communication_domains` | Keep elastic gradient traffic off the core training DP process group |
| `runtime_batch_collection_timeout_s` | Timeout for collecting a training batch |
| `runtime_batch_collection_max_retries` | Retries after a batch collection timeout |
| `runtime_drain_before_reconfigure` | Drain in-flight rollout work before a runtime change |

See the [cluster manual](manual.md) for recommended combinations and launch
examples.

## NPU Rollout Weight Refresh

The NPU path supports both disk-checkpoint reloads and a resident in-place
reload. Each refresh publishes one request for the complete rollout instance
set; workers write ACKs atomically with version, checkpoint, method, strategy,
and timing metadata. Set `rollout_weight_reload_method: inplace` only when the
Ascend vLLM server exposes `/reload_weights`; otherwise use the default
`restart`. `parallel` is the default for independent workers. Set
`rollout_weight_reload_strategy: serial` when several workers share a node and
concurrent model loading is undesirable.

With `weight_sync_mode: hccl` and `rollout_weight_sync_mode: hccl`, Libra uses
vLLM Ascend's official weight-transfer engine. The trainer is rank zero and all
vLLM TP workers receive contiguous ranks across rollout instances. The
communicator is initialized once and reused; a runtime rollout-topology change
therefore requires restarting the HCCL-enabled rollout group.

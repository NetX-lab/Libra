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

When `phase_trace_enabled` is true, each rank writes append-only
`phase_trace_rank*.jsonl` spans. Set `RL_TRAIN_PHASE_TRACE=1` and optionally
`RL_TRAIN_PHASE_TRACE_DIR` for long runs. Convert the files for Perfetto or
Chrome Trace with `scripts/phase_trace_to_chrome.py`; the stable schema and
phase names allow the same converter to compare baseline runs.
| `sync_interval` | Training steps between weight sync attempts |
| `rollout_weight_reload_method` | Use `restart` for process replacement or `inplace` to refresh resident vLLM workers |
| `rollout_weight_reload_strategy` | Reload rollout instances concurrently with `parallel` (default), or use the node-serialized `serial` fallback |

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
| `initial_allocation_strategy` | `grp` runs planning before launch; `configured` preserves an explicitly pinned legacy split |
| `allocation_granularity_gpus` | Initial train/rollout split granularity (normally one node or one DP replica) |
| `min_train_gpus` / `min_rollout_gpus` | Minimum viable capacity retained for each stage during startup planning |
| `runtime_online_replanning` | Use online metrics in planner decisions |
| `runtime_manage_rollout_processes` | Let Libra start, stop, and adopt rollout processes |
| `runtime_rollout_reconfigure_strategy` | `diff`, `restart_all`, `blue_green`, `prewarm`, or `cluster_swap` |
| `runtime_cluster_swap_enabled` | Enable rollout/training pool exchange without spare GPUs |
| `runtime_reconfigure_training` | Enable training-side pool changes |
| `runtime_training_pool_plan_only` | Record training-pool changes without attaching workers |
| `decouple_communication_domains` | Keep elastic gradient traffic off the core training DP process group |
| `elastic_hybrid_replica_size_gpus` | Physical ranks in one complete TP×PP×CP DP replica; zero derives it from training topology |
| `elastic_hybrid_min_rollout_gpus` | Rollout capacity that EHP may never borrow |
| `elastic_hybrid_max_workers` | Deprecated and ignored; EHP has no policy maximum |
| `runtime_batch_collection_timeout_s` | Timeout for collecting a training batch |
| `runtime_batch_collection_max_retries` | Retries after a batch collection timeout |
| `runtime_drain_before_reconfigure` | Drain in-flight rollout work before a runtime change |

See the [cluster manual](manual.md) for recommended combinations and launch
examples.

## Rollout Weight Reloads

When `rollout_weight_sync_mode` is `restart`, the trainer publishes one reload
request for the complete rollout instance set and waits for the whole ACK batch.
`rollout_weight_reload_method` controls how each vLLM instance applies the
checkpoint: `restart` replaces the server process, while `inplace` calls the
guarded reload endpoint and keeps the server resident. The default
`rollout_weight_reload_strategy: parallel` applies the selected method to all
instances concurrently. Use `serial` only as an operational fallback on nodes
that cannot tolerate concurrent model loads.

Each ACK records method, strategy, lock wait, process stop, model load, and total
reload time. The trainer validates every ACK and reports the slowest instance
for each refresh.

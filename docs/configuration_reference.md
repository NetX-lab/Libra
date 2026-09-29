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

## Rollout Scheduling Options

These options live under `heterogeneous_rollout.scheduling` and control how
requests are routed across the heterogeneous TP buckets.

### Load metric

| Option | Meaning |
| --- | --- |
| `load_metric` | Selection signal for least-connections routing: `requests` (legacy request-count, default), `tokens` (estimated in-flight token load = prompt + EMA-expected output), or `kv_tokens` (that load normalized by each TP bucket's KV-cache capacity, i.e. occupancy ratio) |
| `kv_capacity_tokens_by_tp` | Explicit KV capacity per TP degree, e.g. `{1: 120000, 4: 450000}`. When empty and `load_metric: kv_tokens`, capacities are auto-estimated from the hardware / model-arch configs and the rollout `gpu_memory_utilization`; at engine startup they are additionally calibrated against vLLM's own profiled `kv_cache_size_tokens` from `/metrics` |
| `kv_activation_reserve_gib` | Per-GPU reserve subtracted when auto-estimating KV capacity (default 4) |
| `load_balance_strategy` | `least_connections` (default), `round_robin`, or `weighted` |

Under `kv_tokens`, instances whose TP degree has no configured capacity are
excluded from selection (with a warning) rather than comparing raw token
counts against ratios. All strategies rotate among equal-load endpoints to
avoid starving later-registered instances.

The expected-output component of the estimate is a per-category EMA learned
from observed completion lengths; it survives weight-sync rebinding, and is
clamped by each request's `max_new_tokens` (callers that pass exact
`input_tokens` — all bundled workflows do — skip the chars-based fallback).

### Closed-loop /metrics feedback

| Option | Meaning |
| --- | --- |
| `enable_metrics_feedback` | Master switch (default false). An engine-layer poller scrapes each instance's `/metrics`; observed gauges drive an additive bias correction of the local load estimate plus occupancy admission |
| `metrics_poll_interval_s` | Scrape interval per instance (default 3; measured interference is <1% at 5Hz) |
| `metrics_request_timeout_s` | Per-scrape HTTP timeout (default 1) |
| `metrics_staleness_ttl_s` | Snapshots older than this are ignored (default 10) |
| `metrics_admission_enter` / `metrics_admission_exit` | Occupancy hysteresis band: block new routes above `enter` (default 0.90), unblock below `exit` (default 0.75) |
| `metrics_preemption_penalty_ttl_s` | Preemption counters are cumulative; a detected increase penalizes the instance for this window (default 60) instead of blacklisting forever |
| `metrics_bias_alpha` | EMA weight for the additive bias `bias = EMA(observed - local)` (default 0.3) |

The poller runs at the engine layer (it survives scheduler rebinding during
weight sync), staggers endpoints with random phases (multi-rank safety), and
any feed outage degrades transparently back to the open-loop estimate.

### Cross-rank load aggregation

| Option | Meaning |
| --- | --- |
| `shared_load_dir` | Shared directory where every training rank publishes its per-instance in-flight accounting (atomic renames + heartbeat TTL). Schedulers then rank instances by the summed cluster load instead of their own partial view, eliminating multi-rank herd. Empty disables |
| `shared_load_ttl_s` / `shared_load_heartbeat_s` | Liveness window / refresh rate for rank files (defaults 30 / 10) |
| `shared_load_cache_ttl_s` | Aggregate-read cache window (default 1); a rank's own writes invalidate immediately |

### Prefix affinity

| Option | Meaning |
| --- | --- |
| `prefix_affinity` | Bind `prompt_id` -> instance (bounded LRU) so repeated prompts (GRPO `n_samples`, multi-turn replays, shared system prompts) land on the instance whose prefix cache already holds them (default false) |
| `prefix_affinity_max_entries` | LRU bound (default 8192) |

Affinity only reorders preference among candidates that survived the
readiness / queue / capacity / admission filters — a sticky instance at its
queue cap or blocked by occupancy admission is bypassed and the mapping
remaps. The affinity table survives weight-sync rebinding.

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

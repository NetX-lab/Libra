# Tier 1 — 0-GPU Scheduling Replay & GRP Offline Benchmark

Tier 1 runs completely on CPU (**0 GPUs required**) in a couple of seconds using Libra's real production schedulers (`SchedulerFactory`), `CausalPrefixTree`, `RolloutCostModel`, and `GlobalResourcePlanner`.

## Quick Start

```bash
python repro/tier1_zero_gpu_replay/run_replay.py
```

Export structured JSON metrics:

```bash
python repro/tier1_zero_gpu_replay/run_replay.py --output reports/tier1_replay.json
```

## Experimental Setup

1. **Cluster & Cost Model**:
   - Heterogeneous rollout cluster: `TP=[1, 1, 2, 4]` (`8 GPUs` across `4` instances: two `TP=1`, one `TP=2`, one `TP=4`).
   - Hardware & Model profile: `configs/hardware_config/a800_80g.yaml` + `configs/model_arch_config/qwen3-14b.yaml`.
   - Water-filling continuous-batching `RolloutCostModel`: integrates decode steps over surviving concurrent requests (`max_batch_per_tp=4`) so finished short requests leave KV cache immediately while active stragglers govern tail latency and KV occupancy.
2. **Workload Trace ([r2e_agentic_trace.jsonl](sample_traces/r2e_agentic_trace.jsonl))**:
   - `64` multi-turn agentic trajectories (`32` prompts × `2` rollouts): leading `24` warmup trajectories (`r2e_task_00..11`) and `40` unseen-prompt eval trajectories (`r2e_task_12..31`).
   - Overlapping prompt lengths (`980–2291` tokens) across 5 execution patterns across `short` (`<=4000`), `medium` (`4000–10000`), and `long` (`>10000`) output buckets:
     - `short_clean` (15 eval): 1 tool turn (`test/bash:small:success`, `~1.2k–2.0k` tokens).
     - `short_recover` (3 eval): Turn-1 `search:medium:failure` at `~900` tokens that recovers quickly (`~2.8k` total tokens).
     - `medium` (12 eval): 2 tool turns (`edit:medium:failure`, Turn-1 `~950`, Turn-2 `~4.1k`, total `~8.6k–9.4k` tokens).
     - `long_unambig` (8 eval): 3 tool turns (`test/bash:large:failure`, Turn-1 `~950`, Turn-2 `~4.3k`, Turn-3 `~8.0k`, total `~19.2k–21.5k` tokens).
     - `long_cascading` (2 eval): **bimodal** Turn-1 `search:medium:failure` at `~900` tokens (shared with `short_recover`, giving `mean ~ 9.1k [medium]` vs `p90 ~ 18.6k [long]`), followed by cascading failures at Turn 2 (`~4.5k`) and Turn 3 (`~8.4k`, total `~19.1k–21.1k` tokens).
3. **Why the 6 Policies Separate Clearly**:
   - `round_robin`: blind static rotation traps 60% of `long` tails on `TP=1/2`.
   - `least_connections`: under bounded in-flight `concurrency=12`, frees slots slightly faster on `TP=4`, but cannot predict or migrate already-trapped tails.
   - `length_aware`: classifies by initial prompt length (`1550/1950/2400`), missing short prompts that grow into multi-turn long tails (`20%` max-TP tail routing).
   - `la_mlfq`: waits until `>3000` generated tokens before reactive promotion (Turn 2 `~4.1k–4.7k`), pays higher recompute, and collapses `medium` + `long` onto `TP=4` (`TP=2` stays idle).
   - `cmlfq`: causal prefix tree promotes Turn-1 unambiguous patterns (`edit:medium:failure -> TP=2`, `test/bash:large:failure -> TP=4`) after just `~900` tokens, giving clean `18/12/10` separation across `TP=[1,2,4]`, except holding bimodal `search:medium:failure` until Turn 2 due to `mean_p90_disagree`.
   - `cmlfq_cost`: evaluates the full `residual_distribution` against [r2e_cost_profile.json](sample_traces/r2e_cost_profile.json) and migration cost, migrating bimodal `long_cascading` eagerly at Turn 1 for the lowest makespan.

## Common CLI Options

```bash
# Compare KV migration movement backends (recompute / nixl / cpu_offload)
python repro/tier1_zero_gpu_replay/run_replay.py --policies cmlfq cmlfq_cost --kv-backend nixl

# Purely online mode without offline warmup tree building
python repro/tier1_zero_gpu_replay/run_replay.py --online-only --batch-size 16

# Replay a measured rollout history log produced by HistoryDataCollector
python repro/tier1_zero_gpu_replay/run_replay.py --trace logs/history_run1.jsonl --tp-layout 1,1,2,4
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--trace` | `sample_traces/r2e_agentic_trace.jsonl` | Native Tier 1 JSONL or `HistoryDataCollector` `history_*.jsonl` |
| `--policies` | all 6 policies | Subset of `round_robin`, `least_connections`, `length_aware`, `la_mlfq`, `cmlfq`, `cmlfq_cost` |
| `--tp-layout` | `1,1,2,4` | Comma-separated TP degrees for heterogeneous rollout instances |
| `--concurrency` | `12` | Bounded in-flight concurrency for static scheduling slot reuse |
| `--hardware` | `configs/hardware_config/a800_80g.yaml` | Hardware cost-model YAML |
| `--model-arch` | `configs/model_arch_config/qwen3-14b.yaml` | Model architecture cost-model YAML |
| `--total-gpus` | `32` | Cluster size for the offline `GlobalResourcePlanner` benchmark |
| `--kv-backend` | `recompute` | Migration KV movement model (`recompute`, `nixl`, `cpu_offload`) |
| `--cost-profile` | `sample_traces/r2e_cost_profile.json` | Measured `decode_ms` / `migrations` JSON profile for `cmlfq_cost` |
| `--online-only` | `false` | Disable warmup split and replay purely online |

## Metric Definitions

- **`evaluation_mode: "analytic_cost_model_replay"`**: Rollout step makespans, migration recompute/transfer costs, and GRP plan `T_train` / `T_rollout` / `T_global` are **analytic cost-model estimates** (`RolloutCostModel` & `HybridSimulatorCostModel`), not physical GPU wall-clock measurements.
- **`Ctrl Avg/P99 (us)` & `GRP Solve Time`**: Directly measured CPU wall-clock execution time of Libra's scheduling and planning controllers (`time.perf_counter_ns()`).

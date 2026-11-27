# Tier 1 — Scheduling Replay (0-GPU Analytic & Real-GPU Live vLLM) & GRP Offline Benchmark

Tier 1 supports two evaluation pathways using Libra's real production schedulers (`SchedulerFactory`), `CausalPrefixTree`, and `GlobalResourcePlanner`:

1. **Pathway 1 (`--eval-mode analytic`, default 0-GPU)**: Runs completely on CPU (**0 GPUs required**) in a couple of seconds using `RolloutCostModel`. Because `cmlfq_cost` and the analytic evaluator both rely on `RolloutCostModel` assumptions, Pathway 1 is intended for fast 0-GPU control-plane verification rather than final physical-GPU claims.
2. **Pathway 2 (`--eval-mode vllm`, Real-GPU Live vLLM Judge)**: Eliminates the simulator circular dependency by dispatching the exact same trace turns and scheduler decisions to live heterogeneous vLLM servers on physical GPUs and measuring real wall-clock makespan and token throughput (`tok/s`).

## Quick Start

### Pathway 1: 0-GPU CPU Analytic Replay (Default)

```bash
python repro/tier1_zero_gpu_replay/run_replay.py
```

Export structured JSON metrics:

```bash
python repro/tier1_zero_gpu_replay/run_replay.py --output reports/tier1_replay.json
```

### Pathway 2: Real-GPU Live vLLM Judge (`--eval-mode vllm`)

Auto-launch local heterogeneous vLLM servers (`TP=[1, 1, 2, 4]` across 8 GPUs) and evaluate all 6 policies against real physical GPUs:

```bash
python repro/tier1_zero_gpu_replay/run_replay.py \
  --eval-mode vllm \
  --launch-vllm \
  --vllm-gpus 0,1,2,3,4,5,6,7 \
  --vllm-model Qwen/Qwen3-14B \
  --vllm-tokenizer Qwen/Qwen3-14B \
  --tp-layout 1,1,2,4 \
  --hardware configs/hardware_config/h20_3e_141g.yaml \
  --skip-grp \
  --output reports/tier1_vllm_replay.json
```

Or connect to already-running heterogeneous vLLM servers via `--vllm-endpoints`:

```bash
python repro/tier1_zero_gpu_replay/run_replay.py \
  --eval-mode vllm \
  --tp-layout 1,1,2,4 \
  --vllm-endpoints 127.0.0.1:19001,127.0.0.1:19002,127.0.0.1:19003,127.0.0.1:19004 \
  --vllm-model Qwen/Qwen3-14B \
  --vllm-tokenizer Qwen/Qwen3-14B \
  --skip-grp
```

## Experimental Setup

1. **Cluster & Cost Model**:
   - Heterogeneous rollout cluster: `TP=[1, 1, 2, 4]` (`8 GPUs` across `4` instances: two `TP=1`, one `TP=2`, one `TP=4`).
   - Hardware & Model profile: `configs/hardware_config/a800_80g.yaml` + `configs/model_arch_config/qwen3-14b.yaml`.
   - Water-filling continuous-batching `RolloutCostModel`: integrates decode steps over surviving concurrent requests (`max_batch_per_tp=4`) so finished short requests leave KV cache immediately while active stragglers govern tail latency and KV occupancy.
2. **Workload Trace ([r2e_agentic_trace.jsonl](sample_traces/r2e_agentic_trace.jsonl))**:
   - `64` multi-turn agentic trajectories (`32` prompts × `2` rollouts): leading `24` warmup trajectories (`r2e_task_00..11`) and `40` unseen-prompt eval trajectories (`r2e_task_12..31`).
   - Overlapping prompt lengths (`1050–2230` tokens; peak context `<= 23,350 <= 32,768`) across 5 execution patterns spanning `short` (`<=4000`), `medium` (`4000–10000`), and `long` (`>10000`) output buckets:
     - `short_clean` (33 eval): 1 tool turn (`test:small:success`, `~1.05k–1.29k` output tokens, prompt tokens `1600–2230`).
     - `short_recover` (2 eval): Turn-1 `search:medium:failure` at `200` tokens that recovers quickly (`~1.75k–1.91k` output tokens).
     - `medium` (3 eval): 2 tool turns (`edit:medium:failure`, Turn-1 `200`, Turn-2 `4060`, `~8.3k–8.7k` output tokens).
     - `long_unambig` (1 eval): 3 tool turns (`test:large:failure`, Turn-1 `200`, Turn-2 `1500`, Turn-3 `4120`, `~14.4k–14.7k` output tokens).
     - `long_cascading` (1 eval): **bimodal** Turn-1 `search:medium:failure` at `200` tokens (shared with `short_recover`, giving `mean ~ 6.7k [medium]` vs `p90 ~ 14.3k [long]`), followed by cascading failures at Turn 2 (`1500`, `bash:large:failure`) and Turn 3 (`4120`, `test:medium:failure`, `14.5k` output tokens).
3. **Why the 6 Policies Separate Clearly**:
   - `round_robin`: blind static rotation traps both `long` tails on `TP=1` (`0.0%` max-TP tail routing).
   - `least_connections`: under bounded in-flight `concurrency=12`, frees slots slightly faster on `TP=4`, but traps `long` tails routed early onto `TP=1` (`0.0%` max-TP tail routing).
   - `length_aware`: classifies by initial prompt length (`1550/1950/2400`), catching the long-prompt `long_unambig` (`50.0%` max-TP tail routing) while trapping short-prompt `long_cascading` on `TP=1` and misrouting long-prompt `short_clean` requests onto `TP=2/4`.
   - `la_mlfq`: waits until `>3000` generated tokens before reactive promotion (Turn 2 `4060` / Turn 3 `4120`), pays higher recompute, and collapses `medium` + `long` onto `TP=4` (`35/0/5`, `TP=2` stays idle).
   - `cmlfq`: causal prefix tree promotes Turn-1 unambiguous patterns (`edit:medium:failure -> TP=2`, `test:large:failure -> TP=4`) after just `200` tokens, giving clean `35/3/2` separation across `TP=[1,2,4]`, except holding bimodal `search:medium:failure` until Turn 2 (`1500` tokens) due to `mean_p90_disagree`.
   - `cmlfq_cost`: evaluates the full `residual_distribution` against [r2e_cost_profile.json](sample_traces/r2e_cost_profile.json) and migration cost, migrating bimodal `search:medium:failure` eagerly at Turn 1 (`200` tokens) for the lowest makespan (`33/3/4`).

## Verified Results

### Pathway 1: 0-GPU Analytic Cost-Model Replay (`a800_80g.yaml` + `qwen3-14b.yaml`, `TP=[1, 1, 2, 4]`)

| Policy | KV Backend | Makespan (s, Analytic) | Speedup vs RR | Long-Tail Max-TP Rate | Final TP Dist (1/2/4) | Migrations | Mig Cost (s) | Ctrl Avg/P99 (us) |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `round_robin` | `recompute` | 297.906 | 1.00x | 0.0% | `20/10/10` | 0 | 0.000 | 3.4 / 11.4 |
| `least_connections` | `recompute` | 261.096 | 1.14x | 0.0% | `6/14/20` | 0 | 0.000 | 5.5 / 9.0 |
| `length_aware` | `recompute` | 256.550 | 1.16x | 50.0% | `4/20/16` | 0 | 0.000 | 4.8 / 11.9 |
| `la_mlfq` | `recompute` | 162.009 | 1.84x | 100.0% | `35/0/5` | 5 | 1.065 | 5.7 / 21.1 |
| `cmlfq` | `recompute` | 120.348 | 2.48x | 100.0% | `35/3/2` | 5 | 0.540 | 6.5 / 40.8 |
| `cmlfq_cost` | `recompute` | 95.430 | **3.12x** | 100.0% | `33/3/4` | 7 | 0.650 | 24.1 / 94.0 |

- **GRP Offline Benchmark (32 GPUs, 64 trajectories)**: Fixed `16 GPUs (TP=4, DP=4)` + `[8, 8]` (`50.486s`) → GRP Optimal `4 GPUs (TP=4, DP=1)` + `[8, 8, 8, 4]` (`45.984s`, **1.10x**, `-8.9%`).

### Pathway 2: Real-GPU Live vLLM Judge (`8 × NVIDIA H20-3e 141GB`, `TP=[1, 1, 2, 4]`, `Qwen3-14B`, `vLLM 0.9.2`)

Measured end-to-end across 40 multi-turn eval trajectories (`95,890` generated tokens per policy) against live heterogeneous vLLM servers with Automatic Prefix Caching (`--enable-prefix-caching`) and session-isolated prompts (zero `RolloutCostModel` simulator in evaluation):

| Policy | KV Backend | Makespan (s, Real-GPU Wall) | Speedup vs RR | Throughput (tok/s) | Long-Tail Max-TP Rate | Final TP Dist (1/2/4) | Migrations | Mig Cost (s) | Ctrl Avg/P99 (us) |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `round_robin` | `recompute` | 203.266 | 1.00x | 471.8 | 0.0% | `20/10/10` | 0 | 0.000 | 40.9 / 67.4 |
| `least_connections` | `recompute` | 188.620 | 1.08x | 508.4 | 0.0% | `10/15/15` | 0 | 0.000 | 52.0 / 69.9 |
| `length_aware` | `recompute` | 176.131 | 1.15x | 544.4 | 50.0% | `4/20/16` | 0 | 0.000 | 49.0 / 78.4 |
| `la_mlfq` | `recompute` | 138.577 | 1.47x | 692.0 | 100.0% | `35/0/5` | 5 | 228.442 | 29.6 / 71.9 |
| **`cmlfq`** | `recompute` | **110.835** | **1.83x** (+83.4%) | **865.2** | 100.0% | `35/3/2` | 5 | 114.337 | 34.7 / 77.3 |
| **`cmlfq_cost`** | `recompute` | **98.272** | **2.07x** (+106.8%) | **975.8** | 100.0% | `33/3/4` | 7 | 129.690 | 91.3 / 273.9 |

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
| `--eval-mode` | `analytic` | Evaluation pathway: `analytic` (0-GPU CostModel) or `vllm` (live vLLM GPUs) |
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
| `--vllm-endpoints` | `None` | Comma-separated `host:port` list for running vLLM servers matching `--tp-layout` |
| `--vllm-host` | `127.0.0.1` | Host for vLLM servers when `--vllm-endpoints` is omitted |
| `--vllm-base-port` | `8000` | Base port for vLLM servers when `--vllm-endpoints` is omitted |
| `--vllm-model` | `Qwen/Qwen3-14B` | Model name/path sent to vLLM `/v1/completions` |
| `--vllm-tokenizer` | `None` | Optional HF tokenizer path/name for exact token-ID prompt padding |
| `--vllm-concurrency` | `0` | Max concurrent requests in `vllm` mode (`0` = `--concurrency` for static, full batch for dynamic) |
| `--launch-vllm` | `false` | Auto-launch local heterogeneous `vllm.entrypoints.openai.api_server` instances |
| `--vllm-gpus` | `None` | Comma-separated GPU IDs for `--launch-vllm` (default: `0..sum(tp_layout)-1`) |

## Metric Definitions

- **`evaluation_mode: "analytic_cost_model_replay"`**: Rollout step makespans, migration recompute/transfer costs, and GRP plan `T_train` / `T_rollout` / `T_global` are **analytic cost-model estimates** (`RolloutCostModel` & `HybridSimulatorCostModel`), not physical GPU wall-clock measurements.
- **`evaluation_mode: "live_vllm_gpu_replay"`**: Rollout batch makespans (`wall_makespan_s`), token throughput (`throughput_tok_per_s`), and migrated turn timings are **measured wall-clock timings from real vLLM servers on physical GPUs**, with zero `RolloutCostModel` evaluation in scoring.
- **`Ctrl Avg/P99 (us)` & `GRP Solve Time`**: Directly measured CPU wall-clock execution time of Libra's scheduling and planning controllers (`time.perf_counter_ns()`).

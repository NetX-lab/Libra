# Libra Reproduction Suite

This directory organizes Libra into **three progressive reproduction tiers** so researchers and engineers can evaluate, adopt, and extend Libra as a clean baseline without hidden cluster dependencies.

| Tier | Goal | Hardware Requirement | Runtime | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Pathway 1)** | 0-GPU Analytic Scheduling Replay & GRP Offline Benchmark (`round_robin`, `least_connections`, `length_aware`, `la_mlfq`, `cmlfq`, `cmlfq_cost`, plus `GlobalResourcePlanner`) | **0 GPUs** (CPU only) | ~1–3 seconds | Ready ([tier1_zero_gpu_replay/](tier1_zero_gpu_replay/)) |
| **Tier 1 (Pathway 2)** | Real-GPU Live vLLM Judge Replay (eliminates `RolloutCostModel` simulator by dispatching trace turns directly to live heterogeneous `vLLM` servers) | **8 GPUs** (`TP=[1, 1, 2, 4]`) | ~15–20 minutes | Ready ([tier1_zero_gpu_replay/](tier1_zero_gpu_replay/)) |
| **Tier 2** | 4-GPU Real Small-Scale Closed Loop (`DAPO-Math` + `Qwen3-4B`, 1 Train + 3 Heterogeneous Rollout GPUs, plus Google Colab launcher) | **4 GPUs** (or single-node Colab / local server) | ~15–30 minutes | Upcoming (Step 4) |
| **Tier 3** | Paper-Scale Reproduction (`32 GPUs`: 4×8 H20 cluster, `R2E-Gym` + `Qwen3-14B`, `Search-R1` / `DAPO-Math` presets, SSH/`torchrun` + minimal Slurm template) | **32 GPUs** (4×8 H20 / H100 / A800) | Paper scale | Upcoming (Step 5) |

---

## Quick Start (Tier 1)

### Pathway 1: 0-GPU CPU Analytic Replay & GRP Benchmark

Runs completely on CPU (`--eval-mode analytic`, default):

```bash
python repro/tier1_zero_gpu_replay/run_replay.py
```

Export structured JSON metrics:

```bash
python repro/tier1_zero_gpu_replay/run_replay.py --output reports/tier1_replay.json
```

### Pathway 2: Real-GPU Live vLLM Judge (`--eval-mode vllm`)

Auto-launch local heterogeneous `vLLM` servers across 8 GPUs (`TP=[1, 1, 2, 4]`) and measure real physical-GPU wall-clock makespan and token throughput:

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

Or connect to already-running heterogeneous `vLLM` servers via `--vllm-endpoints`:

```bash
python repro/tier1_zero_gpu_replay/run_replay.py \
  --eval-mode vllm \
  --tp-layout 1,1,2,4 \
  --vllm-endpoints 127.0.0.1:19001,127.0.0.1:19002,127.0.0.1:19003,127.0.0.1:19004 \
  --vllm-model Qwen/Qwen3-14B \
  --vllm-tokenizer Qwen/Qwen3-14B \
  --skip-grp \
  --output reports/tier1_vllm_replay.json
```

See [tier1_zero_gpu_replay/README.md](tier1_zero_gpu_replay/README.md) for workload pattern breakdown, KV migration backend ablations (`recompute` / `nixl` / `cpu_offload`), `--online-only`, and full CLI flags.

---

## Experimental Setup & Verified Results

Both experiments evaluate Libra's production schedulers (`SchedulerFactory`, `CausalPrefixTree`, `CMLFQOfflineProfiler`) over the same 64-trajectory multi-turn agentic trace ([r2e_agentic_trace.jsonl](tier1_zero_gpu_replay/sample_traces/r2e_agentic_trace.jsonl)): **24 warmup** trajectories (`r2e_task_00..11`) and **40 unseen evaluation** trajectories (`r2e_task_12..31`, prompt tokens `1050–2230`, peak context `<= 23,350 <= 32,768`, `95,890` total eval generated tokens across `33 short_clean`, `2 short_recover`, `3 medium`, `1 long_unambig`, and `1` bimodal `long_cascading` rollouts).

### Experiment 1: Pathway 1 — 0-GPU CPU Analytic Cost-Model Replay (`--eval-mode analytic`)

- **Experimental Setup**:
  - **Heterogeneous Rollout Cluster**: `TP=[1, 1, 2, 4]` (`8 GPUs` across 4 instances: two `TP=1`, one `TP=2`, one `TP=4`).
  - **Cost Model & Specs**: Water-filling continuous-batching `RolloutCostModel` (`max_batch_per_tp=4`), configured with [a800_80g.yaml](../configs/hardware_config/a800_80g.yaml) and [qwen3-14b.yaml](../configs/model_arch_config/qwen3-14b.yaml).
  - **Scheduler & Workload Settings**: Static bounded `concurrency=12`; `length_aware` prompt cutoffs `(1550/1950/2400)`; MLFQ output buckets `short<=4000 / medium<=10000 / long<=32768`; KV migration backend `recompute`; `cmlfq_cost` cost profile [r2e_cost_profile.json](tier1_zero_gpu_replay/sample_traces/r2e_cost_profile.json). Zero GPU compute used.

- **Rollout Scheduling Policy Comparison**:

| Policy | KV Backend | Makespan (s, Analytic) | Speedup vs RR | Long-Tail Max-TP Rate | Final TP Dist (1/2/4) | Migrations | Mig Cost (s) |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: |
| `round_robin` | `recompute` | 297.906 | 1.00x | 0.0% | `20/10/10` | 0 | 0.000 |
| `least_connections` | `recompute` | 261.096 | 1.14x | 0.0% | `6/14/20` | 0 | 0.000 |
| `length_aware` | `recompute` | 256.550 | 1.16x | 50.0% | `4/20/16` | 0 | 0.000 |
| `la_mlfq` | `recompute` | 162.009 | 1.84x | 100.0% | `35/0/5` | 5 | 1.065 |
| **`cmlfq`** | `recompute` | **120.348** | **2.48x** | 100.0% | `35/3/2` | 5 | 0.540 |
| **`cmlfq_cost`** | `recompute` | **95.430** | **3.12x** | 100.0% | `33/3/4` | 7 | 0.650 |

- **Global Resource Planner (GRP) Offline Benchmark (`32 GPUs`, `64 trajectories`)**:

| Plan | Train Allocation | Rollout TP Layout (GPUs) | T_train (s) | T_rollout (s) | T_global (s) | Expected Gain |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Fixed Baseline | 16 GPUs (TP=4, DP=4) | `[8, 8]` (16 GPUs) | 2.726 | 50.486 | 50.486 | — |
| GRP Optimal (`reconfigure`) | 4 GPUs (TP=4, DP=1) | `[8, 8, 8, 4]` (28 GPUs) | 4.695 | 45.984 | 45.984 | **1.10x** (-8.9%) |

### Experiment 2: Pathway 2 — Real-GPU Live vLLM Judge (`--eval-mode vllm`)

- **Experimental Setup**:
  - **Cluster Hardware**: Single-node `8 × NVIDIA H20-3e (141GB HBM3e)` (`qs-417146-2445617-ai-2084480-master-0`).
  - **Live Serving Pool**: 4 independent `vLLM 0.9.2` OpenAI API servers across `TP=[1, 1, 2, 4]`, serving `Qwen/Qwen3-14B` (`bfloat16`, `--enable-prefix-caching`, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, `--no-enable-chunked-prefill`).
  - **Judgement Protocol**: Each rollout uses session-isolated prompt tokens (`ignore_eos=True`, exact per-turn `max_tokens`); `recompute` migrations re-send accumulated context to the target TP engine. End-to-end batch makespan (`wall_makespan_s`), token throughput (`95,890` eval generated tokens / `wall_makespan_s`), migration wall time, and controller Avg/P99 latency are measured directly via wall-clock timing with **zero `RolloutCostModel` simulator in evaluation**.

- **Verified Real-GPU Results**:

| Policy | KV Backend | Makespan (s, Real-GPU Wall) | Speedup vs RR | Throughput (tok/s) | Long-Tail Max-TP Rate | Final TP Dist (1/2/4) | Migrations | Mig Wall (s) | Ctrl Avg/P99 (us) |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `round_robin` | `recompute` | 203.266 | 1.00x | 471.8 | 0.0% | `20/10/10` | 0 | 0.000 | 40.9 / 67.4 |
| `least_connections` | `recompute` | 188.620 | 1.08x | 508.4 | 0.0% | `10/15/15` | 0 | 0.000 | 52.0 / 69.9 |
| `length_aware` | `recompute` | 176.131 | 1.15x | 544.4 | 50.0% | `4/20/16` | 0 | 0.000 | 49.0 / 78.4 |
| `la_mlfq` | `recompute` | 138.577 | 1.47x | 692.0 | 100.0% | `35/0/5` | 5 | 228.442 | 29.6 / 71.9 |
| **`cmlfq`** | `recompute` | **110.835** | **1.83x** (+83.4%) | **865.2** | 100.0% | `35/3/2` | 5 | 114.337 | 34.7 / 77.3 |
| **`cmlfq_cost`** | `recompute` | **98.272** | **2.07x** (+106.8%) | **975.8** | 100.0% | `33/3/4` | 7 | 129.690 | 91.3 / 273.9 |

---

## Verification & CI Check

```bash
pytest -q tests/test_tier1_replay.py
bash scripts/test_cpu_ci.sh
```

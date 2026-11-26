# Tier 2 — Small-Scale Heterogeneous Benchmarks, Closed-Loop RL & Google Colab Quickstart

Tier 2 provides **Slurm-free** GPU entry points covering:

1. **Mode A — Controlled Multi-Turn Heterogeneous Rollout Scheduling Benchmark (`4×TP=1 + 1×TP=4`)**: End-to-end 6-policy comparison (`round_robin`, `least_connections`, `length_aware`, `la_mlfq`, `cmlfq`, `cmlfq_cost`) against live `vLLM` servers with CUDA graphs, Automatic Prefix Caching (`--enable-prefix-caching`), and online H20-3e cold calibration.
2. **Mode B — 4-GPU Heterogeneous Closed Loop (`1 Train + 3 Rollout`)**: Slurm-free `AsyncRLTrainer` (`FSDPTrainEngine` + `DAPOMathWorkflow` + supervised `vllm_hot_reload_api_server.py` with confirmed in-place weight reloads) across `TP=1` + `TP=2`.
3. **Mode C — Google Colab 1-GPU Co-Located Startup**: Single-GPU closed-loop verification with automatic VRAM/SM tier detection (`T4 16GB`, `L4 24GB`, `A100 40GB`, `A100 80GB`).

---

## Mode A: Controlled Multi-Turn Heterogeneous Rollout Scheduling Benchmark

Reproduce the controlled heterogeneous rollout comparison cleanly on an 8-GPU node (`4×TP=1 + 1×TP=4` by default, configurable via `--topology`):

```bash
# Dry-run: validate topology, unified thresholds, CostModel wiring, and launch commands without GPUs
bash repro/tier2_four_gpu_closed_loop/run_rollout_ab.sh --dry-run

# Full real-GPU 6-policy, 2-round counterbalanced benchmark on 8x H20-3e (Qwen3-14B)
bash repro/tier2_four_gpu_closed_loop/run_rollout_ab.sh \
  --model Qwen/Qwen3-14B \
  --gpus 0,1,2,3,4,5,6,7 \
  --topology 1,1,1,1,4 \
  --output-dir ./logs/tier2_rollout_scheduling
```

### Experimental Setup

- **Hardware & Topology**: Single node `8 × NVIDIA H20-3e (141 GiB)`, `Qwen/Qwen3-14B` (`bfloat16`), deployed as **`4 × TP=1` (GPUs 0–3) + `1 × TP=4` (GPUs 4–7)**.
- **vLLM Serving Configuration**: `vLLM 0.9.2` V1 engine with CUDA graphs and `--enable-prefix-caching`, `max_model_len=24576`, `gpu_memory_utilization=0.85`, `max_num_seqs=32`, and per-instance cache isolation (`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`).
- **Fair Isolation & Unified Thresholds**:
  - Each `(round, policy, phase)` prepends a fixed-width session tag (`W01..W12` / `E01..E12`) at token 0 and fits `prompt0` to `1024` tokens, eliminating cross-policy/cross-round KV cache leakage while preserving real intra-trajectory multi-turn vLLM prefix caching.
  - All bucketed/length-aware schedulers share Libra's default boundary `short = 5000` (`BUCKETS_2`: `short -> TP=1 <= 5000`, `long -> TP=4 <= 24576`; `length_thresholds = {short: 5000, medium: 12000, long: 24576}`).
- **Phase 1 Cold Calibration & Cost Model**: Unique token-0 trial prefixes measure cold prefill (`2048, 4096, 8192, 16384`) and decode (`b=1` / `b=8` at `ctx=2048, 6144`, `gen=1024`) on `TP=1` and `TP=4`, autofitting `CostModel` scales (`prefill_scale`, `decode_scale`) alongside measured `recompute_ms` migration costs (`decode_ms={}` uses continuous-batching makespan estimation).
- **Heavy-Tailed Multi-Turn Workload**: Warmup `13` trajectories (`6 short / 4 quick_fix / 2 deep_fail / 1 deep_timeout`) and Eval `60` trajectories (`48 short / 10 quick_fix / 1 deep_fail / 1 deep_timeout`), 100% randomly shuffled, submitted at full-batch concurrency (`60`) with `temperature=0.0` and natural EOS over 2 counterbalanced rounds (Forward + Reverse):
  - `short`: 1 turn (`max_new_tokens=256`).
  - `quick_fix`: Turn 0 (`96`) -> `test_runner:medium:failure` (`1000`-tok log) -> Turn 1 (`192`).
  - `deep_fail` / `deep_timeout`: Turn 0 (`96`) -> 10 turns of `1000`-tok `test_runner` logs + `640` generated tokens/turn (peak context ~`17.5K`).

### Verified Real-GPU Results (`8 × NVIDIA H20-3e`, `Qwen3-14B`)

**Phase 1 Cold Micro-Calibration**:

| Metric | `TP=1` | `TP=4` | `TP=4` Speedup |
| :--- | ---: | ---: | ---: |
| Prefill `ctx=2048` median | `430.4 ms` | `128.9 ms` | `3.34x` |
| Prefill `ctx=4096` median | `864.2 ms` | `251.5 ms` | `3.44x` |
| Prefill `ctx=8192` median | `1790.3 ms` | `513.9 ms` | `3.48x` |
| Prefill `ctx=16384` median | `3970.8 ms` | `1128.6 ms` | `3.52x` |
| Decode `ctx=2048, gen=1024` (`b=1` / `b=8`) | `10.81s` / `15.94s` | `5.95s` / `7.49s` | `1.82x` / `2.13x` |
| Decode `ctx=6144, gen=1024` (`b=1` / `b=8`) | `12.07s` / `26.38s` | `6.28s` / `11.19s` | `1.92x` / `2.36x` |

Auto-fitted analytical `CostModel` scales: `TP=1: {prefill_scale: 0.6722, decode_scale: 1.4631}`, `TP=4: {prefill_scale: 0.7828, decode_scale: 2.6475}`.

**Phase 2 Six-Policy Counterbalanced Results (Mean Across Forward Round 1 & Reverse Round 2)**:

| Policy | Mean Makespan (R1 / R2) | Throughput (tok/s) | vs `round_robin` (tok/s) | Makespan Reduction | `deep_fail` / `deep_timeout` | Migrations | `TP=1 / TP=4` Calls |
| :--- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `la_mlfq` | `81.14s` (`81.09` / `81.18`) | `347.1` | `-4.0%` (`0.96x`) | `-4.2%` | `81.05s` / `81.13s` | `0` | `90 / 0` |
| `round_robin` | `77.88s` (`77.43` / `78.33`) | `361.6` | `+0.0%` (`1.00x`) | `+0.0%` | `77.78s` / `77.88s` | `0` | `72 / 18` |
| `least_connections` | `73.79s` (`73.48` / `74.09`) | `376.3` | `+4.1%` (`1.06x`) | `+5.3%` | `73.30s` / `70.02s` | `0` | `63 / 27` |
| `cmlfq` | `58.42s` (`58.66` / `58.18`) | `472.7` | `+30.7%` (`1.33x`) | `+25.0%` | `58.42s` / `53.83s` | `14` | `66 / 24` |
| `length_aware` | `52.61s` (`53.19` / `52.03`) | `532.6` | `+47.3%` (`1.48x`) | `+32.4%` | `52.60s` / `51.68s` | `0` | `74 / 16` |
| **`cmlfq_cost`** | **`45.08s`** (`45.09` / `45.08`) | **`624.6`** | **`+72.7%` (`1.73x`)** | **`+42.1%`** | **`45.08s` / `45.08s`** | **`15`** | **`60 / 30`** |

- **`cmlfq_cost` net speedups**: **`+72.7%` tok/s vs `round_robin`** (`1.73x`), **`+66.0%` vs `least_connections`**, **`+32.1%` vs `cmlfq`**, and **`+17.3%` vs `length_aware`**.
- **Routing verification**: Across both rounds, `cmlfq_cost` generated all `28,160` expected tokens (`short=256`, `quick_fix=288`, `deep_fail=6496`, `deep_timeout=6496`), placed short requests cleanly across `4×TP=1` (`13,440` tokens), migrated deep bimodal/timeout tails proactively at Turn 1 to `TP=4` (`reasons={resume: 60, distribution_cost: 30}`, `paths={stay: 78, recompute: 12}`), and retained intra-episode vLLM APC residency (`tp_tokens={1: 13440, 4: 14720}`).

---

## Mode B: 4-GPU Heterogeneous Closed Loop (No Slurm Required)

Runs 1 FSDP trainer on GPU 0 and 2 heterogeneous vLLM rollout instances on GPUs `1` (`TP=1`) and `2,3` (`TP=2`):

```bash
# Single policy run (default: cmlfq_cost on DAPO-Math mini + Qwen3-4B)
bash repro/tier2_four_gpu_closed_loop/run_4gpu.sh \
  --model Qwen/Qwen3-4B \
  --gpus 0,1,2,3

# Full policy sweep across round_robin, least_connections, cmlfq, and cmlfq_cost
bash repro/tier2_four_gpu_closed_loop/run_4gpu.sh \
  --model Qwen/Qwen3-4B \
  --policy all \
  --gpus 0,1,2,3 \
  --output-dir ./logs/tier2_ab

# Example: run on a 4-GPU slice of a multi-GPU H20/A800 server
CUDA_VISIBLE_DEVICES=0,1,2,3 bash repro/tier2_four_gpu_closed_loop/run_4gpu.sh \
  --model /path/to/Qwen3-4B \
  --steps 8 \
  --output-dir ./logs/tier2_h20_4gpu
```

Validate configs and inspect exact commands without starting GPUs:

```bash
python3 repro/tier2_four_gpu_closed_loop/launch_4gpu.py --dry-run
```

---

## Mode C: Google Colab 1-GPU Co-Located Startup

Open [colab_quickstart.ipynb](colab_quickstart.ipynb) directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetX-lab/Libra/blob/main/repro/tier2_four_gpu_closed_loop/colab_quickstart.ipynb)

Or launch locally on any single GPU (`T4` / `L4` / `A100` / `RTX`):

```bash
python3 repro/tier2_four_gpu_closed_loop/launch_4gpu.py \
  --mode colab-1gpu \
  --colab-tier auto \
  --steps 4
```

### Automatic Colab Presets (`--colab-tier auto`)

| Preset | Trigger Condition | Model | vLLM / FSDP Dtype | KV Utilization | `max_model_len` |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `t4` | `SM < 8.0` or `< 20 GiB` | `Qwen/Qwen2.5-0.5B-Instruct` | `half` / `float32` | `0.30` | `1024` |
| `l4` | `>= 20 GiB` (`SM >= 8.0`) | `Qwen/Qwen2.5-1.5B-Instruct` | `bfloat16` / `bfloat16` | `0.35` | `1536` |
| `a100-40g` | `>= 36 GiB` (`SM >= 8.0`) | `Qwen/Qwen3-1.7B` | `bfloat16` / `bfloat16` | `0.38` | `2048` |
| `a100-80g` | `>= 70 GiB` (`SM >= 8.0`) | `Qwen/Qwen3-4B` | `bfloat16` / `bfloat16` | `0.38` | `2048` |

> Why `float32` FSDP on T4? T4 (`sm_75`) lacks hardware `bfloat16` Tensor Cores. `engine/train_engine.py` reads `FSDP_DTYPE` (`auto` falls back to `float32` when `not torch.cuda.is_bf16_supported()`), while vLLM runs in `half`.

---

## Bundled Sample Data, Scripts & Templates

- [benchmark_rollout_scheduling.py](benchmark_rollout_scheduling.py) & [run_rollout_ab.sh](run_rollout_ab.sh): Controlled 6-policy multi-turn rollout benchmark with Phase 1 cold calibration, session-isolated vLLM APC, and unified `short=5000` thresholds.
- [launch_4gpu.py](launch_4gpu.py) & [run_4gpu.sh](run_4gpu.sh): Slurm-free 4-GPU closed-loop RL and 1-GPU Colab launcher.
- [colab_quickstart.ipynb](colab_quickstart.ipynb): End-to-end Google Colab notebook (Tier 1 replay + Tier 2 closed-loop RL).
- [sample_data/dapo_math_mini.jsonl](sample_data/dapo_math_mini.jsonl): 64 deterministic math verification problems (works offline out of the box; pass `--dataset /path/to/dapo-math-17k.parquet` for full DAPO-Math-17K).
- [config_4gpu_qwen3_4b.yaml](config_4gpu_qwen3_4b.yaml): Static 4-GPU configuration template (`1 Train + TP=1 + TP=2`).
- [config_colab_1gpu.yaml](config_colab_1gpu.yaml): Static 1-GPU Colab co-located configuration template.
- **Runtime dependencies**: In addition to CPU CI deps, GPU execution requires `vllm`, `transformers`, `datasets`, `sympy`, `regex`, `latex2sympy2`, and `word2number` (pre-installed by Colab notebook Cell 1).

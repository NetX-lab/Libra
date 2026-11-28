# Tier 2 — 8-GPU & 4-GPU Heterogeneous Closed-Loop RL & Google Colab Quickstart

Tier 2 provides **Slurm-free** real-GPU **end-to-end closed-loop RL** entry points (`AsyncRLTrainer` + `FSDPTrainEngine` + `DAPOMathWorkflow` + supervised `vllm_hot_reload_api_server.py` with confirmed in-place weight reloads):

1. **8-GPU Heterogeneous Closed Loop (`1 Train + 7 Rollout [TP=1, TP=2, TP=4]`)**: Real end-to-end GRPO closed-loop training across three heterogeneous serving tiers (`TP=1` on GPU 1, `TP=2` on GPUs 2–3, `TP=4` on GPUs 4–7) comparing `round_robin`, `least_connections`, `cmlfq`, and `cmlfq_cost`.
2. **4-GPU Heterogeneous Closed Loop (`1 Train + 3 Rollout [TP=1, TP=2]`)**: Smaller 4-GPU heterogeneous closed-loop setup (`TP=1` on GPU 1, `TP=2` on GPUs 2–3).
3. **Google Colab 1-GPU Co-Located Startup**: Single-GPU closed-loop verification with automatic VRAM/SM tier detection (`T4 16GB`, `L4 24GB`, `A100 40GB`, `A100 80GB`).

> Looking for pure rollout scheduling comparison (without FSDP training updates or weight sync)? See Tier 1 Pathway 2 (`Real-GPU Live vLLM Judge`) in [../tier1_zero_gpu_replay/README.md](../tier1_zero_gpu_replay/README.md).

---

## Mode 1: 8-GPU Heterogeneous Closed Loop (`1 Train + TP=1, 2, 4`)

Reproduce the 8-GPU end-to-end closed-loop RL comparison without Slurm:

```bash
# Dry-run: validate topology, CostModel profile, seeded trees, and commands without GPUs
python3 repro/tier2_four_gpu_closed_loop/launch_4gpu.py \
  --mode 8gpu \
  --policy all \
  --gpus 0,1,2,3,4,5,6,7 \
  --dry-run

# Full 4-policy real-GPU end-to-end closed-loop RL sweep on 8x H20-3e
bash repro/tier2_four_gpu_closed_loop/run_8gpu.sh \
  --policy all \
  --gpus 0,1,2,3,4,5,6,7 \
  --model Qwen/Qwen3-4B \
  --output-dir ./logs/tier2_8gpu_ab
```

### Experimental Setup (`8 × NVIDIA H20-3e`, `Qwen/Qwen3-4B`)

- **Hardware & Topology**: Single node `8 × NVIDIA H20-3e (141 GiB)`:
  - **Training**: `1 × FSDPTrainEngine` on GPU `0` (`bfloat16`, `batch_size=16`, `micro_batch_size=1`, `recompute_logprobs=false`, `FSDP_GRADIENT_CHECKPOINTING=0`, `n_samples=2`, `ppo_epochs=1`, `sync_interval=4`).
  - **Heterogeneous Rollout (`7` GPUs)**: `tp1` (`TP=1`, GPU `1`, `max_num_seqs=1`, port `18920`), `tp2` (`TP=2`, GPUs `2,3`, `max_num_seqs=2`, port `18921`), and `tp4` (`TP=4`, GPUs `4,5,6,7`, `max_num_seqs=16`, port `18922`).
- **vLLM Serving & Weight Reload**: `vLLM 0.9.2` V1 engine with CUDA graphs enabled, `--enable-prefix-caching`, `max_model_len=6144`, `gpu_memory_utilization=0.75`, per-instance isolated compile/KV caches, and verified parallel in-place disk weight reload (`rollout_weight_reload_method=inplace`, confirmed via `ack_tp1_4.json`, `ack_tp2_4.json`, `ack_tp4_4.json`).
- **Scheduling Policies & Calibration**:
  - Baselines: `load_balance` (`round_robin`, `least_connections`).
  - Causal schedulers: `cmlfq` and `cmlfq_cost` with unified three-tier buckets (`short -> TP=1 <= 400`, `medium -> TP=2 <= 1200`, `long -> TP=4 <= 6144`), pre-seeded `CausalPrefixTree` (`tree.json`), and empirical tier latency table (`sample_data/dapo_cost_profile.json`).
- **Multi-Turn Closed-Loop Workload (`sample_data/dapo_math_mini.jsonl`)**: `4` RL steps $\times$ `8` prompts/step $\times$ `2` rollouts (`64` trajectories total, `64` FSDP optimizer updates per policy), mixing `short_recover` (`3` turns `[32, 160, 160]`, `small:timeout`), `medium` (`2` turns `[32, 1120]`, `medium:failure`), `long_unambig` (`rollout 0` `[32, 1792, 1792]` + `rollout 1` `[32]`, `large:failure`), and `long_cascading` (Step 0 bimodal causal signal on `small:failure`, where `cmlfq` defers migration at Turn 1 due to `mean_p90_disagree` while `cmlfq_cost` eagerly promotes at Turn 1 via residual distribution + cost model).

### Verified Real-GPU Closed-Loop Results (`8 × NVIDIA H20-3e`, `Qwen/Qwen3-4B`)

All 4 policies completed 4 RL steps (`64` samples, `64` FSDP optimizer updates, `152` generation calls, `3/3` confirmed in-place weight reload acks):

| Policy | End-to-End `loop_seconds` | `rollout_wait_seconds` | `train_seconds` | `sync_seconds` | `samples/s` | vs `round_robin` (`samples/s`) | `gen_tokens/s` | vs `round_robin` (`tok/s`) | Route Calls (`TP=1 / 2 / 4`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `round_robin` | `168.60s` | `122.45s` | `29.94s` | `16.20s` | `0.3796` | `+0.0%` (`1.00x`) | `413.23` | `+0.0%` (`1.00x`) | `51 / 51 / 50` |
| `least_connections` | `121.69s` | `76.02s` | `29.40s` | `16.26s` | `0.5259` | `+38.5%` (`1.39x`) | `574.93` | `+39.1%` (`1.39x`) | `45 / 52 / 55` |
| `cmlfq` | `119.99s` | `73.57s` | `30.24s` | `16.15s` | `0.5334` | `+40.5%` (`1.41x`) | `580.13` | `+40.4%` (`1.40x`) | `113 / 16 / 23` |
| **`cmlfq_cost`** | **`115.38s`** | **`69.98s`** | **`29.35s`** | **`16.02s`** | **`0.5547`** | **`+46.1%` (`1.46x`)** | **`607.22`** | **`+46.9%` (`1.47x`)** | **`100 / 17 / 35`** |

- **`cmlfq_cost` net end-to-end closed-loop RL speedups**: **`+46.1%` (`1.46x`) `samples/s`** and **`+46.9%` (`1.47x`) `generated_tokens/s`** vs `round_robin` (`1.75x` faster rollout collection), **`+5.5%` `samples/s` / `+5.6%` `tok/s`** vs `least_connections`, and **`+4.0%` `samples/s` / `+4.7%` `tok/s`** vs `cmlfq`.
- **Per-step stability & tier placement**: `cmlfq_cost` held steady rollout times across all 4 steps (`[16.97s, 17.73s, 17.62s, 17.66s]`), promoting all long trajectories at Turn 1 to `TP=4` (`35` calls, including Step 0 `long_cascading`), isolating `medium` on `TP=2` (`17` calls), and keeping short turns on `TP=1` (`100` calls), whereas `round_robin` stalled `tp1` with long/medium head-of-line queueing (`36.52s -> 24.93s` per step).

---

## Mode 2: 4-GPU Heterogeneous Closed Loop (No Slurm Required)

Runs 1 FSDP trainer on GPU 0 and 2 heterogeneous vLLM rollout instances on GPUs `1` (`TP=1`) and `2,3` (`TP=2`):

```bash
# Single policy run (default: cmlfq_cost on DAPO-Math mini + Qwen3-4B)
bash repro/tier2_four_gpu_closed_loop/run_4gpu.sh \
  --model Qwen/Qwen3-4B \
  --gpus 0,1,2,3

# Full policy sweep across round_robin, least_connections, and cmlfq_cost
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

## Mode 3: Google Colab 1-GPU Co-Located Startup

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

- [launch_4gpu.py](launch_4gpu.py), [run_8gpu.sh](run_8gpu.sh), & [run_4gpu.sh](run_4gpu.sh): Slurm-free 8-GPU (`--mode 8gpu`) and 4-GPU (`--mode 4gpu`) closed-loop RL and 1-GPU Colab (`--mode colab-1gpu`) launcher.
- [colab_quickstart.ipynb](colab_quickstart.ipynb): End-to-end Google Colab notebook (Tier 1 replay + Tier 2 closed-loop RL).
- [sample_data/dapo_math_mini.jsonl](sample_data/dapo_math_mini.jsonl): 64 deterministic math verification problems with multi-turn rollout profiles (works offline out of the box; pass `--dataset /path/to/dapo-math-17k.parquet` for full DAPO-Math-17K).
- [sample_data/dapo_cost_profile.json](sample_data/dapo_cost_profile.json): Empirical H20-3e tier latency profile for `cmlfq_cost` in 8-GPU closed-loop evaluation.
- [config_8gpu_qwen3_4b.yaml](config_8gpu_qwen3_4b.yaml): Static 8-GPU configuration template (`1 Train + TP=1 + TP=2 + TP=4`).
- [config_4gpu_qwen3_4b.yaml](config_4gpu_qwen3_4b.yaml): Static 4-GPU configuration template (`1 Train + TP=1 + TP=2`).
- [config_colab_1gpu.yaml](config_colab_1gpu.yaml): Static 1-GPU Colab co-located configuration template.
- **Runtime dependencies**: In addition to CPU CI deps, GPU execution requires `vllm`, `transformers`, `datasets`, `sympy`, `regex`, `latex2sympy2`, and `word2number` (pre-installed by Colab notebook Cell 1).

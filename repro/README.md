# Libra Reproduction Suite

This directory organizes Libra into **three progressive reproduction tiers** so researchers and engineers can evaluate, adopt, and extend Libra as a clean baseline without hidden cluster dependencies.

| Tier | Goal | Hardware Requirement | Runtime | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | 0-GPU Scheduling Replay & GRP Offline Baseline (`RoundRobin`, `LeastConnections`, `LengthAware`, `LA-MLFQ`, `C-MLFQ`, `C-MLFQ-Cost`, plus `GlobalResourcePlanner`) | **0 GPUs** (CPU only) | ~1–3 seconds | Ready (`repro/tier1_zero_gpu_replay/`) |
| **Tier 2** | 4-GPU Real Small-Scale Closed Loop (`DAPO-Math` + `Qwen3-4B`, 1 Train + 3 Heterogeneous Rollout GPUs, plus Google Colab launcher) | **4 GPUs** (or single-node Colab / local server) | ~15–30 minutes | Upcoming (Step 4) |
| **Tier 3** | Paper-Scale Reproduction (`32 GPUs`: 4×8 H20 cluster, `R2E-Gym` + `Qwen3-14B`, `Search-R1` / `DAPO-Math` presets, SSH/`torchrun` + minimal Slurm template) | **32 GPUs** (4×8 H20 / H100 / A800) | Paper scale | Upcoming (Step 5) |

---

## Quick Start (Tier 1 — 0 GPUs)

```bash
python repro/tier1_zero_gpu_replay/run_replay.py
```

See [tier1_zero_gpu_replay/README.md](tier1_zero_gpu_replay/README.md) for custom trace inputs, KV migration backend ablations (`recompute` / `nixl` / `cpu_offload`), and JSON export options.

## CPU CI Check

```bash
bash scripts/test_cpu_ci.sh
```

# Megatron-Core training backend

The `megatron_core` backend is the Qwen3-MoE training path used by the
R2E-Gym experiment. It replaces the former FSDP full-state export boundary
with two collective, bounded-memory operations:

1. `model.sharded_state_dict()` and
   `optimizer.sharded_state_dict(...)` are saved with Megatron-Core
   `torch_dist` distributed checkpointing.
2. Megatron Bridge collectively streams converted Hugging Face tensors.
   All ranks participate in TP/EP conversion. For disk sync, rank 0 writes the
   original safetensors shard layout. With `weight_sync_mode=nccl`, rank 0
   broadcasts each tensor directly from GPU memory to all vLLM workers; no
   rollout checkpoint or safetensors file is created.

MCore 0.14's non-Transformer-Engine `GroupedMLP` stores every local expert in
two expert-major tensors, `weight1` and `weight2`. Megatron Bridge 0.2 does not
map those names. `engine/megatron_core_moe_mapping.py` supplies an explicit
Qwen3 mapping which:

- selects only the experts owned by the local EP rank;
- slices gate/up/down projections directly for the local ETP rank;
- preserves GroupedMLP's expert-major in-memory layout;
- gathers one expert at a time across ETP and EP during HF export.

Conversion task coverage is strict. Initialization fails if any local or
placeholder task has `mapping=None`; an unmapped expert can no longer remain
randomly initialized behind a Bridge warning.

Each synchronized version contains:

```text
vN/
  megatron_dist/                  # restartable model + optimizer shards
  model-*.safetensors             # rollout-compatible streaming export
  model.safetensors.index.json
  megatron_checkpoint_manifest.json
  sync_progress/rank_*.json
```

## Runtime topology

The default four-node job uses eight training GPUs and eight rollout GPUs.
The training topology is `TP=2, PP=1, CP=1, DP=4, EP=4, ETP=2`. Expert
weights are split over EP and expert TP ranks; dense weights and optimizer
state use TP and Megatron distributed optimizer sharding.

The default attention implementation starts from the official MCore local
Qwen3-MoE layer spec and replaces only core attention with PyTorch 2.7 SDPA.
This selects CUDA Flash Attention for supported BF16 shapes and avoids the
quadratic attention-score allocation at a 32768-token model limit.
The local spec also supplies PyTorch RMSNorm modules carrying MCore's
`sequence_parallel` parameter metadata, including the decoder's final norm.
Transformer Engine remains available through
`megatron_use_transformer_engine=true`.

Multi-GPU preflights launch one Slurm task per node and let `torchrun` create
the four local workers. Do not assign `LOCAL_RANK=0` to every Slurm task:
that maps multiple ranks to the same CUDA device and can surface as an NCCL
`invalid device ordinal` error.

## Installation

The cluster login node does not need internet access. Put the pinned MCore,
Bridge, and small Python dependency packages in `vendor/megatron`, then run:

```bash
bash scripts/install_megatron_core.sh
```

The installer validates the Qwen3-MoE bridge, distributed checkpointing, and
the SDPA adapter before a Slurm job is submitted.

## Submission

```bash
sbatch scripts/submit_r2e_gym_cmlfq_megatron_core_qwen3_30b_a3b.slurm
```

The native distributed checkpoint is the source of truth for training
resume. The Hugging Face safetensors files are an interoperability artifact
used only to restart vLLM rollout workers at the configured sync interval.

## Direct NCCL rollout refresh

```yaml
train_backend: megatron_core
weight_sync_mode: nccl
rollout_weight_sync_mode: nccl
rollout_weight_sync_control_dir: /path/to/run/rollout_weight_sync
rollout_nccl_host: 10.0.0.10
rollout_nccl_port: 29620
rollout_nccl_chunk_mb: 256
rollout_nccl_rate_limit_gbps: 0.0
```

Both rollout instances enter `/reload_weights_nccl` concurrently. Metadata
uses `StatelessProcessGroup`, while tensors use an isolated
`PyNcclCommunicator`, so this path does not alter Megatron's training process
group. The trainer pauses rollout dispatch and sends bounded chunks outside
gradient collectives. Set a positive `rollout_nccl_rate_limit_gbps` when the
rollout and training jobs share an InfiniBand fabric.

Libra defaults the independent communicator to `NCCL_CUMEM_ENABLE=0` to match
vLLM 0.9.x worker communicators. Do not configure different cuMem modes on the
Megatron sender and vLLM receivers; mixed CUMEM/IPC transports fail during
communicator initialization.

Wire volume is approximately `parameter_count * dtype_bytes` per rollout
worker and refresh. Same-node peers normally use NVLink/PCIe; cross-node peers
use InfiniBand. Two rollout instances increase fan-out traffic, but avoid all
model checkpoint I/O. The refresh log reports bytes and elapsed seconds for
effective-bandwidth and congestion monitoring.

## Verification

Run the focused CPU-friendly tests before submitting a cluster job:

```bash
pytest -q \
  tests/test_megatron_core_attention.py \
  tests/test_megatron_core_checkpointing.py \
  tests/test_megatron_core_config.py \
  tests/test_megatron_core_moe_mapping.py \
  tests/test_megatron_core_train_engine.py
```

The production launcher performs its NCCL all-gather health check before
starting training. Tune `NCCL_PREFLIGHT_ELEMENTS`,
`NCCL_PREFLIGHT_ITERATIONS`, and `NCCL_PREFLIGHT_WALLTIME` when validating a
new cluster or interconnect.

# Libra DeepSeek-V4 MindSpeed / PP=2 integration

This backend is under hardware validation. It is not a completed 96-NPU
C-MLFQ/EHP/GRP benchmark. Do not quote automodelwire's verl results as Libra results.

## Selection and runtime

Use `train_backend: megatron_core` and `megatron_model_provider: mindspeed`.
The existing `bridge` provider retains its PP=1 restriction. See
`configs/deepseek_v4_mindspeed_pp2_96npu.yaml` for the 64-training/32-rollout
budget and model metadata. This is a training integration configuration, not
an independently validated 32-NPU rollout launcher.

Target image: `deepseek-v4-dspark:v30-fixed`, image digest
`sha256:ef517fe815af8c8d68a3fb5c8e2432866843dd7690f163a28266e345f0b3aaf2`.
Source `scripts/mindspeed_v4_env.sh` inside the image after setting
`LIBRA_SOURCE_PARENT` to the parent of the `RL_Framework` source package.
Preserve the image's existing CANN Python/library paths. Mount the host Ascend
driver and the assigned NPU devices. No global Python environment was modified.

`configs/model_arch_config/deepseek_v4_mindspeed_args.json` contains the V4
architecture/kernel settings extracted from automodelwire's
`configs/train_e4_reduce.sh`. Runtime TP/PP/EP/ETP, batch, sequence length,
optimizer offload and tokenizer path override the corresponding template values.
The uneven pipeline split is 21/22 layers, CP=1, MTP training disabled.

## Implemented contracts

- Initialize MindSpeed-LLM patches once, create DeepSeek4Model with Megatron
  `get_model`, and load distributed BF16 weights using the pinned mbridge.
- Execute training and logprob recomputation through Megatron's 1F1B scheduler;
  retain Libra's existing GRPO loss and policy-version API. The scheduler owns
  gradient finalization, avoiding duplicate reduction/scaling.
- Distribute source trajectories across TP then PP, pad microbatches consistently,
  and broadcast final-stage logprobs/statistics back to the first stage.
- Permit decoupled expert TP: the expert world is `world_size / (PP * ETP)`.
  Thus TP4/PP2/DP8/EP32/ETP1 is legal for 64 training ranks.
- Adapt mbridge's export API to Libra checkpoints. Preserve frozen MTP tensors
  in streamed exports; disk exports link original frozen shards and retain V4
  config/remote-code files. These checkpoint links require the original model
  path to remain available. Missing non-MTP weights fail validation.
- Add explicit Ascend NPU/HCCL support to the device helper while retaining
  existing CUDA/CPU behavior.

## Validation and remaining gates

Executed in the target runtime:

- Configuration, training-contract and checkpoint regression tests passed.
- `scripts/validate_mindspeed_pp2.py` passed on two real 910B3 NPUs. Both PP
  stages matched gradients from the same unsplit tiny network; recomputed
  logprobs returned correctly across PP. It uses 4D activations to exercise the
  mHC-compatible communication patch. It does not instantiate the full V4 model.
- Full TP4/PP2/EP32 runtime argument validation and DeepSeek4Model/mbridge imports
  passed. Full-weight 64-rank validation is tracked separately in run logs.

`validate_mindspeed_v4.py` is the full-weight gate: load, recompute logprobs,
perform one GRPO update, assert finite metrics and changed parameter samples,
and record per-rank peak HBM. It uses synthetic trajectories and is not a
throughput benchmark. The dated launcher is restricted to the checked eight
training nodes and refuses busy devices or an existing gate container.

Not yet validated: full-model gradient/update correctness, complete HF export
and reload, vLLM-Ascend EP rollout integration, end-to-end policy version
consistency, production workloads and performance ablations. MindSpeed EHP now
models TP*PP*DP as a complete V4 replica and reduces dense and expert gradients
over their matching DP groups. The 64-rank endpoint mapping, multi-node worker
launch and hardware gradient/update gate still require validation before EHP
can be reported as running.

Remote EHP checks on 2026-10-08 passed 60 relevant pytest cases and a two-NPU
910B3 HCCL gradient gate (`dense=12`, `expert=122`). The gate exercises the new
per-parameter dense/expert reduction on hardware with synthetic gradients; it
does not load V4 weights. A complete V4 EHP experiment needs 64 core ranks plus
a separate 64-rank EHP replica (128 NPUs before rollout capacity), exceeding
the requested 96-NPU pool.

Direct CUDA/NCCL rollout transport is rejected for this provider until a real
HCCL transport is connected and validated. EP-sharded EHP requires a complete
expert-sharded training replica. A 64-training/32-rollout layout alone cannot
demonstrate adding another identical 64-rank training copy.
GRP must respect legal expert and rollout topologies before enabling online
resource changes. These limitations must not be reported as zero speedup or
as successful EHP/GRP execution.

The full Libra minimum NPU count and C-MLFQ/EHP/GRP speedups remain unmeasured.

## Initial 64-rank attempt

The first full-weight gate exited before model construction. Seven nodes hit
Megatron's legacy CLI check requiring precision-aware optimizer with CPU offload;
the automodelwire path configures offload on `OptimizerConfig` separately from
the model/global arguments. The Libra provider now does the same, keeping CPU
offload enabled in the optimizer while leaving precision-aware mode disabled.
Node `.85` could not enumerate NPUs in the eight-device container while the
two-device validation container remained running; the task-owned validation
containers were stopped before retry. No other user's processes were stopped.


The third eight-node gate run completed full 64-rank BF16 model construction and
mbridge weight loading (observed at all ranks). It then exposed an API mismatch:
the pinned Megatron `OptimizerConfig` does not accept `overlap_param_gather`.
That unsupported argument has been removed; the next attempt will reuse the
validated interface and check optimizer creation, GRPO update and HBM peaks.

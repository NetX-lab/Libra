# Ascend NPU support

The `NPU_Support` branch is the Ascend-specific implementation of Libra. It is
kept separate from the NVIDIA GPU implementation on `main` so that CUDA/NCCL
and torch-npu/HCCL dependency stacks can evolve independently.

## Supported path

- Qwen3 dense models, including the validated Qwen3-14B configuration.
- Megatron-Core training with tensor parallelism and HCCL collectives.
- Project-local Megatron-Core compatibility shims; installed packages are not
  patched in place.
- vLLM Ascend OpenAI-compatible rollout workers.
- Global Resource Planner runtime decisions.
- C-MLFQ routing across heterogeneous TP=1, TP=2, and TP=4 rollout workers.
- Elastic Hybrid Pool execution with a decoupled inter-replica gradient domain.
- R2E-Gym asynchronous reinforcement learning.

## Important files

| Path | Purpose |
| --- | --- |
| `engine/device_utils.py` | Backend-neutral device and distributed helpers |
| `engine/megatron_npu_compat.py` | Process-scoped Megatron-Core NPU compatibility layer |
| `engine/megatron_core_train_engine.py` | Megatron-Core training integration |
| `scripts/setup_ascend_megatron_validated_env.sh` | Recreate the validated training environment |
| `scripts/run_vllm_ascend_server.sh` | Start one vLLM Ascend server |
| `scripts/start_r2e_rollout_pool_npu.sh` | Manage a TP 1+1+2+4 rollout pool |
| `scripts/run_6node48_r2e_mcore_npu.sh` | Launch the six-node reference experiment |
| `configs/hardware_config/ascend_910b3_64g.yaml` | Calibratable GRP hardware seed values |
| `configs/r2e_gym_qwen3_14b_mcore_npu_6node48_100step.yaml` | Sanitized reference configuration |

## Branch policy

NPU changes should be developed and validated on `NPU_Support`. GPU-only
changes belong on `main` and are not automatically merged into this branch.
Portable fixes may be applied to both branches as separate, independently
tested commits.

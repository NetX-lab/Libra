# Ascend NPU cluster quick start

This guide describes the reference layout used for Qwen3-14B on 6 nodes and
48 Ascend NPUs: four 8-NPU Megatron-Core training nodes and two 8-NPU rollout
nodes.

## 1. Prepare identical software

Install CANN 8.5.x and torch-npu on every node. Recreate the validated training
environment and keep the vLLM Ascend environment separate:

```bash
bash scripts/setup_ascend_megatron_validated_env.sh
```

See `docs/npu_environment.md` and the validated requirement lock files for the
dependency pins and hardware preflight command.

## 2. Distribute code and models

The nodes do not need a shared local disk, but every node must see the same
source revision, model files, configuration, and Python environments. Choose
one of these approaches:

- mount shared storage at identical paths on every node; or
- synchronize the repository, model, and environments to identical local
  paths before launch.

Runtime coordination directories used for checkpoints, weight synchronization,
planner state, and Elastic Hybrid Pool tasks should be on shared storage. If
shared storage is unavailable, replace those mechanisms with an explicit
replication service before running the multi-node configuration.

## 3. Configure SSH and inventory

The public launcher uses key-based, non-interactive SSH. Configure host keys and
verify `ssh -o BatchMode=yes root@HOST true` for all six nodes. Do not place
passwords or private keys in repository files.

Export your topology:

```bash
export TRAIN_HOSTS="10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13"
export ROLLOUT_HOSTS="10.0.0.20 10.0.0.21"
export MASTER_ADDR="10.0.0.10"
export SSH_USER="root"
export PROJECT_DIR="/opt/libra"
export SHARED_RUN_ROOT="/shared/libra_runs/qwen3_14b_6node48"
export TRAIN_PYTHON="/opt/libra-env/bin/python"
```

Update the sanitized IP addresses and filesystem paths in
`configs/r2e_gym_qwen3_14b_mcore_npu_6node48_100step.yaml`.

## 4. Validate one NPU

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$(dirname "$PWD")"
export DEVICE_BACKEND=npu
export DIST_BACKEND=hccl
ASCEND_RT_VISIBLE_DEVICES=0 \
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
  examples/test_megatron_core_npu_runtime.py \
  --model /models/Qwen3-0.6B --max-seq-length 128 --training-step
```

Require both `MCORE_NPU_PREFLIGHT_OK` and `MCORE_NPU_TRAIN_STEP_OK` before a
multi-node run.

## 5. Launch the reference experiment

```bash
bash scripts/run_6node48_r2e_mcore_npu.sh
```

The launcher checks that every selected training node is idle, starts two
heterogeneous rollout pools, waits for all eight health endpoints, and then
starts four `torch.distributed.run` drivers with eight local ranks each.

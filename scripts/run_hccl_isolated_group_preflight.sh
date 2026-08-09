#!/usr/bin/env bash
set -euo pipefail

runtime_project_dir="${RUNTIME_PROJECT_DIR:-/opt/libra/runtime_sources/RL_Framework_NPU}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-/opt/libra/runtime_sources}"
python_bin="${TRAIN_PYTHON:-/opt/libra/envs/rl_mindspeed/bin/python}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$runtime_project_dir"
export PYTHONPATH="$runtime_pythonpath"
export DEVICE_BACKEND=npu DIST_BACKEND=hccl
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300

exec "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/hccl_isolated_group_preflight.py

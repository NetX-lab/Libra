#!/usr/bin/env bash
# Launch a foreground, password-authenticated torchrun preflight from the jump
# host.  Every selected host must have all eight NPUs idle; this script never
# kills or replaces an existing accelerator process.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

read -r -a hosts <<< "${INTERNAL_HOSTS:-node-internal node-internal node-internal node-internal}"
master_addr="${MASTER_ADDR:-${hosts[0]}}"
master_port="${MASTER_PORT:-29712}"
model_path="${MODEL_PATH:-./models/Qwen3-14B}"
initialize_optimizer="${INITIALIZE_OPTIMIZER:-0}"
max_seq_length="${MAX_SEQ_LENGTH:-512}"
run_root="${RUN_ROOT:-./runs/r2e_gym_qwen3_14b_5node40}"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${run_root}/mcore_init_32_opt${initialize_optimizer}_${timestamp}"
mkdir -p "$run_dir"
printf '%s\n' "$run_dir" > "${run_root}/current_mcore_preflight"

export INTERNAL_SSH_TIMEOUT=30
for host in "${hosts[@]}"; do
  probe="$(
    ./scripts/internal_ssh.sh "$host" -- \
      "npu-smi info | grep -c -F 'No running processes found in NPU'"
  )"
  if ! grep -qE '(^|[[:space:]])8([[:space:]]|$)' <<< "$probe"; then
    printf 'Refusing to launch: %s is not fully idle\n%s\n' "$host" "$probe" >&2
    exit 3
  fi
done

export INTERNAL_SSH_TIMEOUT=-1
pids=()
for node_rank in "${!hosts[@]}"; do
  target="${hosts[$node_rank]}"
  remote_cmd="source ${ASCEND_SET_ENV:?set ASCEND_SET_ENV}; \
cd ./RL_Framework_npu; \
export PYTHONPATH=/root DEVICE_BACKEND=npu DIST_BACKEND=hccl \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=${model_path} TRAIN_TP_SIZE=8 MAX_SEQ_LENGTH=${max_seq_length} \
INITIALIZE_OPTIMIZER=${initialize_optimizer} GLOO_SOCKET_IFNAME=eth0 \
HCCL_SOCKET_IFNAME=eth0 HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800; \
exec ./venv/framework/bin/torchrun \
--nnodes=${#hosts[@]} --nproc_per_node=8 --node_rank=${node_rank} \
--master_addr=${master_addr} --master_port=${master_port} \
scripts/validate_megatron_core_npu_init.py"
  ./scripts/internal_ssh.sh "$target" -- "$remote_cmd" \
    >"$run_dir/driver_node_${node_rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

printf 'RUN_DIR=%s STATUS=%s\n' "$run_dir" "$status"
for log in "$run_dir"/driver_node_*.log; do
  printf 'FILE=%s\n' "$log"
  grep -E 'MCORE_NPU_MODEL_INIT_OK|Traceback|Error|FAILED' "$log" | tail -n 40 || true
done
exit "$status"

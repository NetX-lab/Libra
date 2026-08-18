#!/usr/bin/env bash
# Jump-host launcher for a multi-node Megatron trainer. Selected training hosts
# must be completely idle; set INTERNAL_HOSTS to the exact NPU host list.
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$project_dir"

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

read -r -a hosts <<< "${INTERNAL_HOSTS:?set INTERNAL_HOSTS to the selected NPU hosts}"
master_addr="${MASTER_ADDR:-${hosts[0]}}"
master_port="${MASTER_PORT:-29720}"
config_path="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_5node40_100step.yaml}"
run_root="${RUN_ROOT:-./runs/r2e_gym_qwen3_14b_5node40}"
run_name="${RUN_NAME:-train_$(date +%Y%m%d_%H%M%S)}"
run_dir="${run_root}/${run_name}"
mkdir -p "$run_dir"
printf '%s\n' "$run_dir" > "${run_root}/current_training_run"

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

if [[ "${SKIP_ROLLOUT_HEALTH:-0}" != "1" ]]; then
  for port in 8000 8001 8002 8003; do
    if ! curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null; then
      printf 'Rollout health check failed on jump-host port %s\n' "$port" >&2
      exit 4
    fi
  done
fi

export INTERNAL_SSH_TIMEOUT=-1
nodes="${#hosts[@]}"
nproc_per_node="${NPROC_PER_NODE:-8}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-$project_dir}"
package_root="${PACKAGE_ROOT:-$(dirname "$runtime_project_dir")}"
train_python="${TRAIN_PYTHON:-python3}"
ascend_set_env="${ASCEND_SET_ENV:?set ASCEND_SET_ENV to the Ascend toolkit set_env.sh}"
pids=()
for node_rank in "${!hosts[@]}"; do
  target="${hosts[$node_rank]}"
  remote_cmd="source '$ascend_set_env'; \
cd '$runtime_project_dir'; \
export PYTHONPATH='$package_root' DEVICE_BACKEND=npu DIST_BACKEND=hccl \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GLOO_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0 \
HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 \
TORCH_DISTRIBUTED_TIMEOUT=3600 TOKENIZERS_PARALLELISM=false \
WANDB_MODE=disabled RL_TRAIN_PHASE_TRACE=1 OMP_NUM_THREADS=1 \
JOB_ID=${run_name} \
R2E_EVAL_MAX_SAMPLES=${R2E_EVAL_MAX_SAMPLES:-4} \
R2E_EVAL_CONCURRENCY=${R2E_EVAL_CONCURRENCY:-4} \
R2E_GYM_INDEX='$runtime_project_dir/data/r2e_gym_v1/index.jsonl'; \
exec '$train_python' -m torch.distributed.run \
--nnodes=${nodes} --nproc_per_node=${nproc_per_node} --node_rank=${node_rank} \
--master_addr=${master_addr} --master_port=${master_port} \
examples/r2e_gym_async_rl.py --config ${config_path}"
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
  grep -E 'Step [0-9]+|RL_TRAIN_PHASE|Traceback|Error|FAILED|Training complete' "$log" \
    | tail -n 80 || true
done
exit "$status"

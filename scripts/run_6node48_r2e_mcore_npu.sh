#!/usr/bin/env bash
# Generic six-node / 48-NPU R2E-Gym launcher.
# Configure key-based SSH and the inventory through environment variables.
set -euo pipefail

project_dir="${PROJECT_DIR:-$PWD}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-$project_dir}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-$(dirname "$runtime_project_dir")}"
python_bin="${TRAIN_PYTHON:-python}"
entrypoint="${ENTRYPOINT:-examples/r2e_gym_async_rl.py}"
config_path="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_6node48_100step.yaml}"
run_root="${RUN_ROOT:-$project_dir/runs/r2e_gym_qwen3_14b_6node48}"
run_name="${RUN_NAME:-formal_100step_$(date +%Y%m%d_%H%M%S)}"
run_dir="${run_root}/${run_name}"
master_addr="${MASTER_ADDR:?set MASTER_ADDR}"
master_port="${MASTER_PORT:-29742}"
ssh_user="${SSH_USER:-root}"
read -r -a train_hosts <<< "${TRAIN_HOSTS:?set TRAIN_HOSTS to four training hosts}"
read -r -a rollout_hosts <<< "${ROLLOUT_HOSTS:?set ROLLOUT_HOSTS to two rollout hosts}"
[[ "${#train_hosts[@]}" -eq 4 ]] || { echo "TRAIN_HOSTS must contain four hosts" >&2; exit 2; }
[[ "${#rollout_hosts[@]}" -eq 2 ]] || { echo "ROLLOUT_HOSTS must contain two hosts" >&2; exit 2; }

mkdir -p "$run_dir" "${run_root}/rollout_weight_sync"
printf '%s\n' "$run_dir" >"${run_root}/current_training_run"

ssh_cmd=(ssh -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT:-15}")

for host in "${train_hosts[@]}" "${rollout_hosts[@]}"; do
    "${ssh_cmd[@]}" "${ssh_user}@${host}" true
    if [[ "${REQUIRE_IDLE_NPUS:-1}" == "1" ]]; then
        idle_count="$("${ssh_cmd[@]}" "${ssh_user}@${host}" \
            "npu-smi info | grep -c -F 'No running processes found in NPU'")"
        [[ "$idle_count" -eq 8 ]] || {
            echo "node is not fully idle: $host (idle NPUs: $idle_count/8)" >&2
            exit 3
        }
    fi
done

start_rollout() {
    local host="$1" node_id="$2"
    "${ssh_cmd[@]}" "${ssh_user}@${host}" \
        "cd '$runtime_project_dir' && RUN_ROOT='$run_root' PROJECT_DIR='$runtime_project_dir' bash scripts/start_r2e_rollout_pool_npu.sh start '$node_id'"
}

stop_rollout() {
    local host="$1" node_id="$2"
    "${ssh_cmd[@]}" "${ssh_user}@${host}" \
        "cd '$runtime_project_dir' && RUN_ROOT='$run_root' PROJECT_DIR='$runtime_project_dir' bash scripts/start_r2e_rollout_pool_npu.sh stop '$node_id'" \
        || true
}

cleanup() {
    [[ "${KEEP_ROLLOUT:-0}" == "1" ]] && return
    stop_rollout "${rollout_hosts[0]}" rollout0
    stop_rollout "${rollout_hosts[1]}" rollout1
}
trap cleanup EXIT INT TERM

start_rollout "${rollout_hosts[0]}" rollout0
start_rollout "${rollout_hosts[1]}" rollout1

for host in "${rollout_hosts[@]}"; do
    for port in 8000 8001 8002 8003; do
        ready=0
        for _ in $(seq 1 360); do
            if curl -fsS --max-time 5 "http://${host}:${port}/health" >/dev/null; then
                ready=1
                break
            fi
            sleep 5
        done
        [[ "$ready" -eq 1 ]] || { echo "rollout endpoint failed: ${host}:${port}" >&2; exit 4; }
    done
done

pids=()
for node_rank in "${!train_hosts[@]}"; do
    host="${train_hosts[$node_rank]}"
    remote_cmd="source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true; \
cd '$runtime_project_dir'; \
export PYTHONPATH='$runtime_pythonpath' PYTHONDONTWRITEBYTECODE=1 DEVICE_BACKEND=npu DIST_BACKEND=hccl \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 TORCH_DISTRIBUTED_TIMEOUT=3600 \
TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled RL_TRAIN_PHASE_TRACE=1 OMP_NUM_THREADS=1; \
exec '$python_bin' -m torch.distributed.run --nnodes=4 --nproc_per_node=8 --node_rank=$node_rank \
--master_addr='$master_addr' --master_port='$master_port' '$entrypoint' --config '$config_path'"
    "${ssh_cmd[@]}" "${ssh_user}@${host}" "bash -lc \"$remote_cmd\"" \
        >"${run_dir}/driver_node_${node_rank}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
echo "RUN_DIR=$run_dir STATUS=$status"
exit "$status"

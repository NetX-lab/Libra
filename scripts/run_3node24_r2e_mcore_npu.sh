#!/usr/bin/env bash
# Foreground launcher for the compact 3-node / 24-NPU R2E-Gym A/B run.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-/opt/libra/runtime_sources/RL_Framework_NPU}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-/opt/libra/runtime_sources}"
config_path="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_3node24_100step_no_ehp.yaml}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_3node24}"
run_name="${RUN_NAME:-formal_3node24_$(date +%Y%m%d_%H%M%S)}"
run_dir="${run_root}/${run_name}"
master_addr="${MASTER_ADDR:-192.0.2.10}"
master_port="${MASTER_PORT:-29862}"
read -r -a train_hosts <<< "${TRAIN_HOSTS:-192.0.2.10 192.0.2.11}"

[[ "${#train_hosts[@]}" -eq 2 ]] || {
    echo "compact run requires exactly two training hosts" >&2
    exit 2
}

mkdir -p "$run_dir" "${run_root}/rollout_weight_sync"
printf '%s\n' "$run_dir" >"${run_root}/current_training_run"

cleanup_rollout() {
    [[ "${KEEP_ROLLOUT:-0}" == "1" ]] && return
    PROJECT_DIR="$runtime_project_dir" RUN_ROOT="$run_root" \
        bash "$runtime_project_dir/scripts/start_r2e_rollout_pool_npu.sh" stop rollout_a || true
}
trap cleanup_rollout EXIT INT TERM

export INTERNAL_SSH_TIMEOUT=30
for host in "${train_hosts[@]}"; do
    probe="$("$project_dir/scripts/internal_ssh.sh" "$host" -- \
        "test -f '$runtime_project_dir/examples/r2e_gym_async_rl.py' && test -f '$runtime_project_dir/$config_path' && npu-smi info | grep -c -F 'No running processes found in NPU'")"
    grep -qE '(^|[[:space:]])8([[:space:]]|$)' <<<"$probe" || {
        echo "training node is not fully idle: $host" >&2
        exit 3
    }
done

local_idle="$(npu-smi info | grep -c -F 'No running processes found in NPU')"
[[ "$local_idle" -eq 8 ]] || { echo "rollout node is not fully idle" >&2; exit 3; }

find "${run_root}/rollout_weight_sync" -maxdepth 1 -type f \
    \( -name 'reload_request.json' -o -name 'ack_*.json' -o -name 'error_*.json' \) \
    -delete

PROJECT_DIR="$runtime_project_dir" RUN_ROOT="$run_root" \
    bash "$runtime_project_dir/scripts/start_r2e_rollout_pool_npu.sh" start rollout_a

for port in 8000 8001 8002 8003; do
    ready=0
    for _ in $(seq 1 360); do
        if curl -fsS --max-time 5 "http://192.0.2.20:${port}/health" >/dev/null; then
            ready=1
            break
        fi
        sleep 5
    done
    [[ "$ready" -eq 1 ]] || { echo "rollout endpoint failed: 192.0.2.20:${port}" >&2; exit 4; }
    echo "rollout ready: 192.0.2.20:${port}"
done

export INTERNAL_SSH_TIMEOUT=-1
pids=()
for node_rank in "${!train_hosts[@]}"; do
    target="${train_hosts[$node_rank]}"
    remote_cmd="source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
cd ${runtime_project_dir}; \
export PYTHONPATH=${runtime_pythonpath} PYTHONDONTWRITEBYTECODE=1 DEVICE_BACKEND=npu DIST_BACKEND=hccl \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GLOO_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0 \
HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 \
TORCH_DISTRIBUTED_TIMEOUT=3600 TOKENIZERS_PARALLELISM=false \
WANDB_MODE=disabled RL_TRAIN_PHASE_TRACE=1 OMP_NUM_THREADS=1 \
MCORE_MOE_GROUPED_GEMM=0 JOB_ID=${run_name} \
ELASTIC_TRAINING_STATE_DIR=${run_root}/elastic_training_state \
R2E_GYM_INDEX=${runtime_project_dir}/data/r2e_gym_v1/index.jsonl; \
exec /opt/libra/envs/rl_mindspeed/bin/python -m torch.distributed.run \
--nnodes=2 --nproc_per_node=8 --node_rank=${node_rank} \
--master_addr=${master_addr} --master_port=${master_port} \
examples/r2e_gym_async_rl.py --config ${config_path}"
    "$project_dir/scripts/internal_ssh.sh" "$target" -- "$remote_cmd" \
        >"${run_dir}/driver_node_${node_rank}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
echo "RUN_DIR=$run_dir STATUS=$status"
for log in "$run_dir"/driver_node_*.log; do
    echo "FILE=$log"
    grep -E 'Step [0-9]+|GlobalResourcePlanner|RuntimeElasticExecutor|Traceback|Error|FAILED|Training complete' "$log" \
        | tail -n 160 || true
done
exit "$status"

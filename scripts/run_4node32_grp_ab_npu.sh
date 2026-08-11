#!/usr/bin/env bash
# Run one arm of the controlled four-node GRP ablation in the foreground.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"
: "${ARM:?set ARM to grp or no_grp}"

case "$ARM" in
  grp)
    default_config="configs/r2e_gym_qwen3_14b_mcore_npu_4node32_grp.yaml"
    ;;
  no_grp)
    default_config="configs/r2e_gym_qwen3_14b_mcore_npu_4node32_no_grp.yaml"
    ;;
  *)
    echo "ARM must be grp or no_grp" >&2
    exit 2
    ;;
esac

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-/opt/libra/runtime_sources/RL_Framework_NPU}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-/opt/libra/runtime_sources}"
rollout_project_dir="${ROLLOUT_PROJECT_DIR:-$runtime_project_dir}"
config_path="${CONFIG_PATH:-$default_config}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_4node32_grp_ab}"
run_name="${RUN_NAME:-formal_4node32_${ARM}_$(date +%Y%m%d_%H%M%S)}"
run_dir="${run_root}/${run_name}"
master_port="${MASTER_PORT:-30740}"
config_python="${CONFIG_PYTHON:-/opt/libra/envs/rl_framework_py310/bin/python}"
vllm_venv_dir="${VLLM_VENV_DIR:-/opt/libra/envs/vllm_ascend}"
model_path="${MODEL_PATH:-/opt/libra/models/Qwen3-14B}"

read -r -a available_hosts <<< "${AVAILABLE_HOSTS:-}"
[[ "${#available_hosts[@]}" -eq 4 ]] || {
  echo "AVAILABLE_HOSTS must contain exactly four distinct hosts" >&2
  exit 2
}
declare -A seen_hosts=()
for host in "${available_hosts[@]}"; do
  [[ -z "${seen_hosts[$host]:-}" ]] || {
    echo "duplicate host: $host" >&2
    exit 2
  }
  seen_hosts[$host]=1
done

mkdir -p "$run_dir" "${run_root}/rollout_weight_sync"
printf '%s\n' "$run_dir" >"${run_root}/current_${ARM}_run"

planned_config="${run_dir}/grp_initial_config.yaml"
placement_json="${run_dir}/initial_placement.json"
if [[ "$ARM" == "grp" ]]; then
  grp_history_args=()
  if [[ "${GRP_STARTUP_PROFILE_ENABLED:-1}" == "1" ]]; then
    profile_dataset_jsonl="${GRP_PROFILE_DATASET_JSONL:-$runtime_project_dir/data/r2e_gym_v1/index.jsonl}"
    profile_cache_dir="${GRP_STARTUP_PROFILE_CACHE_DIR:-${GRP_PROFILE_CACHE_DIR:-$run_root/startup_profile_cache}}"
    profile_jsonl="${GRP_STARTUP_PROFILE_JSONL:-${GRP_PROFILE_JSONL:-$profile_cache_dir/grp_startup_profile.jsonl}}"
    profile_summary_json="${GRP_STARTUP_PROFILE_SUMMARY_JSON:-${GRP_PROFILE_SUMMARY_JSON:-$profile_cache_dir/grp_startup_profile_summary.json}}"
    profile_args=(
      --config "$runtime_project_dir/$config_path"
      --dataset-jsonl "$profile_dataset_jsonl"
      --output-jsonl "$profile_jsonl"
      --summary-json "$profile_summary_json"
      --reuse-existing
    )
    [[ -z "${GRP_PROFILE_SAMPLE_SIZE:-}" ]] || profile_args+=(--sample-size "$GRP_PROFILE_SAMPLE_SIZE")
    [[ -z "${GRP_PROFILE_STRATEGY:-}" ]] || profile_args+=(--strategy "$GRP_PROFILE_STRATEGY")
    [[ -z "${GRP_PROFILE_SEED:-}" ]] || profile_args+=(--seed "$GRP_PROFILE_SEED")
    [[ -z "${GRP_PROFILE_SAMPLES_PER_PROMPT:-}" ]] || profile_args+=(--samples-per-prompt "$GRP_PROFILE_SAMPLES_PER_PROMPT")
    PYTHONPATH="$runtime_pythonpath" "$config_python" \
      "$project_dir/scripts/collect_r2e_grp_profile.py" \
      "${profile_args[@]}"
    grp_history_args=(--history-jsonl "$profile_jsonl")
  elif [[ "${GRP_ALLOW_SYNTHETIC_FALLBACK:-0}" == "1" ]]; then
    grp_history_args=(--allow-synthetic-fallback)
  else
    echo "GRP startup profile is disabled and synthetic fallback is not allowed" >&2
    exit 2
  fi
  placement_args=()
  for host in "${available_hosts[@]}"; do placement_args+=(--host "$host"); done
  PYTHONPATH="$runtime_pythonpath" "$config_python" \
    "$project_dir/scripts/plan_initial_grp_placement.py" \
    --config "$runtime_project_dir/$config_path" \
    --output-config "$planned_config" \
    --placement-json "$placement_json" \
    --gpus-per-host 8 \
    "${grp_history_args[@]}" \
    "${placement_args[@]}"
  readarray -t train_hosts < <(
    "$config_python" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["train_hosts"], sep="\n")' "$placement_json"
  )
  readarray -t rollout_hosts < <(
    "$config_python" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["rollout_hosts"], sep="\n")' "$placement_json"
  )
  placement_template="$planned_config"
else
  train_hosts=("${available_hosts[0]}" "${available_hosts[1]}")
  rollout_hosts=("${available_hosts[2]}" "${available_hosts[3]}")
  placement_template="$runtime_project_dir/$config_path"
  "$config_python" -c \
    'import json,sys; json.dump({"strategy":"fixed_equal","gpus_per_host":8,"train_gpus":16,"rollout_gpus":16,"train_hosts":sys.argv[2].split(","),"rollout_hosts":sys.argv[3].split(",")},open(sys.argv[1],"w"),indent=2)' \
    "$placement_json" "$(IFS=,; echo "${train_hosts[*]}")" "$(IFS=,; echo "${rollout_hosts[*]}")"
fi

[[ "${#train_hosts[@]}" -ge 1 && "${#rollout_hosts[@]}" -ge 1 ]] || {
  echo "each arm needs at least one train and one rollout host" >&2
  exit 2
}
master_addr="${MASTER_ADDR:-${train_hosts[0]}}"
gradient_port="${GRADIENT_SERVER_PORT:-$((master_port + 100))}"

effective_config="${run_dir}/effective_config.yaml"
materialize_args=(
  --template "$placement_template"
  --output "$effective_config"
  --master-addr "$master_addr"
  --master-port "$master_port"
  --run-root "$run_root"
  --gradient-port "$gradient_port"
)
for host in "${rollout_hosts[@]}"; do materialize_args+=(--rollout-host "$host"); done
"$config_python" "$project_dir/scripts/materialize_6node48_config.py" "${materialize_args[@]}"

rollout_tp_pattern="$($config_python -c \
  'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(",".join(map(str,c["global_resource_planner"]["rollout_node_tp_pattern"])))' \
  "$effective_config")"
IFS=',' read -r -a rollout_tp_degrees <<< "$rollout_tp_pattern"
rollout_ports=()
for index in "${!rollout_tp_degrees[@]}"; do rollout_ports+=("$((8000 + index))"); done

cleanup_rollout() {
  local host node_id
  for host in "${rollout_hosts[@]}"; do
    node_id="n${host##*.}"
    INTERNAL_SSH_TIMEOUT=30 "$project_dir/scripts/internal_ssh.sh" "$host" -- \
      "cd '$rollout_project_dir' && PROJECT_DIR='$rollout_project_dir' VLLM_VENV_DIR='$vllm_venv_dir' MODEL_PATH='$model_path' RUN_ROOT='$run_root' ROLLOUT_TP_PATTERN='$rollout_tp_pattern' bash scripts/start_r2e_rollout_pool_npu.sh stop '$node_id'" || true
  done
}
trap cleanup_rollout EXIT INT TERM

# Recheck all four nodes immediately before launching.  Never displace a job
# that arrived while the outer capacity watcher was selecting hosts.
export INTERNAL_SSH_TIMEOUT=20
for host in "${available_hosts[@]}"; do
  output="$($project_dir/scripts/internal_ssh.sh "$host" -- \
    "npu-smi info | grep -c -F 'No running processes found in NPU'; true" 2>&1)"
  idle="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
  [[ "$idle" == "8" ]] || {
    echo "host is no longer fully idle: $host idle=${idle:-unreachable}" >&2
    exit 3
  }
done

for host in "${train_hosts[@]}"; do
  output="$($project_dir/scripts/internal_ssh.sh "$host" -- \
    "test -x /opt/libra/envs/rl_mindspeed/bin/python && test -f '$effective_config' && test -f '$runtime_project_dir/examples/r2e_gym_async_rl.py' && echo READY=1 || echo READY=0" 2>&1)"
  grep -q 'READY=1' <<< "$output" || {
    echo "training runtime preflight failed: $host" >&2
    exit 4
  }
done
for host in "${rollout_hosts[@]}"; do
  output="$($project_dir/scripts/internal_ssh.sh "$host" -- \
    "test -x /opt/libra/envs/vllm_ascend/bin/python && test -f /opt/libra/models/Qwen3-14B/config.json && test -f '$rollout_project_dir/scripts/start_r2e_rollout_pool_npu.sh' && echo READY=1 || echo READY=0" 2>&1)"
  grep -q 'READY=1' <<< "$output" || {
    echo "rollout runtime preflight failed: $host" >&2
    exit 4
  }
done

find "${run_root}/rollout_weight_sync" -maxdepth 1 -type f \
  \( -name 'reload_request.json' -o -name 'ack_*.json' -o -name 'error_*.json' \) -delete

for host in "${rollout_hosts[@]}"; do
  node_id="n${host##*.}"
  "$project_dir/scripts/internal_ssh.sh" "$host" -- \
    "cd '$rollout_project_dir' && PROJECT_DIR='$rollout_project_dir' VLLM_VENV_DIR='$vllm_venv_dir' MODEL_PATH='$model_path' RUN_ROOT='$run_root' ROLLOUT_TP_PATTERN='$rollout_tp_pattern' bash scripts/start_r2e_rollout_pool_npu.sh start '$node_id'"
done

for host in "${rollout_hosts[@]}"; do
  for port in "${rollout_ports[@]}"; do
    ready=0
    for _ in $(seq 1 360); do
      if curl -fsS --max-time 5 "http://${host}:${port}/health" >/dev/null; then
        ready=1
        break
      fi
      sleep 5
    done
    [[ "$ready" -eq 1 ]] || {
      echo "rollout endpoint failed: ${host}:${port}" >&2
      exit 5
    }
  done
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
--nnodes=${#train_hosts[@]} --nproc_per_node=8 --node_rank=${node_rank} \
--master_addr=${master_addr} --master_port=${master_port} \
examples/r2e_gym_async_rl.py --config ${effective_config}"
  "$project_dir/scripts/internal_ssh.sh" "$target" -- "$remote_cmd" \
    >"${run_dir}/driver_node_${node_rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
echo "RUN_DIR=$run_dir ARM=$ARM STATUS=$status"
exit "$status"

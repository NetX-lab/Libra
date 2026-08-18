#!/usr/bin/env bash
# Run GRP and fixed 24/24 control arms serially on the same six NPU hosts.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-$project_dir}"
package_root="${PACKAGE_ROOT:-$(dirname "$project_dir")}"
config_python="${CONFIG_PYTHON:-python}"
train_python="${TRAIN_PYTHON:-python}"
run_root="${RUN_ROOT:-./runs/r2e_6node48_grp_vs_no_grp_equal}"
profile_jsonl="${GRP_PROFILE_JSONL:-$run_root/startup_profile_cache/grp_startup_profile.jsonl}"
hccl_if_base_port="${HCCL_IF_BASE_PORT:-52000}"
available_hosts_value="${AVAILABLE_HOSTS:?set AVAILABLE_HOSTS to six selected NPU hosts}"
read -r -a hosts <<< "${available_hosts_value//,/ }"
[[ "${#hosts[@]}" -eq 6 ]] || { echo "AVAILABLE_HOSTS must contain six hosts" >&2; exit 2; }

mkdir -p "$run_root" "$package_root"
ln -sfn "$runtime_project_dir" "${package_root}/RL_Framework"
printf '%s\n' "${hosts[*]}" >"${run_root}/selected_hosts.txt"

active_placement=""
current_run_root="$run_root"
driver_pids=()

check_idle_hosts() {
  local host output idle
  export INTERNAL_SSH_TIMEOUT=20
  for host in "${hosts[@]}"; do
    output="$($project_dir/scripts/internal_ssh.sh "$host" -- \
      "npu-smi info | grep -c -F 'No running processes found in NPU'; true" 2>&1)"
    idle="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
    [[ "$idle" == "8" ]] || { echo "host is not fully idle: $host idle=${idle:-unreachable}" >&2; return 1; }
  done
}

cleanup_arm() {
  local placement="${1:-$active_placement}" pid host
  for pid in "${driver_pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  [[ -n "$placement" ]] || return 0
  for host in "${hosts[@]}"; do
    $project_dir/scripts/internal_ssh.sh "$host" -- \
      "cd '$runtime_project_dir' && PROJECT_DIR='$runtime_project_dir' RUN_ROOT='$current_run_root' bash scripts/start_r2e_rollout_manifest_npu.sh stop '$placement' '$host'" || true
  done
  driver_pids=()
  active_placement=""
}

trap 'cleanup_arm' EXIT
trap 'cleanup_arm; exit 143' INT TERM

run_arm() {
  local mode="$1" config_path="$2" run_name="$3" master_port="$4"
  local run_dir="${run_root}/${run_name}"
  local effective_config="${run_dir}/effective_config.yaml"
  local placement="${run_dir}/device_placement.json"
  local host rank device instance_host instance_port status=0
  mkdir -p "$run_dir"
  current_run_root="$run_dir"
  driver_pids=()

  check_idle_hosts
  planner_args=(
    --mode "$mode"
    --config "$runtime_project_dir/$config_path"
    --output-config "$effective_config"
    --placement-json "$placement"
    --run-root "$run_dir"
    --master-port "$master_port"
  )
  for host in "${hosts[@]}"; do planner_args+=(--host "$host"); done
  [[ "$mode" != "grp" ]] || planner_args+=(--history-jsonl "$profile_jsonl")
  PYTHONPATH="$package_root" "$config_python" \
    "$runtime_project_dir/scripts/plan_unrestricted_grp_device_placement.py" \
    "${planner_args[@]}"

  PYTHONPATH="$package_root" "$config_python" -c \
    'import sys; from RL_Framework.config import AsyncRLConfig; c=AsyncRLConfig.from_yaml(sys.argv[1]); print(f"validated train={c.train_gpus} rollout={c.rollout_gpus} tp={c.train_tp_size} pp={c.train_pp_size} dp={c.train_dp_size}")' \
    "$effective_config"
  [[ "${PREFLIGHT_ONLY:-0}" != "1" ]] || return 0

  active_placement="$placement"
  find "${run_dir}/rollout_weight_sync" -maxdepth 1 -type f \
    \( -name 'reload_request.json' -o -name 'ack_*.json' -o -name 'error_*.json' \) \
    -delete 2>/dev/null || true

  for host in "${hosts[@]}"; do
    $project_dir/scripts/internal_ssh.sh "$host" -- \
      "cd '$runtime_project_dir' && PROJECT_DIR='$runtime_project_dir' RUN_ROOT='$run_dir' MODEL_PATH='${MODEL_PATH:-./models/Qwen3-14B}' bash scripts/start_r2e_rollout_manifest_npu.sh start '$placement' '$host'"
  done
  while IFS=$'\t' read -r instance_host instance_port; do
    ready=0
    for _ in $(seq 1 360); do
      if curl -fsS --max-time 5 "http://${instance_host}:${instance_port}/health" >/dev/null; then ready=1; break; fi
      sleep 5
    done
    [[ "$ready" -eq 1 ]] || { echo "rollout endpoint failed: ${instance_host}:${instance_port}" >&2; return 4; }
  done < <("$config_python" -c 'import json,sys; p=json.load(open(sys.argv[1])); [print(i["host"],i["port"],sep="\t") for i in p["rollout_instances"]]' "$placement")

  master_addr="$($config_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_devices"][0]["host"])' "$placement")"
  world_size="$($config_python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["train_devices"]))' "$placement")"
  export INTERNAL_SSH_TIMEOUT=-1
  while IFS=$'\t' read -r rank host device; do
    remote_cmd="source ${ASCEND_SET_ENV:?set ASCEND_SET_ENV}; \
cd '$runtime_project_dir'; \
export PYTHONPATH='$package_root' PYTHONDONTWRITEBYTECODE=1 DEVICE_BACKEND=npu DIST_BACKEND=hccl \
ASCEND_RT_VISIBLE_DEVICES='$device' RANK='$rank' WORLD_SIZE='$world_size' LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
MASTER_ADDR='$master_addr' MASTER_PORT='$master_port' GLOO_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0 \
HCCL_IF_BASE_PORT='$hccl_if_base_port' HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 TORCH_DISTRIBUTED_TIMEOUT=3600 \
VLLM_REQUEST_RETRIES='${VLLM_REQUEST_RETRIES:-0}' \
TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled RL_TRAIN_PHASE_TRACE=1 OMP_NUM_THREADS=1 \
MCORE_MOE_GROUPED_GEMM=0 JOB_ID='$run_name' R2E_GYM_INDEX='$runtime_project_dir/data/r2e_gym_v1/index.jsonl'; \
exec '$train_python' examples/r2e_gym_async_rl.py --config '$effective_config'"
    setsid $project_dir/scripts/internal_ssh.sh "$host" -- "$remote_cmd" \
      >"${run_dir}/driver_rank_${rank}.log" 2>&1 &
    driver_pids+=("$!")
  done < <("$config_python" -c 'import json,sys; p=json.load(open(sys.argv[1])); [print(i["rank"],i["host"],i["device"],sep="\t") for i in p["train_devices"]]' "$placement")

  for pid in "${driver_pids[@]}"; do wait "$pid" || status=$?; done
  driver_pids=()
  cleanup_arm "$placement"
  echo "RUN_DIR=$run_dir MODE=$mode STATUS=$status"
  return "$status"
}

timestamp="$(date +%Y%m%d_%H%M%S)"
grp_name="formal_6node48_grp_unrestricted_${timestamp}"
no_grp_name="formal_6node48_no_grp_equal_${timestamp}"

case "${RUN_ONLY:-both}" in
  both)
    run_arm grp configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_grp.yaml "$grp_name" 31240
    run_arm fixed configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_no_grp.yaml "$no_grp_name" 31260
    ;;
  fixed)
    grp_name="${GRP_REFERENCE_RUN_NAME:?set GRP_REFERENCE_RUN_NAME for RUN_ONLY=fixed}"
    no_grp_name="${FIXED_RUN_NAME:-$no_grp_name}"
    run_arm fixed "${FIXED_CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_no_grp.yaml}" "$no_grp_name" "${FIXED_MASTER_PORT:-31360}"
    ;;
  *)
    echo "RUN_ONLY must be 'both' or 'fixed'" >&2
    exit 2
    ;;
esac

[[ "${PREFLIGHT_ONLY:-0}" != "1" ]] || exit 0
report="${run_root}/grp_vs_no_grp_${timestamp}.md"
PYTHONPATH="$package_root" "$config_python" "$runtime_project_dir/scripts/analyze_libra_experiment.py" \
  --run "no-GRP-equal=${run_root}/${no_grp_name}" \
  --run "GRP=${run_root}/${grp_name}" \
  --output "$report"
echo "STATUS=complete GRP_RUN=$grp_name NO_GRP_RUN=$no_grp_name REPORT=$report"

#!/usr/bin/env bash
set -euo pipefail

run_name="${1:?usage: start_formal_run_remote.sh RUN_NAME MASTER_PORT}"
master_port="${2:?master port required}"
project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48}"

if [[ ! "${run_name}" =~ ^formal_[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid run name: ${run_name}" >&2
  exit 2
fi
if [[ ! "${master_port}" =~ ^[0-9]+$ ]] || ((master_port < 1024 || master_port > 65535)); then
  echo "invalid master port: ${master_port}" >&2
  exit 2
fi

run_dir="${run_root}/${run_name}"
launcher_log="${run_dir}_launcher.log"
mkdir -p "${run_root}"

if [[ -e "${run_dir}" || -e "${launcher_log}" ]]; then
  echo "refusing to overwrite existing run artifacts: ${run_dir}" >&2
  exit 3
fi

nohup env \
  NODE_PASSWORD="${NODE_PASSWORD:?set NODE_PASSWORD}" \
  INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}" \
  RUN_NAME="${run_name}" \
  MASTER_PORT="${master_port}" \
  bash "${project_dir}/scripts/run_6node48_r2e_mcore_npu.sh" \
  >"${launcher_log}" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "${launcher_pid}" >"${run_root}/current_launcher_pid"
printf '%s\n' "${master_port}" >"${run_root}/current_master_port"
printf 'RUN_NAME=%s MASTER_PORT=%s LAUNCHER_PID=%s LAUNCHER_LOG=%s\n' \
  "${run_name}" "${master_port}" "${launcher_pid}" "${launcher_log}"

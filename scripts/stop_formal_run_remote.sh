#!/usr/bin/env bash
set -euo pipefail

launcher_pid="${1:?usage: stop_formal_run_remote.sh LAUNCHER_PID RUN_NAME}"
run_name="${2:?run name required}"
project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48}"
remote_rollout_host="${REMOTE_ROLLOUT_HOST:-192.0.2.22}"
train_hosts=(192.0.2.10 192.0.2.11 192.0.2.12 192.0.2.13)
export INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}"

launcher_command="$(tr '\0' ' ' <"/proc/${launcher_pid}/cmdline" 2>/dev/null || true)"
run_dir="${run_root}/${run_name}"
if [[ -n "${launcher_command}" ]]; then
  if [[ "${launcher_command}" != *"run_6node48_r2e_mcore_npu.sh"* ]]; then
    echo "refusing to stop unverified live launcher pid=${launcher_pid}: ${launcher_command}" >&2
    exit 2
  fi
else
  if [[ ! "${run_name}" =~ ^formal_[A-Za-z0-9_.-]+$ || ! -d "${run_dir}" ]]; then
    echo "refusing orphan cleanup for unverified run=${run_name} run_dir=${run_dir}" >&2
    exit 2
  fi
  echo "launcher pid=${launcher_pid} is gone; using verified run directory and exact JOB_ID cleanup"
fi

remote_cleanup="target_job='${run_name}'; pids=(); for proc in /proc/[0-9]*; do [[ -r \"\$proc/environ\" ]] || continue; if tr '\\0' '\\n' <\"\$proc/environ\" 2>/dev/null | grep -Fxq \"JOB_ID=\$target_job\"; then pids+=(\"\${proc##*/}\"); fi; done; if ((\${#pids[@]})); then kill -TERM \"\${pids[@]}\" 2>/dev/null || true; sleep 8; for pid in \"\${pids[@]}\"; do kill -0 \"\$pid\" 2>/dev/null && kill -KILL \"\$pid\" 2>/dev/null || true; done; fi; printf 'job=%s cleaned_pids=%s\\n' \"\$target_job\" \"\${pids[*]:-none}\""

for host in "${train_hosts[@]}"; do
  echo "cleaning training job on ${host}"
  "${project_dir}/scripts/internal_ssh.sh" "${host}" -- "${remote_cleanup}" || true
done

descendants=()
collect_descendants() {
  local parent="$1" child
  while read -r child; do
    [[ -n "${child}" ]] || continue
    collect_descendants "${child}"
    descendants+=("${child}")
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
}
collect_descendants "${launcher_pid}"
if ((${#descendants[@]})); then
  kill -TERM "${descendants[@]}" 2>/dev/null || true
fi
kill -TERM "${launcher_pid}" 2>/dev/null || true
sleep 5
for pid in "${descendants[@]}" "${launcher_pid}"; do
  kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
done

RUN_ROOT="${run_root}" bash "${project_dir}/scripts/start_r2e_rollout_pool_npu.sh" stop rollout_a || true
ssh -o BatchMode=yes root@"${remote_rollout_host}" \
  "cd '${project_dir}' && RUN_ROOT='${run_root}' bash scripts/start_r2e_rollout_pool_npu.sh stop n17" || true

echo "stopped run=${run_name} launcher_pid=${launcher_pid}"

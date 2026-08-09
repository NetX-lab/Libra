#!/usr/bin/env bash
set -u

run_dir="${1:?usage: watch_formal_run_remote.sh RUN_DIR LAUNCHER_PID [INTERVAL] [ITERATIONS]}"
launcher_pid="${2:?launcher pid required}"
interval="${3:-45}"
iterations="${4:-30}"
run_root="$(dirname "${run_dir}")"

for ((iteration = 1; iteration <= iterations; iteration++)); do
  now="$(date '+%F %T')"
  if ps -p "${launcher_pid}" >/dev/null 2>&1; then
    launcher_state=alive
  else
    launcher_state=dead
  fi
  step="$(grep -hE '^Step [0-9]+/[0-9]+' "${run_dir}"/driver_node_*.log 2>/dev/null | tail -n 1 || true)"
  phase="$(grep -h '\[TrainPhase\]' "${run_dir}"/driver_node_0.log 2>/dev/null | tail -n 1 || true)"
  fatal_count="$(grep -hEi 'Traceback|RuntimeError|ChildFailedError|HCCL.*(error|fail)|ERR[0-9]{5}' "${run_dir}"/driver_node_*.log 2>/dev/null | wc -l)"
  unhealthy=0
  for host in 192.0.2.20 192.0.2.22; do
    for port in 8000 8001 8002 8003; do
      code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "http://${host}:${port}/health" 2>/dev/null || true)"
      [[ "${code}" == 200 ]] || unhealthy=$((unhealthy + 1))
    done
  done
  applied=no
  [[ -f "${run_root}/logs/runtime_reconfiguration/applied.json" ]] && applied=yes
  printf '%s iteration=%s launcher=%s step=%q unhealthy=%s fatal_count=%s applied=%s phase=%q\n' \
    "${now}" "${iteration}" "${launcher_state}" "${step:-none}" "${unhealthy}" \
    "${fatal_count}" "${applied}" "${phase:-none}"

  if [[ "${launcher_state}" == dead || "${fatal_count}" -gt 0 || "${step}" =~ ^Step[[:space:]]+100/100$ ]]; then
    exit 0
  fi
  sleep "${interval}"
done

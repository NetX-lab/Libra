#!/usr/bin/env bash
set -u

run_dir="${1:?usage: monitor_formal_run_compact_remote.sh RUN_DIR [LAUNCHER_PID]}"
launcher_pid="${2:-}"
run_root="$(dirname "${run_dir}")"

date '+%F %T %z'
if [[ -n "${launcher_pid}" ]]; then
  ps -p "${launcher_pid}" -o pid=,etime=,stat=,cmd= || true
fi

printf 'rollout_health'
for host in 192.0.2.20 192.0.2.22; do
  for port in 8000 8001 8002 8003; do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "http://${host}:${port}/health" 2>/dev/null || true)"
    printf ' %s:%s=%s' "${host}" "${port}" "${code:-000}"
  done
done
printf '\n'

for log in "${run_dir}"/driver_node_*.log; do
  [[ -e "${log}" ]] || continue
  stat -c 'driver mtime=%y size=%s file=%n' "${log}"
  grep -E '\[TrainPhase\]|Step [0-9]+|Training complete|Traceback|RuntimeError|ERROR|HCCL.*(error|fail)' "${log}" \
    | tail -n 12 || true
done

echo 'reconfiguration_files'
find "${run_root}/logs/runtime_reconfiguration" -maxdepth 2 -type f \
  -printf '%y size=%s %p\n' 2>/dev/null | sort | tail -n 30 || true

for log in "${run_root}"/rollout_logs/*.log; do
  [[ -e "${log}" ]] || continue
  requests="$(grep -c 'POST /v1/completions' "${log}" 2>/dev/null || true)"
  errors="$(grep -ciE 'Traceback|RuntimeError|ERROR|failed' "${log}" 2>/dev/null || true)"
  stat -c "rollout mtime=%y size=%s requests=${requests} errors=${errors} file=%n" "${log}"
done

echo 'monitor_complete'

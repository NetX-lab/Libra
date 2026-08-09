#!/usr/bin/env bash
set -u

run_dir="${1:?usage: monitor_formal_run_remote.sh RUN_DIR [LAUNCHER_PID]}"
launcher_pid="${2:-}"

date '+%F %T %z'

echo '=== launcher ==='
if [[ -n "${launcher_pid}" ]]; then
  ps -p "${launcher_pid}" -o pid=,etime=,stat=,cmd= || true
fi

echo '=== launcher log ==='
tail -n 120 "${run_dir}_launcher.log" 2>/dev/null || true

echo '=== rollout health ==='
for host in 192.0.2.20 192.0.2.22; do
  for port in 8000 8001 8002 8003; do
    code="$(curl -sS -o /tmp/rl_monitor_health_body -w '%{http_code}' --max-time 3 "http://${host}:${port}/health" 2>/dev/null || true)"
    body="$(tr '\n' ' ' </tmp/rl_monitor_health_body 2>/dev/null | head -c 120 || true)"
    printf '%s:%s http=%s body=%s\n' "${host}" "${port}" "${code:-000}" "${body}"
  done
done

echo '=== driver stats ==='
for log in "${run_dir}"/driver_node_*.log; do
  [[ -e "${log}" ]] || continue
  stat -c '%y size=%s %n' "${log}"
done

echo '=== progress and errors ==='
grep -RniE 'Traceback|RuntimeError|ERROR|HCCL|MegatronCoreTrainEngine|Total steps|resource_planner|rollout_rank_wait|train_step|step[^[:alpha:]_]*[0-9]+' "${run_dir}"/driver_node_*.log 2>/dev/null | tail -n 300 || true

echo '=== driver tails ==='
for log in "${run_dir}"/driver_node_*.log; do
  [[ -e "${log}" ]] || continue
  echo "--- ${log}"
  tail -n 80 "${log}"
done

echo '=== reconfiguration ==='
find /opt/libra/runs/r2e_gym_qwen3_14b_6node48/elastic_training_state -maxdepth 3 -type f \
  -printf '%y size=%s %p\n' 2>/dev/null | sort | tail -n 100 || true

echo '=== rollout log stats ==='
for log in /opt/libra/runs/r2e_gym_qwen3_14b_6node48/rollout_logs/*.log; do
  [[ -e "${log}" ]] || continue
  stat -c '%y size=%s %n' "${log}"
done

echo '=== rollout requests and errors ==='
grep -hEi 'POST|request|completion|Traceback|RuntimeError|ERROR|failed' \
  /opt/libra/runs/r2e_gym_qwen3_14b_6node48/rollout_logs/*.log \
  2>/dev/null | tail -n 200 || true

#!/usr/bin/env bash
# Manage the rollout instances assigned to one host in a device placement JSON.
set -euo pipefail

action="${1:?usage: $0 start|stop PLACEMENT_JSON HOST}"
placement_json="${2:?placement JSON required}"
host="${3:?host required}"
project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_root="${RUN_ROOT:?set RUN_ROOT}"
venv_dir="${VLLM_VENV_DIR:-./venv/npu}"
model_path="${MODEL_PATH:-./models/Qwen3-14B}"
max_model_len="${MAX_MODEL_LEN:-4096}"
control_dir="${run_root}/rollout_weight_sync"
log_dir="${run_root}/rollout_logs"
node_id="n${host##*.}"
pid_dir="${run_root}/rollout_pids/${node_id}"
mkdir -p "$control_dir" "$log_dir" "$pid_dir"

mapfile -t rows < <(
  python -c '
import json,sys
payload=json.load(open(sys.argv[1]))
for item in payload["rollout_instances"]:
    if item["host"] == sys.argv[2]:
        print("\t".join((item["instance_id"],str(item["tp"]),",".join(map(str,item["gpus"])),str(item["port"]))))
' "$placement_json" "$host"
)

stop_one() {
  local instance_id="$1" pid_file="${pid_dir}/$1.pid" pid command
  [[ -f "$pid_file" ]] || return 0
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    command="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "$command" == *"restartable_vllm_server.py"* && "$command" == *"${instance_id}"* ]] || {
      echo "refusing to stop unverified pid=$pid instance=$instance_id" >&2
      return 1
    }
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

for row in "${rows[@]}"; do
  IFS=$'\t' read -r instance_id tp devices port <<<"$row"
  pid_file="${pid_dir}/${instance_id}.pid"
  case "$action" in
    start)
      if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        continue
      fi
      nohup env \
        PROJECT_DIR="$project_dir" \
        VENV_DIR="$venv_dir" \
        SERVED_MODEL_NAME="$model_path" \
        ASCEND_DEVICES="$devices" \
        TP_SIZE="$tp" \
        PORT="$port" \
        MAX_MODEL_LEN="$max_model_len" \
        GPU_MEMORY_UTILIZATION=0.88 \
        ROLLOUT_WEIGHT_SYNC_MODE="${ROLLOUT_WEIGHT_SYNC_MODE:-none}" \
        TOKENIZERS_PARALLELISM=false \
        "$venv_dir/bin/python" \
        "$project_dir/scripts/restartable_vllm_server.py" \
        --instance-id "$instance_id" \
        --control-dir "$control_dir" \
        --health-url "http://127.0.0.1:${port}/health" \
        --initial-model "$model_path" \
        --ready-timeout 1800 \
        -- bash "$project_dir/scripts/run_vllm_ascend_server.sh" __MODEL_PATH__ \
        >"${log_dir}/${instance_id}.log" 2>&1 </dev/null &
      echo "$!" >"$pid_file"
      sleep 1
      ;;
    stop)
      stop_one "$instance_id"
      ;;
    *)
      echo "unknown action: $action" >&2
      exit 2
      ;;
  esac
done

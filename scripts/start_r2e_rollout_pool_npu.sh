#!/usr/bin/env bash
# Manage one 8-NPU Qwen3-14B vLLM pool.  The default profile is the validated
# heterogeneous TP 1+1+2+4 layout; ROLLOUT_TP_PATTERN=2,2,2,2 selects the
# matched homogeneous baseline.
set -euo pipefail

action="${1:?usage: $0 start|stop|status NODE_ID}"
node_id="${2:?usage: $0 start|stop|status NODE_ID}"
project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
venv_dir="${VLLM_VENV_DIR:-/opt/libra/envs/vllm_ascend}"
model_path="${MODEL_PATH:-/opt/libra/models/Qwen3-14B}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48}"
control_dir="${CONTROL_DIR:-${run_root}/rollout_weight_sync}"
log_dir="${ROLLOUT_LOG_DIR:-${run_root}/rollout_logs}"
pid_dir="${run_root}/rollout_pids/${node_id}"

rollout_tp_pattern="${ROLLOUT_TP_PATTERN:-1,1,2,4}"
IFS=',' read -r -a tps <<< "$rollout_tp_pattern"
[[ "${#tps[@]}" -gt 0 ]] || { echo "empty ROLLOUT_TP_PATTERN" >&2; exit 2; }

ids=()
devices=()
ports=()
declare -A tp_occurrences=()
device_offset=0
for index in "${!tps[@]}"; do
    tp="${tps[$index]}"
    [[ "$tp" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid TP degree in ROLLOUT_TP_PATTERN: $tp" >&2
        exit 2
    }
    occurrence="${tp_occurrences[$tp]:-0}"
    if [[ "$rollout_tp_pattern" == "1,1,2,4" ]]; then
        case "${tp}:${occurrence}" in
            1:0) suffix="short_tp1_0" ;;
            1:1) suffix="short_tp1_1" ;;
            2:0) suffix="medium_tp2" ;;
            4:0) suffix="long_tp4" ;;
            *) echo "invalid validated heterogeneous profile" >&2; exit 2 ;;
        esac
    else
        suffix="tp${tp}_${occurrence}"
    fi
    tp_occurrences[$tp]=$((occurrence + 1))
    ids+=("${node_id}_${suffix}")
    device_list=""
    for ((device=device_offset; device<device_offset+tp; device++)); do
        [[ -z "$device_list" ]] || device_list+=","
        device_list+="$device"
    done
    devices+=("$device_list")
    ports+=("$((8000 + index))")
    device_offset=$((device_offset + tp))
done
[[ "$device_offset" -eq 8 ]] || {
    echo "ROLLOUT_TP_PATTERN must consume exactly 8 NPUs: $rollout_tp_pattern" >&2
    exit 2
}

mkdir -p "$control_dir" "$log_dir" "$pid_dir"

start_one() {
    local index="$1"
    local instance_id="${ids[$index]}"
    local pid_file="${pid_dir}/${instance_id}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "already running: $instance_id pid=$(cat "$pid_file")"
        return
    fi
    nohup env \
        PROJECT_DIR="$project_dir" \
        VENV_DIR="$venv_dir" \
        SERVED_MODEL_NAME="$model_path" \
        ASCEND_DEVICES="${devices[$index]}" \
        TP_SIZE="${tps[$index]}" \
        PORT="${ports[$index]}" \
        MAX_MODEL_LEN=32768 \
        GPU_MEMORY_UTILIZATION=0.88 \
        TOKENIZERS_PARALLELISM=false \
        "$venv_dir/bin/python" \
        "$project_dir/scripts/restartable_vllm_server.py" \
        --instance-id "$instance_id" \
        --control-dir "$control_dir" \
        --health-url "http://127.0.0.1:${ports[$index]}/health" \
        --initial-model "$model_path" \
        --ready-timeout 1800 \
        -- bash "$project_dir/scripts/run_vllm_ascend_server.sh" __MODEL_PATH__ \
        >"${log_dir}/${instance_id}.log" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "started: $instance_id pid=$! devices=${devices[$index]} tp=${tps[$index]}"
}

stop_one() {
    local index="$1"
    local instance_id="${ids[$index]}"
    local pid_file="${pid_dir}/${instance_id}.pid"
    [[ -f "$pid_file" ]] || return
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        local command
        command="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        if [[ "$command" != *"restartable_vllm_server.py"* || "$command" != *"${instance_id}"* ]]; then
            echo "refusing to stop unverified pid=$pid for $instance_id" >&2
            return 1
        fi
        # The restartable wrapper launches the vLLM API server, EngineCore and
        # TP workers as descendants.  Terminating only the wrapper can orphan
        # those processes and leave all NPUs allocated.  Snapshot the complete
        # descendant tree before signalling the root, then terminate leaves
        # first so every process remains attributable to this verified wrapper.
        local descendants=()
        collect_descendants() {
            local parent="$1" child
            while read -r child; do
                [[ -n "$child" ]] || continue
                collect_descendants "$child"
                descendants+=("$child")
            done < <(pgrep -P "$parent" 2>/dev/null || true)
        }
        collect_descendants "$pid"
        if ((${#descendants[@]})); then
            kill -TERM "${descendants[@]}" 2>/dev/null || true
        fi
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 45); do
            local alive=0 target
            for target in "$pid" "${descendants[@]}"; do
                if kill -0 "$target" 2>/dev/null; then
                    alive=1
                    break
                fi
            done
            [[ "$alive" -eq 0 ]] && break
            sleep 1
        done
        local target
        for target in "$pid" "${descendants[@]}"; do
            kill -0 "$target" 2>/dev/null && kill -KILL "$target" 2>/dev/null || true
        done
    fi
    rm -f "$pid_file"
    echo "stopped: $instance_id"
}

case "$action" in
    start)
        # Stagger wrapper creation slightly: launching four TP2 vLLM process
        # trees in the same instant can exhaust transient host resources and
        # make the final wrapper exit before it writes a log.
        for index in "${!ids[@]}"; do
            start_one "$index"
            sleep 1
        done
        # Verify the long-lived restartable wrappers, and idempotently retry
        # only missing instances. Endpoint readiness is checked by the
        # cluster launcher after this local process-level guard.
        for _ in 1 2 3; do
            sleep 5
            missing=0
            for index in "${!ids[@]}"; do
                pid_file="${pid_dir}/${ids[$index]}.pid"
                if [[ ! -f "$pid_file" ]] || ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
                    missing=1
                    start_one "$index"
                fi
            done
            [[ "$missing" -eq 0 ]] && break
        done
        ;;
    stop)
        for index in "${!ids[@]}"; do stop_one "$index"; done
        ;;
    status)
        for index in "${!ids[@]}"; do
            pid_file="${pid_dir}/${ids[$index]}.pid"
            if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
                echo "running ${ids[$index]} pid=$(cat "$pid_file")"
            else
                echo "stopped ${ids[$index]}"
            fi
        done
        ;;
    *)
        echo "unknown action: $action" >&2
        exit 2
        ;;
esac

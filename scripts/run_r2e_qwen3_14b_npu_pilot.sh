#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_VENV_DIR="${TRAIN_VENV_DIR:-${PROJECT_DIR}/.venv-npu}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-/opt/libra/envs/vllm_ascend}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
MODEL_PATH="${MODEL_PATH:-/opt/libra/models/Qwen3-14B}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/r2e_gym_cmlfq_qwen3_14b_npu_pilot.yaml}"
CONTROL_DIR="${CONTROL_DIR:-${PROJECT_DIR}/logs/r2e_gym_qwen3_14b_npu_pilot/rollout_weight_sync}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/r2e_gym_qwen3_14b_npu_pilot/job_manual}"
MAX_WAIT="${MAX_WAIT:-1200}"

mkdir -p "$LOG_DIR" "$CONTROL_DIR"
rm -f \
    "$CONTROL_DIR"/reload_request.json \
    "$CONTROL_DIR"/ack_*.json \
    "$CONTROL_DIR"/error_*.json

PACKAGE_LINK="$(dirname "$PROJECT_DIR")/RL_Framework"
if [ ! -e "$PACKAGE_LINK" ]; then
    ln -s "$PROJECT_DIR" "$PACKAGE_LINK"
elif [ "$(readlink -f "$PACKAGE_LINK" 2>/dev/null || realpath "$PACKAGE_LINK")" != "$PROJECT_DIR" ]; then
    echo "ERROR: $PACKAGE_LINK exists but does not point to $PROJECT_DIR" >&2
    exit 1
fi

VLLM_PIDS=()

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for pid in "${VLLM_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${VLLM_PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

source_env() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        had_nounset=0
        case "$-" in
            *u*)
                had_nounset=1
                set +u
                ;;
        esac
        # shellcheck disable=SC1090
        source "$env_file"
        if [ "$had_nounset" -eq 1 ]; then
            set -u
        fi
    fi
}

wait_for_health() {
    local url="$1"
    local name="$2"
    local elapsed=0
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "READY: $name at $url (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: $name did not become ready: $url" >&2
    return 1
}

start_vllm() {
    local instance_id="$1"
    local devices="$2"
    local tp="$3"
    local port="$4"
    local log_file="$LOG_DIR/vllm_${instance_id}.log"

    echo "Starting rollout $instance_id devices=$devices tp=$tp port=$port"
    (
        export PROJECT_DIR="$PROJECT_DIR"
        export VENV_DIR="$VLLM_VENV_DIR"
        export SERVED_MODEL_NAME="$MODEL_PATH"
        export ASCEND_DEVICES="$devices"
        export TP_SIZE="$tp"
        export PORT="$port"
        export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
        export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
        export TOKENIZERS_PARALLELISM=false
        cd "$PROJECT_DIR"
        exec "$VLLM_VENV_DIR/bin/python" "$PROJECT_DIR/scripts/restartable_vllm_server.py" \
            --instance-id "$instance_id" \
            --control-dir "$CONTROL_DIR" \
            --health-url "http://127.0.0.1:${port}/health" \
            --initial-model "$MODEL_PATH" \
            --ready-timeout "$MAX_WAIT" \
            -- bash "$PROJECT_DIR/scripts/run_vllm_ascend_server.sh" __MODEL_PATH__
    ) >"$log_file" 2>&1 &
    VLLM_PIDS+=("$!")
}

run_eval() {
    local name="$1"
    local output="$LOG_DIR/eval_${name}.json"
    echo "Running R2E eval: $name"
    (
        source "$TRAIN_VENV_DIR/bin/activate"
        source_env "$ASCEND_SET_ENV"
        export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
        export DEVICE_BACKEND=npu
        export R2E_GYM_INDEX="${PROJECT_DIR}/data/r2e_gym_v1/index.jsonl"
        # Leave workflow limits unset unless the launcher explicitly overrides
        # them.  The Python config is the source of truth for reproducible
        # multi-turn experiments.
        export R2E_EVAL_MAX_SAMPLES="${R2E_EVAL_MAX_SAMPLES:-4}"
        export R2E_EVAL_CONCURRENCY="${R2E_EVAL_CONCURRENCY:-1}"
        export R2E_EVAL_MULTI_TURN="${R2E_EVAL_MULTI_TURN:-0}"
        export R2E_EVAL_RECORD_SAMPLES="${R2E_EVAL_RECORD_SAMPLES:-4}"
        export R2E_EVAL_OUTPUT="$output"
        cd "$PROJECT_DIR"
        python examples/r2e_gym_eval.py --config "$CONFIG_PATH"
    ) | tee "$LOG_DIR/eval_${name}.log"
}

compare_eval() {
    python - "$LOG_DIR/eval_baseline.json" "$LOG_DIR/eval_post.json" <<'PY'
import json, sys
before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))
summary = {
    "baseline_accuracy": before.get("eval_accuracy"),
    "post_accuracy": after.get("eval_accuracy"),
    "accuracy_delta": (after.get("eval_accuracy") or 0) - (before.get("eval_accuracy") or 0),
    "baseline_reward_mean": before.get("eval_reward_mean"),
    "post_reward_mean": after.get("eval_reward_mean"),
    "reward_mean_delta": (after.get("eval_reward_mean") or 0) - (before.get("eval_reward_mean") or 0),
    "samples": after.get("eval_samples"),
}
print(json.dumps(summary, indent=2))
PY
}

echo "R2E-Gym Qwen3-14B NPU pilot"
echo "Project: $PROJECT_DIR"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_PATH"
echo "Logs: $LOG_DIR"
echo "Current NPU processes:"
npu-smi info | sed -n '1,120p' || true

cd "$PROJECT_DIR"
source "$TRAIN_VENV_DIR/bin/activate"
source_env "$ASCEND_SET_ENV"
export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
python examples/test_r2e_gym.py
python examples/test_r2e_cmlfq_flow.py

start_vllm short_tp1_0 4 1 8000
start_vllm short_tp1_1 5 1 8001
start_vllm long_tp2 6,7 2 8002
wait_for_health http://127.0.0.1:8000/health short_tp1_0
wait_for_health http://127.0.0.1:8001/health short_tp1_1
wait_for_health http://127.0.0.1:8002/health long_tp2

run_eval baseline

echo "Starting torchrun training on NPU 0-3"
(
    source "$TRAIN_VENV_DIR/bin/activate"
    source_env "$ASCEND_SET_ENV"
    export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
    export ASCEND_RT_VISIBLE_DEVICES="${TRAIN_ASCEND_DEVICES:-0,1,2,3}"
    export DEVICE_BACKEND=npu
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    export TOKENIZERS_PARALLELISM=false
    export FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-1}"
    export R2E_GYM_INDEX="${PROJECT_DIR}/data/r2e_gym_v1/index.jsonl"
    export R2E_EVAL_MAX_SAMPLES="${R2E_EVAL_MAX_SAMPLES:-4}"
    export R2E_EVAL_CONCURRENCY="${R2E_EVAL_CONCURRENCY:-1}"
    export R2E_EVAL_MULTI_TURN="${R2E_EVAL_MULTI_TURN:-0}"
    export RL_TRAIN_PHASE_TRACE="${RL_TRAIN_PHASE_TRACE:-1}"
    cd "$PROJECT_DIR"
    torchrun --standalone --nproc_per_node=4 \
        examples/r2e_gym_async_rl.py --config "$CONFIG_PATH"
) 2>&1 | tee "$LOG_DIR/train.log"

run_eval post
compare_eval | tee "$LOG_DIR/eval_comparison.json"

echo "Pilot complete. Logs: $LOG_DIR"

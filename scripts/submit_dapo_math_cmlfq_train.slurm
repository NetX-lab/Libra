#!/bin/bash
#SBATCH -J dapo_math_cmlfq_train
#SBATCH -o logs/dapo_math_cmlfq_train_%j.out
#SBATCH -e logs/dapo_math_cmlfq_train_%j.err
#SBATCH -N 3
#SBATCH -G 12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --time=72:00:00
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/RL_Framework}"
VENV_PATH="${VENV_PATH:-/path/to/rl_framework_env/bin/activate}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
DAPO_MATH_ROOT="${DAPO_MATH_ROOT:-$PROJECT_DIR/data/DAPO-Math-17K}"
DAPO_MATH_SPLIT="${DAPO_MATH_SPLIT:-all}"
DAPO_MAX_TURNS="${DAPO_MAX_TURNS:-2}"
DAPO_MIN_ROWS="${DAPO_MIN_ROWS:-1000}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-32768}"
EVAL_ACCURACY_THRESHOLD="${EVAL_ACCURACY_THRESHOLD:-1.0}"
EVAL_WAIT_TIMEOUT="${EVAL_WAIT_TIMEOUT:-172800}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_WAIT="${MAX_WAIT:-900}"

source "$VENV_PATH"
cd "$PROJECT_DIR"

mkdir -p logs
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="$PROJECT_DIR/logs/dapo_math_cmlfq/job_${JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${PROJECT_DIR}/wandb_dapo_math_cmlfq"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#NODES[@]}" -lt 3 ]; then
    echo "ERROR: DAPO-Math C-MLFQ training requires 3 nodes; got ${#NODES[@]}" >&2
    exit 2
fi

TRAIN_NODE="${NODES[0]}"
ROLLOUT_NODE_A="${NODES[1]}"
ROLLOUT_NODE_B="${NODES[2]}"
MASTER_ADDR="$TRAIN_NODE"
RUNTIME_CONFIG="$PROJECT_DIR/configs/.dapo_math_cmlfq_${JOB_ID}.yaml"

echo "============================================================"
echo "DAPO-Math-17K full RL post-training with C-MLFQ"
echo "============================================================"
echo "Job: $JOB_ID"
echo "Model: $MODEL_PATH"
echo "Data root: $DAPO_MATH_ROOT"
echo "Data split: $DAPO_MATH_SPLIT"
echo "Train node: $TRAIN_NODE (4 GPUs)"
echo "Rollout node A: $ROLLOUT_NODE_A (TP=4 long bucket)"
echo "Rollout node B: $ROLLOUT_NODE_B (2 x TP=2 short/medium buckets)"
echo "Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Eval samples: ${EVAL_MAX_SAMPLES:-0} (0 means all)"
echo "Logs: $JOB_LOG_DIR"
python3 -c 'import torch, transformers, vllm; print("torch", torch.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)'

VLLM_STEP_PIDS=()
declare -A VLLM_PID_BY_NAME=()

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    echo "[cleanup] stopping vLLM processes..."
    for pid in "${VLLM_STEP_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for node in "$ROLLOUT_NODE_A" "$ROLLOUT_NODE_B"; do
        for port in "$VLLM_BASE_PORT" "$((VLLM_BASE_PORT + 1))" "$((VLLM_BASE_PORT + 2))"; do
            srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
                bash -lc "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${port}' 2>/dev/null || true" \
                >/dev/null 2>&1 || true
        done
    done
    echo "[cleanup] complete, exit=$exit_code"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "[1/7] Writing runtime config"
export PROJECT_DIR MODEL_PATH TOTAL_STEPS MAX_MODEL_LEN MAX_NEW_TOKENS EVAL_INTERVAL
export TRAIN_BATCH_SIZE TRAIN_MICRO_BATCH_SIZE VLLM_BASE_PORT GPU_MEMORY_UTILIZATION JOB_ID
python3 - <<'PY'
import os
from pathlib import Path

import yaml

project = Path(os.environ["PROJECT_DIR"])
base = project / "configs" / "dapo_math_cmlfq_qwen3_4b.yaml"
config = {
    "base_config": str(base),
    "model_path": os.environ["MODEL_PATH"],
    "tokenizer_path": os.environ["MODEL_PATH"],
    "batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["TRAIN_MICRO_BATCH_SIZE"]),
    "total_steps": int(os.environ["TOTAL_STEPS"]),
    "eval_interval": int(os.environ["EVAL_INTERVAL"]),
    "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    "max_seq_length": int(os.environ["MAX_MODEL_LEN"]),
    "sync_path": str(project / "logs" / "dapo_math_cmlfq" / f"weights_{os.environ['JOB_ID']}"),
    "log_dir": str(project / "logs" / "dapo_math_cmlfq"),
    "wandb_run_name": f"dapo-math-cmlfq-{os.environ['JOB_ID']}",
    "heterogeneous_rollout": {
        "vllm_base_port": int(os.environ["VLLM_BASE_PORT"]),
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
        "scheduling": {
            "cmlfq_tree_path": str(project / "logs" / "dapo_math_cmlfq" / f"cmlfq_tree_{os.environ['JOB_ID']}.json")
        },
    },
}
path = project / "configs" / f".dapo_math_cmlfq_{os.environ['JOB_ID']}.yaml"
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(path)
PY

echo "[2/7] Checking DAPO-Math data before reserving rollout services"
export DAPO_MATH_ROOT DAPO_MATH_SPLIT DAPO_MIN_ROWS
python3 examples/test_dapo_math_data.py

start_vllm() {
    local node="$1"
    local tp="$2"
    local port="$3"
    local name="$4"
    local log_file="$JOB_LOG_DIR/vllm_${name}.log"
    echo "Starting $name on $node TP=$tp port=$port"
    srun --exclusive \
        --nodes=1 --ntasks=1 --nodelist="$node" \
        --gpus="$tp" \
        --cpus-per-task=16 \
        bash -lc "
            source '$VENV_PATH'
            export TOKENIZERS_PARALLELISM=false
            export MASTER_ADDR=127.0.0.1
            export MASTER_PORT=$((20000 + port))
            python3 -m vllm.entrypoints.openai.api_server \
                --model '$MODEL_PATH' \
                --host 0.0.0.0 \
                --port '$port' \
                --tensor-parallel-size '$tp' \
                --max-model-len '$MAX_MODEL_LEN' \
                --dtype bfloat16 \
                --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
                --trust-remote-code \
                --enforce-eager \
                --disable-custom-all-reduce
        " >"$log_file" 2>&1 &
    VLLM_STEP_PIDS+=("$!")
    VLLM_PID_BY_NAME["$name"]="$!"
}

wait_for_health() {
    local host="$1"
    local port="$2"
    local name="$3"
    local elapsed=0
    local pid="${VLLM_PID_BY_NAME[$name]}"
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
            echo "READY: $name at $host:$port (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name exited before ready" >&2
            tail -120 "$JOB_LOG_DIR/vllm_${name}.log" || true
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: $name not ready after ${MAX_WAIT}s" >&2
    tail -120 "$JOB_LOG_DIR/vllm_${name}.log" || true
    return 1
}

echo "[3/7] Starting heterogeneous rollout vLLM instances"
start_vllm "$ROLLOUT_NODE_A" 4 "$VLLM_BASE_PORT" "long_tp4"
start_vllm "$ROLLOUT_NODE_B" 2 "$((VLLM_BASE_PORT + 1))" "short_tp2_0"

echo "[4/7] Waiting for initial rollout services"
wait_for_health "$ROLLOUT_NODE_A" "$VLLM_BASE_PORT" "long_tp4"
wait_for_health "$ROLLOUT_NODE_B" "$((VLLM_BASE_PORT + 1))" "short_tp2_0"

start_vllm "$ROLLOUT_NODE_B" 2 "$((VLLM_BASE_PORT + 2))" "short_tp2_1"
wait_for_health "$ROLLOUT_NODE_B" "$((VLLM_BASE_PORT + 2))" "short_tp2_1"

echo "[5/7] OpenAI-compatible API check"
curl -fsS "http://${ROLLOUT_NODE_A}:${VLLM_BASE_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"Compute 1+1. Answer with the number only.\",\"max_tokens\":8,\"temperature\":0}" \
    | python3 -m json.tool

export HETERO_INSTANCE_HOSTS="${ROLLOUT_NODE_A},${ROLLOUT_NODE_B},${ROLLOUT_NODE_B}"
export EVAL_MAX_SAMPLES="$EVAL_MAX_SAMPLES"
export EVAL_CONCURRENCY="$EVAL_CONCURRENCY"
export EVAL_MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS"
export EVAL_ACCURACY_THRESHOLD="$EVAL_ACCURACY_THRESHOLD"
export EVAL_WAIT_TIMEOUT="$EVAL_WAIT_TIMEOUT"
export EVAL_METRICS_FILE="${EVAL_METRICS_FILE:-eval_metrics.jsonl}"

echo "[6/7] Starting full DAPO-Math AsyncRLTrainer run"
srun --exclusive \
    --nodes=1 --ntasks=1 --nodelist="$TRAIN_NODE" \
    --gpus=4 \
    --cpus-per-task=16 \
    bash -lc "
        source '$VENV_PATH'
        cd '$PROJECT_DIR'
        export PYTHONPATH='$PROJECT_DIR':\${PYTHONPATH:-}
        export HETERO_INSTANCE_HOSTS='$HETERO_INSTANCE_HOSTS'
        export DAPO_MATH_ROOT='$DAPO_MATH_ROOT'
        export DAPO_MATH_SPLIT='$DAPO_MATH_SPLIT'
        export DAPO_MAX_TURNS='$DAPO_MAX_TURNS'
        export EVAL_MAX_SAMPLES='$EVAL_MAX_SAMPLES'
        export EVAL_CONCURRENCY='$EVAL_CONCURRENCY'
        export EVAL_MAX_NEW_TOKENS='$EVAL_MAX_NEW_TOKENS'
        export EVAL_ACCURACY_THRESHOLD='$EVAL_ACCURACY_THRESHOLD'
        export EVAL_WAIT_TIMEOUT='$EVAL_WAIT_TIMEOUT'
        export EVAL_METRICS_FILE='$EVAL_METRICS_FILE'
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export TOKENIZERS_PARALLELISM=false
        export WANDB_MODE='$WANDB_MODE'
        export WANDB_DIR='$WANDB_DIR'
        torchrun \
            --nproc_per_node=4 \
            --master_addr='$MASTER_ADDR' \
            --master_port='$MASTER_PORT' \
            examples/dapo_math_async_rl.py \
            --config '$RUNTIME_CONFIG'
    " 2>&1 | tee "$JOB_LOG_DIR/training.log"

echo "[7/7] DAPO-Math C-MLFQ training completed"
echo "Logs: $JOB_LOG_DIR"
echo "C-MLFQ tree: $PROJECT_DIR/logs/dapo_math_cmlfq/cmlfq_tree_${JOB_ID}.json"
echo "History: $PROJECT_DIR/logs/dapo_math_cmlfq/history"
echo "Eval metrics: $PROJECT_DIR/logs/dapo_math_cmlfq/eval/eval_metrics.jsonl"
echo "============================================================"

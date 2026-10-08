#!/bin/bash
# =============================================================================
# Qwen3-14B + C-MLFQ Slurm training
#
# Topology (3 nodes, 4 GPUs per node, 12 GPUs total):
#   node 0: training, 4 x GPU (FSDP)
#   node 1: rollout TP=4
#   node 2: rollout TP=2 + TP=2
#
# Submit:
#   MODEL_PATH=/shared/models/Qwen3-14B \
#   sbatch --export=ALL,MODEL_PATH scripts/submit_qwen3_cmlfq_train.slurm
#
# Optional:
#   SERPER_KEY_ID=... TOTAL_STEPS=1000 MAX_MODEL_LEN=40960 \
#   VENV_PATH=$HOME/rl_framework_env/bin/activate \
#   sbatch --export=ALL scripts/submit_qwen3_cmlfq_train.slurm
# =============================================================================

#SBATCH -J qwen3_cmlfq_train
#SBATCH -o logs/qwen3_cmlfq_train_%j.out
#SBATCH -e logs/qwen3_cmlfq_train_%j.err
#SBATCH -N 3
#SBATCH -G 12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --time=06:00:00
#SBATCH --exclude=gn019

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/RL_Framework}"
VENV_PATH="${VENV_PATH:-$HOME/rl_framework_env/bin/activate}"
MODEL_PATH="${MODEL_PATH:-}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TRAIN_MAX_SEQ_LENGTH="${TRAIN_MAX_SEQ_LENGTH:-40960}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_WAIT="${MAX_WAIT:-900}"

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is required."
    echo "Example:"
    echo "  MODEL_PATH=/shared/models/Qwen3-14B \\"
    echo "  sbatch --export=ALL,MODEL_PATH scripts/submit_qwen3_cmlfq_train.slurm"
    exit 2
fi

source "$VENV_PATH"
cd "$PROJECT_DIR"

mkdir -p logs
JOB_LOG_DIR="$PROJECT_DIR/logs/qwen3_cmlfq_${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#NODES[@]}" -lt 3 ]; then
    echo "ERROR: this training run requires 3 nodes; got ${#NODES[@]}"
    exit 2
fi

TRAIN_NODE="${NODES[0]}"
ROLLOUT_NODE_A="${NODES[1]}"
ROLLOUT_NODE_B="${NODES[2]}"
MASTER_ADDR="$TRAIN_NODE"

echo "============================================================"
echo "Qwen3-14B + C-MLFQ training"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID}"
echo "Model: $MODEL_PATH"
echo "Train node: $TRAIN_NODE (4 GPUs, FSDP)"
echo "Rollout node A: $ROLLOUT_NODE_A (TP4)"
echo "Rollout node B: $ROLLOUT_NODE_B (TP2 + TP2)"
echo "Steps: $TOTAL_STEPS"
echo "Logs: $JOB_LOG_DIR"
echo "Python: $(which python3)"
python3 -c 'import torch, transformers, vllm; print("torch", torch.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)'

VLLM_STEP_PIDS=()
declare -A VLLM_PID_BY_NAME=()
RUNTIME_CONFIG=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    echo ""
    echo "[cleanup] stopping training and vLLM processes..."

    for pid in "${VLLM_STEP_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done

    for node in "$ROLLOUT_NODE_A" "$ROLLOUT_NODE_B"; do
        for port in \
            "$VLLM_BASE_PORT" \
            "$((VLLM_BASE_PORT + 1))" \
            "$((VLLM_BASE_PORT + 2))"; do
            srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
                bash -lc "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${port}' 2>/dev/null || true" \
                >/dev/null 2>&1 || true
        done
    done

    if [ -n "$RUNTIME_CONFIG" ]; then
        rm -f "$RUNTIME_CONFIG"
    fi
    echo "[cleanup] complete, exit=$exit_code"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo ""
echo "[1/4] Writing runtime configuration"
RUNTIME_CONFIG="$PROJECT_DIR/configs/.qwen3_cmlfq_train_${SLURM_JOB_ID}.yaml"
export PROJECT_DIR MODEL_PATH TOTAL_STEPS MAX_MODEL_LEN VLLM_BASE_PORT
export TRAIN_MAX_SEQ_LENGTH TRAIN_BATCH_SIZE TRAIN_MICRO_BATCH_SIZE
export MAX_NEW_TOKENS
python3 - <<'PY'
import os
from pathlib import Path

import yaml

project = Path(os.environ["PROJECT_DIR"])
config = {
    "base_config": str(
        project / "configs" / "search_r1_cmlfq_qwen3_14b.yaml"
    ),
    "model_path": os.environ["MODEL_PATH"],
    "tokenizer_path": os.environ["MODEL_PATH"],
    "train_gpus": 4,
    "tp_size": 1,
    "train_tp_size": 1,
    "train_pp_size": 1,
    "train_dp_size": 4,
    "batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["TRAIN_MICRO_BATCH_SIZE"]),
    "total_steps": int(os.environ["TOTAL_STEPS"]),
    "max_seq_length": int(os.environ["TRAIN_MAX_SEQ_LENGTH"]),
    "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    "n_samples": 1,
    "max_concurrent_rollouts": int(os.environ["TRAIN_BATCH_SIZE"]),
    "rollout_gpus": 8,
    "eval_interval": 1000,
    "save_interval": 1000,
    "sync_path": str(
        project / "logs" / "qwen3_cmlfq_train_weights"
    ),
    "log_dir": str(project / "logs" / "qwen3_cmlfq_train"),
    "heterogeneous_rollout": {
        "total_gpus": 8,
        "vllm_host": "0.0.0.0",
        "vllm_base_port": int(os.environ["VLLM_BASE_PORT"]),
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "instances": [
            {
                "instance_id": "short_tp4",
                "tp": 4,
                "gpus": [0, 1, 2, 3],
            },
            {
                "instance_id": "medium_tp2_0",
                "tp": 2,
                "gpus": [0, 1],
            },
            {
                "instance_id": "medium_tp2_1",
                "tp": 2,
                "gpus": [2, 3],
            },
        ],
        "scheduling": {
            "cmlfq_tree_path": str(
                project / "logs" / "qwen3_cmlfq_train" / "cmlfq_tree.json"
            ),
        },
    },
}

path = project / "configs" / (
    f".qwen3_cmlfq_train_{os.environ['SLURM_JOB_ID']}.yaml"
)
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"Wrote runtime config: {path}")
PY

start_vllm() {
    local node="$1"
    local gpu_ids="$2"
    local tp="$3"
    local port="$4"
    local name="$5"
    local log_file="$JOB_LOG_DIR/vllm_${name}.log"

    echo "Starting $name on $node: GPUs=$gpu_ids TP=$tp port=$port"
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

echo ""
echo "[2/4] Starting heterogeneous vLLM rollout instances"
start_vllm "$ROLLOUT_NODE_A" "0,1,2,3" 4 "$VLLM_BASE_PORT" "short_tp4"
start_vllm "$ROLLOUT_NODE_B" "0,1" 2 "$((VLLM_BASE_PORT + 1))" "medium_tp2_0"

wait_for_health() {
    local host="$1"
    local port="$2"
    local name="$3"
    local elapsed=0
    local pid="${VLLM_PID_BY_NAME[$name]}"

    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
            echo "  READY: $name at $host:$port (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name exited before becoming ready"
            tail -100 "$JOB_LOG_DIR/vllm_${name}.log" || true
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo "ERROR: $name did not become ready in ${MAX_WAIT}s"
    echo "Log: $JOB_LOG_DIR/vllm_${name}.log"
    return 1
}

echo ""
echo "[3/4] Waiting for vLLM health endpoints"
wait_for_health "$ROLLOUT_NODE_A" "$VLLM_BASE_PORT" "short_tp4"
wait_for_health "$ROLLOUT_NODE_B" "$((VLLM_BASE_PORT + 1))" "medium_tp2_0"

# Start the second service only after the first TP=2 step owns its two GPUs.
# Slurm then assigns and remaps the remaining two GPUs to local IDs 0 and 1.
start_vllm "$ROLLOUT_NODE_B" "2,3" 2 "$((VLLM_BASE_PORT + 2))" "medium_tp2_1"
wait_for_health "$ROLLOUT_NODE_B" "$((VLLM_BASE_PORT + 2))" "medium_tp2_1"

export HETERO_INSTANCE_HOSTS="${ROLLOUT_NODE_A},${ROLLOUT_NODE_B},${ROLLOUT_NODE_B}"

echo ""
echo "[4/4] Starting ${TOTAL_STEPS}-step Search-R1 training"
srun --exclusive \
    --nodes=1 --ntasks=1 --nodelist="$TRAIN_NODE" \
    --gpus=4 \
    --cpus-per-task=16 \
    bash -lc "
        source '$VENV_PATH'
        cd '$PROJECT_DIR'
        export PYTHONPATH='$(dirname "$PROJECT_DIR")':\${PYTHONPATH:-}
        export HETERO_INSTANCE_HOSTS='$HETERO_INSTANCE_HOSTS'
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export TOKENIZERS_PARALLELISM=false
        export WANDB_MODE='$WANDB_MODE'
        export SERPER_KEY_ID='${SERPER_KEY_ID:-}'
        export SEARCH_BACKEND='${SEARCH_BACKEND:-serper}'
        export SEARXNG_URL='${SEARXNG_URL:-http://127.0.0.1:18080}'
        torchrun \
            --nproc_per_node=4 \
            --master_addr='$MASTER_ADDR' \
            --master_port='$MASTER_PORT' \
            examples/search_r1_async_rl.py \
            --config '$RUNTIME_CONFIG'
    " 2>&1 | tee "$JOB_LOG_DIR/training.log"

echo ""
echo "============================================================"
echo "Training completed successfully"
echo "Logs: $JOB_LOG_DIR"
echo "Trie: $PROJECT_DIR/logs/qwen3_cmlfq_train/cmlfq_tree.json"
echo "============================================================"

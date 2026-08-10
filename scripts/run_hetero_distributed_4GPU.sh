#!/bin/bash
# =============================================================================

#


#


#

#   bash scripts/run_hetero_distributed_4GPU.sh --config configs/gsm8k_hetero_4gpu.yaml

#



# =============================================================================

set -e


cd "$(dirname "$0")/.."


MODEL_PATH="/path/to/Qwen3-4B"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline

export TOKENIZERS_PARALLELISM=false
# Training GPU
TRAIN_GPUS="0,1"
TRAIN_NPROC=2

# =============================================================================

#




# =============================================================================
HETERO_INSTANCES=(
    "tp1_inst0:1:2"
    "tp1_inst1:1:3"
)


VLLM_HOST="127.0.0.1"
VLLM_BASE_PORT=8000
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.90


MASTER_ADDR="localhost"
MASTER_PORT="29500"


LOG_DIR="/path/to/RL_Framework/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/training_hetero_${TIMESTAMP}.log"

# PYTHONPATH
export PYTHONPATH=/path/to/user:$PYTHONPATH
cd /path/to/user



PROCESSED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"

            if [[ "$CONFIG_PATH" != /* ]]; then
                CONFIG_PATH="$(pwd)/$CONFIG_PATH"
            fi
            PROCESSED_ARGS+=("$1" "$CONFIG_PATH")
            shift 2
            ;;
        *)
            PROCESSED_ARGS+=("$1")
            shift
            ;;
    esac
done


NUM_INSTANCES=${#HETERO_INSTANCES[@]}

echo "=========================================="
echo "Heterogeneous rollout cluster summary"
echo "=========================================="
echo "  Training GPU: $TRAIN_GPUS (${TRAIN_NPROC} processes)"
echo "  Rollout Number of instances: $NUM_INSTANCES"
echo ""

for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS <<< "${HETERO_INSTANCES[$i]}"
    INST_PORT=$((VLLM_BASE_PORT + i))
    echo "  Instance $i ($INST_NAME): TP=$INST_TP, GPU=$INST_GPUS, port=$INST_PORT"
done
echo ""


echo "=========================================="
echo "[Stage 1] Cleaning residual processes"
echo "=========================================="
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
echo "Cleanup complete"


echo ""
echo "=========================================="
echo "[Stage 2] Starting $NUM_INSTANCES heterogeneous vLLM instances"
echo "=========================================="

VLLM_PIDS=()

for ((i=0; i<NUM_INSTANCES; i++)); do

    IFS=':' read -r INST_NAME INST_TP INST_GPUS <<< "${HETERO_INSTANCES[$i]}"
    INST_PORT=$((VLLM_BASE_PORT + i))
    VLLM_LOG="$LOG_DIR/vllm_${INST_NAME}_${TIMESTAMP}.log"

    echo "  Instance $i ($INST_NAME): TP=$INST_TP, GPU=$INST_GPUS, port=$INST_PORT"

    CUDA_VISIBLE_DEVICES=$INST_GPUS nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host "$VLLM_HOST" \
        --port "$INST_PORT" \
        --tensor-parallel-size "$INST_TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enforce-eager \
        --trust-remote-code \
        > "$VLLM_LOG" 2>&1 &

    VLLM_PIDS+=($!)
    echo "    PID: ${VLLM_PIDS[$i]}, Log: $VLLM_LOG"
done


echo "${VLLM_PIDS[@]}" > "$LOG_DIR/.vllm_pids_hetero"


echo ""
echo "=========================================="
echo "[Stage 3] Waiting for all heterogeneous vLLM instances"
echo "=========================================="

MAX_WAIT=300
for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS <<< "${HETERO_INSTANCES[$i]}"
    INST_PORT=$((VLLM_BASE_PORT + i))
    PID=${VLLM_PIDS[$i]}
    ELAPSED=0

    echo "  Waiting for instance $i ($INST_NAME, TP=$INST_TP, port=$INST_PORT, PID=$PID)..."

    while [ $ELAPSED -lt $MAX_WAIT ]; do

        if ! kill -0 $PID 2>/dev/null; then
            echo "  ERROR: Instance $INST_NAME exited!"
            echo "  See log: $LOG_DIR/vllm_${INST_NAME}_${TIMESTAMP}.log"
            for pid in "${VLLM_PIDS[@]}"; do
                kill $pid 2>/dev/null || true
            done
            exit 1
        fi


        if curl -s -o /dev/null -w "%{http_code}" "http://${VLLM_HOST}:${INST_PORT}/health" 2>/dev/null | grep -q "200"; then
            echo "  Instance $INST_NAME is ready! (waited${ELAPSED}s)"
            break
        fi

        if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
            echo "    Waiting... (${ELAPSED}s)"
        fi

        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "  ERROR: Instance $INST_NAME startup timed out (${MAX_WAIT}s)"
        for pid in "${VLLM_PIDS[@]}"; do
            kill $pid 2>/dev/null || true
        done
        exit 1
    fi
done

echo ""
echo "All $NUM_INSTANCES heterogeneous vLLM instancesis ready!"
echo "  TP layout: $(for inst in "${HETERO_INSTANCES[@]}"; do IFS=':' read -r _ tp _ <<< "$inst"; printf "%s " "$tp"; done)"



HETERO_INSTANCE_HOSTS=""
for ((i=0; i<NUM_INSTANCES; i++)); do
    if [ -n "$HETERO_INSTANCE_HOSTS" ]; then
        HETERO_INSTANCE_HOSTS="${HETERO_INSTANCE_HOSTS},"
    fi
    HETERO_INSTANCE_HOSTS="${HETERO_INSTANCE_HOSTS}127.0.0.1"
done
export HETERO_INSTANCE_HOSTS

echo ""
echo "=========================================="
echo "[Stage 4] Starting FSDP training"
echo "=========================================="
echo "  Training GPUs: $TRAIN_GPUS"
echo "  Processes: $TRAIN_NPROC"
echo "  Log: $TRAIN_LOG"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
nohup torchrun \
    --nproc_per_node=$TRAIN_NPROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    RL_Framework/examples/gsm8k_async_rl.py "${PROCESSED_ARGS[@]}" \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "  PID: $TRAIN_PID"
echo "$TRAIN_PID" > "$LOG_DIR/.train_pid_hetero"


echo ""
echo "=========================================="
echo "Heterogeneous cluster started"
echo "=========================================="
echo ""
echo "  Heterogeneous vLLM instances:"
for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS <<< "${HETERO_INSTANCES[$i]}"
    echo "    ${INST_NAME} (TP=${INST_TP}): PID=${VLLM_PIDS[$i]} port=$((VLLM_BASE_PORT + i))"
done
echo ""
echo "  Training process: PID=$TRAIN_PID"
echo "    Log: tail -f $TRAIN_LOG"
echo ""
echo "Stop all services:"
echo "  kill ${VLLM_PIDS[*]} $TRAIN_PID"
echo "  or: pkill -f vllm.entrypoints && kill $TRAIN_PID"

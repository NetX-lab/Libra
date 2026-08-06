#!/bin/bash
# =============================================================================

#


#



# =============================================================================

set -e


cd "$(dirname "$0")/.."


MODEL_PATH="/path/to/Qwen3-4B"
export WANDB_API_KEY="bfee760e4e267073a9eeadca3169b8bc2b2cacbe"

TRAIN_GPUS="0,1"
ROLLOUT_GPUS="2,3"
TRAIN_NPROC=2


VLLM_HOST="127.0.0.1"
VLLM_BASE_PORT=8000
VLLM_TP_SIZE=1
MAX_MODEL_LEN=2048


MASTER_ADDR="localhost"
MASTER_PORT="29500"


LOG_DIR="/path/to/RL_Framework/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/training_${TIMESTAMP}.log"

# PYTHONPATH
export PYTHONPATH=/path/to/user:$PYTHONPATH
cd /path/to/user



IFS=',' read -ra GPU_ARRAY <<< "$ROLLOUT_GPUS"
NUM_ROLLOUT_GPUS=${#GPU_ARRAY[@]}
NUM_INSTANCES=$((NUM_ROLLOUT_GPUS / VLLM_TP_SIZE))

if [ "$NUM_INSTANCES" -le 0 ]; then
    echo "ERROR: GPU count ($NUM_ROLLOUT_GPUS) is insufficient to start a TP=$VLLM_TP_SIZE instance"
    exit 1
fi

if [ $((NUM_ROLLOUT_GPUS % VLLM_TP_SIZE)) -ne 0 ]; then
    echo "WARNING: GPU count($NUM_ROLLOUT_GPUS)is not divisible by TP($VLLM_TP_SIZE); extra GPUs will remain idle"
fi

echo "=========================================="
echo "Multi-instance vLLM configuration summary"
echo "=========================================="
echo "  Rollout GPUs: $ROLLOUT_GPUS ($NUM_ROLLOUT_GPUS )"
echo "  TP size: $VLLM_TP_SIZE"
echo "  Number of instances: $NUM_INSTANCES"
echo "  Port range: ${VLLM_BASE_PORT}-$((VLLM_BASE_PORT + NUM_INSTANCES - 1))"
echo ""


echo "=========================================="
echo "[Stage 1] Cleaning residual processes"
echo "=========================================="
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
echo "Cleanup complete"


echo ""
echo "=========================================="
echo "[Stage 2] Starting $NUM_INSTANCES vLLM inference instances"
echo "=========================================="

VLLM_PIDS=()

for ((i=0; i<NUM_INSTANCES; i++)); do

    START_IDX=$((i * VLLM_TP_SIZE))
    INSTANCE_GPUS=""
    for ((j=0; j<VLLM_TP_SIZE; j++)); do
        GPU_IDX=$((START_IDX + j))
        if [ -n "$INSTANCE_GPUS" ]; then
            INSTANCE_GPUS="${INSTANCE_GPUS},"
        fi
        INSTANCE_GPUS="${INSTANCE_GPUS}${GPU_ARRAY[$GPU_IDX]}"
    done

    INSTANCE_PORT=$((VLLM_BASE_PORT + i))
    VLLM_LOG="$LOG_DIR/vllm_instance${i}_${TIMESTAMP}.log"

    echo "  Instance $i: GPU=$INSTANCE_GPUS, port=$INSTANCE_PORT, Log=$VLLM_LOG"

    CUDA_VISIBLE_DEVICES=$INSTANCE_GPUS nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host "$VLLM_HOST" \
        --port "$INSTANCE_PORT" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --trust-remote-code \
        > "$VLLM_LOG" 2>&1 &

    VLLM_PIDS+=($!)
    echo "    PID: ${VLLM_PIDS[$i]}"
done


echo "${VLLM_PIDS[@]}" > "$LOG_DIR/.vllm_pids"


echo ""
echo "=========================================="
echo "[Stage 3] Waiting for all vLLM instances"
echo "=========================================="

MAX_WAIT=300
for ((i=0; i<NUM_INSTANCES; i++)); do
    INSTANCE_PORT=$((VLLM_BASE_PORT + i))
    PID=${VLLM_PIDS[$i]}
    ELAPSED=0

    echo "  Waiting for instance $i (port=$INSTANCE_PORT, PID=$PID)..."

    while [ $ELAPSED -lt $MAX_WAIT ]; do

        if ! kill -0 $PID 2>/dev/null; then
            echo "  ERROR: vLLMInstance $i exited, See log: $LOG_DIR/vllm_instance${i}_${TIMESTAMP}.log"

            for pid in "${VLLM_PIDS[@]}"; do
                kill $pid 2>/dev/null || true
            done
            exit 1
        fi


        if curl -s -o /dev/null -w "%{http_code}" "http://${VLLM_HOST}:${INSTANCE_PORT}/health" 2>/dev/null | grep -q "200"; then
            echo "  Instance $i is ready! (waited${ELAPSED}s)"
            break
        fi

        if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
            echo "    Waiting... (${ELAPSED}s)"
        fi

        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "  ERROR: vLLMInstance $i startup timed out (${MAX_WAIT}s)"
        for pid in "${VLLM_PIDS[@]}"; do
            kill $pid 2>/dev/null || true
        done
        exit 1
    fi
done

echo "All $NUM_INSTANCES vLLM instances are ready!"


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
    RL_Framework/examples/gsm8k_async_rl.py "$@" \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "  PID: $TRAIN_PID"
echo "$TRAIN_PID" > "$LOG_DIR/.train_pid"


echo ""
echo "=========================================="
echo "All services started"
echo "=========================================="
echo "  vLLM instances: $NUM_INSTANCES"
for ((i=0; i<NUM_INSTANCES; i++)); do
    echo "    Instance$i PID=${VLLM_PIDS[$i]} port=$((VLLM_BASE_PORT + i)) | tail -f $LOG_DIR/vllm_instance${i}_${TIMESTAMP}.log"
done
echo "  Training PID: $TRAIN_PID | tail -f $TRAIN_LOG"
echo ""
echo "Stop all services:"
echo "  kill ${VLLM_PIDS[*]} $TRAIN_PID"
echo "  or: pkill -f vllm.entrypoints && kill $TRAIN_PID"

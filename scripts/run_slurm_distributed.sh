#!/bin/bash
# =============================================================================

#



#



#


#   bash run_slurm_distributed.sh
#

#   NUM_NODES=4 GPUS_PER_NODE=4 bash run_slurm_distributed.sh
#

#   MODEL_PATH, TRAIN_SCRIPT, CONFIG_FILE,
#   TRAIN_GPUS_PER_NODE, ROLLOUT_GPUS_PER_NODE, GPUS_PER_NODE,
#   VLLM_TP_SIZE, VLLM_BASE_PORT, MAX_MODEL_LEN,
#   MASTER_PORT, LOG_DIR, PYTHONPATH
# =============================================================================

set -e


cd "$(dirname "$0")/.."



MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/gsm8k_async_rl.py}"
CONFIG_FILE="${CONFIG_FILE:-}"


GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-2}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-2}"


VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"


MASTER_PORT="${MASTER_PORT:-29500}"


LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


MAX_WAIT="${MAX_WAIT:-600}"


if [ -n "$SLURM_NODELIST" ]; then

    mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
    NUM_NODES=${#NODES[@]}
    HEAD_NODE="${NODES[0]}"
    IS_SLURM=true
    echo "=========================================="
    echo "Slurm environment"
    echo "=========================================="
    echo "  Job ID: $SLURM_JOBID"
    echo "  Nodes: $NUM_NODES"
    echo "  Node list: ${NODES[*]}"
    echo "  Head node: $HEAD_NODE"
    echo "  SLURM_GPUS: ${SLURM_GPUS:-N/A}"
else

    NODES=("$(hostname)")
    NUM_NODES=1
    HEAD_NODE="${NODES[0]}"
    IS_SLURM=false
    echo "=========================================="
    echo "Non-Slurm environment; using single-node mode"
    echo "=========================================="
    echo "  node: $HEAD_NODE"
fi


ROLLOUT_INSTANCES_PER_NODE=$((ROLLOUT_GPUS_PER_NODE / VLLM_TP_SIZE))
TOTAL_ROLLOUT_INSTANCES=$((NUM_NODES * ROLLOUT_INSTANCES_PER_NODE))
TOTAL_TRAIN_GPUS=$((NUM_NODES * TRAIN_GPUS_PER_NODE))


if [ "$ROLLOUT_INSTANCES_PER_NODE" -le 0 ]; then
    echo "ERROR: Rollout GPU count ($ROLLOUT_GPUS_PER_NODE) is insufficient to start a TP=$VLLM_TP_SIZE instance"
    exit 1
fi

if [ $((ROLLOUT_GPUS_PER_NODE % VLLM_TP_SIZE)) -ne 0 ]; then
    echo "WARNING: Rollout GPUs count($ROLLOUT_GPUS_PER_NODE)is not divisible by TP($VLLM_TP_SIZE)exactly"
fi


VLLM_ENDPOINTS=""
INSTANCE_GLOBAL_ID=0
for ((n=0; n<NUM_NODES; n++)); do
    NODE="${NODES[$n]}"
    for ((i=0; i<ROLLOUT_INSTANCES_PER_NODE; i++)); do
        PORT=$((VLLM_BASE_PORT + INSTANCE_GLOBAL_ID))
        if [ -n "$VLLM_ENDPOINTS" ]; then
            VLLM_ENDPOINTS="${VLLM_ENDPOINTS},"
        fi
        VLLM_ENDPOINTS="${VLLM_ENDPOINTS}${NODE}:${PORT}"
        INSTANCE_GLOBAL_ID=$((INSTANCE_GLOBAL_ID + 1))
    done
done

echo ""
echo "=========================================="
echo "Multi-node configuration summary"
echo "=========================================="
echo "  Total nodes: $NUM_NODES"
echo "  GPUs per node: $GPUS_PER_NODE (Training=$TRAIN_GPUS_PER_NODE, Rollout=$ROLLOUT_GPUS_PER_NODE)"
echo "  vLLM TP: $VLLM_TP_SIZE"
echo "  Rollout instances per node: $ROLLOUT_INSTANCES_PER_NODE"
echo "  Total rollout instances: $TOTAL_ROLLOUT_INSTANCES"
echo "  Total training GPUs: $TOTAL_TRAIN_GPUS"
echo "  vLLM endpoints: $VLLM_ENDPOINTS"
echo "  Master: $HEAD_NODE:$MASTER_PORT"
echo ""


echo "=========================================="
echo "[Stage 1] Cleaning residual processes on all nodes"
echo "=========================================="

if [ "$IS_SLURM" = true ]; then

    for NODE in "${NODES[@]}"; do
        echo "  Cleaning node $NODE ..."
        srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
            bash -c '
                pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
                pkill -9 -f "EngineCore" 2>/dev/null || true
            ' 2>/dev/null &
    done
    wait
    sleep 3


    INSTANCE_ID=0
    for ((n=0; n<NUM_NODES; n++)); do
        NODE="${NODES[$n]}"
        for ((i=0; i<ROLLOUT_INSTANCES_PER_NODE; i++)); do
            PORT=$((VLLM_BASE_PORT + INSTANCE_ID))
            srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
                bash -c "pids=\$(lsof -t -i:${PORT} 2>/dev/null); [ -n \"\$pids\" ] && echo \"\$pids\" | xargs kill -9 2>/dev/null || true" 2>/dev/null &
            INSTANCE_ID=$((INSTANCE_ID + 1))
        done
    done
    wait
else

    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
fi

sleep 2
echo "Cleanup complete"


echo ""
echo "=========================================="
echo "[Stage 2] Starting $TOTAL_ROLLOUT_INSTANCES vLLM inference instances"
echo "=========================================="

WORK_DIR="$(pwd)"
VLLM_PIDS=()
INSTANCE_GLOBAL_ID=0

for ((n=0; n<NUM_NODES; n++)); do
    NODE="${NODES[$n]}"
    for ((i=0; i<ROLLOUT_INSTANCES_PER_NODE; i++)); do

        START_GPU_IDX=$((TRAIN_GPUS_PER_NODE + i * VLLM_TP_SIZE))
        INSTANCE_GPUS=""
        for ((j=0; j<VLLM_TP_SIZE; j++)); do
            GPU_IDX=$((START_GPU_IDX + j))
            if [ -n "$INSTANCE_GPUS" ]; then
                INSTANCE_GPUS="${INSTANCE_GPUS},"
            fi
            INSTANCE_GPUS="${INSTANCE_GPUS}${GPU_IDX}"
        done

        PORT=$((VLLM_BASE_PORT + INSTANCE_GLOBAL_ID))
        VLLM_LOG="$LOG_DIR/vllm_${NODE}_instance${i}_${TIMESTAMP}.log"

        echo "  Instance $INSTANCE_GLOBAL_ID: node=$NODE, GPU=$INSTANCE_GPUS, port=$PORT"

        if [ "$IS_SLURM" = true ]; then

            srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
                bash -c "cd ${WORK_DIR} && \
                    export CUDA_VISIBLE_DEVICES=${INSTANCE_GPUS} && \
                    export NO_PROXY=localhost,127.0.0.1 && \
                    python3 -m vllm.entrypoints.openai.api_server \
                        --model ${MODEL_PATH} \
                        --host 0.0.0.0 \
                        --port ${PORT} \
                        --tensor-parallel-size ${VLLM_TP_SIZE} \
                        --max-model-len ${MAX_MODEL_LEN} \
                        --dtype bfloat16 \
                        --gpu-memory-utilization ${GPU_MEM_UTIL} \
                        --enforce-eager \
                        --trust-remote-code \
                    > ${VLLM_LOG} 2>&1" &
        else

            CUDA_VISIBLE_DEVICES=$INSTANCE_GPUS nohup python3 -m vllm.entrypoints.openai.api_server \
                --model "$MODEL_PATH" \
                --host 0.0.0.0 \
                --port "$PORT" \
                --tensor-parallel-size "$VLLM_TP_SIZE" \
                --max-model-len "$MAX_MODEL_LEN" \
                --dtype bfloat16 \
                --gpu-memory-utilization "$GPU_MEM_UTIL" \
                --enforce-eager \
                --trust-remote-code \
                > "$VLLM_LOG" 2>&1 &
        fi

        VLLM_PIDS+=($!)
        INSTANCE_GLOBAL_ID=$((INSTANCE_GLOBAL_ID + 1))
    done
done

echo "  Started ${#VLLM_PIDS[@]} vLLM instances"
echo "${VLLM_PIDS[@]}" > "$LOG_DIR/.vllm_pids_${TIMESTAMP}"


echo ""
echo "=========================================="
echo "[Stage 3] Waiting for all vLLM instances"
echo "=========================================="

IFS=',' read -ra EP_ARRAY <<< "$VLLM_ENDPOINTS"

for ((idx=0; idx<${#EP_ARRAY[@]}; idx++)); do
    EP="${EP_ARRAY[$idx]}"
    HOST_PART=$(echo "$EP" | cut -d: -f1)
    PORT_PART=$(echo "$EP" | cut -d: -f2)
    ELAPSED=0

    echo "  Waiting for instance $idx ($EP)..."

    while [ $ELAPSED -lt $MAX_WAIT ]; do

        if curl -s -o /dev/null -w "%{http_code}" "http://${EP}/health" 2>/dev/null | grep -q "200"; then
            echo "  Instance $idx ($EP) is ready! (${ELAPSED}s)"
            break
        fi

        if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
            echo "    Waiting... (${ELAPSED}s)"
        fi

        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "  ERROR: Instance $idx ($EP) startup timed out (${MAX_WAIT}s)"
        echo "  See log: $LOG_DIR/vllm_*_${TIMESTAMP}.log"

        bash scripts/cleanup_slurm.sh 2>/dev/null || true
        exit 1
    fi
done

echo "All $TOTAL_ROLLOUT_INSTANCES vLLM instancesis ready!"


echo ""
echo "=========================================="
echo "[Stage 4] Starting multi-node FSDP training"
echo "=========================================="


TRAIN_GPU_IDS=""
for ((g=0; g<TRAIN_GPUS_PER_NODE; g++)); do
    if [ -n "$TRAIN_GPU_IDS" ]; then
        TRAIN_GPU_IDS="${TRAIN_GPU_IDS},"
    fi
    TRAIN_GPU_IDS="${TRAIN_GPU_IDS}${g}"
done

echo "  Training nodes: $NUM_NODES"
echo "  Training GPUs per node: $TRAIN_GPU_IDS ($TRAIN_GPUS_PER_NODE )"
echo "  Master: $HEAD_NODE:$MASTER_PORT"


TRAIN_ARGS=""
if [ -n "$CONFIG_FILE" ]; then
    TRAIN_ARGS="--config $CONFIG_FILE"
fi

TRAIN_LOG="$LOG_DIR/training_${TIMESTAMP}.log"


export VLLM_ENDPOINTS="$VLLM_ENDPOINTS"

if [ "$IS_SLURM" = true ] && [ "$NUM_NODES" -gt 1 ]; then


    NODE_RANK=0
    for NODE in "${NODES[@]}"; do
        NODE_TRAIN_LOG="$LOG_DIR/training_${NODE}_${TIMESTAMP}.log"

        srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
            bash -c "cd ${WORK_DIR} && \
                export CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDS} && \
                export VLLM_ENDPOINTS='${VLLM_ENDPOINTS}' && \
                export NO_PROXY=localhost,127.0.0.1 && \
                torchrun \
                    --nnodes=${NUM_NODES} \
                    --nproc_per_node=${TRAIN_GPUS_PER_NODE} \
                    --node_rank=${NODE_RANK} \
                    --rdzv_backend=c10d \
                    --rdzv_endpoint=${HEAD_NODE}:${MASTER_PORT} \
                    ${TRAIN_SCRIPT} ${TRAIN_ARGS} \
                > ${NODE_TRAIN_LOG} 2>&1" &

        NODE_RANK=$((NODE_RANK + 1))
    done

    TRAIN_PID=$!
    echo "  Multi-node training started (last node PID: $TRAIN_PID)"
else

    CUDA_VISIBLE_DEVICES=$TRAIN_GPU_IDS \
    VLLM_ENDPOINTS="$VLLM_ENDPOINTS" \
    MASTER_ADDR=$HEAD_NODE \
    MASTER_PORT=$MASTER_PORT \
    nohup torchrun \
        --nproc_per_node=$TRAIN_GPUS_PER_NODE \
        --master_addr=$HEAD_NODE \
        --master_port=$MASTER_PORT \
        $TRAIN_SCRIPT $TRAIN_ARGS \
        > "$TRAIN_LOG" 2>&1 &

    TRAIN_PID=$!
    echo "  Single-node training PID: $TRAIN_PID"
fi

echo "$TRAIN_PID" > "$LOG_DIR/.train_pid_${TIMESTAMP}"


echo ""
echo "=========================================="
echo "All services started!"
echo "=========================================="
echo "  vLLM instances: $TOTAL_ROLLOUT_INSTANCES"
echo "  vLLM endpoints: $VLLM_ENDPOINTS"
echo "  Training nodes: $NUM_NODES"
echo "  Training GPUs: $TOTAL_TRAIN_GPUS"
echo ""
echo "Log directory: $LOG_DIR"
echo "  vLLM: $LOG_DIR/vllm_*_${TIMESTAMP}.log"
echo "  Training: $LOG_DIR/training_*${TIMESTAMP}.log"
echo ""
echo "Stop all services:"
echo "  bash scripts/cleanup_slurm.sh"
echo ""


wait $TRAIN_PID 2>/dev/null
TRAIN_EXIT_CODE=$?
echo "Training process exited with code: $TRAIN_EXIT_CODE"


echo "Stopping vLLM instances..."
bash scripts/cleanup_slurm.sh 2>/dev/null || true

echo "Job complete! $(date)"
exit $TRAIN_EXIT_CODE

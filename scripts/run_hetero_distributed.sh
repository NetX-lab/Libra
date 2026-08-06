#!/bin/bash
# =============================================================================

#

#

#   Node 0: GPU 0,1 -> Training (FSDP)




#


# =============================================================================

set -e


cd "$(dirname "$0")/.."


MODEL_PATH="/path/to/Qwen3-4B"
export WANDB_API_KEY="bfee760e4e267073a9eeadca3169b8bc2b2cacbe"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

TRAIN_GPUS="0,1"
TRAIN_NPROC=2

# =============================================================================

# =============================================================================

if [ -n "$SLURM_NODELIST" ]; then
    NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
else
    echo "WARNING: SLURM_NODELIST is not set; using localhost"
    NODES=("$(hostname)")
fi

NODE0=${NODES[0]}
NODE1=${NODES[1]:-$NODE0}

echo "========================================="
echo "Node information"
echo "========================================="
echo "  Node 0 (Master): $NODE0"
echo "  Node 1 (Rollout): $NODE1"
echo ""

# =============================================================================

#



# =============================================================================
HETERO_INSTANCES=(
    "tp1_inst0:1:2:0"       # Node 0, GPU 2, TP=1
    "tp1_inst1:1:3:0"       # Node 0, GPU 3, TP=1
    "tp2_inst0:2:0,1:1"     # Node 1, GPU 0,1, TP=2
    "tp2_inst1:2:2,3:1"     # Node 1, GPU 2,3, TP=2
)


VLLM_HOST="0.0.0.0"
VLLM_BASE_PORT=8000
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.90


MASTER_ADDR="$NODE0"
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
echo "Heterogeneous rollout cluster summary (multi-node)"
echo "=========================================="
echo "  Training: Node 0 ($NODE0), GPU $TRAIN_GPUS (${TRAIN_NPROC} processes)"
echo "  Rollout Number of instances: $NUM_INSTANCES"
echo ""

for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS INST_NODE_IDX <<< "${HETERO_INSTANCES[$i]}"
    INST_NODE=${NODES[$INST_NODE_IDX]}
    INST_PORT=$((VLLM_BASE_PORT + i))
    echo "  Instance $i ($INST_NAME): TP=$INST_TP, GPU=$INST_GPUS, node=$INST_NODE, port=$INST_PORT"
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
echo "[Stage 2] Starting $NUM_INSTANCES heterogeneous vLLM instances across nodes"
echo "=========================================="

VLLM_PIDS=()
VLLM_HOSTS=()

for ((i=0; i<NUM_INSTANCES; i++)); do

    IFS=':' read -r INST_NAME INST_TP INST_GPUS INST_NODE_IDX <<< "${HETERO_INSTANCES[$i]}"
    INST_NODE=${NODES[$INST_NODE_IDX]}
    INST_PORT=$((VLLM_BASE_PORT + i))
    VLLM_LOG="$LOG_DIR/vllm_${INST_NAME}_${TIMESTAMP}.log"
    VLLM_HOSTS+=($INST_NODE)

    echo "  Instance $i ($INST_NAME): TP=$INST_TP, GPU=$INST_GPUS, node=$INST_NODE, port=$INST_PORT"

    if [ "$INST_NODE" = "$(hostname)" ] || [ "$INST_NODE_IDX" = "0" ]; then

        echo "    local node $(hostname) starting instance $i ($INST_NAME)..."

        CUDA_VISIBLE_DEVICES=$INST_GPUS MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + i)) TOKENIZERS_PARALLELISM=false \
            nohup python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --host $VLLM_HOST \
            --port $INST_PORT \
            --tensor-parallel-size $INST_TP \
            --max-model-len $MAX_MODEL_LEN \
            --dtype float16 \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --enforce-eager \
            --trust-remote-code \
            --disable-custom-all-reduce \
            > $VLLM_LOG 2>&1 &
        PID=$!
    else



        GPU_COUNT=$(echo "$INST_GPUS" | tr ',' '\n' | wc -l)
        echo "    remote node $INST_NODE starting instance $i ($INST_NAME), GPU=$INST_GPUS, gres=gpu:$GPU_COUNT"

        srun --exact --nodelist="$INST_NODE" --nodes=1 --ntasks=1 -c 16 \
            -G $GPU_COUNT \
            --export=ALL,CUDA_VISIBLE_DEVICES=$INST_GPUS,MASTER_ADDR=127.0.0.1,MASTER_PORT=$((20000 + i)),TOKENIZERS_PARALLELISM=false \
            python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --host $VLLM_HOST \
            --port $INST_PORT \
            --tensor-parallel-size $INST_TP \
            --max-model-len $MAX_MODEL_LEN \
            --dtype float16 \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --enforce-eager \
            --trust-remote-code \
            --disable-custom-all-reduce \
            > $VLLM_LOG 2>&1 &

        PID=$!
    fi

    VLLM_PIDS+=($PID)
    echo "    PID: $PID, Log: $VLLM_LOG"



    if [ "$INST_NODE" != "$(hostname)" ] && [ "$INST_NODE_IDX" != "0" ]; then
        echo "    Waiting for remote instance $INST_NAME before continuing..."
        WAIT_ELAPSED=0
        while [ $WAIT_ELAPSED -lt 300 ]; do
            if ! kill -0 $PID 2>/dev/null; then
                echo "  ERROR: Instance $INST_NAME (node $INST_NODE) exited!"
                echo "  See log: $VLLM_LOG"
                exit 1
            fi
            if curl -s -o /dev/null -w "%{http_code}" "http://${INST_NODE}:${INST_PORT}/health" 2>/dev/null | grep -q "200"; then
                echo "    Remote instance $INST_NAME is ready!"
                break
            fi
            if [ $((WAIT_ELAPSED % 30)) -eq 0 ] && [ $WAIT_ELAPSED -gt 0 ]; then
                echo "      Waiting... (${WAIT_ELAPSED}s)"
            fi
            sleep 2
            WAIT_ELAPSED=$((WAIT_ELAPSED + 2))
        done
        if [ $WAIT_ELAPSED -ge 300 ]; then
            echo "  ERROR: Remote instance $INST_NAME startup timed out"
            exit 1
        fi
    fi
done


echo "${VLLM_PIDS[@]}" > "$LOG_DIR/.vllm_pids_hetero"


echo ""
echo "=========================================="
echo "[Stage 3] Waiting for all heterogeneous vLLM instances"
echo "=========================================="

MAX_WAIT=300
for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS INST_NODE_IDX <<< "${HETERO_INSTANCES[$i]}"
    INST_NODE=${NODES[$INST_NODE_IDX]}
    INST_PORT=$((VLLM_BASE_PORT + i))
    PID=${VLLM_PIDS[$i]}
    ELAPSED=0

    echo "  Waiting for instance $i ($INST_NAME, TP=$INST_TP, node=$INST_NODE, port=$INST_PORT, PID=$PID)..."

    while [ $ELAPSED -lt $MAX_WAIT ]; do


        if ! kill -0 $PID 2>/dev/null; then
            if [ "$INST_NODE" = "$(hostname)" ]; then
                echo "  ERROR: Instance $INST_NAME exited!"
            else
                echo "  ERROR: Instance $INST_NAME (node $INST_NODE) exited!"
            fi
            echo "  See log: $LOG_DIR/vllm_${INST_NAME}_${TIMESTAMP}.log"
            exit 1
        fi


        if curl -s -o /dev/null -w "%{http_code}" "http://${INST_NODE}:${INST_PORT}/health" 2>/dev/null | grep -q "200"; then
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
        exit 1
    fi
done

echo ""
echo "All $NUM_INSTANCES heterogeneous vLLM instancesis ready!"
echo "  TP layout: $(for inst in "${HETERO_INSTANCES[@]}"; do IFS=':' read -r _ tp _ _ <<< "$inst"; printf "%s " "$tp"; done)"


echo ""
echo "=========================================="
echo "[Stage 4] Starting FSDP training (Node 0: $NODE0)"
echo "=========================================="
echo "  Training GPUs: $TRAIN_GPUS"
echo "  Processes: $TRAIN_NPROC"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  Log: $TRAIN_LOG"


HETERO_INSTANCE_HOSTS=""
for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS INST_NODE_IDX <<< "${HETERO_INSTANCES[$i]}"
    INST_NODE=${NODES[$INST_NODE_IDX]}
    if [ -n "$HETERO_INSTANCE_HOSTS" ]; then
        HETERO_INSTANCE_HOSTS="${HETERO_INSTANCE_HOSTS},"
    fi
    HETERO_INSTANCE_HOSTS="${HETERO_INSTANCE_HOSTS}${INST_NODE}"
done
export HETERO_INSTANCE_HOSTS

echo "  HETERO_INSTANCE_HOSTS: $HETERO_INSTANCE_HOSTS"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
HETERO_INSTANCE_HOSTS=$HETERO_INSTANCE_HOSTS \
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
echo "Heterogeneous cluster started across nodes"
echo "=========================================="
echo ""
echo "  node: $NODE0 (Master), $NODE1 (Rollout)"
echo "  Heterogeneous vLLM instances:"
for ((i=0; i<NUM_INSTANCES; i++)); do
    IFS=':' read -r INST_NAME INST_TP INST_GPUS INST_NODE_IDX <<< "${HETERO_INSTANCES[$i]}"
    echo "    ${INST_NAME} (TP=${INST_TP}): node=${NODES[$INST_NODE_IDX]} PID=${VLLM_PIDS[$i]} port=$((VLLM_BASE_PORT + i))"
done
echo ""
echo "  Training process: PID=$TRAIN_PID (node=$NODE0)"
echo "    Log: tail -f $TRAIN_LOG"
echo ""
echo "Stop all services:"
echo "  kill ${VLLM_PIDS[*]} $TRAIN_PID"
echo "  or: pkill -f vllm.entrypoints && kill $TRAIN_PID"

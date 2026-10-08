#!/bin/bash
# =============================================================================

#


#



# =============================================================================

CLEAN_GPU=false
if [ "$1" = "--gpu" ]; then
    CLEAN_GPU=true
fi

echo "=== Starting multi-node cleanup ==="


if [ -n "$SLURM_NODELIST" ]; then
    mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
    IS_SLURM=true
    echo "Slurm environment, nodes: ${NODES[*]}"
else
    NODES=("$(hostname)")
    IS_SLURM=false
    echo "Single-node mode: ${NODES[0]}"
fi


echo ""
echo "--- Stopping vLLM-related processes ---"

if [ "$IS_SLURM" = true ]; then
    for NODE in "${NODES[@]}"; do
        echo "  node $NODE: stopping vLLM/EngineCore processes..."
        srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
            bash -c '
                pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
                pkill -9 -f "vllm" 2>/dev/null || true
                pkill -9 -f "EngineCore" 2>/dev/null || true
            ' 2>/dev/null &
    done
    wait
else
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
fi


echo ""
echo "--- Stopping training processes ---"

if [ "$IS_SLURM" = true ]; then
    for NODE in "${NODES[@]}"; do
        echo "  node $NODE: stopping torchrun/training processes..."
        srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
            bash -c '
                pkill -9 -f "torch.distributed.run" 2>/dev/null || true
                pkill -9 -f "torchrun" 2>/dev/null || true
                pkill -9 -f "trainer.py" 2>/dev/null || true
                pkill -9 -f "gsm8k_async_rl" 2>/dev/null || true
            ' 2>/dev/null &
    done
    wait
else
    pkill -9 -f "torch.distributed.run" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "trainer.py" 2>/dev/null || true
    pkill -9 -f "gsm8k_async_rl" 2>/dev/null || true
fi

sleep 2


if [ "$CLEAN_GPU" = true ]; then
    echo ""
    echo "--- Stopping processes using GPUs ---"

    if [ "$IS_SLURM" = true ]; then
        for NODE in "${NODES[@]}"; do
            echo "  node $NODE: stopping GPU workloads..."
            srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
                bash -c '
                    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
                        awk "{print \$1}" | xargs -r kill -9 2>/dev/null || true
                ' 2>/dev/null &
        done
        wait
    else
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
            awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
    fi

    sleep 2
fi


echo ""
echo "--- Checking for residual processes ---"

if [ "$IS_SLURM" = true ]; then
    for NODE in "${NODES[@]}"; do
        REMAINING=$(srun --overlap --nodes=1 --nodelist="$NODE" --ntasks=1 \
            bash -c 'ps aux | grep -E "(vllm|trainer.py|EngineCore|torchrun)" | grep -v grep | wc -l' 2>/dev/null || echo "0")
        REMAINING=$(echo "$REMAINING" | tr -d '[:space:]')
        if [ "$REMAINING" -eq 0 ] 2>/dev/null; then
            echo "  node $NODE: all processes stopped"
        else
            echo "  node $NODE: still has $REMAINING residual processes"
        fi
    done
else
    REMAINING=$(ps aux | grep -E "(vllm|trainer.py|EngineCore|torchrun)" | grep -v grep | wc -l)
    if [ "$REMAINING" -eq 0 ]; then
        echo "  all processes stopped"
    else
        echo "  still has $REMAINING residual processes:"
        ps aux | grep -E "(vllm|trainer.py|EngineCore|torchrun)" | grep -v grep
    fi
fi

echo ""
echo "=== Cleanup complete ==="

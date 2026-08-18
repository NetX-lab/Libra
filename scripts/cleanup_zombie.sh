#!/bin/bash


echo "=== Starting residual-process cleanup ==="


echo "Stopping vLLM-related processes..."
pkill -9 -f "vllm" 2>/dev/null
pkill -9 -f "EngineCore" 2>/dev/null
pkill -9 -f "trainer.py" 2>/dev/null

pkill -9 -f "torch.distributed.run" 2>/dev/null
pkill -9 -f "pilot.trainer" 2>/dev/null

echo "Stopping torchrun-related processes..."
pkill -9 -f "torch.distributed.run" 2>/dev/null
pkill -9 -f "torchrun" 2>/dev/null


sleep 2


echo "Checking GPU utilization..."
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Stopping processes using GPUs..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
        awk '{print $1}' | xargs -r kill -9 2>/dev/null
fi


sleep 2


echo "=== Cleanup complete; current GPU status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader


echo ""
echo "=== Checking for residual processes ==="
REMAINING=$(ps aux | grep -E "(vllm|trainer.py|EngineCore)" | grep -v grep | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✓ All residual processes have been stopped"
else
    echo "⚠ still has $REMAINING processes remain; inspect them manually："
    ps aux | grep -E "(vllm|trainer.py|EngineCore)" | grep -v grep
fi

echo ""
echo "=== Cleanup complete ==="

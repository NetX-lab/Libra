#!/bin/bash
# =============================================================================

#

#   sbatch scripts/submit_hetero_4gpu.slurm
#




# =============================================================================










#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=you@example.com

# =============================================================================

# =============================================================================
echo "=========================================="
echo "Job information"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# =============================================================================

# =============================================================================


source ~/rl_framework_env/bin/activate

echo "Python: $(which python3)"
echo "Python version: $(python3 --version 2>&1)"
echo "CUDA devices: $(python3 -c 'import torch; print(torch.cuda.device_count(), "GPUs")' 2>/dev/null || echo 'torch not available')"


PROJECT_DIR="/path/to/RL_Framework"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1"


mkdir -p logs

# =============================================================================

# =============================================================================
echo ""
echo "=========================================="
echo "Starting heterogeneous rollout training on 4 GPUs"
echo "=========================================="
echo "  Configuration: configs/gsm8k_hetero_4gpu.yaml"
echo "  Training: GPU 0,1 (FSDP, 2 processes)"
echo "  Rollout:  GPU 2 (TP=1), GPU 3 (TP=1)"
echo ""

bash scripts/run_hetero_distributed_4GPU.sh --config /path/to/RL_Framework/configs/gsm8k_hetero_4gpu.yaml


TRAIN_PID=$(cat logs/.train_pid_hetero 2>/dev/null || echo "")
if [ -n "$TRAIN_PID" ]; then
    echo "Waiting for training process (PID=$TRAIN_PID) to complete..."
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 10
    done
    echo "Training process ended"
else
    echo "WARNING: Training process PID was not found"
fi

# =============================================================================

# =============================================================================
echo ""
echo "=========================================="
echo "Job ended; cleaning processes"
echo "=========================================="
pkill -f "vllm.entrypoints" 2>/dev/null || true
echo "Job finished at: $(date)"

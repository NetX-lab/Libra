#!/bin/bash
# =============================================================================

#

#   sbatch submit_slurm.slurm
#




# =============================================================================









#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

# =============================================================================

# =============================================================================
echo "=========================================="
echo "Job information"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo ""


scontrol show hostnames "$SLURM_NODELIST"

# =============================================================================

# =============================================================================
echo "Activating virtual environment..."
source ~/rl_framework_env/bin/activate

echo "Python: $(which python3)"
echo "Python version: $(python3 --version 2>&1)"


PROJECT_DIR="/path/to/RL_Framework"
cd "$PROJECT_DIR"

export NO_PROXY="localhost,127.0.0.1"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"


mkdir -p logs

# =============================================================================

# =============================================================================


export MODEL_PATH="/path/to/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c"


export TRAIN_SCRIPT="examples/gsm8k_async_rl.py"


export CONFIG_FILE="configs/gsm8k_slurm.yaml"



export GPUS_PER_NODE=4
export TRAIN_GPUS_PER_NODE=2
export ROLLOUT_GPUS_PER_NODE=2


export VLLM_TP_SIZE=1
export VLLM_BASE_PORT=8000
export MAX_MODEL_LEN=2048


export MASTER_PORT=29500


export LOG_DIR="./logs"


export MAX_WAIT=600

# =============================================================================

# =============================================================================
echo ""
echo "=========================================="
echo "Starting multi-node asynchronous RL training"
echo "=========================================="
echo "  Model: ${MODEL_PATH}"
echo "  Training script: ${TRAIN_SCRIPT}"
echo "  Configuration file: ${CONFIG_FILE:-none}"
echo "  Nodes: ${SLURM_NNODES}"
echo "  GPUs per node: ${GPUS_PER_NODE} (Training=${TRAIN_GPUS_PER_NODE}, Rollout=${ROLLOUT_GPUS_PER_NODE})"
echo "  vLLM TP: ${VLLM_TP_SIZE}"
echo ""


bash scripts/run_slurm_distributed.sh

echo ""
echo "Job finished at: $(date)"

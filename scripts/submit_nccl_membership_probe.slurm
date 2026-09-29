#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=nsgm_xu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/nccl_membership_%j.log
set -euo pipefail
source /home-ssd/Users/nsgm_xu/rl_framework_env/bin/activate
cd /home-ssd/Users/nsgm_xu/kwchen/RL_Framework
export PYTHONPATH=/home-ssd/Users/nsgm_xu/kwchen:${PYTHONPATH:-}
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
mapfile -t probe_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
timeout 240 srun --kill-on-bad-exit=1 python scripts/nccl_membership_probe.py --host "${probe_nodes[0]}"

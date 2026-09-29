#!/bin/bash
#SBATCH --job-name=libra_sched_rl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home-ssd/Users/nsgm_xu/kwchen/RL_Framework
export MODEL_PATH=/home-ssd/Users/nsgm_xu/kwchen/hf_models/Qwen3-4B
export PYTHONPATH=/home-ssd/Users/nsgm_xu/kwchen
exec /home-ssd/Users/nsgm_xu/rl_framework_env/bin/python3 scripts/run_scheduler_rl_comparison.py

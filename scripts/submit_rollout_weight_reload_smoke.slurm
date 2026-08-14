#!/usr/bin/env bash
#SBATCH -J rollout_reload_smoke
#SBATCH -o logs/rollout_reload_smoke_%j.out
#SBATCH -e logs/rollout_reload_smoke_%j.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
VENV_PATH="${VENV_PATH:-${HOME}/rl_framework_env/bin/activate}"
STARTUP_DELAY="${STARTUP_DELAY:-3}"

source "$VENV_PATH"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH="$(dirname "$PROJECT_DIR"):$PROJECT_DIR:${PYTHONPATH:-}"

python scripts/rollout_weight_reload_smoke.py \
    --startup-delay "$STARTUP_DELAY" \
    --timeout 90

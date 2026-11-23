#!/usr/bin/env bash
#SBATCH --job-name=libra-r2e-14b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@90

set -euo pipefail

if [[ $# -lt 2 || $1 != --source ]]; then
    echo "Usage: sbatch [site options] $0 --source SOURCE --model MODEL --tasks TASKS --run-dir RUN_DIR [--config CONFIG]"
    [[ ${1:-} == --help || ${1:-} == -h ]] && exit 0
    exit 2
fi

exec "${PYTHON:-python3}" -B "$2/scripts/run_r2e_qwen3_14b_16gpu.py" "$@"

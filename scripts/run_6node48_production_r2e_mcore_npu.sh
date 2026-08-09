#!/usr/bin/env bash
# Select one arm of the production-scale six-node EHP A/B experiment, then use
# the validated launcher and its non-destructive node preflight.
set -euo pipefail

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
arm="${EHP_MODE:-ehp}"

case "$arm" in
    ehp)
        config="configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml"
        ;;
    no_ehp)
        config="configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_no_ehp.yaml"
        ;;
    *)
        echo "EHP_MODE must be 'ehp' or 'no_ehp'" >&2
        exit 2
        ;;
esac

export CONFIG_PATH="${CONFIG_PATH:-$config}"
export RUN_ROOT="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48_production_${arm}}"
export RUN_NAME="${RUN_NAME:-production_${arm}_$(date +%Y%m%d_%H%M%S)}"

exec bash "$project_dir/scripts/run_6node48_r2e_mcore_npu.sh"

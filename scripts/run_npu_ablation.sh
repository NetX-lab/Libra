#!/usr/bin/env bash
# NPU-only entry point for the three requested A/B experiments.
# Usage: MODE=cmlfq|grp|ehp ARM=with|without ./scripts/run_npu_ablation.sh
set -euo pipefail

mode="${MODE:?set MODE to cmlfq, grp, or ehp}"
arm="${ARM:?set ARM to with or without}"
project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

case "$mode:$arm" in
  cmlfq:with)
    export CONFIG_PATH="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_5node40_100step.yaml}"
    exec bash "$project_dir/scripts/run_multinode_r2e_mcore_npu.sh"
    ;;
  cmlfq:without)
    export CONFIG_PATH="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_5node40_no_cmlfq.yaml}"
    exec bash "$project_dir/scripts/run_multinode_r2e_mcore_npu.sh"
    ;;
  grp:with)
    export RUN_ONLY=both
    exec bash "$project_dir/scripts/run_6node48_grp_vs_no_grp_equal.sh"
    ;;
  grp:without)
    export RUN_ONLY=fixed
    export FIXED_CONFIG_PATH="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_no_grp.yaml}"
    exec bash "$project_dir/scripts/run_6node48_grp_vs_no_grp_equal.sh"
    ;;
  ehp:with)
    export INTERNAL_HOSTS="${INTERNAL_HOSTS:-node-internal node-internal}"
    export SKIP_ROLLOUT_HEALTH=1
    export CONFIG_PATH="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_2node16_ehp_docker.yaml}"
    exec bash "$project_dir/scripts/run_multinode_r2e_mcore_npu.sh"
    ;;
  ehp:without)
    export INTERNAL_HOSTS="${INTERNAL_HOSTS:-node-internal node-internal}"
    export SKIP_ROLLOUT_HEALTH=1
    export CONFIG_PATH="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_2node16_no_ehp_docker.yaml}"
    exec bash "$project_dir/scripts/run_multinode_r2e_mcore_npu.sh"
    ;;
  *)
    echo "usage: MODE={cmlfq|grp|ehp} ARM={with|without} $0" >&2
    exit 2
    ;;
esac

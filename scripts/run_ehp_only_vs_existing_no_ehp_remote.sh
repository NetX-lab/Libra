#!/usr/bin/env bash
# Wait for one controlled six-node pool, run only the EHP arm, then compare it
# with an existing no-EHP history directory. Passwords stay in the process env.
set -Eeuo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD only in this process environment}"
: "${AVAILABLE_HOSTS:?set AVAILABLE_HOSTS to the six authorized candidates}"
: "${BASELINE_RUN:?set BASELINE_RUN to the existing no-EHP run directory}"

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-$project_dir}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-/opt/libra/runtime_sources}"
config_python="${CONFIG_PYTHON:-/opt/libra/envs/rl_framework_py310/bin/python}"
config_path="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48_ehp_only}"
run_name="${RUN_NAME:-formal_ehp_only_$(date +%Y%m%d_%H%M%S)}"
report_path="${REPORT_PATH:-$run_root/ehp_vs_existing_no_ehp.md}"
poll_seconds="${POLL_SECONDS:-300}"
internal_ssh="${INTERNAL_SSH:-$project_dir/scripts/internal_ssh.sh}"
ascend_env_script="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/cann-8.5.2/set_env.sh}"

read -r -a hosts <<< "${AVAILABLE_HOSTS//,/ }"
(( ${#hosts[@]} == 6 )) || { echo "exactly six candidate hosts are required" >&2; exit 2; }

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

all_hosts_idle() {
    local host output count
    for host in "${hosts[@]}"; do
        output="$($internal_ssh "$host" -- \
            "npu-smi info | grep -c -F 'No running processes found in NPU'; true" 2>&1 || true)"
        count="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
        log "host=$host idle_npus=${count:-unreachable}"
        [[ "$count" == "8" ]] || return 1
    done
}

while ! all_hosts_idle; do
    log "waiting_for_same_six_hosts_to_be_fully_idle"
    sleep "$poll_seconds"
done

mkdir -p "$run_root"
export PROJECT_DIR="$project_dir"
export RUNTIME_PROJECT_DIR="$runtime_project_dir"
export RUNTIME_PYTHONPATH="$runtime_pythonpath"
export CONFIG_PYTHON="$config_python"
export CONFIG_PATH="$config_path"
export RUN_ROOT="$run_root"
export RUN_NAME="$run_name"
export MASTER_PORT="${MASTER_PORT:-30950}"
export GRADIENT_SERVER_PORT="${GRADIENT_SERVER_PORT:-30951}"
export AVAILABLE_HOSTS
export EHP_MODE=ehp

source "$ascend_env_script"

log "running_ehp_preflight"
PREFLIGHT_ONLY=1 bash "$project_dir/scripts/run_6node48_production_r2e_mcore_npu.sh"
log "starting_ehp_arm"
PREFLIGHT_ONLY=0 bash "$project_dir/scripts/run_6node48_production_r2e_mcore_npu.sh"
log "ehp_arm_completed"

"$config_python" "$project_dir/scripts/analyze_libra_experiment.py" \
    --run "existing-no-EHP=$BASELINE_RUN" \
    --run "EHP=$run_root" \
    --output "$report_path"

cat >>"$report_path" <<EOF

## Controlled Experiment Notes

- EHP arm: 200 steps, rollout/training weight sync interval 5.
- Initial placement: selected by GRP from the six idle candidates; no fixed
  training GPU count is used by the launcher.
- Existing no-EHP baseline is reused as requested. Its recorded effective
  config must be checked separately before treating this as a strict control.
- Candidate hosts: `${AVAILABLE_HOSTS}`.
- EHP run directory: `$run_root`.
EOF

log "comparison_complete report=$report_path"

#!/usr/bin/env bash
# Wait for four idle nodes, then run the two controlled arms serially on the
# same hosts and generate a Markdown comparison report.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"
: "${CANDIDATE_HOSTS:?set CANDIDATE_HOSTS to the authorized host list}"

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-/opt/libra/runtime_sources/RL_Framework_NPU}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_4node32_grp_ab}"
poll_seconds="${POLL_SECONDS:-300}"
candidate_hosts_value="$CANDIDATE_HOSTS"
read -r -a candidate_hosts <<< "$candidate_hosts_value"
mkdir -p "$run_root"

select_idle_hosts() {
  local host output idle
  selected_hosts=()
  export INTERNAL_SSH_TIMEOUT=12
  for host in "${candidate_hosts[@]}"; do
    output="$($project_dir/scripts/internal_ssh.sh "$host" -- \
      "npu-smi info | grep -c -F 'No running processes found in NPU'; true" 2>&1 || true)"
    idle="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
    printf '%s idle_npus=%s\n' "$host" "${idle:-unreachable}"
    if [[ "$idle" == "8" ]]; then
      selected_hosts+=("$host")
      [[ "${#selected_hosts[@]}" -eq 4 ]] && return 0
    fi
  done
  return 1
}

while ! select_idle_hosts; do
  echo "$(date --iso-8601=seconds) waiting_for_four_idle_nodes"
  sleep "$poll_seconds"
done
fixed_hosts="${selected_hosts[*]}"
echo "$(date --iso-8601=seconds) selected_hosts=$fixed_hosts"
printf '%s\n' "$fixed_hosts" >"${run_root}/selected_hosts.txt"

timestamp="$(date +%Y%m%d_%H%M%S)"
grp_name="formal_4node32_grp_${timestamp}"
no_grp_name="formal_4node32_no_grp_${timestamp}"

# Run GRP first so its placement decision is made from an entirely idle pool;
# the no-GRP arm then reuses the exact same physical hosts.
ARM=grp AVAILABLE_HOSTS="$fixed_hosts" RUN_NAME="$grp_name" MASTER_PORT=30740 \
  RUN_ROOT="$run_root" bash "$project_dir/scripts/run_4node32_grp_ab_npu.sh"

while ! select_idle_hosts || [[ "${selected_hosts[*]}" != "$fixed_hosts" ]]; do
  echo "$(date --iso-8601=seconds) waiting_for_same_hosts_before_no_grp"
  sleep 60
done

ARM=no_grp AVAILABLE_HOSTS="$fixed_hosts" RUN_NAME="$no_grp_name" MASTER_PORT=30760 \
  RUN_ROOT="$run_root" bash "$project_dir/scripts/run_4node32_grp_ab_npu.sh"

report="${run_root}/grp_vs_no_grp_${timestamp}.md"
/opt/libra/envs/rl_framework_py310/bin/python \
  "$project_dir/scripts/analyze_libra_experiment.py" \
  --run "no-GRP-equal=${run_root}/${no_grp_name}" \
  --run "GRP=${run_root}/${grp_name}" \
  --history-root "${run_root}/history" \
  --output "$report"
printf 'STATUS=complete GRP_RUN=%s NO_GRP_RUN=%s REPORT=%s\n' \
  "$grp_name" "$no_grp_name" "$report"

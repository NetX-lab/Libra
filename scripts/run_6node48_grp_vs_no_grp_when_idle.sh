#!/usr/bin/env bash
# Wait for the authorized six-host pool, then run the controlled comparison.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
candidate_hosts_value="${AVAILABLE_HOSTS:?set AVAILABLE_HOSTS to the authorized six-host pool}"
poll_seconds="${POLL_SECONDS:-60}"
read -r -a hosts <<< "${candidate_hosts_value//,/ }"
[[ "${#hosts[@]}" -eq 6 ]] || { echo "AVAILABLE_HOSTS must contain six hosts" >&2; exit 2; }

all_idle() {
  local host output idle
  export INTERNAL_SSH_TIMEOUT=15
  for host in "${hosts[@]}"; do
    output="$($project_dir/scripts/internal_ssh.sh "$host" -- \
      "npu-smi info | grep -c -F 'No running processes found in NPU'; true" 2>&1 || true)"
    idle="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
    printf '%s idle_npus=%s\n' "$host" "${idle:-unreachable}"
    [[ "$idle" == "8" ]] || return 1
  done
}

while ! all_idle; do
  echo "$(date --iso-8601=seconds) waiting_for_six_idle_hosts"
  sleep "$poll_seconds"
done

echo "$(date --iso-8601=seconds) launching_grp_vs_no_grp hosts=${hosts[*]}"
exec env AVAILABLE_HOSTS="${hosts[*]}" \
  bash "$project_dir/scripts/run_6node48_grp_vs_no_grp_equal.sh"

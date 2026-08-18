#!/usr/bin/env bash
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_root="${RUN_ROOT:-./runs/r2e_6node48_grp_vs_no_grp_equal}"
poll_seconds="${POLL_SECONDS:-120}"
probe_script="${PROBE_SCRIPT:?set PROBE_SCRIPT to a host-idleness probe}"
candidate_hosts="${CANDIDATE_HOSTS:?set CANDIDATE_HOSTS}"
vllm_hosts="${VLLM_HOSTS:?set VLLM_HOSTS}"
reference_run="${GRP_REFERENCE_RUN_NAME:?set GRP_REFERENCE_RUN_NAME}"
fixed_config_path="${FIXED_CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_6node48_no_grp_v25_optimizer_reset_75step.yaml}"
model_path="${MODEL_PATH:?set MODEL_PATH}"

while true; do
  probe_output="$(python3 "$probe_script" 2>/dev/null || true)"
  mapfile -t idle_hosts < <(
    awk -v allowed="$candidate_hosts" '
      BEGIN { split(allowed, a, " "); for (i in a) ok[a[i]]=1 }
      $2 == "idle=8" && ok[$1] { print $1 }
    ' <<<"$probe_output"
  )
  mapfile -t rollout_hosts < <(
    awk -v allowed="$vllm_hosts" '
      BEGIN { split(allowed, a, " "); for (i in a) ok[a[i]]=1 }
      $2 == "idle=8" && ok[$1] { print $1 }
    ' <<<"$probe_output" | head -n 3
  )
  train_hosts=()
  for host in "${idle_hosts[@]}"; do
    [[ " ${rollout_hosts[*]} " == *" $host "* ]] && continue
    train_hosts+=("$host")
    [[ "${#train_hosts[@]}" -eq 3 ]] && break
  done
  if [[ "${#rollout_hosts[@]}" -eq 3 && "${#train_hosts[@]}" -eq 3 ]]; then
    # The fixed placement planner assigns the first 24 devices to Training and
    # the final 24 to Rollout, so vLLM-capable hosts must be listed last.
    hosts="${train_hosts[*]} ${rollout_hosts[*]}"
    echo "$(date --iso-8601=seconds) launching_fixed_24_24 hosts=$hosts"
    exec env \
      AVAILABLE_HOSTS="$hosts" \
      RUN_ONLY=fixed \
      GRP_REFERENCE_RUN_NAME="$reference_run" \
      FIXED_CONFIG_PATH="$fixed_config_path" \
      FIXED_RUN_NAME="${FIXED_RUN_NAME:-formal_6node48_no_grp_v25_optimizer_reset_75step_$(date +%Y%m%d_%H%M%S)}" \
      MODEL_PATH="$model_path" \
      RUN_ROOT="$run_root" \
      FIXED_MASTER_PORT="${FIXED_MASTER_PORT:-31460}" \
      HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-53000}" \
      VLLM_REQUEST_RETRIES="${VLLM_REQUEST_RETRIES:-3}" \
      bash "$project_dir/scripts/run_6node48_grp_vs_no_grp_equal.sh"
  fi
  echo "$(date --iso-8601=seconds) waiting_for_compatible_hosts idle_count=${#idle_hosts[@]} rollout_capable_idle=${#rollout_hosts[@]} train_candidates=${#train_hosts[@]}"
  sleep "$poll_seconds"
done

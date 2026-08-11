#!/usr/bin/env bash
set -u

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
job_pattern="${1:-formal_100step_6node48}"
train_hosts=(192.0.2.10 192.0.2.11 192.0.2.12 192.0.2.13)
export INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}"

check_command="npu-smi info | grep -c -F 'No running processes found in NPU'; pgrep -af '${job_pattern}|elastic_hybrid_worker.py|master_port=29747' || true"

echo '=== jump rollout node 192.0.2.20 ==='
npu-smi info | grep -c -F 'No running processes found in NPU' || true
pgrep -af "${job_pattern}|elastic_hybrid_worker.py|master_port=29747" || true

echo '=== rollout node 192.0.2.21 ==='
NODE_PASSWORD="${NODE_PASSWORD:-}" INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}" \
  "${project_dir}/scripts/internal_ssh.sh" 192.0.2.21 -- "${check_command}" || true

for host in "${train_hosts[@]}"; do
  echo "=== training node ${host} ==="
  "${project_dir}/scripts/internal_ssh.sh" "${host}" -- "${check_command}" || true
done

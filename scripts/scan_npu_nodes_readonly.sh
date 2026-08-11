#!/usr/bin/env bash
# Read-only NPU capacity probe through the project jump-host SSH helper.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD only in the invoking process environment}"
project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
export INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-15}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 host [host ...]" >&2
  exit 2
fi

probe_command="hostname; printf '__IDLE_NPU__='; npu-smi info 2>/dev/null | grep -c -F 'No running processes found in NPU' || true; printf '__NPU_PROCESSES__='; npu-smi info 2>/dev/null | grep -E 'python|vllm|ray|torch|mind' | wc -l || true; test -x /opt/libra/envs/vllm_ascend/bin/python && echo '__VLLM_ENV__=1' || echo '__VLLM_ENV__=0'; test -f /opt/libra/models/Qwen3-14B/config.json && echo '__MODEL__=1' || echo '__MODEL__=0'; test -f /opt/libra/runtime_sources/RL_Framework_NPU/examples/r2e_gym_async_rl.py && echo '__RUNTIME__=1' || echo '__RUNTIME__=0'; ip -o link show eth0 >/dev/null 2>&1 && echo '__HCCL_IFACE__=1' || echo '__HCCL_IFACE__=0'"

for host in "$@"; do
  echo "===NODE ${host}==="
  NODE_PASSWORD="$NODE_PASSWORD" \
    "$project_dir/scripts/internal_ssh.sh" "$host" -- "$probe_command" \
    2>&1 || echo "__PROBE_FAILED__=1"
done

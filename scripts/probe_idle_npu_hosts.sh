#!/usr/bin/env bash
# Read-only helper: report the number of NPUs without running processes.
set -u

: "${NODE_PASSWORD:?set NODE_PASSWORD}"
internal_ssh="${INTERNAL_SSH:-/opt/libra/RL_Framework_NPU/scripts/internal_ssh.sh}"
export INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-20}"

for host in "$@"; do
    output="$($internal_ssh "$host" -- \
        "npu-smi info | grep -c -F 'No running processes found in NPU'; true" 2>&1)"
    count="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
    printf '%s idle_npus=%s\n' "$host" "${count:-unreachable}"
done

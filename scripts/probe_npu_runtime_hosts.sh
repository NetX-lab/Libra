#!/usr/bin/env bash
# Read-only inventory for selecting training and rollout nodes.
set -u

: "${NODE_PASSWORD:?set NODE_PASSWORD}"
internal_ssh="${INTERNAL_SSH:-/opt/libra/RL_Framework_NPU/scripts/internal_ssh.sh}"
export INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-20}"

for host in "$@"; do
    output="$($internal_ssh "$host" -- \
        "idle=\$(npu-smi info | grep -c -F 'No running processes found in NPU'); train=0; rollout=0; test -x /opt/libra/envs/rl_mindspeed/bin/python && test -f /opt/libra/runtime_sources/RL_Framework_NPU/examples/r2e_gym_async_rl.py && train=1; test -x /opt/libra/envs/vllm_ascend/bin/python && test -f /opt/libra/runtime_sources/RL_Framework_NPU/scripts/start_r2e_rollout_pool_npu.sh && rollout=1; echo RUNTIME_PROBE idle=\$idle train=\$train rollout=\$rollout; true" 2>&1)"
    summary="$(printf '%s\n' "$output" | tr -d '\r' | grep 'RUNTIME_PROBE' | tail -n 1)"
    printf '%s %s\n' "$host" "${summary:-RUNTIME_PROBE unreachable}"
done

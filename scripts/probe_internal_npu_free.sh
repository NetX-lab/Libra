#!/usr/bin/env bash
# Read-only probe for Ascend devices on hosts reachable from a jump host.
# Requires NODE_PASSWORD in the environment only when SSH keys are unavailable.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD before invoking this script}"

for host in "$@"; do
  TARGET_IP="$host" expect <<'EXPECT'
set timeout 20
set host $env(TARGET_IP)
set password $env(NODE_PASSWORD)
set command "npu-smi info | grep -c -F 'No running processes found in NPU'"
spawn ssh -tt -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$host $command
expect {
  "password:" { send -- "$password\r"; exp_continue }
  eof {}
}
EXPECT
done

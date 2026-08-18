#!/usr/bin/env bash
# Execute one explicitly supplied command on each internal host through a
# password-authenticated jump host.  No command is run unless the caller
# supplies it after `--`.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD before invoking this script}"
: "${INTERNAL_SSH_TIMEOUT:=120}"

hosts=()
while [[ "$#" -gt 0 && "$1" != "--" ]]; do
  hosts+=("$1")
  shift
done

if [[ "$#" -eq 0 || "${#hosts[@]}" -eq 0 ]]; then
  echo "usage: $0 host [host ...] -- command" >&2
  exit 2
fi
shift

for host in "${hosts[@]}"; do
  TARGET_IP="$host" REMOTE_COMMAND="$*" expect <<'EXPECT'
set timeout $env(INTERNAL_SSH_TIMEOUT)
set host $env(TARGET_IP)
set command $env(REMOTE_COMMAND)
set password $env(NODE_PASSWORD)
spawn ssh -tt \
  -o StrictHostKeyChecking=no \
  -o ConnectTimeout=15 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=120 \
  -o TCPKeepAlive=yes \
  root@$host $command
expect {
  "password:" { send -- "$password\r"; exp_continue }
  eof {}
}
catch wait result
exit [lindex $result 3]
EXPECT
done

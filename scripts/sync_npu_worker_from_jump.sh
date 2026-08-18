#!/usr/bin/env bash
# Synchronize the already-validated local Python environment and source tree
# from the jump host to one otherwise-idle NPU worker.  This intentionally
# never uses --delete and therefore cannot remove user files on a worker.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD before invoking this script}"
host="${1:?usage: sync_npu_worker_from_jump.sh INTERNAL_HOST}"
remote_root="${REMOTE_WORKSPACE:-.}"

run_rsync() {
  local source_path="$1"
  local destination_path="$2"
  shift 2
  TARGET_IP="$host" SOURCE_PATH="$source_path" DESTINATION_PATH="$destination_path" \
    RSYNC_EXCLUDES="$*" expect <<'EXPECT'
set timeout -1
set host $env(TARGET_IP)
set source_path $env(SOURCE_PATH)
set destination_path $env(DESTINATION_PATH)
set password $env(NODE_PASSWORD)
set excludes [split $env(RSYNC_EXCLUDES) "|"]
set command [list rsync -a --human-readable --info=name,stats]
foreach entry $excludes {
  if {$entry ne ""} { lappend command "--exclude=$entry" }
}
lappend command -e {ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15}
lappend command $source_path root@$host:$destination_path
eval spawn $command
expect {
  "password:" { send -- "$password\r"; exp_continue }
  eof {}
}
catch wait result
exit [lindex $result 3]
EXPECT
}

run_rsync ./venv/framework/ "${remote_root}/venv/framework/" ""
run_rsync ./ "${remote_root}/RL_Framework_npu/" \
  ".git|.venv|.venv-npu|__pycache__|*.pyc|logs|wandb*|runs|data_nv2"

# The package import root is configurable so the script does not assume a
# particular account home directory on worker nodes.
TARGET_IP="$host" expect <<'EXPECT'
set timeout 30
set host $env(TARGET_IP)
set password $env(NODE_PASSWORD)
spawn ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$host \
  {test -d RL_Framework_npu}
expect {
  "password:" { send -- "$password\r"; exp_continue }
  eof {}
}
catch wait result
exit [lindex $result 3]
EXPECT

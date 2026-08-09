#!/usr/bin/env bash
set -u
for pid in $(npu-smi info 2>/dev/null | awk -F'|' '/python|VLLM|rayWorker|rayCheckpoint/ {gsub(/ /,"",$3); if ($3 ~ /^[0-9]+$/) print $3}' | sort -u); do
  ps -o user=,pid=,ppid=,etime=,cmd= -p "$pid" 2>/dev/null || true
done

#!/usr/bin/env bash
# Discover SSH-speaking hosts in an explicitly authorized NPU subnet.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 subnet-prefix" >&2
  echo "example: $0 192.0.2" >&2
  exit 2
fi

subnet="$1"
first="${SCAN_FIRST:-1}"
last="${SCAN_LAST:-254}"
parallelism="${SCAN_PARALLELISM:-48}"

seq "$first" "$last" | xargs -P "$parallelism" -I{} bash -c '
  ip="$1.$2"
  if timeout 2 bash -c ">/dev/tcp/${ip}/22" 2>/dev/null; then
    printf "%s\n" "$ip"
  fi
' _ "$subnet" {}

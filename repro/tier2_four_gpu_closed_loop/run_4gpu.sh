#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/repro/tier2_four_gpu_closed_loop/launch_4gpu.py" "$@"

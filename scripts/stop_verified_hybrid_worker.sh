#!/usr/bin/env bash
# Stop only the exact stale worker from the prior validated run.
set -euo pipefail
pid="${1:?pid required}"
cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
case "$cmd" in
  *elastic_hybrid_worker.py*"/opt/libra/runs/r2e_gym_qwen3_14b_6node48/elastic_training_state"*)
    echo "verified pid=${pid}: ${cmd}"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || exit 0
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    ;;
  *)
    echo "refusing unverified pid=${pid}: ${cmd}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
arm="${1:?usage: collect_deepseek_v4_r2e_112_metrics.sh baseline|cmlfq|grp|ehp}"
case "$arm" in baseline|cmlfq|grp|ehp) ;; *) echo "unsupported arm: $arm" >&2; exit 2 ;; esac
base=/data/qianzhirong/runtime_sources/Libra_Benchmark_20261009
image=libra-v4-cann91-unified:20261009
log_dir="/libra-work/benchmark_r2e_112_20261010/$arm/logs"
output="/libra-work/benchmark_r2e_112_20261010/$arm/metrics.json"
if [[ "$arm" == baseline ]]; then
  baseline_arg=()
else
  baseline_arg=(--baseline /libra-work/benchmark_r2e_112_20261010/baseline/metrics.json)
fi
docker run --rm --network host \
  -v "$base":/libra-work \
  --entrypoint /bin/bash "$image" -lc \
  "python3 /libra-work/RL_Framework/examples/collect_r2e_metrics.py '$log_dir' --warmup-steps 1 ${baseline_arg[*]} --output '$output'"

#!/usr/bin/env bash
set -euo pipefail
arm="${1:?baseline, cmlfq, grp, or ehp}"
rank="${2:?node rank 0-7}"
master_addr="${3:-${LIBRA_MASTER_ADDR:-192.168.0.50}}"
case "$arm" in baseline|cmlfq|grp|ehp) ;; *) echo "unsupported arm: $arm" >&2; exit 2 ;; esac
base=/data/qianzhirong/runtime_sources/Libra_Benchmark_20261009
image=libra-v4-cann91-unified:20261009
name="libra-v4-r2e-112-${arm}-20261010"
run_dir="$base/benchmark_r2e_112_20261010/$arm"
[[ "$(npu-smi info | grep -c 'No running processes found in NPU')" -eq 8 ]] || {
  echo "busy NPUs on $(hostname)" >&2; exit 3;
}
docker image inspect "$image" >/dev/null
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "container already exists: $name" >&2; exit 4
fi
npu_devices=(--device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
for i in {0..7}; do
  optical="$(hccn_tool -i "$i" -optical -g 2>/dev/null)"
  lldp="$(hccn_tool -i "$i" -lldp -g 2>/dev/null)"
  grep -Eq 'present[[:space:]]*:[[:space:]]*present' <<<"$optical" || {
    echo "NPU $i optical module is absent on $(hostname)" >&2; exit 5;
  }
  grep -q 'System Name TLV' <<<"$lldp" || {
    echo "NPU $i HCCN link has no LLDP neighbor on $(hostname)" >&2; exit 6;
  }
done
docker run --rm --privileged --network host --shm-size 8g "${npu_devices[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /data:/data:ro \
  -v "$base":/libra-work -e LIBRA_SOURCE_PARENT=/libra-work \
  -e LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 \
  -e LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 \
  -e LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3 \
  --entrypoint /bin/bash "$image" -lc \
  'source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh >/dev/null; python3 -c '\''import torch, torch_npu; n=torch.npu.device_count(); assert torch.npu.is_available() and n == 8, f"expected 8 available NPUs, got {n}"; print(f"NPU preflight passed: {n} devices")'\'''
mkdir -p "$run_dir/logs"
docker run -d --name "$name" --privileged --network host --shm-size 64g \
  "${npu_devices[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /data:/data:ro \
  -v "$base":/libra-work -e LIBRA_SOURCE_PARENT=/libra-work \
  -e NODE_RANK="$rank" -e MASTER_ADDR="$master_addr" -e LIBRA_BENCH_ARM="$arm" -e OMP_NUM_THREADS=4 -e R2E_MAX_TURNS=2 \
  -e R2E_GYM_INDEX=/data/qianzhirong/runtime_sources/RL_Framework_npu_validated_20260805/data/r2e_gym_v1/index.jsonl \
  --entrypoint /bin/bash "$image" -c \
  'set -o pipefail; log="/libra-work/benchmark_r2e_112_20261010/'"$arm"'/logs/node_${NODE_RANK}.log"; export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3; export HCCL_IF_BASE_PORT=$((60000 + NODE_RANK * 100)); export HCCL_HOST_SOCKET_PORT_RANGE=auto; source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh > "$log" 2>&1; python3 /libra-work/RL_Framework/scripts/patch_mindspeed_gloo_groups.py >> "$log" 2>&1; cfg=/libra-work/RL_Framework/configs/deepseek_v4_r2e_112_'"$arm"'.yaml; if [ "$NODE_RANK" = 0 ] && { [ "$LIBRA_BENCH_ARM" = grp ] || [ "$LIBRA_BENCH_ARM" = ehp ]; }; then python3 /libra-work/RL_Framework/scripts/deepseek_v4_grp_preflight.py --config "$cfg" --output /libra-work/benchmark_r2e_112_20261010/'"$arm"'/planned.yaml >> "$log" 2>&1 || exit 11; fi; if [ "$LIBRA_BENCH_ARM" = grp ] || [ "$LIBRA_BENCH_ARM" = ehp ]; then for i in $(seq 1 1800); do [ -f /libra-work/benchmark_r2e_112_20261010/'"$arm"'/planned.yaml ] && break; sleep 1; done; test -s /libra-work/benchmark_r2e_112_20261010/'"$arm"'/planned.yaml || { echo GRP_PREFLIGHT_MISSING; exit 10; }; cfg=/libra-work/benchmark_r2e_112_20261010/'"$arm"'/planned.yaml; fi; timeout 10800 python3 -m torch.distributed.run --master-addr="$MASTER_ADDR" --master-port=29759 --nnodes=8 --nproc_per_node=8 --node_rank="$NODE_RANK" /libra-work/RL_Framework/examples/r2e_gym_async_rl.py --config "$cfg" >> "$log" 2>&1; code=$?; echo "$code" > "/libra-work/benchmark_r2e_112_20261010/'"$arm"'/logs/node_${NODE_RANK}.exit"; exit "$code"'

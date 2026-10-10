#!/usr/bin/env bash
set -euo pipefail

arm="${1:?baseline, cmlfq, ehp, or grp}"
rank="${2:?node rank}"
case "$arm" in baseline|cmlfq|ehp|grp) ;; *) echo "invalid arm: $arm" >&2; exit 2 ;; esac

base=/data/qianzhirong/runtime_sources/Libra_Benchmark_20261009
image=libra-v4-cann91-unified:20261009
container="libra-v4-128-${arm}-20261009"
run_dir="$base/benchmark_128_tp8_20261009/$arm"
[[ "$(npu-smi info | grep -c 'No running processes found in NPU')" -eq 8 ]] || {
  echo "busy NPUs on $(hostname)" >&2
  exit 3
}
docker image inspect "$image" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "container already exists: $container" >&2
  exit 4
fi
mkdir -p "$run_dir/logs"

docker run -d --name "$container" --network host --shm-size 64g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /data:/data:ro \
  -v "$base":/libra-work -e LIBRA_SOURCE_PARENT=/libra-work \
  -e NODE_RANK="$rank" -e OMP_NUM_THREADS=4 \
  -e LIBRA_BENCH_ARM="$arm" \
  -e LIBRA_BENCH_RESULT="/libra-work/benchmark_128_tp8_20261009/$arm/result.json" \
  --entrypoint /bin/bash "$image" -c \
  'set -o pipefail; log="/libra-work/benchmark_128_tp8_20261009/'"$arm"'/logs/node_${NODE_RANK}.log"; export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3; source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh > "$log" 2>&1; python3 /libra-work/RL_Framework/scripts/patch_mindspeed_gloo_groups.py >> "$log" 2>&1; timeout 10800 python3 -m torch.distributed.run --master-addr=192.168.0.50 --master-port=29749 --nnodes=8 --nproc_per_node=8 --node_rank="$NODE_RANK" /libra-work/RL_Framework/examples/benchmark_deepseek_v4_128.py --config /libra-work/RL_Framework/configs/deepseek_v4_128_'"$arm"'.yaml >> "$log" 2>&1; code=$?; echo "$code" > "/libra-work/benchmark_128_tp8_20261009/'"$arm"'/logs/node_${NODE_RANK}.exit"; exit "$code"'

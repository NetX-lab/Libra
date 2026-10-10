#!/usr/bin/env bash
set -euo pipefail
arm="${1:?baseline or cmlfq}"
rank="${2:?node rank 0-7}"
case "$arm" in baseline|cmlfq) ;; *) echo "unsupported arm: $arm" >&2; exit 2 ;; esac
base=/data/qianzhirong/runtime_sources/Libra_Benchmark_20261009
image=libra-v4-cann91-unified:20261009
name="libra-v4-r2e-120-${arm}-20261010"
run_dir="$base/benchmark_r2e_120_20261010/$arm"
[[ "$(npu-smi info | grep -c 'No running processes found in NPU')" -eq 8 ]] || {
  echo "busy NPUs on $(hostname)" >&2; exit 3;
}
docker image inspect "$image" >/dev/null
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "container already exists: $name" >&2; exit 4
fi
mkdir -p "$run_dir/logs"
docker run -d --name "$name" --network host --shm-size 64g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /data:/data:ro \
  -v "$base":/libra-work -e LIBRA_SOURCE_PARENT=/libra-work \
  -e NODE_RANK="$rank" -e OMP_NUM_THREADS=4 -e R2E_MAX_TURNS=2 \
  -e R2E_GYM_INDEX=/data/qianzhirong/runtime_sources/RL_Framework_npu_validated_20260805/data/r2e_gym_v1/index.jsonl \
  --entrypoint /bin/bash "$image" -c \
  'set -o pipefail; log="/libra-work/benchmark_r2e_120_20261010/'"$arm"'/logs/node_${NODE_RANK}.log"; export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3; export HCCL_IF_BASE_PORT=$((60000 + NODE_RANK * 100)); export HCCL_HOST_SOCKET_PORT_RANGE=auto; source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh > "$log" 2>&1; python3 /libra-work/RL_Framework/scripts/patch_mindspeed_gloo_groups.py >> "$log" 2>&1; timeout 10800 python3 -m torch.distributed.run --master-addr=192.168.0.50 --master-port=29759 --nnodes=8 --nproc_per_node=8 --node_rank="$NODE_RANK" /libra-work/RL_Framework/examples/r2e_gym_async_rl.py --config /libra-work/RL_Framework/configs/deepseek_v4_r2e_120_'"$arm"'.yaml >> "$log" 2>&1; code=$?; echo "$code" > "/libra-work/benchmark_r2e_120_20261010/'"$arm"'/logs/node_${NODE_RANK}.exit"; exit "$code"'

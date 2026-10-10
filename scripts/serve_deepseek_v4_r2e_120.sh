#!/usr/bin/env bash
set -euo pipefail
cluster="${1:?a or b}"
case "$cluster" in
  a) head_ip=192.168.0.217; ray_port=6769; dp=4; api_port=8010; dp_nodes=192.168.0.217,192.168.0.2,192.168.0.89,192.168.0.71 ;;
  b) head_ip=192.168.0.109; ray_port=6779; dp=3; api_port=8011; dp_nodes=192.168.0.109,192.168.0.51,192.168.0.192 ;;
  *) exit 2 ;;
esac
base=/data/qianzhirong/runtime_sources/Libra_Benchmark_20261009
image=libra-v4-cann91-unified:20261009
name="libra-v4-r2e-120-api-${cluster}-20261010"
if docker container inspect "$name" >/dev/null 2>&1; then exit 4; fi
mkdir -p "$base/benchmark_r2e_120_20261010/rollout/logs"
mkdir -p "/tmp/libra-r2e-120-ray-${cluster}"
docker run -d --name "$name" --privileged --network host --shm-size 64g \
  -v "/tmp/libra-r2e-120-ray-${cluster}:/tmp/ray" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /data:/data:ro \
  -v "$base":/libra-work -e LIBRA_SOURCE_PARENT=/libra-work \
  -e VLLM_HOST_IP="$head_ip" -e RAY_ADDRESS="${head_ip}:${ray_port}" \
  -e DP="$dp" -e API_PORT="$api_port" -e LIBRA_RAY_DP_NODE_IPS="$dp_nodes" --entrypoint /bin/bash "$image" -c \
  'set -o pipefail; log="/libra-work/benchmark_r2e_120_20261010/rollout/logs/api_${VLLM_HOST_IP}.log"; export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3; source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh > "$log" 2>&1; export VLLM_USE_V1=1 HCCL_OP_EXPANSION_MODE=AIV; python3 /libra-work/RL_Framework/scripts/patch_vllm_ray_dp_anchor.py >> "$log" 2>&1; exec python3 -m vllm.entrypoints.openai.api_server --model /data/l00619320/models/DeepSeek-V4-Flash-DSpark-BF16 --served-model-name DeepSeek-V4-Flash --tensor-parallel-size 8 --data-parallel-size "$DP" --data-parallel-address "$VLLM_HOST_IP" --enable-expert-parallel --distributed-executor-backend ray --gpu-memory-utilization 0.72 --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 32 --trust-remote-code --enforce-eager --host 0.0.0.0 --port "$API_PORT" >> "$log" 2>&1'

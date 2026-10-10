#!/usr/bin/env bash
# Run inside the Ray head container so Ray, vLLM and HCCL share namespaces.
set -euo pipefail
cluster="${1:?a or b}"
case "$cluster" in
  a) head_ip=192.168.0.2; ray_port=6769; dp=4; api_port=8010; dp_nodes=192.168.0.2,192.168.0.89,192.168.0.71,192.168.0.217 ;;
  b) head_ip=192.168.0.109; ray_port=6779; dp=2; api_port=8011; dp_nodes=192.168.0.109,192.168.0.192 ;;
  *) exit 2 ;;
esac
export LIBRA_SOURCE_PARENT=/libra-work
export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0
export LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0
export LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3
source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh
export VLLM_HOST_IP="$head_ip" RAY_ADDRESS="${head_ip}:${ray_port}"
export LIBRA_RAY_DP_NODE_IPS="$dp_nodes" VLLM_USE_V1=1 HCCL_OP_EXPANSION_MODE=AIV
python3 /libra-work/RL_Framework/scripts/patch_vllm_ray_dp_anchor.py
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /data/l00619320/models/DeepSeek-V4-Flash-DSpark-BF16 \
  --served-model-name DeepSeek-V4-Flash \
  --tensor-parallel-size 8 --data-parallel-size "$dp" \
  --data-parallel-address "$head_ip" \
  --enable-expert-parallel --distributed-executor-backend ray \
  --gpu-memory-utilization 0.72 --max-model-len 4096 \
  --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --trust-remote-code --enforce-eager --host 0.0.0.0 --port "$api_port"

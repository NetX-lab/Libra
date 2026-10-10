#!/usr/bin/env bash
set -euo pipefail
cluster="${1:?a, b, or c}"
node_ip="${2:?node IP}"
role="${3:?head or worker}"
case "$cluster" in
  a) head_ip=192.168.0.2; ray_port=6769 ;;
  b) head_ip=192.168.0.109; ray_port=6779 ;;
  c) head_ip=192.168.0.192; ray_port=6789 ;;
  *) echo "invalid cluster: $cluster" >&2; exit 2 ;;
esac
case "$role" in head|worker) ;; *) exit 2 ;; esac
base=/data/qianzhirong/runtime_sources/Libra_Benchmark_20261009
image=libra-v4-cann91-unified:20261009
name="libra-v4-r2e-120-ray-${cluster}-20261010"
[[ "$(npu-smi info | grep -c 'No running processes found in NPU')" -eq 8 ]] || exit 3
docker image inspect "$image" >/dev/null
if docker container inspect "$name" >/dev/null 2>&1; then exit 4; fi
mkdir -p "$base/benchmark_r2e_120_20261010/rollout/logs"
mkdir -p "/tmp/libra-r2e-120-ray-${cluster}"
docker run -d --name "$name" --privileged --network host --shm-size 64g \
  -v "/tmp/libra-r2e-120-ray-${cluster}:/tmp/ray" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /data:/data:ro \
  -v "$base":/libra-work -e LIBRA_SOURCE_PARENT=/libra-work \
  -e NODE_IP="$node_ip" -e VLLM_HOST_IP="$node_ip" -e HEAD_IP="$head_ip" \
  -e RAY_PORT="$ray_port" -e RAY_ROLE="$role" --entrypoint /bin/bash "$image" -c \
  'set -e; export LIBRA_CANN_TOOLKIT_ROOT=/data/image/libra-cann91-toolkit-installed/cann-9.1.0 LIBRA_CANN_OPS_ROOT=/tmp/libra-cann91-ops/cann-9.1.0 LIBRA_CANN_KB_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3; source /libra-work/RL_Framework/scripts/mindspeed_v4_env.sh; log="/libra-work/benchmark_r2e_120_20261010/rollout/logs/ray_${NODE_IP}.log"; if [ "$RAY_ROLE" = head ]; then ray start --head --port="$RAY_PORT" --node-ip-address="$NODE_IP" --resources="{\"NPU\":8,\"rollout_node\":1}" > "$log" 2>&1; else ray start --address="${HEAD_IP}:${RAY_PORT}" --node-ip-address="$NODE_IP" --resources="{\"NPU\":8,\"rollout_node\":1}" > "$log" 2>&1; fi; tail -f /dev/null'

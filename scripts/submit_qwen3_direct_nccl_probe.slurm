#!/usr/bin/env bash
#SBATCH -J qwen3-direct-nccl
#SBATCH -A nsgm_xu
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --exclude=gn021
#SBATCH -N 1
#SBATCH -G 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=02:00:00

set -euo pipefail

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
TRAIN_TP="${TRAIN_TP:-2}"
TRAIN_EP="${TRAIN_EP:-1}"
BASE_PORT="${BASE_PORT:-18500}"
TRANSPORT_PORT="${TRANSPORT_PORT:-29620}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
PROBE_REPEATS="${PROBE_REPEATS:-1}"
ROOT="$HOME/kwchen/rlinf_baseline"
PROJECT="$ROOT/libra_nccl_dev/RL_Framework"
MODEL="$HOME/kwchen/hf_models/$MODEL_NAME"
RESULTS="$ROOT/libra_results/direct_nccl_${MODEL_NAME}_${SLURM_JOB_ID}"

source "$HOME/rl_framework_env/bin/activate"
export PYTHONPATH="$ROOT/libra_nccl_dev:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
mkdir -p "$RESULTS"

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$ALLOCATED_CUDA_VISIBLE_DEVICES"
REQUIRED_GPUS=$((2 + TRAIN_TP))
if ((${#GPU_IDS[@]} < REQUIRED_GPUS)); then
  echo "Expected $REQUIRED_GPUS allocated GPUs, got: $ALLOCATED_CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

server_steps=()
cleanup() {
  if ((${#server_steps[@]})); then
    kill "${server_steps[@]}" >/dev/null 2>&1 || true
    wait "${server_steps[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for instance in 0 1; do
  port=$((BASE_PORT + instance))
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$instance]}" \
    python "$PROJECT/scripts/vllm_hot_reload_api_server.py" \
      --model "$MODEL" \
      --served-model-name qwen3-probe \
      --host 0.0.0.0 \
      --port "$port" \
      --tensor-parallel-size 1 \
      --max-model-len 512 \
      --max-num-seqs 4 \
      --dtype bfloat16 \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      --enable-prefix-caching \
      --trust-remote-code \
      --worker-extension-cls RL_Framework.vllm_hot_reload.InplaceReloadWorkerExtension \
      >"$RESULTS/vllm_${instance}.log" 2>&1 &
  server_steps+=("$!")
done

for instance in 0 1; do
  port=$((BASE_PORT + instance))
  for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
done

NODE_IP="127.0.0.1"
MASTER_PORT=$((TRANSPORT_PORT + 1000))
TRAINER_CUDA_VISIBLE_DEVICES="${GPU_IDS[2]}"
if ((TRAIN_TP > 1)); then
  for ((index = 3; index < REQUIRED_GPUS; index++)); do
    TRAINER_CUDA_VISIBLE_DEVICES+=",${GPU_IDS[$index]}"
  done
fi
CUDA_VISIBLE_DEVICES="$TRAINER_CUDA_VISIBLE_DEVICES" \
  torchrun --standalone --nnodes=1 --nproc-per-node="$TRAIN_TP" \
    --master-port="$MASTER_PORT" \
    "$PROJECT/scripts/direct_nccl_qwen3_probe.py" \
      --model "$MODEL" \
      --endpoints "http://127.0.0.1:${BASE_PORT},http://127.0.0.1:$((BASE_PORT + 1))" \
      --transport-host "$NODE_IP" \
      --transport-port "$TRANSPORT_PORT" \
      --train-tp "$TRAIN_TP" \
      --train-ep "$TRAIN_EP" \
      --repeats "$PROBE_REPEATS" \
      --output "$RESULTS/result.json" \
      >"$RESULTS/trainer.log" 2>&1

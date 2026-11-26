#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_SCRIPT="$ROOT/repro/tier2_four_gpu_closed_loop/benchmark_rollout_scheduling.py"

MODEL="Qwen/Qwen3-14B"
GPUS="0,1,2,3,4,5,6,7"
TOPOLOGY="1,1,1,1,4"
BASE_PORT="19001"
HOST="127.0.0.1"
MAX_MODEL_LEN="24576"
GPU_MEM_UTIL="0.85"
MAX_NUM_SEQS="32"
OUTPUT_DIR="./logs/tier2_rollout_scheduling"
HARDWARE_CONFIG="$ROOT/configs/hardware_config/h20_3e_141g.yaml"
MODEL_ARCH_CONFIG="$ROOT/configs/model_arch_config/qwen3-14b.yaml"
ROUNDS="2"
POLICIES="all"
PYTHON="python3"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: bash repro/tier2_four_gpu_closed_loop/run_rollout_ab.sh [OPTIONS]

Launch heterogeneous vLLM servers (--enable-prefix-caching) and run the Tier 2
6-policy multi-turn rollout scheduling benchmark.

Options:
  --model PATH_OR_ID             HF model ID or local path (default: Qwen/Qwen3-14B)
  --gpus IDS                     Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  --topology TPS                 Comma-separated TP degrees (default: 1,1,1,1,4)
  --base-port PORT               Starting vLLM HTTP port (default: 19001)
  --host HOST                    vLLM bind host (default: 127.0.0.1)
  --max-model-len N              Max sequence length (default: 24576)
  --gpu-memory-utilization F     vLLM GPU memory fraction (default: 0.85)
  --max-num-seqs N               vLLM max concurrent sequences (default: 32)
  --output-dir DIR               Output directory (default: ./logs/tier2_rollout_scheduling)
  --hardware-config PATH         Hardware YAML for CostModel (default: h20_3e_141g.yaml)
  --model-arch-config PATH       Model arch YAML for CostModel (default: qwen3-14b.yaml)
  --rounds N                     Counterbalanced rounds (default: 2)
  --policies LIST                Comma-separated policy names or 'all' (default: all)
  --python BIN                   Python executable to run vLLM & benchmark (default: python3)
  --dry-run                      Print vLLM launch commands and validate schedulers without GPUs
  -h, --help                     Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --hardware-config) HARDWARE_CONFIG="$2"; shift 2 ;;
    --model-arch-config) MODEL_ARCH_CONFIG="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

IFS=',' read -ra TP_ARR <<< "$TOPOLOGY"
if [[ ${#TP_ARR[@]} -lt 2 ]]; then
  echo "ERROR: --topology requires at least 2 instances, got '$TOPOLOGY'" >&2
  exit 1
fi

TOTAL_NEEDED=0
for tp in "${TP_ARR[@]}"; do
  TOTAL_NEEDED=$((TOTAL_NEEDED + tp))
done

if [[ -z "$GPUS" ]]; then
  GPU_ARR=()
  for ((g=0; g<TOTAL_NEEDED; g++)); do
    GPU_ARR+=("$g")
  done
  GPUS="$(IFS=','; echo "${GPU_ARR[*]}")"
else
  IFS=',' read -ra GPU_ARR <<< "$GPUS"
fi

if [[ ${#GPU_ARR[@]} -ne $TOTAL_NEEDED ]]; then
  echo "ERROR: --topology '$TOPOLOGY' needs $TOTAL_NEEDED GPUs, got ${#GPU_ARR[@]} ('$GPUS')" >&2
  exit 1
fi

if [[ -d /dev/shm && -w /dev/shm ]]; then
  CACHE_BASE="/dev/shm"
else
  CACHE_BASE="${TMPDIR:-/tmp}"
fi
RUN_TAG="$$"

INST_NAMES=()
INST_GPUS=()
INST_TPS=()
INST_PORTS=()

cursor=0
for ((idx=0; idx<${#TP_ARR[@]}; idx++)); do
  tp="${TP_ARR[$idx]}"
  sub_gpus=""
  for ((j=0; j<tp; j++)); do
    g="${GPU_ARR[$((cursor + j))]}"
    if [[ -z "$sub_gpus" ]]; then
      sub_gpus="$g"
    else
      sub_gpus="$sub_gpus,$g"
    fi
  done
  cursor=$((cursor + tp))
  # Count how many instances of this TP appeared before idx
  suffix=0
  for ((p=0; p<idx; p++)); do
    if [[ "${TP_ARR[$p]}" -eq "$tp" ]]; then
      suffix=$((suffix + 1))
    fi
  done
  INST_NAMES+=("tp${tp}_${suffix}")
  INST_GPUS+=("$sub_gpus")
  INST_TPS+=("$tp")
  INST_PORTS+=("$((BASE_PORT + idx))")
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== Tier 2 Heterogeneous Rollout Dry-Run ==="
  for ((i=0; i<${#INST_NAMES[@]}; i++)); do
    name="${INST_NAMES[$i]}"
    cache_dir="${CACHE_BASE}/libra_vllm_${RUN_TAG}_${name}"
    echo "[$name] CUDA_VISIBLE_DEVICES=${INST_GPUS[$i]} VLLM_CACHE_ROOT=${cache_dir}/vllm TORCHINDUCTOR_CACHE_DIR=${cache_dir}/inductor TRITON_CACHE_DIR=${cache_dir}/triton $PYTHON -m vllm.entrypoints.openai.api_server --model $MODEL --host $HOST --port ${INST_PORTS[$i]} --tensor-parallel-size ${INST_TPS[$i]} --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL --max-num-seqs $MAX_NUM_SEQS --enable-prefix-caching --disable-log-requests --seed 42"
  done
  exec "$PYTHON" "$BENCH_SCRIPT" \
    --model "$MODEL" \
    --topology "$TOPOLOGY" \
    --gpus "$GPUS" \
    --host "$HOST" \
    --base-port "$BASE_PORT" \
    --hardware-config "$HARDWARE_CONFIG" \
    --model-arch-config "$MODEL_ARCH_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --rounds "$ROUNDS" \
    --policies "$POLICIES" \
    --dry-run
fi

mkdir -p "$OUTPUT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PIDS=()
cleanup() {
  local rc=$?
  for pid in "${PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    sleep 6
  fi
  for pid in "${PIDS[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  rm -rf "${CACHE_BASE}/libra_vllm_${RUN_TAG}_"*
  exit "$rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap cleanup EXIT

for ((i=0; i<${#INST_NAMES[@]}; i++)); do
  name="${INST_NAMES[$i]}"
  gpus="${INST_GPUS[$i]}"
  tp="${INST_TPS[$i]}"
  port="${INST_PORTS[$i]}"
  cache_dir="${CACHE_BASE}/libra_vllm_${RUN_TAG}_${name}"
  mkdir -p "${cache_dir}/vllm" "${cache_dir}/inductor" "${cache_dir}/triton"
  env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_CACHE_ROOT="${cache_dir}/vllm" \
    TORCHINDUCTOR_CACHE_DIR="${cache_dir}/inductor" \
    TRITON_CACHE_DIR="${cache_dir}/triton" \
    setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --host "$HOST" \
      --port "$port" \
      --tensor-parallel-size "$tp" \
      --dtype bfloat16 \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enable-prefix-caching \
      --disable-log-requests \
      --seed 42 \
      > "$OUTPUT_DIR/vllm_${name}.log" 2>&1 < /dev/null &
  PIDS+=($!)
  echo "Launched $name (gpus=$gpus, tp=$tp, port=$port, pid=${PIDS[-1]})"
done

deadline=$((SECONDS + 600))
for port in "${INST_PORTS[@]}"; do
  while true; do
    if "$PYTHON" -c "import urllib.request; r=urllib.request.urlopen('http://${HOST}:${port}/health', timeout=2); raise SystemExit(0 if r.status == 200 else 1)" 2>/dev/null; then
      echo "Server on ${HOST}:${port} is healthy."
      break
    fi
    for pid in "${PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: vLLM server pid $pid exited early; inspect $OUTPUT_DIR/vllm_*.log" >&2
        exit 2
      fi
    done
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "ERROR: Timed out waiting for vLLM server on ${HOST}:${port}" >&2
      exit 3
    fi
    sleep 3
  done
done

echo "All ${#INST_NAMES[@]} vLLM servers ready. Running rollout scheduling benchmark..."
"$PYTHON" -u "$BENCH_SCRIPT" \
  --model "$MODEL" \
  --topology "$TOPOLOGY" \
  --gpus "$GPUS" \
  --host "$HOST" \
  --base-port "$BASE_PORT" \
  --hardware-config "$HARDWARE_CONFIG" \
  --model-arch-config "$MODEL_ARCH_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --rounds "$ROUNDS" \
  --policies "$POLICIES"

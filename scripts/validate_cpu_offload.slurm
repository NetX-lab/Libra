#!/bin/bash
#SBATCH --job-name=libra_cpu_offload
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=00:35:00

# Three disjoint GPUs: source TP1 and target TP2. Set SOURCE_TP=2 TARGET_TP=1
# to check the reverse resharding direction. No training jobs are modified.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON="${PYTHON:-$HOME/rl_framework_env/bin/python3}"
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3 model directory}"
SOURCE_TP="${SOURCE_TP:-1}"
TARGET_TP="${TARGET_TP:-2}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/logs/cmlfq_offload/${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
export VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN OMP_NUM_THREADS=4
export LIBRA_KV_STORAGE_PATH="/dev/shm/libra-kv-${SLURM_JOB_ID}"
mkdir -p "$LIBRA_KV_STORAGE_PATH"
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPUS[@]} < SOURCE_TP + TARGET_TP )); then
    echo "Need disjoint GPUs for source TP${SOURCE_TP} and target TP${TARGET_TP}" >&2
    exit 2
fi
SOURCE_GPUS=$(IFS=,; echo "${GPUS[*]:0:SOURCE_TP}")
TARGET_GPUS=$(IFS=,; echo "${GPUS[*]:SOURCE_TP:TARGET_TP}")
KV="{\"kv_connector\":\"LibraCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"RL_Framework.infra.scheduling.vllm_cpu_offload\",\"kv_connector_extra_config\":{\"storage_path\":\"$LIBRA_KV_STORAGE_PATH\"}}"
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait || true
    rm -rf -- "$LIBRA_KV_STORAGE_PATH"
}
trap cleanup EXIT INT TERM
"$PYTHON" -c 'import torch,vllm,json,socket;print(json.dumps({"host":socket.gethostname(),"torch":torch.__version__,"vllm":vllm.__version__,"cuda":torch.version.cuda,"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))' > "$RUN_DIR/environment.json"
CUDA_VISIBLE_DEVICES="$SOURCE_GPUS" "$PYTHON" scripts/vllm_hot_reload_api_server.py \
    --model "$MODEL_PATH" --host 127.0.0.1 --port 18910 --tensor-parallel-size "$SOURCE_TP" \
    --dtype bfloat16 --max-model-len 10000 --gpu-memory-utilization 0.50 \
    --max-num-seqs 4 --enforce-eager --disable-log-requests --no-enable-prefix-caching --kv-transfer-config "$KV" \
    > "$RUN_DIR/source.log" 2>&1 &
PIDS+=("$!")
CUDA_VISIBLE_DEVICES="$TARGET_GPUS" "$PYTHON" scripts/vllm_hot_reload_api_server.py \
    --model "$MODEL_PATH" --host 127.0.0.1 --port 18911 --tensor-parallel-size "$TARGET_TP" \
    --dtype bfloat16 --max-model-len 10000 --gpu-memory-utilization 0.50 \
    --max-num-seqs 4 --enforce-eager --disable-log-requests --no-enable-prefix-caching --kv-transfer-config "$KV" \
    > "$RUN_DIR/target.log" 2>&1 &
PIDS+=("$!")
"$PYTHON" -m pytest -q tests/test_cmlfq_cost_scheduler.py tests/test_cmlfq_scheduler.py \
    tests/test_hetero_cmlfq_integration.py tests/test_rollout_engine.py \
    tests/test_cpu_offload_backend.py tests/test_vllm_cpu_offload.py \
    > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" scripts/benchmark_cpu_offload.py --model "$MODEL_PATH" \
    --source-tp "$SOURCE_TP" --target-tp "$TARGET_TP" \
    --words ${BENCH_WORDS:-64 256 1024} --tool-ms ${TOOL_MS:-0 100 500} \
    --trials "${TRIALS:-3}" --output "$RUN_DIR/results.json" \
    > "$RUN_DIR/benchmark.log" 2>&1

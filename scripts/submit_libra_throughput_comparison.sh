#!/usr/bin/env bash
# Submit a long-output Libra throughput comparison suite.
#
# The suite is designed to magnify the benefit of C-MLFQ, Global Resource
# Planner, and Elastic Hybrid Pool under long, skewed rollout workloads.
#
# Runs:
#   1. fixed_tp2_rr:       fixed TP2 rollout, round-robin routing, no GRP/EHP.
#   2. hetero_length:      static heterogeneous rollout, length-aware routing.
#   3. hetero_cmlfq:       static heterogeneous rollout, C-MLFQ routing.
#   4. full_libra_dynamic: C-MLFQ + online GRP + EHP cluster swap.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SLURM_NODES="${SLURM_NODES:-4}"
SLURM_GPUS="${SLURM_GPUS:-16}"
SLURM_TIME="${SLURM_TIME:-08:00:00}"
SUBMIT_FULL_DYNAMIC="${SUBMIT_FULL_DYNAMIC:-1}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
MODEL_ARCH_CONFIG="${MODEL_ARCH_CONFIG:-model_arch_config/qwen3-4b.yaml}"

TOTAL_STEPS="${TOTAL_STEPS:-100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30000}"
R2E_MAX_PROMPT_TOKENS="${R2E_MAX_PROMPT_TOKENS:-2048}"
R2E_MAX_TURNS="${R2E_MAX_TURNS:-3}"

TRAIN_GPUS="${TRAIN_GPUS:-8}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-2}"

MAX_CONCURRENT_ROLLOUTS="${MAX_CONCURRENT_ROLLOUTS:-12}"
QUEUE_SIZE="${QUEUE_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
SYNC_INTERVAL="${SYNC_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
RUN_NCCL_PREFLIGHT="${RUN_NCCL_PREFLIGHT:-0}"
MAX_WAIT="${MAX_WAIT:-3600}"
VLLM_READ_TIMEOUT="${VLLM_READ_TIMEOUT:-3600}"
VLLM_WRITE_TIMEOUT="${VLLM_WRITE_TIMEOUT:-3600}"
VLLM_CONNECT_TIMEOUT="${VLLM_CONNECT_TIMEOUT:-60}"
VLLM_POOL_TIMEOUT="${VLLM_POOL_TIMEOUT:-3600}"

ROLLOUT_WEIGHT_SYNC_MODE="${ROLLOUT_WEIGHT_SYNC_MODE:-restart}"
REQUIRE_ROLLOUT_WEIGHT_SYNC="${REQUIRE_ROLLOUT_WEIGHT_SYNC:-1}"
ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY="${ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY:-1}"

COMMON_ENV=(
  PROJECT_DIR="$PROJECT_DIR"
  MODEL_PATH="$MODEL_PATH"
  MODEL_ARCH_CONFIG="$MODEL_ARCH_CONFIG"
  TOTAL_STEPS="$TOTAL_STEPS"
  MAX_MODEL_LEN="$MAX_MODEL_LEN"
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
  R2E_MAX_PROMPT_TOKENS="$R2E_MAX_PROMPT_TOKENS"
  R2E_MAX_TURNS="$R2E_MAX_TURNS"
  TRAIN_GPUS="$TRAIN_GPUS"
  TRAIN_TP_SIZE="$TRAIN_TP_SIZE"
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
  TRAIN_MICRO_BATCH_SIZE="$TRAIN_MICRO_BATCH_SIZE"
  N_SAMPLES="$N_SAMPLES"
  MAX_CONCURRENT_ROLLOUTS="$MAX_CONCURRENT_ROLLOUTS"
  QUEUE_SIZE="$QUEUE_SIZE"
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
  SYNC_INTERVAL="$SYNC_INTERVAL"
  EVAL_INTERVAL="$EVAL_INTERVAL"
  RUN_NCCL_PREFLIGHT="$RUN_NCCL_PREFLIGHT"
  MAX_WAIT="$MAX_WAIT"
  VLLM_READ_TIMEOUT="$VLLM_READ_TIMEOUT"
  VLLM_WRITE_TIMEOUT="$VLLM_WRITE_TIMEOUT"
  VLLM_CONNECT_TIMEOUT="$VLLM_CONNECT_TIMEOUT"
  VLLM_POOL_TIMEOUT="$VLLM_POOL_TIMEOUT"
  ROLLOUT_WEIGHT_SYNC_MODE="$ROLLOUT_WEIGHT_SYNC_MODE"
  REQUIRE_ROLLOUT_WEIGHT_SYNC="$REQUIRE_ROLLOUT_WEIGHT_SYNC"
  ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY="$ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY"
  CMLFQ_INITIAL_BUCKET=short
  ALLOW_QWEN30_FALLBACK=1
)

submit_run() {
  local tag="$1"
  local script="$2"
  shift 2
  local cmd=(
    env
    "${COMMON_ENV[@]}"
    EXPERIMENT_TAG="$tag"
    "$@"
    sbatch
    --nodes="$SLURM_NODES"
    --gpus="$SLURM_GPUS"
    --time="$SLURM_TIME"
    "$PROJECT_DIR/$script"
  )
  echo
  echo "Submitting $tag"
  printf '  %q' "${cmd[@]}"
  echo
  if [ "$DRY_RUN" = "0" ]; then
    "${cmd[@]}"
  fi
}

echo "Libra throughput comparison suite"
echo "  nodes/gpus: ${SLURM_NODES}/${SLURM_GPUS}"
echo "  steps: ${TOTAL_STEPS}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
echo "  max_concurrent_rollouts: ${MAX_CONCURRENT_ROLLOUTS}"
echo "  sync_interval: ${SYNC_INTERVAL}"

submit_run \
  "throughput_fixed_tp2_rr_len30000" \
  "scripts/submit_libra_validation_baseline.slurm" \
  SCHEDULER_TYPE=length_aware \
  SCHEDULER_LOAD_BALANCE_STRATEGY=round_robin \
  GRP_INITIAL_ROLLOUT_TP_LIST=2:2:2:2 \
  GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP=0

submit_run \
  "throughput_hetero_length_len30000" \
  "scripts/submit_libra_validation_baseline.slurm" \
  SCHEDULER_TYPE=length_aware \
  SCHEDULER_LOAD_BALANCE_STRATEGY=least_connections \
  GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2 \
  GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP=1

submit_run \
  "throughput_hetero_cmlfq_len30000" \
  "scripts/submit_libra_validation_baseline.slurm" \
  SCHEDULER_TYPE=cmlfq \
  SCHEDULER_LOAD_BALANCE_STRATEGY=least_connections \
  CMLFQ_INITIAL_BUCKET=long \
  CMLFQ_LONG_TP_DEGREES=4 \
  GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2 \
  GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP=1

if [ "$SUBMIT_FULL_DYNAMIC" = "1" ]; then
  submit_run \
    "throughput_full_libra_dynamic_len30000" \
    "scripts/submit_libra_validation_full.slurm" \
    SCHEDULER_TYPE=cmlfq \
    SCHEDULER_LOAD_BALANCE_STRATEGY=least_connections \
    CMLFQ_INITIAL_BUCKET=long \
    CMLFQ_LONG_TP_DEGREES=4 \
    GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2 \
    GRP_FORCE_TRAIN_GPUS=4 \
    GRP_FORCE_ROLLOUT_TP_LIST=4:2:2 \
    GRP_TRAINING_POOL_TARGET_GPUS=4 \
    GRP_REPLAN_COOLDOWN_STEPS=10 \
    GRP_INTERVAL=5 \
    GRP_WARMUP_STEPS=5 \
    GRP_MIN_HISTORY=8 \
    GRP_MIN_GAIN_RATIO=0.0 \
    GRP_QUEUE_PRESSURE_THRESHOLD=0.50 \
    GRP_ACTIVE_ROLLOUT_PRESSURE_THRESHOLD=0.60 \
    GRP_ROLLOUT_TRAIN_IMBALANCE_THRESHOLD=1.05 \
    GRP_FORCE_RUNTIME_RECONFIGURE=1 \
    GRP_DYNAMIC_RECONFIG_ENABLED=1 \
    GRP_ONLINE_REPLANNING=1 \
    GRP_MANAGE_ROLLOUT_PROCESSES=1 \
    GRP_RECONFIGURE_TRAINING=1 \
    GRP_TRAINING_POOL_PLAN_ONLY=0 \
    GRP_TRAINING_HANDOFF_ENABLED=1 \
    GRP_TRAINING_HANDOFF_TIMEOUT_S=1800 \
    GRP_CLUSTER_SWAP_ENABLED=1 \
    GRP_ROLLOUT_RECONFIGURE_STRATEGY=cluster_swap \
    GRP_DECOUPLE_COMMUNICATION_DOMAINS=1 \
    GRP_BATCH_COLLECTION_TIMEOUT_S=3600 \
    GRP_BATCH_COLLECTION_MAX_RETRIES=1
fi

echo
echo "After the jobs finish, run:"
echo "  python scripts/analyze_libra_experiment.py \\"
echo "    --run fixed_tp2=logs/r2e_gym_cmlfq_24gpu/job_<fixed_job> \\"
echo "    --run hetero_length=logs/r2e_gym_cmlfq_24gpu/job_<length_job> \\"
echo "    --run hetero_cmlfq=logs/r2e_gym_cmlfq_24gpu/job_<cmlfq_job> \\"
echo "    --run full_libra=logs/r2e_gym_cmlfq_24gpu/job_<full_job> \\"
echo "    --output logs/r2e_gym_cmlfq_24gpu/libra_throughput_len30000_report.md"

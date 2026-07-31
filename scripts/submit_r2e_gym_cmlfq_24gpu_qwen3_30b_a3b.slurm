#!/bin/bash
#SBATCH -J r2e_qwen30_cmlfq
#SBATCH -o logs/r2e_qwen30_cmlfq_%j.out
#SBATCH -e logs/r2e_qwen30_cmlfq_%j.err
#SBATCH -N 4
#SBATCH -G 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH -p gpu
#SBATCH --time=24:00:00

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load compiler/gcc/10.1.0 || true
if [ -x /usr/local/cuda-12.3/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-12.3
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
else
    export CUDA_HOME=/usr/local/cuda-12.1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${HOME}/rl_framework_env/bin/activate}"
MODEL_PATH="${MODEL_PATH:-${HOME}/hf_models/Qwen3-30B-A3B}"
MODEL_ARCH_CONFIG="${MODEL_ARCH_CONFIG:-model_arch_config/qwen3-30b-a3b.yaml}"
if [ -f "$PROJECT_DIR/.slurm/slurm.conf" ]; then
    export SLURM_CONF="$PROJECT_DIR/.slurm/slurm.conf"
fi
TOTAL_STEPS="${TOTAL_STEPS:-100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-4}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
TRAIN_BACKEND="${TRAIN_BACKEND:-megatron_core}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-4}"
TRAIN_PP_SIZE="${TRAIN_PP_SIZE:-1}"
TRAIN_CP_SIZE="${TRAIN_CP_SIZE:-1}"
TRAIN_EP_SIZE="${TRAIN_EP_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-1}"
MEGATRON_USE_CPU_INITIALIZATION="${MEGATRON_USE_CPU_INITIALIZATION:-0}"
MCORE_MOE_GROUPED_GEMM="${MCORE_MOE_GROUPED_GEMM:-1}"
MEGATRON_USE_PRECISION_AWARE_OPTIMIZER="${MEGATRON_USE_PRECISION_AWARE_OPTIMIZER:-1}"
MEGATRON_OPTIMIZER_CPU_OFFLOAD="${MEGATRON_OPTIMIZER_CPU_OFFLOAD:-1}"
MEGATRON_OPTIMIZER_OFFLOAD_FRACTION="${MEGATRON_OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
ALLOW_QWEN30_FALLBACK="${ALLOW_QWEN30_FALLBACK:-0}"
MAX_CONCURRENT_ROLLOUTS="${MAX_CONCURRENT_ROLLOUTS:-32}"
QUEUE_SIZE="${QUEUE_SIZE:-256}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
RL_TRAIN_PHASE_TRACE="${RL_TRAIN_PHASE_TRACE:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-32}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
SYNC_INTERVAL="${SYNC_INTERVAL:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_WAIT="${MAX_WAIT:-1800}"
R2E_MAX_TURNS="${R2E_MAX_TURNS:-4}"
R2E_MAX_PROMPT_TOKENS="${R2E_MAX_PROMPT_TOKENS:-4096}"
R2E_EVAL_MULTI_TURN="${R2E_EVAL_MULTI_TURN:-1}"
R2E_EVAL_RECORD_SAMPLES="${R2E_EVAL_RECORD_SAMPLES:-16}"
R2E_EVAL_STRATEGY="${R2E_EVAL_STRATEGY:-spread}"
LONG_CONTEXT_TP_FIRST="${LONG_CONTEXT_TP_FIRST:-0}"
VLLM_READ_TIMEOUT="${VLLM_READ_TIMEOUT:-600}"
VLLM_CONNECT_TIMEOUT="${VLLM_CONNECT_TIMEOUT:-30}"
VLLM_WRITE_TIMEOUT="${VLLM_WRITE_TIMEOUT:-30}"
VLLM_POOL_TIMEOUT="${VLLM_POOL_TIMEOUT:-30}"
VLLM_NCCL_P2P_DISABLE="${VLLM_NCCL_P2P_DISABLE:-1}"
VLLM_NCCL_IB_DISABLE="${VLLM_NCCL_IB_DISABLE:-0}"
VLLM_CUDA_LAUNCH_BLOCKING="${VLLM_CUDA_LAUNCH_BLOCKING:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
FSDP_ALIGN_SEQUENCE_LENGTHS="${FSDP_ALIGN_SEQUENCE_LENGTHS:-1}"
FSDP_LOCKSTEP_BARRIERS="${FSDP_LOCKSTEP_BARRIERS:-1}"
FSDP_GRADIENT_CHECKPOINTING="${FSDP_GRADIENT_CHECKPOINTING:-0}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-1}"
CMLFQ_SHARED_LOAD_TTL="${CMLFQ_SHARED_LOAD_TTL:-60}"
CMLFQ_SHARED_LOAD_HEARTBEAT="${CMLFQ_SHARED_LOAD_HEARTBEAT:-10}"
CMLFQ_TREE_PERSIST_INTERVAL="${CMLFQ_TREE_PERSIST_INTERVAL:-8}"
CMLFQ_WARM_START_TREE="${CMLFQ_WARM_START_TREE:-}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-r2e_qwen_cmlfq}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-cmlfq}"
GRP_ENABLED="${GRP_ENABLED:-1}"
GRP_DYNAMIC_RECONFIG_ENABLED="${GRP_DYNAMIC_RECONFIG_ENABLED:-0}"
GRP_RUNTIME_ASYNC_PLANNING="${GRP_RUNTIME_ASYNC_PLANNING:-1}"
GRP_ONLINE_REPLANNING="${GRP_ONLINE_REPLANNING:-1}"
GRP_REPLAN_COOLDOWN_STEPS="${GRP_REPLAN_COOLDOWN_STEPS:-5}"
GRP_QUEUE_PRESSURE_THRESHOLD="${GRP_QUEUE_PRESSURE_THRESHOLD:-0.75}"
GRP_ACTIVE_ROLLOUT_PRESSURE_THRESHOLD="${GRP_ACTIVE_ROLLOUT_PRESSURE_THRESHOLD:-0.85}"
GRP_REJECTED_ROLLOUT_DELTA_THRESHOLD="${GRP_REJECTED_ROLLOUT_DELTA_THRESHOLD:-8}"
GRP_ROLLOUT_TRAIN_IMBALANCE_THRESHOLD="${GRP_ROLLOUT_TRAIN_IMBALANCE_THRESHOLD:-1.25}"
GRP_INTERVAL="${GRP_INTERVAL:-20}"
GRP_WARMUP_STEPS="${GRP_WARMUP_STEPS:-5}"
GRP_MIN_HISTORY="${GRP_MIN_HISTORY:-16}"
GRP_MIN_GAIN_RATIO="${GRP_MIN_GAIN_RATIO:-${GRP_GAIN_THRESHOLD:-0.02}}"
GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP="${GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP:-1}"
GRP_MANAGE_ROLLOUT_PROCESSES="${GRP_MANAGE_ROLLOUT_PROCESSES:-0}"
GRP_RECONFIGURE_TRAINING="${GRP_RECONFIGURE_TRAINING:-0}"
GRP_TRAINING_POOL_ONLY="${GRP_TRAINING_POOL_ONLY:-1}"
GRP_TRAINING_POOL_TARGET_GPUS="${GRP_TRAINING_POOL_TARGET_GPUS:-0}"
GRP_TRAINING_POOL_PLAN_ONLY="${GRP_TRAINING_POOL_PLAN_ONLY:-1}"
GRP_CLUSTER_SWAP_ENABLED="${GRP_CLUSTER_SWAP_ENABLED:-0}"
GRP_HYBRID_WORKER_LAUNCH_ENABLED="${GRP_HYBRID_WORKER_LAUNCH_ENABLED:-0}"
GRP_HYBRID_TRAINING_PREWARM_ENABLED="${GRP_HYBRID_TRAINING_PREWARM_ENABLED:-0}"
GRP_HYBRID_TRAINING_PREWARM_COUNT="${GRP_HYBRID_TRAINING_PREWARM_COUNT:-0}"
GRP_HYBRID_TRAINING_PREWARM_WORKER_IDS="${GRP_HYBRID_TRAINING_PREWARM_WORKER_IDS:-}"
GRP_HYBRID_WORKER_TASK_DIR="${GRP_HYBRID_WORKER_TASK_DIR:-}"
GRP_HYBRID_WORKER_COMMAND_TEMPLATE="${GRP_HYBRID_WORKER_COMMAND_TEMPLATE:-}"
GRP_HYBRID_WORKER_READY_TIMEOUT_S="${GRP_HYBRID_WORKER_READY_TIMEOUT_S:-120}"
GRP_GRADIENT_TRANSPORT_BACKEND="${GRP_GRADIENT_TRANSPORT_BACKEND:-tcp}"
GRP_DECOUPLE_COMMUNICATION_DOMAINS="${GRP_DECOUPLE_COMMUNICATION_DOMAINS:-1}"
GRP_GRADIENT_SERVER_HOST="${GRP_GRADIENT_SERVER_HOST:-127.0.0.1}"
GRP_GRADIENT_SERVER_PUBLIC_HOST="${GRP_GRADIENT_SERVER_PUBLIC_HOST:-}"
GRP_GRADIENT_SERVER_PORT="${GRP_GRADIENT_SERVER_PORT:-0}"
GRP_GRADIENT_SERVER_AUTHKEY="${GRP_GRADIENT_SERVER_AUTHKEY:-rl-framework-elastic}"
GRP_ROLLOUT_RECONFIGURE_STRATEGY="${GRP_ROLLOUT_RECONFIGURE_STRATEGY:-blue_green}"
GRP_PREWARM_NO_SPARE_FALLBACK_STRATEGY="${GRP_PREWARM_NO_SPARE_FALLBACK_STRATEGY:-restart_all}"
GRP_DRAIN_BEFORE_RECONFIGURE="${GRP_DRAIN_BEFORE_RECONFIGURE:-1}"
GRP_RUNTIME_PEER_REQUEST_WAIT_S="${GRP_RUNTIME_PEER_REQUEST_WAIT_S:-45}"
GRP_BATCH_COLLECTION_TIMEOUT_S="${GRP_BATCH_COLLECTION_TIMEOUT_S:-300}"
GRP_BATCH_COLLECTION_MAX_RETRIES="${GRP_BATCH_COLLECTION_MAX_RETRIES:-2}"
GRP_USE_NCCL_BARRIER_AFTER_ROLLOUT="${GRP_USE_NCCL_BARRIER_AFTER_ROLLOUT:-0}"
GRP_USE_NCCL_BARRIER_BEFORE_WEIGHT_SYNC="${GRP_USE_NCCL_BARRIER_BEFORE_WEIGHT_SYNC:-0}"
GRP_INITIAL_ROLLOUT_TP_LIST="${GRP_INITIAL_ROLLOUT_TP_LIST:-}"
GRP_FORCE_TRAIN_GPUS="${GRP_FORCE_TRAIN_GPUS:-}"
GRP_ROLLOUT_MANIFEST_PATH="${GRP_ROLLOUT_MANIFEST_PATH:-}"
GRP_ROLLOUT_PID_REGISTRY_PATH="${GRP_ROLLOUT_PID_REGISTRY_PATH:-}"
GRP_WAIT_ROLLOUT_PROCESS_START_READY="${GRP_WAIT_ROLLOUT_PROCESS_START_READY:-1}"
GRP_POST_ROLLOUT_STOP_GRACE_S="${GRP_POST_ROLLOUT_STOP_GRACE_S:-45}"
GRP_VLLM_LAUNCH_COMMAND_TEMPLATE="${GRP_VLLM_LAUNCH_COMMAND_TEMPLATE:-}"
GRP_VLLM_STOP_COMMAND_TEMPLATE="${GRP_VLLM_STOP_COMMAND_TEMPLATE:-}"
ROLLOUT_WEIGHT_SYNC_MODE="${ROLLOUT_WEIGHT_SYNC_MODE:-restart}"
REQUIRE_ROLLOUT_WEIGHT_SYNC="${REQUIRE_ROLLOUT_WEIGHT_SYNC:-1}"
ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY="${ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY:-1}"
MEGATRON_STREAMING_EXPORT="${MEGATRON_STREAMING_EXPORT:-1}"
PREFERRED_TRAIN_NODE="${PREFERRED_TRAIN_NODE:-}"
AVOID_TRAIN_NODE="${AVOID_TRAIN_NODE:-}"
RUN_NCCL_PREFLIGHT="${RUN_NCCL_PREFLIGHT:-1}"
NCCL_PREFLIGHT_REQUIRED="${NCCL_PREFLIGHT_REQUIRED:-1}"
NCCL_PREFLIGHT_ELEMENTS="${NCCL_PREFLIGHT_ELEMENTS:-1048576}"
NCCL_PREFLIGHT_ITERATIONS="${NCCL_PREFLIGHT_ITERATIONS:-1}"
NCCL_PREFLIGHT_WALLTIME="${NCCL_PREFLIGHT_WALLTIME:-300}"

source "$VENV_PATH"
cd "$PROJECT_DIR"

mkdir -p logs
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="$PROJECT_DIR/logs/r2e_gym_cmlfq_24gpu/job_${JOB_ID}"
ROLLOUT_SYNC_DIR="$JOB_LOG_DIR/rollout_weight_sync"
CMLFQ_TREE_PATH="$PROJECT_DIR/logs/r2e_gym_cmlfq_24gpu/cmlfq_tree_${JOB_ID}.json"
CMLFQ_SHARED_LOAD_DIR="$JOB_LOG_DIR/cmlfq_shared_load"
GRP_ROLLOUT_MANIFEST_PATH="${GRP_ROLLOUT_MANIFEST_PATH:-$JOB_LOG_DIR/global_resource_rollout_manifest.json}"
GRP_ROLLOUT_PID_REGISTRY_PATH="${GRP_ROLLOUT_PID_REGISTRY_PATH:-$JOB_LOG_DIR/rollout_process_registry.jsonl}"
GRP_HYBRID_WORKER_TASK_DIR="${GRP_HYBRID_WORKER_TASK_DIR:-$JOB_LOG_DIR/elastic_training_tasks}"
GRP_VLLM_READY_TIMEOUT_S="${GRP_VLLM_READY_TIMEOUT_S:-$MAX_WAIT}"
VLLM_CUSTOM_ALL_REDUCE_FLAG=""
if [ "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
    VLLM_CUSTOM_ALL_REDUCE_FLAG="--disable-custom-all-reduce"
fi
if [ "$GRP_MANAGE_ROLLOUT_PROCESSES" = "1" ] && [ -z "$GRP_VLLM_LAUNCH_COMMAND_TEMPLATE" ]; then
    GRP_VLLM_LAUNCH_COMMAND_TEMPLATE="srun --exclusive --nodes=1 --ntasks=1 --nodelist={host} --gpus={tp} --cpus-per-task=16 bash -lc \"source '$VENV_PATH' && export TOKENIZERS_PARALLELISM=false && export PLANNED_CUDA_VISIBLE_DEVICES={gpus} && if [ -z \\\"\\\${{CUDA_VISIBLE_DEVICES:-}}\\\" ]; then export CUDA_VISIBLE_DEVICES=\\\"\\\$PLANNED_CUDA_VISIBLE_DEVICES\\\"; fi && echo \\\"CUDA_VISIBLE_DEVICES=\\\${{CUDA_VISIBLE_DEVICES}} planned=\\\${{PLANNED_CUDA_VISIBLE_DEVICES}}\\\" && export MASTER_ADDR=127.0.0.1 && export MASTER_PORT=\\\$((20000 + {port})) && export NCCL_P2P_DISABLE='$VLLM_NCCL_P2P_DISABLE' && export NCCL_IB_DISABLE='$VLLM_NCCL_IB_DISABLE' && export CUDA_LAUNCH_BLOCKING='$VLLM_CUDA_LAUNCH_BLOCKING' && {python} '$PROJECT_DIR/scripts/restartable_vllm_server.py' --instance-id {instance_id} --control-dir {control_dir} --health-url {local_health_url} --initial-model {initial_model} --ready-timeout '$MAX_WAIT' -- python3 -m vllm.entrypoints.openai.api_server --model __MODEL_PATH__ --served-model-name {model_path} --host 0.0.0.0 --port {port} --tensor-parallel-size {tp} --max-model-len {max_model_len} --dtype bfloat16 --gpu-memory-utilization {gpu_memory_utilization} --trust-remote-code --enforce-eager $VLLM_CUSTOM_ALL_REDUCE_FLAG\""
fi
if [ "$GRP_MANAGE_ROLLOUT_PROCESSES" = "1" ] && [ -z "$GRP_VLLM_STOP_COMMAND_TEMPLATE" ]; then
    GRP_VLLM_STOP_COMMAND_TEMPLATE="srun --overlap --nodes=1 --ntasks=1 --nodelist={host} bash -lc \"pkill -TERM -f 'restartable_vllm_server.py.*--instance-id {instance_id}' || pkill -TERM -f 'vllm.entrypoints.openai.api_server.*--port {port}' || true\""
fi
if [ "$GRP_HYBRID_WORKER_LAUNCH_ENABLED" = "1" ] && [ -z "$GRP_HYBRID_WORKER_COMMAND_TEMPLATE" ]; then
    GRP_HYBRID_WORKER_COMMAND_TEMPLATE="srun --overlap --nodes=1 --ntasks=1 --nodelist={host} --gpus=1 --cpus-per-task=8 bash -lc \"source '$VENV_PATH' && export TOKENIZERS_PARALLELISM=false && export PLANNED_CUDA_VISIBLE_DEVICES={gpus} && if [ -z \\\"\\\${{CUDA_VISIBLE_DEVICES:-}}\\\" ]; then export CUDA_VISIBLE_DEVICES=\\\"\\\$PLANNED_CUDA_VISIBLE_DEVICES\\\"; fi && {python} {rl_framework_path}/scripts/elastic_hybrid_worker.py --worker-id {worker_id} --target-core-id {target_core_id} --snapshot-path {snapshot_path} --gradient-host {gradient_host} --gradient-port {gradient_port} --authkey {authkey} --task-dir {task_dir} --gradient-transport {gradient_transport_backend} --native-rdma-device {native_rdma_device} --native-rdma-gid-index {native_rdma_gid_index} --native-rdma-ib-port {native_rdma_ib_port} --native-rdma-max-bytes {native_rdma_max_bytes}\""
fi
export GRP_DYNAMIC_RECONFIG_ENABLED
export GRP_RUNTIME_ASYNC_PLANNING
export GRP_ONLINE_REPLANNING
export GRP_REPLAN_COOLDOWN_STEPS
export GRP_QUEUE_PRESSURE_THRESHOLD
export GRP_ACTIVE_ROLLOUT_PRESSURE_THRESHOLD
export GRP_REJECTED_ROLLOUT_DELTA_THRESHOLD
export GRP_ROLLOUT_TRAIN_IMBALANCE_THRESHOLD
export GRP_INTERVAL
export GRP_WARMUP_STEPS
export GRP_MIN_HISTORY
export GRP_MIN_GAIN_RATIO
export GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP
export GRP_MANAGE_ROLLOUT_PROCESSES
export GRP_RECONFIGURE_TRAINING
export GRP_TRAINING_POOL_ONLY
export GRP_TRAINING_POOL_TARGET_GPUS
export GRP_TRAINING_POOL_PLAN_ONLY
export GRP_CLUSTER_SWAP_ENABLED
export GRP_HYBRID_WORKER_LAUNCH_ENABLED
export GRP_HYBRID_TRAINING_PREWARM_ENABLED
export GRP_HYBRID_TRAINING_PREWARM_COUNT
export GRP_HYBRID_TRAINING_PREWARM_WORKER_IDS
export GRP_HYBRID_WORKER_TASK_DIR
export GRP_HYBRID_WORKER_COMMAND_TEMPLATE
export GRP_HYBRID_WORKER_READY_TIMEOUT_S
export GRP_GRADIENT_TRANSPORT_BACKEND
export GRP_DECOUPLE_COMMUNICATION_DOMAINS
export GRP_GRADIENT_SERVER_HOST
export GRP_GRADIENT_SERVER_PUBLIC_HOST
export GRP_GRADIENT_SERVER_PORT
export GRP_GRADIENT_SERVER_AUTHKEY
export GRP_ROLLOUT_RECONFIGURE_STRATEGY
export GRP_PREWARM_NO_SPARE_FALLBACK_STRATEGY
export GRP_DRAIN_BEFORE_RECONFIGURE
export GRP_RUNTIME_PEER_REQUEST_WAIT_S
export GRP_BATCH_COLLECTION_TIMEOUT_S
export GRP_BATCH_COLLECTION_MAX_RETRIES
export GRP_USE_NCCL_BARRIER_AFTER_ROLLOUT
export GRP_USE_NCCL_BARRIER_BEFORE_WEIGHT_SYNC
export GRP_INITIAL_ROLLOUT_TP_LIST
export GRP_FORCE_TRAIN_GPUS
export GRP_ROLLOUT_MANIFEST_PATH
export GRP_ROLLOUT_PID_REGISTRY_PATH
export GRP_WAIT_ROLLOUT_PROCESS_START_READY
export GRP_POST_ROLLOUT_STOP_GRACE_S
export GRP_VLLM_READY_TIMEOUT_S
export GRP_VLLM_LAUNCH_COMMAND_TEMPLATE
export GRP_VLLM_STOP_COMMAND_TEMPLATE
export ROLLOUT_WEIGHT_SYNC_MODE
export REQUIRE_ROLLOUT_WEIGHT_SYNC
export ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY
export MEGATRON_STREAMING_EXPORT
mkdir -p "$JOB_LOG_DIR"
mkdir -p "$ROLLOUT_SYNC_DIR"
mkdir -p "$CMLFQ_SHARED_LOAD_DIR"

if [ -n "$CMLFQ_WARM_START_TREE" ] && [ ! -e "$CMLFQ_TREE_PATH" ]; then
    if [ ! -s "$CMLFQ_WARM_START_TREE" ]; then
        echo "ERROR: CMLFQ warm-start tree does not exist or is empty: $CMLFQ_WARM_START_TREE" >&2
        exit 2
    fi
    cp "$CMLFQ_WARM_START_TREE" "$CMLFQ_TREE_PATH"
    echo "C-MLFQ warm-start tree: $CMLFQ_WARM_START_TREE -> $CMLFQ_TREE_PATH"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${PROJECT_DIR}/wandb_r2e_gym_cmlfq_24gpu"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
export TORCH_NCCL_DUMP_ON_TIMEOUT=0
export TORCH_NCCL_DESYNC_DEBUG=0
export TORCH_NCCL_ENABLE_TIMING=0
export TORCH_NCCL_HIGH_PRIORITY=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST" 2>/dev/null || true)
if [ "${#NODES[@]}" -eq 0 ]; then
    mapfile -t NODES < <(python3 - <<'PY'
import os
import re

value = os.environ.get("SLURM_NODELIST", "")

def expand_part(part: str) -> list[str]:
    match = re.fullmatch(r"([^\[]+)\[(.+)\]", part)
    if not match:
        return [part] if part else []
    prefix, body = match.groups()
    out: list[str] = []
    for item in body.split(","):
        if "-" in item:
            start, end = item.split("-", 1)
            width = max(len(start), len(end))
            for number in range(int(start), int(end) + 1):
                out.append(f"{prefix}{number:0{width}d}")
        else:
            out.append(f"{prefix}{item}")
    return out

for chunk in re.split(r",(?![^\[]*\])", value):
    for node in expand_part(chunk):
        print(node)
PY
)
fi
MIN_NODE_COUNT=$((TRAIN_GPUS / 4 + 1))
if [ "${#NODES[@]}" -lt "$MIN_NODE_COUNT" ]; then
    echo "ERROR: R2E-Gym C-MLFQ training requires at least ${MIN_NODE_COUNT} nodes for TRAIN_GPUS=${TRAIN_GPUS}; got ${#NODES[@]}" >&2
    exit 2
fi

# Keep a known cross-node participant in the FSDP group when Slurm allocates
# it, regardless of lexical node ordering. This makes communication regressions
# reproducible while still allowing Slurm to choose the remaining nodes.
if [ -n "$PREFERRED_TRAIN_NODE" ] && [ "${NODES[0]}" != "$PREFERRED_TRAIN_NODE" ]; then
    preferred_found=0
    reordered_nodes=("${NODES[0]}")
    for node in "${NODES[@]}"; do
        if [ "$node" = "$PREFERRED_TRAIN_NODE" ]; then
            reordered_nodes+=("$node")
            preferred_found=1
            break
        fi
    done
    if [ "$preferred_found" = "1" ]; then
        for node in "${NODES[@]:1}"; do
            if [ "$node" != "$PREFERRED_TRAIN_NODE" ]; then
                reordered_nodes+=("$node")
            fi
        done
        NODES=("${reordered_nodes[@]}")
    fi
fi

# Keep a suspect-but-usable node in the rollout pool. vLLM health and API
# checks validate it before training, while FSDP remains on known-good nodes.
if [ -n "$AVOID_TRAIN_NODE" ]; then
    reordered_nodes=()
    deferred_nodes=()
    for node in "${NODES[@]}"; do
        if [ "$node" = "$AVOID_TRAIN_NODE" ]; then
            deferred_nodes+=("$node")
        else
            reordered_nodes+=("$node")
        fi
    done
    NODES=("${reordered_nodes[@]}" "${deferred_nodes[@]}")
fi

if [ $((TRAIN_GPUS % 4)) -ne 0 ]; then
    echo "ERROR: TRAIN_GPUS must be a multiple of 4 for full-node training placement; got ${TRAIN_GPUS}" >&2
    exit 2
fi
TRAIN_NODE_COUNT=$((TRAIN_GPUS / 4))
if [ "$TRAIN_NODE_COUNT" -lt 1 ] || [ "$TRAIN_NODE_COUNT" -ge "${#NODES[@]}" ]; then
    echo "ERROR: TRAIN_GPUS=${TRAIN_GPUS} leaves no rollout node in ${#NODES[@]} allocated nodes" >&2
    exit 2
fi
TRAIN_MODEL_PARALLEL_SIZE=$((TRAIN_TP_SIZE * TRAIN_PP_SIZE * TRAIN_CP_SIZE))
if [ $((TRAIN_GPUS % TRAIN_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "ERROR: TRAIN_GPUS must be divisible by TP*PP*CP: ${TRAIN_GPUS} vs ${TRAIN_MODEL_PARALLEL_SIZE}" >&2
    exit 2
fi
TRAIN_DP_SIZE=$((TRAIN_GPUS / TRAIN_MODEL_PARALLEL_SIZE))
if [ $((TRAIN_BATCH_SIZE % (TRAIN_DP_SIZE * TRAIN_MICRO_BATCH_SIZE))) -ne 0 ]; then
    echo "ERROR: TRAIN_BATCH_SIZE must be divisible by DP * TRAIN_MICRO_BATCH_SIZE: ${TRAIN_BATCH_SIZE} vs ${TRAIN_DP_SIZE}*${TRAIN_MICRO_BATCH_SIZE}" >&2
    exit 2
fi
LOCAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE / TRAIN_DP_SIZE))
if [ $((LOCAL_BATCH_SIZE % N_SAMPLES)) -ne 0 ]; then
    echo "ERROR: local batch per DP replica must be divisible by N_SAMPLES: local_batch_size=${LOCAL_BATCH_SIZE}, N_SAMPLES=${N_SAMPLES}" >&2
    exit 2
fi
TRAIN_NODES=("${NODES[@]:0:$TRAIN_NODE_COUNT}")
ROLLOUT_NODES=("${NODES[@]:$TRAIN_NODE_COUNT}")
ROLLOUT_GPUS=$((${#ROLLOUT_NODES[@]} * 4))
TOTAL_GPUS=$((TRAIN_GPUS + ROLLOUT_GPUS))
MASTER_ADDR="${TRAIN_NODES[0]}"
RUNTIME_CONFIG="$PROJECT_DIR/configs/.r2e_gym_cmlfq_24gpu_${JOB_ID}.yaml"
PLAN_ENV="$JOB_LOG_DIR/planned_rollout.env"

echo "============================================================"
echo "R2E-Gym full RL post-training with C-MLFQ, Qwen3-30B-A3B, ${TOTAL_GPUS} GPUs"
echo "============================================================"
echo "Job: $JOB_ID"
echo "Model: $MODEL_PATH"
echo "Model arch config: $MODEL_ARCH_CONFIG"
echo "Train backend: $TRAIN_BACKEND"
echo "Train topology: TP=$TRAIN_TP_SIZE PP=$TRAIN_PP_SIZE CP=$TRAIN_CP_SIZE EP=$TRAIN_EP_SIZE"
echo "Megatron CPU initialization: $MEGATRON_USE_CPU_INITIALIZATION"
echo "Megatron grouped GEMM: ${MCORE_MOE_GROUPED_GEMM:-1}"
echo "Train nodes: ${TRAIN_NODES[*]} (${TRAIN_GPUS} GPUs)"
echo "Rollout nodes: ${ROLLOUT_NODES[*]} (GlobalResourcePlanner decides TP layout)"
echo "Total GPUs: $TOTAL_GPUS (train=${TRAIN_GPUS}, rollout=${ROLLOUT_GPUS})"
echo "Total steps: $TOTAL_STEPS"
echo "Max model len: $MAX_MODEL_LEN"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Train batch size: $TRAIN_BATCH_SIZE"
echo "Local batch per DP replica: $LOCAL_BATCH_SIZE"
echo "N samples: $N_SAMPLES"
echo "Max concurrent rollouts: $MAX_CONCURRENT_ROLLOUTS"
echo "Queue size: $QUEUE_SIZE"
echo "R2E max turns: $R2E_MAX_TURNS"
echo "R2E max prompt tokens: $R2E_MAX_PROMPT_TOKENS"
echo "Eval samples: $EVAL_MAX_SAMPLES ($R2E_EVAL_STRATEGY)"
echo "Long-context TP first: $LONG_CONTEXT_TP_FIRST"
echo "vLLM read timeout: $VLLM_READ_TIMEOUT"
echo "vLLM NCCL_P2P_DISABLE: $VLLM_NCCL_P2P_DISABLE"
echo "Train NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "Train NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "Logs: $JOB_LOG_DIR"
python3 -c 'import torch, transformers, vllm; print("torch", torch.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)'

VLLM_STEP_PIDS=()
declare -A VLLM_PID_BY_NAME=()
ROLLOUT_NODE_LIST=()
ROLLOUT_TP_LIST=()
ROLLOUT_PORT_LIST=()
ROLLOUT_NAME_LIST=()
ROLLOUT_GPU_LIST=()

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    echo "[cleanup] stopping vLLM processes..."
    for pid in "${VLLM_STEP_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for i in "${!ROLLOUT_NODE_LIST[@]}"; do
        local node="${ROLLOUT_NODE_LIST[$i]}"
        local port="${ROLLOUT_PORT_LIST[$i]}"
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
            bash -lc "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${port}' 2>/dev/null || true" \
            >/dev/null 2>&1 || true
    done
    python3 "$PROJECT_DIR/scripts/merge_cmlfq_prefix_trees.py" \
        --output "$CMLFQ_TREE_PATH" || true
    echo "[cleanup] complete, exit=$exit_code"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "[1/6] Running GlobalResourcePlanner preflight for ${TOTAL_GPUS}-GPU runtime config"
export PROJECT_DIR MODEL_PATH MODEL_ARCH_CONFIG TOTAL_STEPS MAX_MODEL_LEN MAX_NEW_TOKENS EVAL_INTERVAL MAX_WAIT SYNC_INTERVAL RL_TRAIN_PHASE_TRACE
export TRAIN_BATCH_SIZE TRAIN_MICRO_BATCH_SIZE N_SAMPLES MAX_CONCURRENT_ROLLOUTS QUEUE_SIZE VLLM_BASE_PORT GPU_MEMORY_UTILIZATION JOB_ID
export JOB_LOG_DIR EXPERIMENT_TAG SCHEDULER_TYPE GRP_ENABLED
export LONG_CONTEXT_TP_FIRST VLLM_READ_TIMEOUT VLLM_CONNECT_TIMEOUT VLLM_WRITE_TIMEOUT VLLM_POOL_TIMEOUT
export VLLM_NCCL_P2P_DISABLE VLLM_NCCL_IB_DISABLE VLLM_CUDA_LAUNCH_BLOCKING
export NCCL_P2P_DISABLE NCCL_IB_DISABLE
export TRAIN_GPUS ROLLOUT_GPUS TOTAL_GPUS
export TRAIN_BACKEND TRAIN_TP_SIZE TRAIN_PP_SIZE TRAIN_CP_SIZE TRAIN_EP_SIZE
export EXPERT_TENSOR_PARALLEL_SIZE USE_DISTRIBUTED_OPTIMIZER
export MEGATRON_USE_PRECISION_AWARE_OPTIMIZER MEGATRON_OPTIMIZER_CPU_OFFLOAD
export MEGATRON_OPTIMIZER_OFFLOAD_FRACTION
export RUNTIME_CONFIG PLAN_ENV
export ALLOW_QWEN30_FALLBACK
export CMLFQ_TREE_PATH CMLFQ_SHARED_LOAD_DIR CMLFQ_SHARED_LOAD_TTL CMLFQ_SHARED_LOAD_HEARTBEAT CMLFQ_TREE_PERSIST_INTERVAL
ROLLOUT_NODES_CSV="$(IFS=,; echo "${ROLLOUT_NODES[*]}")"
export ROLLOUT_NODES_CSV
python3 - <<'PY'
import os
import shlex
from pathlib import Path
import yaml

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.preflight_planner import (
    PreflightPlanner,
    synthetic_history,
)

project = Path(os.environ["PROJECT_DIR"])
base = project / "configs" / "r2e_gym_cmlfq_qwen3_4b.yaml"
seed_path = project / "configs" / f".r2e_gym_cmlfq_24gpu_seed_{os.environ['JOB_ID']}.yaml"
runtime_path = Path(os.environ["RUNTIME_CONFIG"])
plan_env = Path(os.environ["PLAN_ENV"])
rollout_nodes = [n for n in os.environ["ROLLOUT_NODES_CSV"].split(",") if n]
base_port = int(os.environ["VLLM_BASE_PORT"])
max_model_len = int(os.environ["MAX_MODEL_LEN"])
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
train_gpus = int(os.environ["TRAIN_GPUS"])
rollout_gpus = int(os.environ["ROLLOUT_GPUS"])
total_gpus = int(os.environ["TOTAL_GPUS"])
train_tp = int(os.environ["TRAIN_TP_SIZE"])
train_pp = int(os.environ["TRAIN_PP_SIZE"])
train_cp = int(os.environ["TRAIN_CP_SIZE"])
train_dp = train_gpus // (train_tp * train_pp * train_cp)

seed = {
    "base_config": str(base),
    "model_arch_config": os.environ["MODEL_ARCH_CONFIG"],
    "model_path": os.environ["MODEL_PATH"],
    "tokenizer_path": os.environ["MODEL_PATH"],
    "train_gpus": train_gpus,
    "rollout_gpus": rollout_gpus,
    "n_total_gpus": total_gpus,
    "train_backend": os.environ["TRAIN_BACKEND"],
    "train_tp_size": train_tp,
    "train_pp_size": train_pp,
    "train_cp_size": train_cp,
    "train_ep_size": int(os.environ["TRAIN_EP_SIZE"]),
    "expert_tensor_parallel_size": int(os.environ["EXPERT_TENSOR_PARALLEL_SIZE"]),
    "train_dp_size": train_dp,
    "use_distributed_optimizer": os.environ["USE_DISTRIBUTED_OPTIMIZER"] == "1",
    "megatron_grad_reduce_in_fp32": (
        os.environ.get("MEGATRON_GRAD_REDUCE_IN_FP32", "0") == "1"
    ),
    "megatron_use_precision_aware_optimizer": (
        os.environ.get("MEGATRON_USE_PRECISION_AWARE_OPTIMIZER", "0") == "1"
    ),
    "megatron_optimizer_cpu_offload": (
        os.environ.get("MEGATRON_OPTIMIZER_CPU_OFFLOAD", "0") == "1"
    ),
    "megatron_optimizer_offload_fraction": float(
        os.environ.get("MEGATRON_OPTIMIZER_OFFLOAD_FRACTION", "0.0")
    ),
    "megatron_optimizer_pin_cpu_grads": (
        os.environ.get("MEGATRON_OPTIMIZER_PIN_CPU_GRADS", "1") == "1"
    ),
    "megatron_optimizer_pin_cpu_params": (
        os.environ.get("MEGATRON_OPTIMIZER_PIN_CPU_PARAMS", "1") == "1"
    ),
    "megatron_checkpoint_format": "torch_dist",
    "megatron_fully_parallel_save": True,
    "megatron_async_save": False,
    "megatron_streaming_export": os.environ.get("MEGATRON_STREAMING_EXPORT", "1") == "1",
    "megatron_use_transformer_engine": False,
    "megatron_use_cpu_initialization": (
        os.environ.get("MEGATRON_USE_CPU_INITIALIZATION", "0") == "1"
    ),
    "megatron_recompute_num_layers": 1,
    "batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["TRAIN_MICRO_BATCH_SIZE"]),
    "n_samples": int(os.environ["N_SAMPLES"]),
    "total_steps": int(os.environ["TOTAL_STEPS"]),
    "sync_interval": int(os.environ["SYNC_INTERVAL"]),
    "eval_interval": int(os.environ["EVAL_INTERVAL"]),
    "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    "max_seq_length": int(os.environ["MAX_MODEL_LEN"]),
    "max_concurrent_rollouts": int(os.environ["MAX_CONCURRENT_ROLLOUTS"]),
    "queue_size": int(os.environ["QUEUE_SIZE"]),
    "sync_path": str(project / "logs" / "r2e_gym_cmlfq_24gpu" / f"weights_{os.environ['JOB_ID']}"),
    "rollout_weight_sync_mode": os.environ["ROLLOUT_WEIGHT_SYNC_MODE"],
    "rollout_weight_sync_control_dir": str(project / "logs" / "r2e_gym_cmlfq_24gpu" / f"job_{os.environ['JOB_ID']}" / "rollout_weight_sync"),
    "rollout_weight_sync_timeout_s": int(os.environ["MAX_WAIT"]),
    "rollout_weight_sync_export_only": os.environ.get("ROLLOUT_WEIGHT_SYNC_EXPORT_ONLY", "1") == "1",
    "require_rollout_weight_sync": os.environ.get("REQUIRE_ROLLOUT_WEIGHT_SYNC", "1") == "1",
    "log_dir": str(project / "logs" / "r2e_gym_cmlfq_24gpu"),
    "wandb_run_name": f"{os.environ['EXPERIMENT_TAG']}-{os.environ['JOB_ID']}",
    "enable_history_collection": True,
    "history_output_dir": str(Path(os.environ["JOB_LOG_DIR"]) / "history"),
    "history_save_raw_lengths": True,
    "history_flush_interval": 1,
    "history_experiment_name": os.environ["EXPERIMENT_TAG"],
    "heterogeneous_rollout": {
        "enabled": True,
        "total_gpus": rollout_gpus,
        "available_gpus": list(range(train_gpus, total_gpus)),
        "vllm_base_port": int(os.environ["VLLM_BASE_PORT"]),
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
        "instances": [],
        "scheduling": {
            "scheduler_type": os.environ["SCHEDULER_TYPE"],
            "load_balance_strategy": "least_connections",
            "max_queue_length": int(os.environ["QUEUE_SIZE"]),
            "enable_fallback": True,
            "cmlfq_tree_path": os.environ["CMLFQ_TREE_PATH"],
            "cmlfq_tree_persist_interval": int(os.environ["CMLFQ_TREE_PERSIST_INTERVAL"]),
            "cmlfq_shared_load_dir": os.environ["CMLFQ_SHARED_LOAD_DIR"],
            "cmlfq_shared_load_ttl_s": float(os.environ["CMLFQ_SHARED_LOAD_TTL"]),
            "cmlfq_shared_load_heartbeat_s": float(os.environ["CMLFQ_SHARED_LOAD_HEARTBEAT"]),
            "cmlfq_buckets": {
                "short": {
                    "tp_degrees": [2],
                    "max_tokens": 3000,
                },
                "medium": {
                    "tp_degrees": [2],
                    "max_tokens": 6000,
                },
                "long": {"tp_degrees": [4, 2], "max_tokens": max_model_len},
            },
        },
    },
    "global_resource_planner": {
        "enabled": os.environ["GRP_ENABLED"] == "1",
        "train_backend": "analytic",
        "rollout_backend": "analytic",
        "plan_interval": int(os.environ["GRP_INTERVAL"]),
        "warmup_steps": int(os.environ["GRP_WARMUP_STEPS"]),
        "min_history_size": int(os.environ["GRP_MIN_HISTORY"]),
        "min_gain_ratio": float(os.environ["GRP_MIN_GAIN_RATIO"]),
        "reconfiguration_cost_s": 15.0,
        "allowed_rollout_tp": [2, 4],
        "require_heterogeneous_rollout_tp": (
            os.environ["GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP"] == "1"
        ),
        "allowed_train_tp": [train_tp],
        "allowed_train_pp": [train_pp],
        "fixed_train_gpus": train_gpus,
        "micro_batch_sizes": [1],
        "apply_to_runtime": True,
        "verbose": True,
        "runtime_async_planning": os.environ["GRP_RUNTIME_ASYNC_PLANNING"] == "1",
        "runtime_max_pending_plans": 1,
        "runtime_dynamic_reconfiguration_enabled": os.environ["GRP_DYNAMIC_RECONFIG_ENABLED"] == "1",
        "runtime_online_replanning": os.environ["GRP_ONLINE_REPLANNING"] == "1",
        "runtime_replan_cooldown_steps": int(os.environ["GRP_REPLAN_COOLDOWN_STEPS"]),
        "runtime_queue_pressure_threshold": float(os.environ["GRP_QUEUE_PRESSURE_THRESHOLD"]),
        "runtime_active_rollout_pressure_threshold": float(os.environ["GRP_ACTIVE_ROLLOUT_PRESSURE_THRESHOLD"]),
        "runtime_rejected_rollout_delta_threshold": int(os.environ["GRP_REJECTED_ROLLOUT_DELTA_THRESHOLD"]),
        "runtime_rollout_train_imbalance_threshold": float(os.environ["GRP_ROLLOUT_TRAIN_IMBALANCE_THRESHOLD"]),
        "runtime_manage_rollout_processes": os.environ["GRP_MANAGE_ROLLOUT_PROCESSES"] == "1",
        "runtime_reconfigure_training": os.environ["GRP_RECONFIGURE_TRAINING"] == "1",
        "runtime_training_pool_only": os.environ["GRP_TRAINING_POOL_ONLY"] == "1",
        "runtime_training_pool_target_gpus": int(os.environ["GRP_TRAINING_POOL_TARGET_GPUS"]),
        "runtime_training_pool_plan_only": os.environ["GRP_TRAINING_POOL_PLAN_ONLY"] == "1",
        "runtime_cluster_swap_enabled": os.environ["GRP_CLUSTER_SWAP_ENABLED"] == "1",
        "hybrid_worker_launch_enabled": os.environ["GRP_HYBRID_WORKER_LAUNCH_ENABLED"] == "1",
        "hybrid_training_prewarm_enabled": os.environ["GRP_HYBRID_TRAINING_PREWARM_ENABLED"] == "1",
        "hybrid_training_prewarm_count": int(os.environ["GRP_HYBRID_TRAINING_PREWARM_COUNT"]),
        "hybrid_training_prewarm_worker_ids": [
            item.strip()
            for item in os.environ["GRP_HYBRID_TRAINING_PREWARM_WORKER_IDS"].split(",")
            if item.strip()
        ],
        "hybrid_worker_task_dir": os.environ["GRP_HYBRID_WORKER_TASK_DIR"],
        "hybrid_worker_command_template": os.environ["GRP_HYBRID_WORKER_COMMAND_TEMPLATE"],
        "hybrid_worker_ready_timeout_s": float(os.environ["GRP_HYBRID_WORKER_READY_TIMEOUT_S"]),
        "gradient_transport_backend": os.environ["GRP_GRADIENT_TRANSPORT_BACKEND"],
        "decouple_communication_domains": os.environ["GRP_DECOUPLE_COMMUNICATION_DOMAINS"] == "1",
        "gradient_server_host": os.environ["GRP_GRADIENT_SERVER_HOST"],
        "gradient_server_public_host": os.environ["GRP_GRADIENT_SERVER_PUBLIC_HOST"],
        "gradient_server_port": int(os.environ["GRP_GRADIENT_SERVER_PORT"]),
        "gradient_server_authkey": os.environ["GRP_GRADIENT_SERVER_AUTHKEY"],
        "runtime_rollout_reconfigure_strategy": os.environ["GRP_ROLLOUT_RECONFIGURE_STRATEGY"],
        "runtime_prewarm_no_spare_fallback_strategy": os.environ["GRP_PREWARM_NO_SPARE_FALLBACK_STRATEGY"],
        "runtime_drain_before_reconfigure": os.environ["GRP_DRAIN_BEFORE_RECONFIGURE"] == "1",
        "runtime_peer_request_wait_s": float(os.environ["GRP_RUNTIME_PEER_REQUEST_WAIT_S"]),
        "runtime_batch_collection_timeout_s": float(os.environ["GRP_BATCH_COLLECTION_TIMEOUT_S"]),
        "runtime_batch_collection_max_retries": int(os.environ["GRP_BATCH_COLLECTION_MAX_RETRIES"]),
        "runtime_use_nccl_barrier_after_rollout": os.environ["GRP_USE_NCCL_BARRIER_AFTER_ROLLOUT"] == "1",
        "runtime_use_nccl_barrier_before_weight_sync": os.environ["GRP_USE_NCCL_BARRIER_BEFORE_WEIGHT_SYNC"] == "1",
        "runtime_rollout_manifest_path": os.environ["GRP_ROLLOUT_MANIFEST_PATH"],
        "runtime_rollout_pid_registry_path": os.environ["GRP_ROLLOUT_PID_REGISTRY_PATH"],
        "runtime_wait_rollout_process_start_ready": os.environ["GRP_WAIT_ROLLOUT_PROCESS_START_READY"] == "1",
        "runtime_post_rollout_stop_grace_s": float(os.environ["GRP_POST_ROLLOUT_STOP_GRACE_S"]),
        "runtime_rollout_log_dir": str(project / "logs" / "r2e_gym_cmlfq_24gpu" / f"job_{os.environ['JOB_ID']}" / "runtime_rollout"),
        "vllm_ready_timeout_s": float(os.environ["GRP_VLLM_READY_TIMEOUT_S"]),
        "vllm_launch_command_template": os.environ["GRP_VLLM_LAUNCH_COMMAND_TEMPLATE"],
        "vllm_stop_command_template": os.environ["GRP_VLLM_STOP_COMMAND_TEMPLATE"],
        "simulator_allow_fallback": True,
        "simulator_timeout_s": 240.0,
    },
}
seed_path.write_text(yaml.safe_dump(seed, sort_keys=False), encoding="utf-8")

cfg = AsyncRLConfig.from_yaml(str(seed_path))
history = synthetic_history(
    num_requests=max(64, int(os.environ["TRAIN_BATCH_SIZE"]) * 4),
    input_len=max(512, max_model_len // 2),
    output_len=max(32, max_new_tokens),
)
result = PreflightPlanner(cfg, apply_best_candidate=True).run(history)
planned = result.planned_config
planned.train_backend = os.environ["TRAIN_BACKEND"]
planned.train_tp_size = train_tp
planned.tp_size = train_tp
planned.train_pp_size = train_pp
planned.train_cp_size = train_cp
planned.train_ep_size = int(os.environ["TRAIN_EP_SIZE"])
planned.expert_tensor_parallel_size = int(
    os.environ["EXPERT_TENSOR_PARALLEL_SIZE"]
)
planned.train_dp_size = train_dp
planned.use_distributed_optimizer = (
    os.environ["USE_DISTRIBUTED_OPTIMIZER"] == "1"
)
planned.megatron_grad_reduce_in_fp32 = (
    os.environ.get("MEGATRON_GRAD_REDUCE_IN_FP32", "0") == "1"
)
planned.megatron_use_precision_aware_optimizer = (
    os.environ.get("MEGATRON_USE_PRECISION_AWARE_OPTIMIZER", "0") == "1"
)
planned.megatron_optimizer_cpu_offload = (
    os.environ.get("MEGATRON_OPTIMIZER_CPU_OFFLOAD", "0") == "1"
)
planned.megatron_optimizer_offload_fraction = float(
    os.environ.get("MEGATRON_OPTIMIZER_OFFLOAD_FRACTION", "0.0")
)
planned.megatron_optimizer_pin_cpu_grads = (
    os.environ.get("MEGATRON_OPTIMIZER_PIN_CPU_GRADS", "1") == "1"
)
planned.megatron_optimizer_pin_cpu_params = (
    os.environ.get("MEGATRON_OPTIMIZER_PIN_CPU_PARAMS", "1") == "1"
)
planned.megatron_use_cpu_initialization = (
    os.environ.get("MEGATRON_USE_CPU_INITIALIZATION", "0") == "1"
)

if not planned.heterogeneous_rollout.instances:
    if os.environ.get("ALLOW_QWEN30_FALLBACK", "0") != "1":
        raise RuntimeError(
            "GlobalResourcePlanner produced no feasible candidate. "
            "Refusing Qwen30 fallback because it can hide train-memory OOM. "
            "Set ALLOW_QWEN30_FALLBACK=1 only for diagnostic rollout runs."
        )
    fallback_layout_by_gpus = {
        4: [2, 2],
        8: [4, 2, 2],
        12: [4, 4, 2, 2],
        20: [4, 4, 4, 4, 2, 2],
    }
    fallback_tp_layout = fallback_layout_by_gpus.get(rollout_gpus)
    if fallback_tp_layout is None:
        fallback_tp_layout = []
        remaining_gpus_for_layout = rollout_gpus
        while remaining_gpus_for_layout >= 4:
            fallback_tp_layout.append(4)
            remaining_gpus_for_layout -= 4
        if remaining_gpus_for_layout == 2:
            fallback_tp_layout.append(2)
        elif remaining_gpus_for_layout:
            raise RuntimeError(f"Cannot create Qwen30 fallback TP layout for rollout_gpus={rollout_gpus}")
    print(
        "GlobalResourcePlanner produced no feasible rollout candidate; "
        f"using Qwen30 fallback TP layout {fallback_tp_layout}"
    )
    planned.train_gpus = train_gpus
    planned.rollout_gpus = rollout_gpus
    planned.n_total_gpus = total_gpus
    planned.train_tp_size = train_tp
    planned.tp_size = train_tp
    planned.train_pp_size = train_pp
    planned.train_cp_size = train_cp
    planned.train_dp_size = train_dp
    planned.heterogeneous_rollout.enabled = True
    planned.heterogeneous_rollout.total_gpus = rollout_gpus
    planned.heterogeneous_rollout.instances = []
    from RL_Framework.config import HeterogeneousInstanceConfig
    for idx, tp in enumerate(fallback_tp_layout):
        planned.heterogeneous_rollout.instances.append(
            HeterogeneousInstanceConfig(
                instance_id=f"qwen30_fallback_tp{tp}_{idx}",
                tp=tp,
                gpus=list(range(tp)),
                host=planned.heterogeneous_rollout.vllm_host,
                description="qwen30_fallback_after_no_feasible_grp_candidate",
            )
        )
if os.environ.get("GRP_INITIAL_ROLLOUT_TP_LIST", "").strip():
    from RL_Framework.config import HeterogeneousInstanceConfig
    override_tps = [
        int(item.strip())
        for item in os.environ["GRP_INITIAL_ROLLOUT_TP_LIST"].replace(";", ":").replace(",", ":").split(":")
        if item.strip()
    ]
    if sum(override_tps) != rollout_gpus:
        raise RuntimeError(
            "GRP_INITIAL_ROLLOUT_TP_LIST must sum to rollout_gpus="
            f"{rollout_gpus}, got {override_tps} sum={sum(override_tps)}"
        )
    planned.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(
            instance_id=f"initial_override_tp{tp}_{idx}",
            tp=tp,
            gpus=[],
            host=planned.heterogeneous_rollout.vllm_host,
            description="initial_rollout_layout_override_for_online_replanning",
        )
        for idx, tp in enumerate(override_tps)
    ]
    planned.train_gpus = train_gpus
    planned.rollout_gpus = rollout_gpus
    planned.n_total_gpus = total_gpus
    planned.train_tp_size = train_tp
    planned.train_pp_size = train_pp
    planned.train_dp_size = train_dp
    print(f"Using initial rollout TP override: {override_tps}")
tp_list = [int(inst.tp) for inst in planned.heterogeneous_rollout.instances]
require_heterogeneous_rollout_tp = (
    os.environ.get("GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP", "1") == "1"
)
if (
    require_heterogeneous_rollout_tp
    and len(set(tp_list)) <= 1
    and not os.environ.get("GRP_INITIAL_ROLLOUT_TP_LIST", "").strip()
):
    raise RuntimeError(f"GlobalResourcePlanner/fallback did not produce heterogeneous TP degrees: {tp_list}")
if sum(tp_list) != rollout_gpus:
    raise RuntimeError(f"GlobalResourcePlanner/fallback rollout GPUs must sum to {rollout_gpus}, got {sum(tp_list)} from {tp_list}")
if planned.train_gpus != train_gpus:
    raise RuntimeError(f"GlobalResourcePlanner/fallback must keep train_gpus={train_gpus}, got {planned.train_gpus}")

remaining = {node: 4 for node in rollout_nodes}
next_gpu = {node: 0 for node in rollout_nodes}
assigned_nodes: list[str] = []
assigned_tps: list[int] = []
assigned_ports: list[int] = []
assigned_names: list[str] = []
assigned_gpus: list[str] = []

for idx, inst in enumerate(planned.heterogeneous_rollout.instances):
    tp = int(inst.tp)
    node = next((candidate for candidate in rollout_nodes if remaining[candidate] >= tp), None)
    if node is None:
        raise RuntimeError(f"Cannot pack planned TP list {tp_list} onto nodes {rollout_nodes}")
    start_gpu = next_gpu[node]
    inst.gpus = list(range(start_gpu, start_gpu + tp))
    inst.host = node
    if not inst.instance_id:
        inst.instance_id = f"grp_tp{tp}_{idx}"
    remaining[node] -= tp
    next_gpu[node] += tp
    assigned_nodes.append(node)
    assigned_tps.append(tp)
    assigned_ports.append(base_port + idx)
    assigned_names.append(inst.instance_id)
    assigned_gpus.append(",".join(str(gpu) for gpu in inst.gpus))

planned.vllm_port = base_port
planned.vllm_num_instances = len(assigned_tps)
planned.rollout_gpus = sum(assigned_tps)
planned.n_total_gpus = planned.train_gpus + planned.rollout_gpus
planned.max_concurrent_rollouts = int(os.environ["MAX_CONCURRENT_ROLLOUTS"])
planned.queue_size = int(os.environ["QUEUE_SIZE"])
planned.heterogeneous_rollout.vllm_base_port = base_port
planned.heterogeneous_rollout.total_gpus = planned.rollout_gpus

grp = planned.global_resource_planner
grp.train_backend = "sailor"
grp.rollout_backend = "vidur"
grp.runtime_async_planning = os.environ["GRP_RUNTIME_ASYNC_PLANNING"] == "1"
grp.runtime_max_pending_plans = 1
grp.simulator_allow_fallback = True
grp.simulator_timeout_s = 240.0
grp.allowed_rollout_tp = [2, 4]
grp.require_heterogeneous_rollout_tp = (
    os.environ["GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP"] == "1"
)
grp.allowed_train_tp = [train_tp]
grp.allowed_train_pp = [train_pp]
grp.fixed_train_gpus = train_gpus
grp.micro_batch_sizes = [1]

planned.to_yaml(str(runtime_path))

def bash_array(values):
    return "(" + " ".join(shlex.quote(str(v)) for v in values) + ")"

plan_env.write_text(
    "\n".join(
        [
            f"ROLLOUT_COUNT={len(assigned_tps)}",
            f"ROLLOUT_NODE_LIST={bash_array(assigned_nodes)}",
            f"ROLLOUT_TP_LIST={bash_array(assigned_tps)}",
            f"ROLLOUT_PORT_LIST={bash_array(assigned_ports)}",
            f"ROLLOUT_NAME_LIST={bash_array(assigned_names)}",
            f"ROLLOUT_GPU_LIST={bash_array(assigned_gpus)}",
            f"HETERO_INSTANCE_HOSTS={shlex.quote(','.join(assigned_nodes))}",
            f"PLANNED_ROLLOUT_TP_LIST={shlex.quote(','.join(str(v) for v in assigned_tps))}",
            f"PLANNED_RUNTIME_CONFIG={shlex.quote(str(runtime_path))}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(f"seed_config={seed_path}")
print(f"runtime_config={runtime_path}")
print(f"decision={result.decision.to_dict()}")
print(f"planned_rollout_tp={assigned_tps}")
print(f"planned_rollout_nodes={assigned_nodes}")
PY

source "$PLAN_ENV"
echo "GlobalResourcePlanner rollout TP list: ${PLANNED_ROLLOUT_TP_LIST}"
echo "GlobalResourcePlanner rollout nodes: ${ROLLOUT_NODE_LIST[*]}"
TRAIN_NODE_LIST_CSV="$(IFS=,; echo "${TRAIN_NODES[*]}")"

if [ "$RUN_NCCL_PREFLIGHT" = "1" ]; then
    echo "[2/7] Running cross-node NCCL all-gather preflight"
    PREFLIGHT_PORT=$((MASTER_PORT + 1))
    PREFLIGHT_STATUS=0
    timeout "$((NCCL_PREFLIGHT_WALLTIME + 30))" srun --overlap \
        --nodes="$TRAIN_NODE_COUNT" \
        --ntasks="$TRAIN_NODE_COUNT" \
        --ntasks-per-node=1 \
        --nodelist="$TRAIN_NODE_LIST_CSV" \
        --gpus-per-task=4 \
        --cpus-per-task=8 \
        --mem=32G \
        bash -lc "
            source '$VENV_PATH'
            cd '$PROJECT_DIR'
            node_rank=\"\${SLURM_PROCID:-0}\"
            preflight_log='$JOB_LOG_DIR/nccl_preflight_rank'\${node_rank}'_'\${HOSTNAME}'.log'
            {
                export PYTHONPATH='$PROJECT_DIR':\${PYTHONPATH:-}
                export MASTER_ADDR='$MASTER_ADDR'
                export MASTER_PORT='$PREFLIGHT_PORT'
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                echo \"[NCCL preflight launcher] node_rank=\${node_rank} host=\${HOSTNAME} cuda_visible_devices=\${CUDA_VISIBLE_DEVICES}\"
                export NCCL_PREFLIGHT_ELEMENTS='$NCCL_PREFLIGHT_ELEMENTS'
                export NCCL_PREFLIGHT_ITERATIONS='$NCCL_PREFLIGHT_ITERATIONS'
                export NCCL_PREFLIGHT_TIMEOUT='$NCCL_PREFLIGHT_WALLTIME'
                export NCCL_P2P_DISABLE='$NCCL_P2P_DISABLE'
                export NCCL_IB_DISABLE='$NCCL_IB_DISABLE'
                export TORCH_NCCL_ASYNC_ERROR_HANDLING='$TORCH_NCCL_ASYNC_ERROR_HANDLING'
                export TORCH_NCCL_TRACE_BUFFER_SIZE='$TORCH_NCCL_TRACE_BUFFER_SIZE'
                timeout '$NCCL_PREFLIGHT_WALLTIME' torchrun \
                    --nnodes='$TRAIN_NODE_COUNT' \
                    --nproc_per_node=4 \
                    --node_rank=\"\${node_rank}\" \
                    --master_addr='$MASTER_ADDR' \
                    --master_port='$PREFLIGHT_PORT' \
                    scripts/nccl_allgather_preflight.py
            } >\"\$preflight_log\" 2>&1
        " || PREFLIGHT_STATUS=$?
    if compgen -G "$JOB_LOG_DIR/nccl_preflight_rank*.log" >/dev/null; then
        cat "$JOB_LOG_DIR"/nccl_preflight_rank*.log
    fi
    if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
        if [ "$NCCL_PREFLIGHT_REQUIRED" = "1" ]; then
            echo "ERROR: NCCL preflight failed with status ${PREFLIGHT_STATUS}" >&2
            exit "$PREFLIGHT_STATUS"
        fi
        echo "WARNING: NCCL preflight failed with status ${PREFLIGHT_STATUS}; continuing because NCCL_PREFLIGHT_REQUIRED=0" >&2
    fi
fi

start_vllm() {
    local node="$1"
    local tp="$2"
    local port="$3"
    local name="$4"
    local gpu_list="$5"
    local log_file="$JOB_LOG_DIR/vllm_${name}.log"
    echo "Starting $name on $node TP=$tp planned_GPUs=$gpu_list port=$port"
    srun --exclusive \
        --nodes=1 --ntasks=1 --nodelist="$node" \
        --gpus="$tp" \
        --cpus-per-task=16 \
        bash -lc "
            source '$VENV_PATH'
            export TOKENIZERS_PARALLELISM=false
            export PLANNED_CUDA_VISIBLE_DEVICES='$gpu_list'
            if [ -z \"\${CUDA_VISIBLE_DEVICES:-}\" ]; then
                export CUDA_VISIBLE_DEVICES=\"\$PLANNED_CUDA_VISIBLE_DEVICES\"
            fi
            echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES} planned=\${PLANNED_CUDA_VISIBLE_DEVICES}\"
            export MASTER_ADDR=127.0.0.1
            export MASTER_PORT=$((20000 + port))
            export NCCL_P2P_DISABLE='$VLLM_NCCL_P2P_DISABLE'
            export NCCL_IB_DISABLE='$VLLM_NCCL_IB_DISABLE'
            export CUDA_LAUNCH_BLOCKING='$VLLM_CUDA_LAUNCH_BLOCKING'
            python3 '$PROJECT_DIR/scripts/restartable_vllm_server.py' \
                --instance-id '$name' \
                --control-dir '$ROLLOUT_SYNC_DIR' \
                --health-url 'http://127.0.0.1:$port/health' \
                --initial-model '$MODEL_PATH' \
                --ready-timeout '$MAX_WAIT' \
                -- python3 -m vllm.entrypoints.openai.api_server \
                --model '__MODEL_PATH__' \
                --served-model-name '$MODEL_PATH' \
                --host 0.0.0.0 \
                --port '$port' \
                --tensor-parallel-size '$tp' \
                --max-model-len '$MAX_MODEL_LEN' \
                --dtype bfloat16 \
                --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
                --trust-remote-code \
                --enforce-eager \
                $VLLM_CUSTOM_ALL_REDUCE_FLAG
        " >"$log_file" 2>&1 &
    VLLM_STEP_PIDS+=("$!")
    VLLM_PID_BY_NAME["$name"]="$!"
    python3 - <<PY
import json
from pathlib import Path

path = Path("$GRP_ROLLOUT_PID_REGISTRY_PATH")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps({
        "instance_id": "$name",
        "host": "$node",
        "tp": int("$tp"),
        "port": int("$port"),
        "gpus": [int(x) for x in "$gpu_list".split(",") if x],
        "pid": int("${VLLM_PID_BY_NAME[$name]}"),
        "log_path": "$log_file",
    }, sort_keys=True) + "\n")
PY
}

wait_for_health() {
    local host="$1"
    local port="$2"
    local name="$3"
    local elapsed=0
    local pid="${VLLM_PID_BY_NAME[$name]}"
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
            echo "READY: $name at $host:$port (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name exited before ready" >&2
            tail -120 "$JOB_LOG_DIR/vllm_${name}.log" || true
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: $name not ready after ${MAX_WAIT}s" >&2
    tail -120 "$JOB_LOG_DIR/vllm_${name}.log" || true
    return 1
}

echo "[3/7] Starting GlobalResourcePlanner rollout vLLM instances"
PRIMARY_INDICES=()
DEFERRED_INDICES=()
declare -A NODE_HAS_PRIMARY=()
for i in "${!ROLLOUT_NODE_LIST[@]}"; do
    node="${ROLLOUT_NODE_LIST[$i]}"
    if [ -z "${NODE_HAS_PRIMARY[$node]:-}" ]; then
        NODE_HAS_PRIMARY["$node"]=1
        PRIMARY_INDICES+=("$i")
    else
        DEFERRED_INDICES+=("$i")
    fi
done

for i in "${PRIMARY_INDICES[@]}"; do
    start_vllm \
        "${ROLLOUT_NODE_LIST[$i]}" \
        "${ROLLOUT_TP_LIST[$i]}" \
        "${ROLLOUT_PORT_LIST[$i]}" \
        "${ROLLOUT_NAME_LIST[$i]}" \
        "${ROLLOUT_GPU_LIST[$i]}"
done

echo "[4/7] Waiting for primary rollout services"
for i in "${PRIMARY_INDICES[@]}"; do
    wait_for_health \
        "${ROLLOUT_NODE_LIST[$i]}" \
        "${ROLLOUT_PORT_LIST[$i]}" \
        "${ROLLOUT_NAME_LIST[$i]}"
done

if [ "${#DEFERRED_INDICES[@]}" -gt 0 ]; then
    echo "[3b/6] Starting deferred same-node rollout services"
    for i in "${DEFERRED_INDICES[@]}"; do
        start_vllm \
            "${ROLLOUT_NODE_LIST[$i]}" \
            "${ROLLOUT_TP_LIST[$i]}" \
            "${ROLLOUT_PORT_LIST[$i]}" \
            "${ROLLOUT_NAME_LIST[$i]}" \
            "${ROLLOUT_GPU_LIST[$i]}"
        wait_for_health \
            "${ROLLOUT_NODE_LIST[$i]}" \
            "${ROLLOUT_PORT_LIST[$i]}" \
            "${ROLLOUT_NAME_LIST[$i]}"
    done
fi

echo "[5/7] Checking R2E-Gym data and C-MLFQ flow"
python3 examples/test_r2e_gym.py
python3 examples/test_r2e_cmlfq_flow.py

echo "[6/7] OpenAI-compatible API check"
curl -fsS "http://${ROLLOUT_NODE_LIST[0]}:${ROLLOUT_PORT_LIST[0]}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"Write one short sentence.\",\"max_tokens\":8,\"temperature\":0}" \
    | python3 -m json.tool

export HETERO_INSTANCE_HOSTS
export R2E_GYM_INDEX="${PROJECT_DIR}/data/r2e_gym_v1/index.jsonl"
export R2E_MAX_TURNS
export R2E_MAX_PROMPT_TOKENS
export R2E_EVAL_MAX_SAMPLES="$EVAL_MAX_SAMPLES"
export R2E_EVAL_CONCURRENCY="$EVAL_CONCURRENCY"
export R2E_EVAL_MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS"
export R2E_EVAL_MULTI_TURN
export R2E_EVAL_RECORD_SAMPLES
export R2E_EVAL_STRATEGY
export VLLM_READ_TIMEOUT
export VLLM_CONNECT_TIMEOUT
export VLLM_WRITE_TIMEOUT
export VLLM_POOL_TIMEOUT
export TORCH_DISTRIBUTED_TIMEOUT
export FSDP_ALIGN_SEQUENCE_LENGTHS
export FSDP_LOCKSTEP_BARRIERS
export FSDP_GRADIENT_CHECKPOINTING
export FSDP_ACTIVATION_CHECKPOINTING
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[7/7] Starting full R2E-Gym AsyncRLTrainer run on ${TOTAL_GPUS} GPUs"
TRAIN_STEP_PIDS=()
for node_rank in "${!TRAIN_NODES[@]}"; do
    node="${TRAIN_NODES[$node_rank]}"
    train_log="$JOB_LOG_DIR/training_rank${node_rank}_${node}.log"
    echo "Starting training rank $node_rank on $node"
    srun --overlap --exact \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$node" \
        --gpus=4 \
        --cpus-per-task=16 \
        bash -lc "
            source '$VENV_PATH'
            cd '$PROJECT_DIR'
            export PYTHONPATH='$PROJECT_DIR':\${PYTHONPATH:-}
            export HETERO_INSTANCE_HOSTS='$HETERO_INSTANCE_HOSTS'
            export R2E_GYM_INDEX='$R2E_GYM_INDEX'
            export R2E_MAX_TURNS='$R2E_MAX_TURNS'
            export R2E_MAX_PROMPT_TOKENS='$R2E_MAX_PROMPT_TOKENS'
            export R2E_EVAL_MAX_SAMPLES='$R2E_EVAL_MAX_SAMPLES'
            export R2E_EVAL_CONCURRENCY='$R2E_EVAL_CONCURRENCY'
            export R2E_EVAL_MAX_NEW_TOKENS='$R2E_EVAL_MAX_NEW_TOKENS'
            export R2E_EVAL_MULTI_TURN='$R2E_EVAL_MULTI_TURN'
            export R2E_EVAL_RECORD_SAMPLES='$R2E_EVAL_RECORD_SAMPLES'
            export R2E_EVAL_STRATEGY='$R2E_EVAL_STRATEGY'
            export VLLM_READ_TIMEOUT='$VLLM_READ_TIMEOUT'
            export VLLM_CONNECT_TIMEOUT='$VLLM_CONNECT_TIMEOUT'
            export VLLM_WRITE_TIMEOUT='$VLLM_WRITE_TIMEOUT'
            export VLLM_POOL_TIMEOUT='$VLLM_POOL_TIMEOUT'
            export TORCH_DISTRIBUTED_TIMEOUT='$TORCH_DISTRIBUTED_TIMEOUT'
            export FSDP_ALIGN_SEQUENCE_LENGTHS='$FSDP_ALIGN_SEQUENCE_LENGTHS'
            export FSDP_LOCKSTEP_BARRIERS='$FSDP_LOCKSTEP_BARRIERS'
            export FSDP_GRADIENT_CHECKPOINTING='$FSDP_GRADIENT_CHECKPOINTING'
            export FSDP_ACTIVATION_CHECKPOINTING='$FSDP_ACTIVATION_CHECKPOINTING'
            export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'
            export MCORE_MOE_GROUPED_GEMM='$MCORE_MOE_GROUPED_GEMM'
            export FSDP_PROGRESS_DIR='$JOB_LOG_DIR/fsdp_progress'
            export TORCH_NCCL_ASYNC_ERROR_HANDLING='$TORCH_NCCL_ASYNC_ERROR_HANDLING'
            export TORCH_NCCL_TRACE_BUFFER_SIZE='$TORCH_NCCL_TRACE_BUFFER_SIZE'
            export TORCH_NCCL_DUMP_ON_TIMEOUT='$TORCH_NCCL_DUMP_ON_TIMEOUT'
            export TORCH_NCCL_DESYNC_DEBUG='$TORCH_NCCL_DESYNC_DEBUG'
            export TORCH_NCCL_ENABLE_MONITORING='$TORCH_NCCL_ENABLE_MONITORING'
            export TORCH_NCCL_ENABLE_TIMING='$TORCH_NCCL_ENABLE_TIMING'
            export TORCH_NCCL_HIGH_PRIORITY='$TORCH_NCCL_HIGH_PRIORITY'
            export NCCL_DEBUG='$NCCL_DEBUG'
            export NCCL_P2P_DISABLE='$NCCL_P2P_DISABLE'
            export NCCL_IB_DISABLE='$NCCL_IB_DISABLE'
            export PYTHONUNBUFFERED=1
            export MASTER_ADDR='$MASTER_ADDR'
            export MASTER_PORT='$MASTER_PORT'
            export TOKENIZERS_PARALLELISM=false
            export WANDB_MODE='$WANDB_MODE'
            export WANDB_DIR='$WANDB_DIR'
            torchrun \
                --nnodes='$TRAIN_NODE_COUNT' \
                --nproc_per_node=4 \
                --node_rank='$node_rank' \
                --master_addr='$MASTER_ADDR' \
                --master_port='$MASTER_PORT' \
                examples/r2e_gym_async_rl.py \
                --config '$RUNTIME_CONFIG'
        " >"$train_log" 2>&1 &
    TRAIN_STEP_PIDS+=("$!")
done

train_exit=0
for pid in "${TRAIN_STEP_PIDS[@]}"; do
    if ! wait "$pid"; then
        train_exit=1
    fi
done
cat "$JOB_LOG_DIR"/training_rank*.log > "$JOB_LOG_DIR/training.log" 2>/dev/null || true
if [ "$train_exit" -ne 0 ]; then
    echo "ERROR: one or more training ranks failed" >&2
    exit "$train_exit"
fi

echo "============================================================"
echo "R2E-Gym C-MLFQ ${TOTAL_GPUS}-GPU training completed"
echo "Logs: $JOB_LOG_DIR"
echo "C-MLFQ tree: $CMLFQ_TREE_PATH"
echo "History: $PROJECT_DIR/logs/r2e_gym_cmlfq/history and logs/r2e_gym_cmlfq_24gpu"
echo "============================================================"

#!/bin/bash
#SBATCH -J r2e_cmlfq_24gpu
#SBATCH -o logs/r2e_cmlfq_24gpu_%j.out
#SBATCH -e logs/r2e_cmlfq_24gpu_%j.err
#SBATCH -N 6
#SBATCH -G 24
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/RL_Framework}"
VENV_PATH="${VENV_PATH:-/path/to/rl_framework_env/bin/activate}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_WAIT="${MAX_WAIT:-1200}"
R2E_MAX_TURNS="${R2E_MAX_TURNS:-3}"
R2E_MAX_PROMPT_TOKENS="${R2E_MAX_PROMPT_TOKENS:-0}"
GRP_PROFILE_DATASET_JSONL="${GRP_PROFILE_DATASET_JSONL:-$PROJECT_DIR/data/r2e_gym_v1/index.jsonl}"
GRP_PROFILE_SAMPLE_SIZE="${GRP_PROFILE_SAMPLE_SIZE:-64}"
GRP_PROFILE_STRATEGY="${GRP_PROFILE_STRATEGY:-spread}"
GRP_PROFILE_SEED="${GRP_PROFILE_SEED:-0}"
GRP_PROFILE_SAMPLES_PER_PROMPT="${GRP_PROFILE_SAMPLES_PER_PROMPT:-1}"

source "$VENV_PATH"
cd "$PROJECT_DIR"

mkdir -p logs
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="$PROJECT_DIR/logs/r2e_gym_cmlfq_24gpu/job_${JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${PROJECT_DIR}/wandb_r2e_gym_cmlfq_24gpu"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#NODES[@]}" -lt 6 ]; then
    echo "ERROR: 24-GPU R2E-Gym C-MLFQ training requires 6 nodes; got ${#NODES[@]}" >&2
    exit 2
fi

TRAIN_NODE="${NODES[0]}"
ROLLOUT_NODES=("${NODES[@]:1:5}")
MASTER_ADDR="$TRAIN_NODE"
RUNTIME_CONFIG="$PROJECT_DIR/configs/.r2e_gym_cmlfq_24gpu_${JOB_ID}.yaml"
PLAN_ENV="$JOB_LOG_DIR/planned_rollout.env"

echo "============================================================"
echo "R2E-Gym full RL post-training with C-MLFQ, 24 GPUs"
echo "============================================================"
echo "Job: $JOB_ID"
echo "Model: $MODEL_PATH"
echo "Train node: $TRAIN_NODE (4 GPUs)"
echo "Rollout nodes: ${ROLLOUT_NODES[*]} (GlobalResourcePlanner decides TP layout)"
echo "Total GPUs: 24 (train fixed to 4, rollout planned by GRP)"
echo "Total steps: $TOTAL_STEPS"
echo "Max model len: $MAX_MODEL_LEN"
echo "Max new tokens: $MAX_NEW_TOKENS"
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
    echo "[cleanup] complete, exit=$exit_code"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "[1/6] Running GlobalResourcePlanner preflight for 24-GPU runtime config"
export PROJECT_DIR MODEL_PATH TOTAL_STEPS MAX_MODEL_LEN MAX_NEW_TOKENS EVAL_INTERVAL
export TRAIN_BATCH_SIZE TRAIN_MICRO_BATCH_SIZE VLLM_BASE_PORT GPU_MEMORY_UTILIZATION JOB_ID
export RUNTIME_CONFIG PLAN_ENV JOB_LOG_DIR
export R2E_MAX_TURNS R2E_MAX_PROMPT_TOKENS
export GRP_PROFILE_DATASET_JSONL GRP_PROFILE_SAMPLE_SIZE GRP_PROFILE_STRATEGY GRP_PROFILE_SEED GRP_PROFILE_SAMPLES_PER_PROMPT
ROLLOUT_NODES_CSV="$(IFS=,; echo "${ROLLOUT_NODES[*]}")"
export ROLLOUT_NODES_CSV
python3 - <<'PY'
import os
import shlex
from pathlib import Path
import yaml

from transformers import AutoTokenizer

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.preflight_planner import PreflightPlanner
from RL_Framework.infra.cost_model.startup_profile import (
    load_jsonl_rows,
    load_profile_summary,
    profile_jsonl_has_records,
    profile_metadata_matches,
    startup_profile_metadata,
    summarize_length_profile,
    write_profile_jsonl,
    write_profile_summary,
)
from RL_Framework.workflow.r2e_gym_profile import build_r2e_gym_length_profile

project = Path(os.environ["PROJECT_DIR"])
base = project / "configs" / "r2e_gym_cmlfq_qwen3_4b.yaml"
seed_path = project / "configs" / f".r2e_gym_cmlfq_24gpu_seed_{os.environ['JOB_ID']}.yaml"
runtime_path = Path(os.environ["RUNTIME_CONFIG"])
plan_env = Path(os.environ["PLAN_ENV"])
job_log_dir = Path(os.environ["JOB_LOG_DIR"])
rollout_nodes = [n for n in os.environ["ROLLOUT_NODES_CSV"].split(",") if n]
base_port = int(os.environ["VLLM_BASE_PORT"])
max_model_len = int(os.environ["MAX_MODEL_LEN"])
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
r2e_max_turns = int(os.environ["R2E_MAX_TURNS"])
r2e_max_prompt_tokens = int(os.environ.get("R2E_MAX_PROMPT_TOKENS", "0") or "0")
profile_dataset = Path(os.environ["GRP_PROFILE_DATASET_JSONL"])
profile_cache_dir = project / "logs" / "r2e_gym_cmlfq_24gpu" / "startup_profile_cache"
profile_jsonl = profile_cache_dir / "grp_startup_profile.jsonl"
profile_summary_json = profile_cache_dir / "grp_startup_profile_summary.json"

seed = {
    "base_config": str(base),
    "model_path": os.environ["MODEL_PATH"],
    "tokenizer_path": os.environ["MODEL_PATH"],
    "train_gpus": 4,
    "rollout_gpus": 20,
    "n_total_gpus": 24,
    "train_tp_size": 1,
    "train_pp_size": 1,
    "train_dp_size": 4,
    "batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["TRAIN_MICRO_BATCH_SIZE"]),
    "total_steps": int(os.environ["TOTAL_STEPS"]),
    "eval_interval": int(os.environ["EVAL_INTERVAL"]),
    "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    "max_seq_length": int(os.environ["MAX_MODEL_LEN"]),
    "r2e_max_turns": r2e_max_turns,
    "r2e_max_prompt_tokens": r2e_max_prompt_tokens,
    "max_concurrent_rollouts": 32,
    "sync_path": str(project / "logs" / "r2e_gym_cmlfq_24gpu" / f"weights_{os.environ['JOB_ID']}"),
    "log_dir": str(project / "logs" / "r2e_gym_cmlfq_24gpu"),
    "wandb_run_name": f"r2e-gym-cmlfq-24gpu-{os.environ['JOB_ID']}",
    "enable_history_collection": True,
    "history_output_dir": str(job_log_dir / "history"),
    "history_save_raw_lengths": True,
    "history_flush_interval": 1,
    "history_experiment_name": f"r2e-gym-cmlfq-24gpu-{os.environ['JOB_ID']}",
    "heterogeneous_rollout": {
        "enabled": True,
        "total_gpus": 20,
        "available_gpus": list(range(4, 24)),
        "vllm_base_port": int(os.environ["VLLM_BASE_PORT"]),
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
        "instances": [],
        "scheduling": {
            "scheduler_type": "cmlfq",
            "load_balance_strategy": "least_connections",
            "max_queue_length": 64,
            "enable_fallback": True,
            "cmlfq_tree_path": str(project / "logs" / "r2e_gym_cmlfq_24gpu" / f"cmlfq_tree_{os.environ['JOB_ID']}.json"),
            "cmlfq_buckets": {
                "short": {"tp_degrees": [2, 4], "max_tokens": 3000},
                "medium": {"tp_degrees": [2, 4], "max_tokens": 6000},
                "long": {"tp_degrees": [4, 2], "max_tokens": max_model_len},
            },
        },
    },
    "global_resource_planner": {
        "enabled": True,
        "train_backend": "analytic",
        "rollout_backend": "analytic",
        "plan_interval": 20,
        "warmup_steps": 5,
        "min_history_size": 16,
        "min_gain_ratio": 0.02,
        "reconfiguration_cost_s": 15.0,
        "allowed_rollout_tp": [2, 4],
        "require_heterogeneous_rollout_tp": True,
        "allowed_train_tp": [1],
        "allowed_train_pp": [1],
        "fixed_train_gpus": 4,
        "micro_batch_sizes": [1],
        "startup_profile_enabled": True,
        "startup_profile_sample_size": int(os.environ["GRP_PROFILE_SAMPLE_SIZE"]),
        "startup_profile_strategy": os.environ["GRP_PROFILE_STRATEGY"],
        "startup_profile_seed": int(os.environ["GRP_PROFILE_SEED"]),
        "startup_profile_samples_per_prompt": int(os.environ["GRP_PROFILE_SAMPLES_PER_PROMPT"]),
        "startup_profile_dataset_jsonl": str(profile_dataset),
        "startup_profile_history_jsonl": str(profile_jsonl),
        "startup_profile_summary_json": str(profile_summary_json),
        "runtime_length_profile_enabled": True,
        "runtime_length_profile_jsonl": str(job_log_dir / "runtime_length_profile.jsonl"),
        "apply_to_runtime": True,
        "verbose": True,
        "runtime_async_planning": True,
        "runtime_max_pending_plans": 1,
        "simulator_allow_fallback": True,
        "simulator_timeout_s": 240.0,
    },
}
seed_path.write_text(yaml.safe_dump(seed, sort_keys=False), encoding="utf-8")

cfg = AsyncRLConfig.from_yaml(str(seed_path))
profile_metadata = startup_profile_metadata(
    dataset_jsonl=profile_dataset,
    model_path=cfg.model_path,
    tokenizer_path=cfg.tokenizer_path if cfg.tokenizer_path else cfg.model_path,
    sample_size=int(os.environ["GRP_PROFILE_SAMPLE_SIZE"]),
    strategy=os.environ["GRP_PROFILE_STRATEGY"],
    seed=int(os.environ["GRP_PROFILE_SEED"]),
    samples_per_prompt=int(os.environ["GRP_PROFILE_SAMPLES_PER_PROMPT"]),
    max_new_tokens=max_new_tokens,
    max_seq_length=max_model_len,
    r2e_max_turns=r2e_max_turns,
    r2e_max_prompt_tokens=r2e_max_prompt_tokens,
)
history = None
summary = None
if profile_jsonl_has_records(profile_jsonl):
    existing_summary = load_profile_summary(profile_summary_json)
    if (
        existing_summary is None
        or profile_metadata_matches(existing_summary, profile_metadata)
    ):
        history = load_jsonl_rows(profile_jsonl)
        if not history:
            raise RuntimeError(f"startup profile JSONL is empty: {profile_jsonl}")
        summary = summarize_length_profile(history)
        summary.update(profile_metadata)
        summary["reused"] = True
        write_profile_summary(summary, profile_summary_json)
if history is None:
    rows = load_jsonl_rows(profile_dataset)
    if not rows:
        raise RuntimeError(f"GRP startup profile dataset is empty: {profile_dataset}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_path if cfg.tokenizer_path else cfg.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    history = build_r2e_gym_length_profile(
        rows,
        tokenizer=tokenizer,
        sample_size=int(os.environ["GRP_PROFILE_SAMPLE_SIZE"]),
        strategy=os.environ["GRP_PROFILE_STRATEGY"],
        seed=int(os.environ["GRP_PROFILE_SEED"]),
        samples_per_prompt=int(os.environ["GRP_PROFILE_SAMPLES_PER_PROMPT"]),
        max_turns=r2e_max_turns,
        max_new_tokens=max_new_tokens,
        max_seq_length=max_model_len,
        max_prompt_tokens=r2e_max_prompt_tokens or None,
    )
    summary = summarize_length_profile(history)
    summary.update(profile_metadata)
    write_profile_jsonl(history, profile_jsonl)
    write_profile_summary(summary, profile_summary_json)
result = PreflightPlanner(cfg, apply_best_candidate=True).run(history)
planned = result.planned_config

if not planned.heterogeneous_rollout.instances:
    raise RuntimeError("GlobalResourcePlanner produced no rollout instances")
tp_list = [int(inst.tp) for inst in planned.heterogeneous_rollout.instances]
if len(set(tp_list)) <= 1:
    raise RuntimeError(f"GlobalResourcePlanner did not produce heterogeneous TP degrees: {tp_list}")
if sum(tp_list) != 20:
    raise RuntimeError(f"GlobalResourcePlanner rollout GPUs must sum to 20, got {sum(tp_list)} from {tp_list}")
if planned.train_gpus != 4:
    raise RuntimeError(f"GlobalResourcePlanner must keep train_gpus=4, got {planned.train_gpus}")

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
planned.max_concurrent_rollouts = max(planned.max_concurrent_rollouts, len(assigned_tps) * planned.batch_size)
planned.heterogeneous_rollout.vllm_base_port = base_port
planned.heterogeneous_rollout.total_gpus = planned.rollout_gpus

grp = planned.global_resource_planner
grp.train_backend = "sailor"
grp.rollout_backend = "vidur"
grp.runtime_async_planning = True
grp.runtime_max_pending_plans = 1
grp.simulator_allow_fallback = True
grp.simulator_timeout_s = 240.0
grp.allowed_rollout_tp = [2, 4]
grp.require_heterogeneous_rollout_tp = True
grp.allowed_train_tp = [1]
grp.allowed_train_pp = [1]
grp.fixed_train_gpus = 4
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
print(f"profile_jsonl={profile_jsonl}")
print(f"profile_summary={summary}")
print(f"decision={result.decision.to_dict()}")
print(f"planned_rollout_tp={assigned_tps}")
print(f"planned_rollout_nodes={assigned_nodes}")
PY

source "$PLAN_ENV"
echo "GlobalResourcePlanner rollout TP list: ${PLANNED_ROLLOUT_TP_LIST}"
echo "GlobalResourcePlanner rollout nodes: ${ROLLOUT_NODE_LIST[*]}"

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
            python3 -m vllm.entrypoints.openai.api_server \
                --model '$MODEL_PATH' \
                --host 0.0.0.0 \
                --port '$port' \
                --tensor-parallel-size '$tp' \
                --max-model-len '$MAX_MODEL_LEN' \
                --dtype bfloat16 \
                --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
                --trust-remote-code \
                --enforce-eager \
                --disable-custom-all-reduce
        " >"$log_file" 2>&1 &
    VLLM_STEP_PIDS+=("$!")
    VLLM_PID_BY_NAME["$name"]="$!"
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

echo "[2/6] Starting GlobalResourcePlanner rollout vLLM instances"
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

echo "[3/6] Waiting for primary rollout services"
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

echo "[4/6] Checking R2E-Gym data and C-MLFQ flow"
python3 examples/test_r2e_gym.py
python3 examples/test_r2e_cmlfq_flow.py

echo "[5/6] OpenAI-compatible API check"
curl -fsS "http://${ROLLOUT_NODE_LIST[0]}:${ROLLOUT_PORT_LIST[0]}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"Write one short sentence.\",\"max_tokens\":8,\"temperature\":0}" \
    | python3 -m json.tool

export HETERO_INSTANCE_HOSTS
export R2E_GYM_INDEX="${PROJECT_DIR}/data/r2e_gym_v1/index.jsonl"
export R2E_MAX_TURNS
export R2E_EVAL_MAX_SAMPLES="$EVAL_MAX_SAMPLES"
export R2E_EVAL_CONCURRENCY="$EVAL_CONCURRENCY"
export R2E_EVAL_MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS"

echo "[6/6] Starting full R2E-Gym AsyncRLTrainer run on 24 GPUs"
srun --exclusive \
    --nodes=1 --ntasks=1 --nodelist="$TRAIN_NODE" \
    --gpus=4 \
    --cpus-per-task=16 \
    bash -lc "
        source '$VENV_PATH'
        cd '$PROJECT_DIR'
        export PYTHONPATH='$(dirname "$PROJECT_DIR")':\${PYTHONPATH:-}
        export HETERO_INSTANCE_HOSTS='$HETERO_INSTANCE_HOSTS'
        export R2E_GYM_INDEX='$R2E_GYM_INDEX'
        export R2E_MAX_TURNS='$R2E_MAX_TURNS'
        export R2E_EVAL_MAX_SAMPLES='$R2E_EVAL_MAX_SAMPLES'
        export R2E_EVAL_CONCURRENCY='$R2E_EVAL_CONCURRENCY'
        export R2E_EVAL_MAX_NEW_TOKENS='$R2E_EVAL_MAX_NEW_TOKENS'
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export TOKENIZERS_PARALLELISM=false
        export WANDB_MODE='$WANDB_MODE'
        export WANDB_DIR='$WANDB_DIR'
        torchrun \
            --nproc_per_node=4 \
            --master_addr='$MASTER_ADDR' \
            --master_port='$MASTER_PORT' \
            examples/r2e_gym_async_rl.py \
            --config '$RUNTIME_CONFIG'
    " 2>&1 | tee "$JOB_LOG_DIR/training.log"

echo "============================================================"
echo "R2E-Gym C-MLFQ 24-GPU training completed"
echo "Logs: $JOB_LOG_DIR"
echo "C-MLFQ tree: $PROJECT_DIR/logs/r2e_gym_cmlfq_24gpu/cmlfq_tree_${JOB_ID}.json"
echo "History: $PROJECT_DIR/logs/r2e_gym_cmlfq/history and logs/r2e_gym_cmlfq_24gpu"
echo "============================================================"

#!/usr/bin/env bash
# Foreground launcher for the validated R2E-Gym topology. The filename is kept
# for compatibility; AVAILABLE_HOSTS may contain the full 20-node pool and GRP
# chooses the initial whole-node train/rollout split.
set -euo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD on the jump host}"

project_dir="${PROJECT_DIR:-/opt/libra/RL_Framework_NPU}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-/opt/libra/runtime_sources/RL_Framework_NPU}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-/opt/libra/runtime_sources}"
config_path="${CONFIG_PATH:-configs/r2e_gym_qwen3_14b_mcore_npu_6node48_100step.yaml}"
run_root="${RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48}"
run_name="${RUN_NAME:-formal_200step_$(date +%Y%m%d_%H%M%S)}"
run_dir="${run_root}/${run_name}"
master_addr="${MASTER_ADDR:-}"
master_port="${MASTER_PORT:-29742}"
gradient_server_port="${GRADIENT_SERVER_PORT:-29852}"
rollout_project_dir="${ROLLOUT_PROJECT_DIR:-$runtime_project_dir}"
reuse_rollout="${REUSE_ROLLOUT:-0}"
preflight_only="${PREFLIGHT_ONLY:-0}"
initial_placement_mode="${INITIAL_PLACEMENT_MODE:-grp}"
resume_model_path="${RESUME_MODEL_PATH:-}"
train_start_step="${TRAIN_START_STEP:-0}"
initial_model_version="${INITIAL_MODEL_VERSION:-$train_start_step}"
mkdir -p "$run_dir" "${run_root}/rollout_weight_sync"
printf '%s\n' "$run_dir" >"${run_root}/current_training_run"

# Hosts are candidates, not preassigned roles. GRP selects the initial split
# before torchrun or any rollout service is launched.
default_hosts="192.0.2.10 192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.20 192.0.2.21"
available_hosts_value="${AVAILABLE_HOSTS:-$default_hosts}"
read -r -a available_hosts <<< "${available_hosts_value//,/ }"
[[ "${#available_hosts[@]}" -ge 2 ]] || {
    echo "at least two candidate hosts are required" >&2
    exit 2
}

if [[ -n "${CONFIG_PYTHON:-}" ]]; then
    config_python="$CONFIG_PYTHON"
elif [[ -x /opt/libra/envs/rl_framework_py310/bin/python ]]; then
    config_python="/opt/libra/envs/rl_framework_py310/bin/python"
else
    config_python="/data/qianzhirong/envs/rl_framework_py310/bin/python"
fi
if [[ -n "${TRAIN_PYTHON:-}" ]]; then
    train_python="$TRAIN_PYTHON"
elif [[ -x /opt/libra/envs/rl_mindspeed/bin/python ]]; then
    train_python="/opt/libra/envs/rl_mindspeed/bin/python"
else
    train_python="/data/qianzhirong/envs/rl_mindspeed_260/bin/python"
fi
cluster_socket_iface="${CLUSTER_SOCKET_IFNAME:-eth0}"
planned_config="${run_dir}/grp_initial_config.yaml"
placement_json="${run_dir}/grp_initial_placement.json"
if [[ "$initial_placement_mode" == "configured" ]]; then
    : "${TRAIN_HOSTS:?set TRAIN_HOSTS when INITIAL_PLACEMENT_MODE=configured}"
    : "${ROLLOUT_HOSTS:?set ROLLOUT_HOSTS when INITIAL_PLACEMENT_MODE=configured}"
    read -r -a train_hosts <<< "${TRAIN_HOSTS//,/ }"
    read -r -a rollout_hosts <<< "${ROLLOUT_HOSTS//,/ }"
    [[ "${#train_hosts[@]}" -eq 4 && "${#rollout_hosts[@]}" -eq 2 ]] || {
        echo "configured six-node baseline requires 4 training and 2 rollout hosts" >&2
        exit 2
    }
    declare -A placement_seen=()
    for host in "${train_hosts[@]}" "${rollout_hosts[@]}"; do
        [[ -z "${placement_seen[$host]:-}" ]] || {
            echo "duplicate configured host: $host" >&2
            exit 2
        }
        placement_seen[$host]=1
    done
    placement_template="$runtime_project_dir/$config_path"
    "$config_python" -c \
        'import json,sys; json.dump({"strategy":"configured","gpus_per_host":8,"train_gpus":32,"rollout_gpus":16,"train_hosts":sys.argv[2].split(","),"rollout_hosts":sys.argv[3].split(",")},open(sys.argv[1],"w"),indent=2)' \
        "$placement_json" "$(IFS=,; echo "${train_hosts[*]}")" "$(IFS=,; echo "${rollout_hosts[*]}")"
    echo "[ConfiguredPlacement] train_hosts=${train_hosts[*]} rollout_hosts=${rollout_hosts[*]}"
elif [[ "$initial_placement_mode" == "grp" ]]; then
    grp_history_args=()
    if [[ "${GRP_STARTUP_PROFILE_ENABLED:-1}" == "1" ]]; then
        profile_dataset_jsonl="${GRP_PROFILE_DATASET_JSONL:-$runtime_project_dir/data/r2e_gym_v1/index.jsonl}"
        profile_cache_dir="${GRP_STARTUP_PROFILE_CACHE_DIR:-${GRP_PROFILE_CACHE_DIR:-$run_root/startup_profile_cache}}"
        profile_jsonl="${GRP_STARTUP_PROFILE_JSONL:-${GRP_PROFILE_JSONL:-$profile_cache_dir/grp_startup_profile.jsonl}}"
        profile_summary_json="${GRP_STARTUP_PROFILE_SUMMARY_JSON:-${GRP_PROFILE_SUMMARY_JSON:-$profile_cache_dir/grp_startup_profile_summary.json}}"
        profile_args=(
            --config "$runtime_project_dir/$config_path"
            --dataset-jsonl "$profile_dataset_jsonl"
            --output-jsonl "$profile_jsonl"
            --summary-json "$profile_summary_json"
            --reuse-existing
        )
        [[ -z "${GRP_PROFILE_SAMPLE_SIZE:-}" ]] || profile_args+=(--sample-size "$GRP_PROFILE_SAMPLE_SIZE")
        [[ -z "${GRP_PROFILE_STRATEGY:-}" ]] || profile_args+=(--strategy "$GRP_PROFILE_STRATEGY")
        [[ -z "${GRP_PROFILE_SEED:-}" ]] || profile_args+=(--seed "$GRP_PROFILE_SEED")
        [[ -z "${GRP_PROFILE_SAMPLES_PER_PROMPT:-}" ]] || profile_args+=(--samples-per-prompt "$GRP_PROFILE_SAMPLES_PER_PROMPT")
        PYTHONPATH="$runtime_pythonpath" "$config_python" \
            "$project_dir/scripts/collect_r2e_grp_profile.py" \
            "${profile_args[@]}"
        grp_history_args=(--history-jsonl "$profile_jsonl")
    elif [[ "${GRP_ALLOW_SYNTHETIC_FALLBACK:-0}" == "1" ]]; then
        grp_history_args=(--allow-synthetic-fallback)
    else
        echo "GRP startup profile is disabled and synthetic fallback is not allowed" >&2
        exit 2
    fi
    placement_args=()
    for host in "${available_hosts[@]}"; do placement_args+=(--host "$host"); done
    PYTHONPATH="$runtime_pythonpath" "$config_python" \
        "$project_dir/scripts/plan_initial_grp_placement.py" \
        --config "$runtime_project_dir/$config_path" \
        --output-config "$planned_config" \
        --placement-json "$placement_json" \
        --gpus-per-host 8 \
        "${grp_history_args[@]}" \
        "${placement_args[@]}"
    readarray -t train_hosts < <(
        "$config_python" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["train_hosts"], sep="\n")' "$placement_json"
    )
    readarray -t rollout_hosts < <(
        "$config_python" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["rollout_hosts"], sep="\n")' "$placement_json"
    )
    placement_template="$planned_config"
else
    echo "INITIAL_PLACEMENT_MODE must be grp or configured" >&2
    exit 2
fi
[[ "${#train_hosts[@]}" -gt 0 && "${#rollout_hosts[@]}" -gt 0 ]] || {
    echo "GRP must allocate at least one complete node to each stage" >&2
    exit 2
}
master_addr="${master_addr:-${train_hosts[0]}}"

effective_config="${run_dir}/effective_config.yaml"
materialize_args=(
    --template "$placement_template" \
    --output "$effective_config" \
    --master-addr "$master_addr" \
    --master-port "$master_port" \
    --run-root "$run_root" \
    --gradient-port "$gradient_server_port"
)
for host in "${rollout_hosts[@]}"; do
    materialize_args+=(--rollout-host "$host")
done
if [[ -n "$resume_model_path" ]]; then
    materialize_args+=(--model-path "$resume_model_path")
fi
"$config_python" \
    "$project_dir/scripts/materialize_6node48_config.py" \
    "${materialize_args[@]}"

rollout_tp_pattern="$($config_python -c \
    'import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])); print(",".join(map(str,cfg["global_resource_planner"]["rollout_node_tp_pattern"])))' \
    "$effective_config")"
IFS=',' read -r -a rollout_tp_degrees <<< "$rollout_tp_pattern"
rollout_ports=()
for index in "${!rollout_tp_degrees[@]}"; do
    rollout_ports+=("$((8000 + index))")
done

coord_dir="${run_root}/logs/runtime_reconfiguration"
mkdir -p "$coord_dir/ready"
rm -f "$coord_dir/request.json" "$coord_dir/applied.json" \
    "$coord_dir/aborted.json" "$coord_dir/planning_pending.json"
find "$coord_dir/ready" -maxdepth 1 -type f -name 'rank_*.json' -delete

cleanup_rollout() {
    [[ "${KEEP_ROLLOUT:-0}" == "1" ]] && return
    local rollout_host rollout_node_id
    for rollout_host in "${rollout_hosts[@]}"; do
        rollout_node_id="n${rollout_host##*.}"
        NODE_PASSWORD="$NODE_PASSWORD" INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}" \
            "$project_dir/scripts/internal_ssh.sh" "$rollout_host" -- \
            "cd '$rollout_project_dir' && PROJECT_DIR='$rollout_project_dir' RUN_ROOT='$run_root' ROLLOUT_TP_PATTERN='$rollout_tp_pattern' bash scripts/start_r2e_rollout_pool_npu.sh stop '$rollout_node_id'" || true
    done
}

hybrid_controller_pid=""
pids=()

cleanup_train_drivers() {
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # Each driver is launched with setsid, so signal its complete
            # internal_ssh -> expect -> ssh process group.  Signalling only the
            # wrapper shell leaves the SSH PTY (and remote torchrun) alive.
            kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    pids=()
}

cleanup_hybrid_controller() {
    [[ -n "$hybrid_controller_pid" ]] || return
    if kill -0 "$hybrid_controller_pid" 2>/dev/null; then
        kill "$hybrid_controller_pid" 2>/dev/null || true
        wait "$hybrid_controller_pid" 2>/dev/null || true
    fi
    hybrid_controller_pid=""
}

cleanup_all() {
    # Prevent the EXIT trap from recursively re-entering cleanup when a signal
    # handler exits.  Training SSH drivers must be stopped before rollout
    # endpoints so in-flight clients are not left retrying dead services.
    trap - EXIT INT TERM
    cleanup_train_drivers
    cleanup_hybrid_controller
    cleanup_rollout
}

export INTERNAL_SSH_TIMEOUT=30
for host in "${train_hosts[@]}"; do
    if ! probe="$("$project_dir/scripts/internal_ssh.sh" "$host" -- \
        "if test -f '$runtime_project_dir/examples/r2e_gym_async_rl.py' && test -f '$effective_config'; then npu-smi info | grep -c -F 'No running processes found in NPU'; else echo missing_shared_runtime_or_config; fi; true")"; then
        echo "training node SSH preflight failed: $host probe=$probe" >&2
        exit 3
    fi
    if ! grep -qE '(^|[[:space:]])8([[:space:]]|$)' <<<"$probe"; then
        echo "training node preflight failed: $host probe=$probe" >&2
        exit 3
    fi
done

if [[ "$reuse_rollout" != "1" ]]; then
    for rollout_host in "${rollout_hosts[@]}"; do
        rollout_idle="$(NODE_PASSWORD="$NODE_PASSWORD" INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}" \
            "$project_dir/scripts/internal_ssh.sh" "$rollout_host" -- \
            "test -f '$rollout_project_dir/scripts/start_r2e_rollout_pool_npu.sh' && (test -x '/opt/libra/envs/vllm_ascend/bin/python' || test -x '/root/vllm_ascend_env/bin/python') && (test -f '/opt/libra/models/Qwen3-14B/config.json' || test -f '/data/qianzhirong/models/Qwen3-14B/config.json') && npu-smi info | grep -c -F 'No running processes found in NPU'; true" \
            | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
        [[ "$rollout_idle" -eq 8 ]] || { echo "rollout node preflight failed: $rollout_host idle=$rollout_idle" >&2; exit 3; }
    done
fi

if [[ "$preflight_only" == "1" ]]; then
    echo "preflight complete: train_hosts=${train_hosts[*]} rollout_hosts=${rollout_hosts[*]} effective_config=$effective_config"
    exit 0
fi

# Install cleanup only after the read-only preflight path has returned. This
# guarantees PREFLIGHT_ONLY never signals or stops processes on any node.
trap 'cleanup_all' EXIT
trap 'cleanup_all; exit 143' INT TERM

hybrid_task_dir="${run_root}/elastic_training_tasks"
mkdir -p "$hybrid_task_dir"
find "$hybrid_task_dir" -maxdepth 1 -type f \
    \( -name '.launch_*.json' -o -name '.launch_*.running' \
       -o -name '.launch_*.started' -o -name '.launch_*.error' \
       -o -name '.stop_*.json' -o -name '*.ready' \) -delete
NODE_PASSWORD="$NODE_PASSWORD" INTERNAL_SSH_TIMEOUT=30 \
    nohup "$config_python" \
    "$project_dir/scripts/elastic_hybrid_controller.py" \
    --task-dir "$hybrid_task_dir" \
    --project-dir "$project_dir" \
    >"${run_dir}/elastic_hybrid_controller.log" 2>&1 &
hybrid_controller_pid="$!"
printf '%s\n' "$hybrid_controller_pid" >"${run_dir}/elastic_hybrid_controller.pid"
sleep 1
kill -0 "$hybrid_controller_pid" 2>/dev/null || {
    echo "elastic hybrid controller failed to start" >&2
    exit 5
}

# The run root is new, so these exact handshake files cannot belong to another run.
find "${run_root}/rollout_weight_sync" -maxdepth 1 -type f \
    \( -name 'reload_request.json' -o -name 'ack_*.json' -o -name 'error_*.json' \) \
    -delete

if [[ "$reuse_rollout" != "1" ]]; then
    for rollout_host in "${rollout_hosts[@]}"; do
        rollout_node_id="n${rollout_host##*.}"
        NODE_PASSWORD="$NODE_PASSWORD" INTERNAL_SSH_TIMEOUT="${INTERNAL_SSH_TIMEOUT:-30}" \
            "$project_dir/scripts/internal_ssh.sh" "$rollout_host" -- \
            "cd '$rollout_project_dir' && PROJECT_DIR='$rollout_project_dir' RUN_ROOT='$run_root' ROLLOUT_TP_PATTERN='$rollout_tp_pattern' bash scripts/start_r2e_rollout_pool_npu.sh start '$rollout_node_id'"
    done
else
    echo "reusing existing rollout pool after endpoint health validation"
fi

for host in "${rollout_hosts[@]}"; do
    for port in "${rollout_ports[@]}"; do
        ready=0
        for _ in $(seq 1 360); do
            if curl -fsS --max-time 5 "http://${host}:${port}/health" >/dev/null; then
                ready=1
                break
            fi
            sleep 5
        done
        [[ "$ready" -eq 1 ]] || { echo "rollout endpoint failed: ${host}:${port}" >&2; exit 4; }
        echo "rollout ready: ${host}:${port}"
    done
done

export INTERNAL_SSH_TIMEOUT=-1
for node_rank in "${!train_hosts[@]}"; do
    target="${train_hosts[$node_rank]}"
    remote_cmd="source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
cd ${runtime_project_dir}; \
export PYTHONPATH=${runtime_pythonpath} PYTHONDONTWRITEBYTECODE=1 DEVICE_BACKEND=npu DIST_BACKEND=hccl \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GLOO_SOCKET_IFNAME=${cluster_socket_iface} HCCL_SOCKET_IFNAME=${cluster_socket_iface} \
HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 \
TORCH_DISTRIBUTED_TIMEOUT=3600 TOKENIZERS_PARALLELISM=false \
WANDB_MODE=disabled RL_TRAIN_PHASE_TRACE=1 OMP_NUM_THREADS=1 \
MCORE_MOE_GROUPED_GEMM=0 JOB_ID=${run_name} \
RL_TRAIN_START_STEP=${train_start_step} RL_INITIAL_MODEL_VERSION=${initial_model_version} \
ELASTIC_TRAINING_STATE_DIR=${run_root}/elastic_training_state \
R2E_GYM_INDEX=${runtime_project_dir}/data/r2e_gym_v1/index.jsonl; \
exec ${train_python} -m torch.distributed.run \
--nnodes=${#train_hosts[@]} --nproc_per_node=8 --node_rank=${node_rank} \
--master_addr=${master_addr} --master_port=${master_port} \
examples/r2e_gym_async_rl.py --config ${effective_config}"
    setsid "$project_dir/scripts/internal_ssh.sh" "$target" -- "$remote_cmd" \
        >"${run_dir}/driver_node_${node_rank}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
echo "RUN_DIR=$run_dir STATUS=$status"
for log in "$run_dir"/driver_node_*.log; do
    echo "FILE=$log"
    grep -E 'Step [0-9]+|GlobalResourcePlanner|RuntimeElasticExecutor|Traceback|Error|FAILED|Training complete' "$log" \
        | tail -n 120 || true
done
exit "$status"

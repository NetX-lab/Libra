#!/usr/bin/env bash
# Fail-closed, sequential EHP A/B supervisor for the six-node NPU cluster.
# NODE_PASSWORD is inherited from the one supervisor process and is never
# persisted by this script.
set -Eeuo pipefail

: "${NODE_PASSWORD:?set NODE_PASSWORD only in this process environment}"
: "${AVAILABLE_HOSTS:?set AVAILABLE_HOSTS to the authorized comma-separated host list}"

project_dir="${PROJECT_DIR:-/opt/libra/runtime_sources/RL_Framework_NPU}"
runtime_project_dir="${RUNTIME_PROJECT_DIR:-$project_dir}"
runtime_pythonpath="${RUNTIME_PYTHONPATH:-/opt/libra/runtime_sources/grp_replica_pythonpath}"
internal_ssh="$project_dir/scripts/internal_ssh.sh"
config_python="${CONFIG_PYTHON:-/opt/libra/envs/rl_framework_py310/bin/python}"

# Ordering is intentional: hosts with rollout runtimes are at the end so any
# split chosen by GRP can map rollout work onto capable nodes. The split itself
# is not fixed in either arm and is checked from each placement artifact.
available_hosts="$AVAILABLE_HOSTS"
read -r -a hosts <<< "${available_hosts//,/ }"

ehp_root="${EHP_RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48_grp_replica_ab_ehp}"
no_ehp_root="${NO_EHP_RUN_ROOT:-/opt/libra/runs/r2e_gym_qwen3_14b_6node48_grp_replica_ab_no_ehp}"
ehp_name="${EHP_RUN_NAME:-formal_grp_replica_ehp_20260809_ab1_200step}"
no_ehp_name="${NO_EHP_RUN_NAME:-formal_grp_replica_no_ehp_20260809_ab1_200step}"
report_dir="${REPORT_DIR:-/opt/libra/runs/grp_replica_ehp_ab_20260809}"
report_path="$report_dir/FINAL_VALIDATION_REPORT.md"

mkdir -p "$report_dir"
exec 9>"$report_dir/supervisor.lock"
flock -n 9 || {
    echo "$(date -Is) another A/B supervisor owns $report_dir/supervisor.lock"
    exit 2
}

log() {
    printf '%s %s\n' "$(date -Is)" "$*"
}

idle_npus() {
    local host="$1" output count
    output="$(
        NODE_PASSWORD="$NODE_PASSWORD" INTERNAL_SSH_TIMEOUT=25 \
            "$internal_ssh" "$host" -- \
            "npu-smi info | grep -c -F 'No running processes found in NPU'; true" \
            2>&1
    )" || true
    count="$(printf '%s\n' "$output" | tr -d '\r' | grep -E '^[0-8]$' | tail -n 1)"
    printf '%s' "${count:-unreachable}"
}

all_hosts_idle() {
    local host count ok=0
    for host in "${hosts[@]}"; do
        count="$(idle_npus "$host")"
        log "host=$host idle_npus=$count"
        [[ "$count" == "8" ]] || ok=1
    done
    return "$ok"
}

wait_for_all_hosts_idle() {
    local phase="$1"
    while ! all_hosts_idle; do
        log "phase=$phase waiting_for_same_six_hosts_to_be_fully_idle"
        sleep 120
    done
    log "phase=$phase all_six_hosts_idle"
}

run_launcher() {
    local mode="$1" config="$2" root="$3" name="$4" master_port="$5" gradient_port="$6" preflight="$7"
    PROJECT_DIR="$project_dir" \
    RUNTIME_PROJECT_DIR="$runtime_project_dir" \
    RUNTIME_PYTHONPATH="$runtime_pythonpath" \
    EHP_MODE="$mode" \
    CONFIG_PATH="$config" \
    RUN_ROOT="$root" \
    RUN_NAME="$name" \
    MASTER_PORT="$master_port" \
    GRADIENT_SERVER_PORT="$gradient_port" \
    AVAILABLE_HOSTS="$available_hosts" \
    PREFLIGHT_ONLY="$preflight" \
        bash "$project_dir/scripts/run_6node48_production_r2e_mcore_npu.sh"
}

placement_path() {
    local root="$1" name="$2"
    printf '%s/%s/grp_initial_placement.json' "$root" "$name"
}

assert_valid_placement() {
    local placement="$1"
    "$config_python" - "$placement" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
train = data.get("train_hosts", [])
rollout = data.get("rollout_hosts", [])
if not train or not rollout or len(train) + len(rollout) != 6:
    raise SystemExit(f"fail-closed: invalid six-node GRP placement: train={train} rollout={rollout}")
if set(train) & set(rollout):
    raise SystemExit(f"fail-closed: overlapping GRP placement: train={train} rollout={rollout}")
if data.get("train_gpus") != 8 * len(train) or data.get("rollout_gpus") != 8 * len(rollout):
    raise SystemExit(f"fail-closed: placement is not in whole eight-NPU nodes: {data}")
print(f"validated GRP placement: train={train} rollout={rollout}")
PY
}

assert_same_placement() {
    local left="$1" right="$2"
    "$config_python" - "$left" "$right" <<'PY'
import json
import sys

left = json.load(open(sys.argv[1], encoding="utf-8"))
right = json.load(open(sys.argv[2], encoding="utf-8"))
keys = ("train_gpus", "rollout_gpus", "train_hosts", "rollout_hosts")
diff = {key: (left.get(key), right.get(key)) for key in keys if left.get(key) != right.get(key)}
if diff:
    raise SystemExit(f"fail-closed: A/B initial placements differ: {diff}")
print("validated identical EHP/no-EHP initial placements")
PY
}

log "validating controlled EHP/no-EHP source configurations"
PYTHONPATH="$runtime_pythonpath" "$config_python" "$project_dir/scripts/validate_ehp_ab_config.py"

wait_for_all_hosts_idle "before_ehp"
log "running EHP preflight"
run_launcher ehp configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml \
    "$ehp_root" "$ehp_name" 30750 30751 1
ehp_placement="$(placement_path "$ehp_root" "$ehp_name")"
assert_valid_placement "$ehp_placement"

log "starting EHP arm"
run_launcher ehp configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_ehp.yaml \
    "$ehp_root" "$ehp_name" 30750 30751 0
log "EHP arm completed successfully"

wait_for_all_hosts_idle "between_arms"
log "running no-EHP preflight"
run_launcher no_ehp configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_no_ehp.yaml \
    "$no_ehp_root" "$no_ehp_name" 30850 30851 1
no_ehp_placement="$(placement_path "$no_ehp_root" "$no_ehp_name")"
assert_valid_placement "$no_ehp_placement"
assert_same_placement "$ehp_placement" "$no_ehp_placement"

log "starting no-EHP arm"
run_launcher no_ehp configs/r2e_gym_qwen3_14b_mcore_npu_6node48_production_no_ehp.yaml \
    "$no_ehp_root" "$no_ehp_name" 30850 30851 0
log "no-EHP arm completed successfully"

wait_for_all_hosts_idle "after_no_ehp"
log "running experiment analyzer"
"$config_python" "$project_dir/scripts/analyze_libra_experiment.py" \
    --run "no-EHP=$no_ehp_root/$no_ehp_name" \
    --run "EHP=$ehp_root/$ehp_name" \
    --output "$report_path"

{
    printf '\n## Controlled Experiment Configuration\n\n'
    printf -- '- Steps per arm: 200\n'
    printf -- '- Candidate hosts (identical order): `%s`\n' "$available_hosts"
    printf -- '- Initial placement: selected independently by GRP and required to match across arms.\n'
    printf -- '- Common scheduling: GRP enabled; C-MLFQ enabled.\n'
    printf -- '- Runtime: `%s`\n' "$runtime_project_dir"
    printf -- '- EHP-only difference validated by `validate_ehp_ab_config.py`.\n'
    printf -- '- EHP placement: `%s`\n' "$ehp_placement"
    printf -- '- no-EHP placement: `%s`\n' "$no_ehp_placement"
    printf -- '- Supervisor completed: `%s`\n' "$(date -Is)"
} >>"$report_path"

log "A/B analysis complete report=$report_path"

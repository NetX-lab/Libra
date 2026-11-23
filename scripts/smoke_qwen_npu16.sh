#!/usr/bin/env bash
# Run inside an Ascend training environment with torch_npu and vLLM Ascend.
# MODEL_PATH and DAPO_MATH_PATH must point to local files on the test host.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-4B checkpoint}"
: "${DAPO_MATH_PATH:?Set DAPO_MATH_PATH to a local DAPO JSONL or Parquet file}"
export MODEL_PATH DAPO_MATH_PATH
export ACCELERATOR_BACKEND=npu DIST_BACKEND=hccl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

# Reserve one 8-card training group and two 4-card rollout groups.
npu_ids="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export ASCEND_RT_VISIBLE_DEVICES="$npu_ids"
export BASE_PORT="${BASE_PORT:-28160}"
export RUN_DIR
RUN_DIR=$(mktemp -d "${LIBRA_SMOKE_ROOT:-/tmp}/libra-qwen-npu16.XXXXXXXX")
echo "Smoke test artifacts: $RUN_DIR"

python - <<'PY'
import os
import socket
from pathlib import Path

import torch
import torch_npu
import yaml

from RL_Framework.config import load_config

model = Path(os.environ["MODEL_PATH"]).resolve(strict=True)
dataset = Path(os.environ["DAPO_MATH_PATH"]).resolve(strict=True)
if not (model / "config.json").is_file() or not dataset.is_file():
    raise SystemExit("Expected a local HF checkpoint directory and a DAPO dataset file")
devices = [int(device) for device in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
if len(devices) != 16 or len(set(devices)) != 16 or min(devices) < 0 or torch.npu.device_count() != 16:
    raise SystemExit("Expected exactly 16 visible Ascend devices")
port = int(os.environ["BASE_PORT"])
for candidate in (port, port + 1):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", candidate))
path = Path(os.environ["RUN_DIR"])
config = dict(
    base_config=str(Path("configs/base.yaml").resolve()),
    model_arch_config=str(Path("configs/model_arch_config/qwen3-4b.yaml").resolve()),
    model_path=str(model), tokenizer_path=str(model), vllm_served_model_name="libra-qwen-smoke",
    train_backend="fsdp", train_gpus=8, train_dp_size=8, rollout_gpus=8, vllm_tp_size=4,
    vllm_endpoints=f"127.0.0.1:{port},127.0.0.1:{port + 1}",
    batch_size=16, n_samples=2, micro_batch_size=1, total_steps=2, sync_interval=1,
    max_new_tokens=128, max_seq_length=1024, max_concurrent_rollouts=4,
    dataset_name="dapo_math_17k", eval_interval=0, save_interval=0, wandb_project="",
    require_rollout_weight_sync=True, rollout_weight_sync_mode="restart",
    rollout_weight_sync_control_dir=str(path / "control"),
    rollout_weight_sync_timeout_s=600,
    sync_path=str(path / "weights"), log_dir=str(path / "logs"),
)
config_path = path / "config.yaml"
config_path.write_text(yaml.safe_dump(config))
load_config(str(config_path))
PY

IFS=, read -r -a devices <<< "$npu_ids"
device_list() {
    local IFS=,
    echo "$*"
}
pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for index in 0 1; do
    port=$((BASE_PORT + index))
    devices_for_rollout=$(device_list "${devices[@]:$((8 + 4 * index)):4}")
    ASCEND_RT_VISIBLE_DEVICES="$devices_for_rollout" \
        python -m RL_Framework.scripts.restartable_vllm_server \
        --instance-id "instance_$index" --control-dir "$RUN_DIR/control" \
        --health-url "http://127.0.0.1:$port/health" --initial-model "$MODEL_PATH" --ready-timeout 600 \
        -- python -m vllm.entrypoints.openai.api_server \
        --model __MODEL_PATH__ --served-model-name libra-qwen-smoke \
        --host 127.0.0.1 --port "$port" --tensor-parallel-size 4 \
        --dtype bfloat16 --max-model-len 1024 --max-num-seqs 16 --enforce-eager \
        > "$RUN_DIR/rollout-$index.log" 2>&1 &
    pids+=("$!")
done

deadline=$((SECONDS + 600))
for index in 0 1; do
    until curl -fsS --max-time 2 "http://127.0.0.1:$((BASE_PORT + index))/health" >/dev/null 2>&1; do
        if ! kill -0 "${pids[$index]}" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "Rollout instance $index did not become ready. See $RUN_DIR/rollout-$index.log" >&2
            exit 1
        fi
        sleep 2
    done
done

export DAPO_MAX_TURNS=1 DAPO_MAX_PROMPT_TOKENS=896 FSDP_PROGRESS_DIR="$RUN_DIR/fsdp-progress"
ASCEND_RT_VISIBLE_DEVICES="$(device_list "${devices[@]:0:8}")" \
    timeout --signal=TERM --kill-after=30s 30m \
    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        -m RL_Framework.examples.dapo_math_async_rl --config "$RUN_DIR/config.yaml" \
    2>&1 | tee "$RUN_DIR/train.log"

python - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
for step in range(2):
    timing = json.loads((run / "logs" / "step_timings" / f"step_{step}.json").read_text())
    if timing["step"] != step or timing["step_latency_s"] <= 0:
        raise SystemExit(f"Invalid optimizer step receipt for step {step}")
    snapshots = set()
    for instance in ("instance_0", "instance_1"):
        ack = json.loads((run / "control" / f"ack_{instance}_{step + 1}.json").read_text())
        if ack["instance_id"] != instance or ack["version"] != step + 1:
            raise SystemExit(f"Invalid rollout ACK for {instance} at version {step + 1}")
        snapshots.add(ack["snapshot_id"])
    if len(snapshots) != 1 or not all(snapshots):
        raise SystemExit(f"Rollout ACKs disagree at version {step + 1}")
print(f"PASS: two training steps and two weight publications on 16 NPUs. {run}")
PY

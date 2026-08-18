#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-npu}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-${ASCEND_SET_ENV:?set ASCEND_SET_ENV}}"
NNAL_ATB_SET_ENV="${NNAL_ATB_SET_ENV:-${NNAL_ATB_SET_ENV:?set NNAL_ATB_SET_ENV}}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
ASCEND_DEVICES="${ASCEND_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ROLLOUT_WEIGHT_SYNC_MODE="${ROLLOUT_WEIGHT_SYNC_MODE:-none}"

if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is required, or pass it as the first argument" >&2
    exit 2
fi

# A Docker image can expose its vLLM interpreter as ``VENV_DIR/bin/python``
# without providing a virtualenv activation script.  In that case use the
# interpreter's bin directory directly; this keeps PATH/python coherent while
# retaining the normal virtualenv path for host deployments.
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
elif [ -x "$VENV_DIR/bin/python" ]; then
    export PATH="$VENV_DIR/bin:$PATH"
else
    echo "VENV_DIR has neither bin/activate nor bin/python: $VENV_DIR" >&2
    exit 2
fi

if [ -f "$ASCEND_SET_ENV" ]; then
    had_nounset=0
    case "$-" in
        *u*)
            had_nounset=1
            set +u
            ;;
    esac
    # shellcheck disable=SC1090
    source "$ASCEND_SET_ENV"
    if [ "$had_nounset" -eq 1 ]; then
        set -u
    fi
fi

if [ -f "$NNAL_ATB_SET_ENV" ]; then
    had_nounset=0
    case "$-" in
        *u*)
            had_nounset=1
            set +u
            ;;
    esac
    # Source after activating the venv. The ATB script uses python3 to infer
    # the torch CXX11 ABI and must see the vLLM environment's torch package.
    # shellcheck disable=SC1090
    source "$NNAL_ATB_SET_ENV"
    if [ "$had_nounset" -eq 1 ]; then
        set -u
    fi
fi

export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICES"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

server_args=(
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enforce-eager
)

if [ "$ROLLOUT_WEIGHT_SYNC_MODE" = "hccl" ]; then
    # vLLM Ascend maps the upstream "nccl" backend name to its official HCCL
    # engine. Dev mode exposes the pause/update/resume control-plane endpoints.
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_ASCEND_ENABLE_NZ=0
    server_args+=(--weight-transfer-config '{"backend":"nccl"}')
fi

exec python -m vllm.entrypoints.openai.api_server "${server_args[@]}"

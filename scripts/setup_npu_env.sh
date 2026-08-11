#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-npu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
INSTALL_MEGATRON_NPU="${INSTALL_MEGATRON_NPU:-0}"
INSTALL_VLLM_ASCEND="${INSTALL_VLLM_ASCEND:-0}"

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
elif [ -f /usr/local/Ascend/cann-8.5.2/set_env.sh ]; then
    had_nounset=0
    case "$-" in
        *u*)
            had_nounset=1
            set +u
            ;;
    esac
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-8.5.2/set_env.sh
    if [ "$had_nounset" -eq 1 ]; then
        set -u
    fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_DIR/requirements-npu.txt"

if [ "$INSTALL_MEGATRON_NPU" = "1" ]; then
    python -m pip install -r "$PROJECT_DIR/requirements-megatron-npu.txt"
fi

if [ "$INSTALL_VLLM_ASCEND" = "1" ]; then
    python -m pip install --no-deps -r "$PROJECT_DIR/requirements-vllm-ascend-npu.txt"
fi

export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"

python - <<'PY'
import torch

import torch_npu

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available())
print("npu_device_count", torch.npu.device_count())
if torch.npu.is_available():
    x = torch.ones((2, 2), device="npu")
    print("npu_tensor_sum", float(x.sum().cpu()))
PY

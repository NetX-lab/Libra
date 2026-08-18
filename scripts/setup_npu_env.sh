#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:?set ASCEND_SET_ENV to the Ascend toolkit set_env.sh}"
INSTALL_MEGATRON_NPU="${INSTALL_MEGATRON_NPU:-0}"
INSTALL_VLLM_ASCEND="${INSTALL_VLLM_ASCEND:-0}"
NPU_STACK_PROFILE="${NPU_STACK_PROFILE:-legacy}"
if [ "$NPU_STACK_PROFILE" = "hccl" ]; then
    default_venv_dir="${PROJECT_DIR}/.venv-npu-hccl"
else
    default_venv_dir="${PROJECT_DIR}/.venv-npu"
fi
VENV_DIR="${VENV_DIR:-$default_venv_dir}"

if [ "$NPU_STACK_PROFILE" = "hccl" ]; then
    "$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13), "official HCCL profile requires Python 3.10-3.12"'
fi

used_legacy_cann_fallback=0
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

if [ "$NPU_STACK_PROFILE" = "hccl" ] && [ "$used_legacy_cann_fallback" -eq 1 ]; then
    echo "NPU_STACK_PROFILE=hccl requires a CANN 9.x toolkit; refusing the CANN 8.5 fallback" >&2
    exit 2
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
case "$NPU_STACK_PROFILE" in
    legacy)
        python -m pip install -r "$PROJECT_DIR/requirements-npu.txt"
        ;;
    hccl)
        # The public upstream vLLM wheel currently declares torch==2.11.0,
        # while the official vLLM-Ascend HCCL wheel is pinned to torch 2.10.0.
        # Install the platform stack first, then add upstream vLLM without
        # dependency re-resolution so pip cannot replace torch-npu.
        hccl_requirements="$(mktemp)"
        trap 'rm -f "$hccl_requirements"' EXIT
        awk '!/^vllm==/ && !/^vllm-ascend==/ && !/^megatron-core==/ && !/^megatron-bridge==/' "$PROJECT_DIR/requirements-npu-hccl.txt" > "$hccl_requirements"
        python -m pip install -r "$hccl_requirements"
        python -m pip install --no-deps megatron-core==0.14.0 megatron-bridge==0.2.0rc6
        python -m pip install --no-deps vllm-ascend==0.22.1rc1
        python -m pip install --no-deps vllm==0.22.1
        ;;
    *)
        echo "Unsupported NPU_STACK_PROFILE=$NPU_STACK_PROFILE (legacy|hccl)" >&2
        exit 2
        ;;
esac

if [ "$INSTALL_MEGATRON_NPU" = "1" ] && [ "$NPU_STACK_PROFILE" = "legacy" ]; then
    python -m pip install -r "$PROJECT_DIR/requirements-megatron-npu.txt"
fi

if [ "$INSTALL_VLLM_ASCEND" = "1" ] && [ "$NPU_STACK_PROFILE" = "legacy" ]; then
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

if [ "$NPU_STACK_PROFILE" = "hccl" ]; then
    python "$PROJECT_DIR/scripts/validate_hccl_weight_transfer_env.py"
fi

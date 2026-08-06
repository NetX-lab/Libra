#!/bin/bash

# Recreate the exact Python overlay used by the validated Ascend MCore
# preflight.  The base Python must already contain the matching torch-npu wheel;
# this deliberately prevents a package resolver from replacing the vendor wheel.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_PYTHON="${BASE_PYTHON:-/data/qianzhirong/envs/rl_framework_py310/bin/python}"
VENV_DIR="${VENV_DIR:-/data/qianzhirong/envs/rl_mindspeed_260}"
UV_BIN="${UV_BIN:-uv}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [ ! -x "$BASE_PYTHON" ]; then
    echo "Missing Ascend base Python: $BASE_PYTHON" >&2
    exit 1
fi
if [ ! -f "$CANN_ENV" ]; then
    echo "Missing CANN environment script: $CANN_ENV" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "$CANN_ENV"
set -u

"$BASE_PYTHON" - <<'PY'
import sys
import torch
import torch_npu

expected = {"torch": "2.7.1+cpu", "torch_npu": "2.7.1.post2"}
actual = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
if actual != expected:
    raise SystemExit(f"Ascend base mismatch: expected {expected}, got {actual}")
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required, got {sys.version}")
PY

if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$UV_BIN" venv \
        --python "$BASE_PYTHON" \
        --system-site-packages \
        "$VENV_DIR"
fi

"$UV_BIN" pip install \
    --python "$VENV_DIR/bin/python" \
    --no-deps \
    "setuptools==79.0.1" \
    "megatron-bridge==0.2.0rc6"

export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
export DEVICE_BACKEND=npu
export DIST_BACKEND=hccl
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data_nv2/qianzhirong/torch_extensions}"
export MCORE_MOE_GROUPED_GEMM=0
export TOKENIZERS_PARALLELISM=false

"$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import version
import torch
import torch_npu

expected = {
    "torch": "2.7.1",
    "torch-npu": "2.7.1.post2",
    "megatron-core": "0.14.0",
    "megatron-bridge": "0.2.0rc6",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "datasets": "2.21.0",
    "accelerate": "1.14.0",
    "setuptools": "79.0.1",
    "packaging": "26.2",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Validated environment mismatch: {actual}")
if not torch.npu.is_available():
    raise SystemExit("torch-npu imported but no NPU is available")
x = torch.ones((2, 2), device="npu")
print("ASCEND_MEGATRON_ENV_OK", actual, "tensor_sum", float(x.sum().cpu()))
PY

echo "Environment ready: $VENV_DIR"
echo "Run: source $CANN_ENV"
echo "Run: export PYTHONPATH=$(dirname "$PROJECT_DIR")"

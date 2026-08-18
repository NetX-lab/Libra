#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${HOME}/rl_framework_env/bin/activate}"
WHEELHOUSE="${MCORE_WHEELHOUSE:-$PROJECT_DIR/vendor/megatron}"
MCORE_ARCHIVE="${MEGATRON_CORE_ARCHIVE:-$WHEELHOUSE/megatron_core-0.14.0.tar.gz}"
BRIDGE_WHEEL="${MEGATRON_BRIDGE_WHEEL:-$WHEELHOUSE/megatron_bridge-0.2.0rc6-py3-none-any.whl}"
GROUPED_GEMM_ARCHIVE="${GROUPED_GEMM_ARCHIVE:-$WHEELHOUSE/grouped_gemm-1.1.4-with-cutlass.tar.gz}"
GROUPED_GEMM_WHEEL="${GROUPED_GEMM_WHEEL:-$WHEELHOUSE/grouped_gemm-1.1.4-cp312-cp312-linux_x86_64.whl}"

source "$VENV_PATH"
export PYTHONPATH="$(dirname "$PROJECT_DIR"):${PYTHONPATH:-}"
if [ -f "$MCORE_ARCHIVE" ] && [ -f "$BRIDGE_WHEEL" ]; then
    python -m pip install --no-index --no-build-isolation --no-deps \
        --find-links "$WHEELHOUSE" \
        "pybind11==2.13.6" \
        "$WHEELHOUSE/antlr4-python3-runtime-4.7.2.tar.gz"
    # Bridge only uses OmegaConf's object/config APIs on this path. Keeping
    # ANTLR 4.7.2 also preserves latex2sympy2 compatibility in math rewards.
    python -m pip install --no-index --no-deps \
        --find-links "$WHEELHOUSE" \
        "omegaconf==2.3.0" \
        "hydra-core==1.3.2" \
        "accelerate==1.10.1"
    python -m pip install --no-build-isolation --no-deps "$MCORE_ARCHIVE"
    python -m pip install --no-deps "$BRIDGE_WHEEL"
    if [ -f "$GROUPED_GEMM_WHEEL" ]; then
        python -m pip install --no-deps "$GROUPED_GEMM_WHEEL"
    elif [ -f "$GROUPED_GEMM_ARCHIVE" ]; then
        if [ -f /opt/rh/gcc-toolset-11/enable ]; then
            source /opt/rh/gcc-toolset-11/enable
        fi
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
        export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
            MAX_JOBS="${MAX_JOBS:-1}" \
            python -m pip install --no-build-isolation --no-deps \
                "$GROUPED_GEMM_ARCHIVE"
    fi
else
    python -m pip install -r "$PROJECT_DIR/requirements-megatron.txt"
    python -m pip install --no-deps "megatron-bridge==0.2.0rc6"
fi

python - <<'PY'
import importlib.metadata as metadata

expected = {
    "megatron-core": "0.14.0",
    "megatron-bridge": "0.2.0rc6",
    "grouped-gemm": "1.1.4",
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise RuntimeError(f"{package}: expected {version}, found {actual}")
    print(f"{package}={actual}")

from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core import dist_checkpointing
from RL_Framework.engine.megatron_core_attention import (
    TorchSDPADotProductAttention,
)

print("Qwen3MoEBridge", Qwen3MoEBridge.__name__)
print("dist_checkpointing", dist_checkpointing.__name__)
print("attention", TorchSDPADotProductAttention.__name__)
PY

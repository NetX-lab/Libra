#!/bin/bash
#SBATCH -J install_grouped_gemm
#SBATCH -o logs/install_grouped_gemm_%j.out
#SBATCH -e logs/install_grouped_gemm_%j.err
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -p gpu
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${HOME}/rl_framework_env/bin/activate}"
ARCHIVE="$PROJECT_DIR/vendor/megatron/grouped_gemm-1.1.4-with-cutlass.tar.gz"
WHEEL="$PROJECT_DIR/vendor/megatron/grouped_gemm-1.1.4-cp312-cp312-linux_x86_64.whl"

source "$VENV_PATH"
cd "$PROJECT_DIR"
mkdir -p logs

if [ -f "$WHEEL" ]; then
    python -m pip install --no-deps --force-reinstall "$WHEEL"
else
    python -m pip install --no-build-isolation --no-deps --force-reinstall "$ARCHIVE"
fi

python - <<'PY'
import importlib.metadata as metadata

import grouped_gemm
import torch
from megatron.core.transformer.moe.grouped_gemm_util import (
    grouped_gemm_is_available,
)

assert grouped_gemm_is_available()
assert metadata.version("grouped-gemm") == "1.1.4"
a = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(
    2,
    16,
    8,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
batch_sizes = torch.tensor([4, 4], dtype=torch.long)
actual = grouped_gemm.ops.gmm(a, b, batch_sizes)
expected = torch.cat((a[:4] @ b[0], a[4:] @ b[1]))
torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
actual.float().sum().backward()
assert a.grad is not None and b.grad is not None
print("GROUPED_GEMM_INSTALL_OK", grouped_gemm.__file__)
PY

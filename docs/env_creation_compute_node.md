# Compute-Node Environment Checklist

Complete the installation in [env_creation.md](env_creation.md), then validate
from an actual compute-node allocation rather than only on the login node.

## Single-node validation

```bash
srun --nodes=1 --ntasks=1 --gpus=1 bash -lc '
  set -euo pipefail
  source "$HOME/libra-env/bin/activate"

  echo "Python: $(command -v python)"
  python --version
  nvidia-smi

  python - <<"PY"
import importlib.metadata as metadata

import pyarrow
import torch
import transformers
import vllm
import xformers
from word2number import w2n
import RL_Framework

assert torch.cuda.is_available(), "CUDA is not visible"
x = torch.ones(1024, device="cuda")
print("Libra", RL_Framework.__version__)
print("Torch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(), "sum", x.sum().item())
print("Transformers", transformers.__version__)
print("vLLM", vllm.__version__)
print("xFormers", xformers.__version__)
print("PyArrow", pyarrow.__version__)
print("word2number", w2n.word_to_num("two"))
PY
'
```

## Megatron-Core validation

Run this when `train_backend: megatron_core`:

```bash
srun --nodes=1 --ntasks=1 --gpus=1 bash -lc '
  set -euo pipefail
  source "$HOME/libra-env/bin/activate"
  python - <<"PY"
import importlib.metadata as metadata
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core import dist_checkpointing
from RL_Framework.engine.megatron_core_attention import TorchSDPADotProductAttention

for package in ("megatron-core", "megatron-bridge", "grouped-gemm"):
    print(package, metadata.version(package))
print(Qwen3MoEBridge.__name__)
print(dist_checkpointing.__name__)
print(TorchSDPADotProductAttention.__name__)
PY
'
```

## Multi-node NCCL validation

Run one `torchrun` worker per GPU using the same Slurm allocation, modules,
network interface variables, and launch geometry as the real job. The provided
probe performs a BF16 all-gather and validates its contents:

```bash
torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$SLURM_NODEID" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  scripts/nccl_allgather_preflight.py
```

On clusters where `srun` starts one task per node, place this `torchrun` command
inside that `srun` step. Verify the measured bandwidth is plausible and that no
rank maps to the same CUDA device.

## Storage and service checklist

- The base Python interpreter and standard library are visible on every node.
- The virtual environment is shared or reproduced identically per node.
- The CUDA driver supports the runtime used by PyTorch and vLLM.
- Model, dataset, checkpoint, log, and weight-sync paths are visible and
  writable from every participating node.
- The repository has been installed with `python -m pip install -e .`.
- `PROJECT_DIR` points to the checkout; launchers prepend it to `PYTHONPATH`.
- Required model repositories are already cached or compute nodes can reach
  them.
- R2E container images are available to the configured container runtime.
- Search-R1 can reach SearXNG or has `SERPER_KEY_ID` configured.
- The selected NCCL socket/InfiniBand interfaces exist on every node.
- Log and checkpoint filesystems have enough capacity for the full run.

If a venv created on the login node reports `No module named 'encodings'`, its
base interpreter or standard library is unavailable on the compute node.
Recreate it from a site-provided or shared Python installation.

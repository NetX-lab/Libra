# No-Slurm Quick Start

This guide validates Libra on a workstation or manually managed Linux server
without Slurm. It is useful for development, artifact review, and first-contact
checks before moving to a managed cluster.

The CPU-friendly path below does not launch vLLM servers or distributed
training. A short optional GPU section is included for machines where you manage
CUDA devices directly.

## 1. Install

```bash
git clone https://github.com/NetX-lab/Libra.git
cd Libra

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If you plan to use the default Megatron-Core backend on this machine, install
the pinned Megatron dependencies as well:

```bash
bash scripts/install_megatron_core.sh
```

Verify the editable package:

```bash
python -c "import RL_Framework; print(RL_Framework.__version__)"
```

## 2. Import Sanity Check

```bash
python - <<'PY'
import importlib
import sys

modules = [
    "RL_Framework",
    "RL_Framework.config",
    "RL_Framework.infra.cost_model.preflight_planner",
    "RL_Framework.infra.scheduling.cmlfq_scheduler",
    "torch",
    "transformers",
    "pyarrow",
]

for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "ok")
    print(f"{name}: {version}")

print("python:", sys.executable)
PY
```

Expected result: every import succeeds. If the `RL_Framework` import fails,
rerun `python -m pip install -e .` from the repository root.

## 3. Planner Preflight

This check exercises strict config loading and the Global Resource Planner with
synthetic history. It uses the analytic in-repository cost model, so it does not
require Sailor, Vidur, Slurm, vLLM servers, or GPUs.

```bash
mkdir -p logs/quick_start

python - <<'PY'
from pathlib import Path

root = Path.cwd()
base = (root / "configs" / "r2e_gym_cmlfq_qwen3_4b.yaml").resolve()
out = root / "logs" / "quick_start" / "no_slurm_preflight.yaml"
out.write_text(
    f"""base_config: {base}
model_path: /tmp/libra-quickstart-model-placeholder
tokenizer_path: /tmp/libra-quickstart-model-placeholder
train_gpus: 3
rollout_gpus: 1
train_tp_size: 1
train_pp_size: 1
train_dp_size: 3
batch_size: 12
micro_batch_size: 1
max_seq_length: 4096
max_new_tokens: 1024
global_resource_planner:
  enabled: true
  train_backend: analytic
  rollout_backend: analytic
  allowed_train_tp: [1, 2]
  allowed_train_pp: [1]
  allowed_rollout_tp: [1, 2, 4]
  micro_batch_sizes: [1]
  min_history_size: 4
  min_gain_ratio: 0.0
  reconfiguration_cost_s: 0.0
  apply_to_runtime: true
""",
    encoding="utf-8",
)
print(out)
PY

python scripts/run_global_resource_preflight.py \
  --config logs/quick_start/no_slurm_preflight.yaml \
  --output-config logs/quick_start/no_slurm_planned.yaml \
  --decision-json logs/quick_start/no_slurm_decision.json \
  --synthetic-requests 8 \
  --synthetic-input-len 512 \
  --synthetic-output-len 1024
```

Expected result: the command prints a line beginning with
`[PreflightPlanner] applied` and writes:

```text
logs/quick_start/no_slurm_planned.yaml
logs/quick_start/no_slurm_decision.json
```

## 4. CPU-Friendly Tests

Run the tests that cover scheduler behavior, planner behavior, GRPO grouping,
and elastic runtime hooks without starting a cluster job:

```bash
pytest -q \
  tests/test_cmlfq_scheduler.py \
  tests/test_preflight_planner.py \
  tests/test_global_resource_planner_simulators.py \
  tests/test_grpo_grouping.py \
  tests/test_runtime_elastic_executor.py
```

These tests are the quickest way to verify that the package import path,
configuration system, scheduler, planner, and CPU-side elastic code are working.

## 5. Optional Single-Machine GPU Checks

If the machine has NVIDIA GPUs and the full runtime dependencies installed,
check CUDA and NCCL without Slurm:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.ones(1, device="cuda")
    print("cuda_tensor:", x.item())
PY
```

For a four-GPU local NCCL preflight:

```bash
export NCCL_PREFLIGHT_ELEMENTS=1048576
export NCCL_PREFLIGHT_ITERATIONS=1

torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/nccl_allgather_preflight.py
```

Expected result: every rank prints a line beginning with `[NCCL preflight]`.

## 6. Optional Local vLLM API Smoke

Use this only after `MODEL_PATH` points to a local Hugging Face model directory
and one GPU is available:

```bash
export MODEL_PATH=/local/models/Qwen3-4B
export VLLM_PORT=8000

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enforce-eager
```

In another terminal:

```bash
curl -fsS "http://127.0.0.1:${VLLM_PORT}/health"
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"Write one short sentence.\",\"max_tokens\":8,\"temperature\":0}"
```

This confirms that the local model and vLLM installation are usable before you
attempt a full RL training run.

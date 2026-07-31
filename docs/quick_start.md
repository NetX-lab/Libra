# Quick Start on a Slurm Cluster

This guide gives a short validation path for running Libra on a Linux NVIDIA
GPU cluster managed by Slurm. It starts with checks that do not allocate GPUs,
then moves to NCCL and a short pilot run.

Use the no-Slurm guide if you are testing on a workstation or a manually
managed server: [No-Slurm quick start](quick_start_no_slurm.md).

## 1. Set Common Paths

Log in to the cluster and set the paths for your installation:

```bash
export PROJECT_DIR=/shared/path/Libra
export VENV_PATH=/shared/path/libra-env/bin/activate
export MODEL_PATH=/shared/models/Qwen3-4B

cd "$PROJECT_DIR"
source "$VENV_PATH"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
```

If the virtual environment does not exist yet, create it with the installation
steps in [Environment setup](env_creation.md), then install Libra in editable
mode:

```bash
python -m pip install -e "$PROJECT_DIR"
bash "$PROJECT_DIR/scripts/install_megatron_core.sh"
```

## 2. Login-Node Sanity Check

These checks do not allocate GPUs and should finish quickly:

```bash
python - <<'PY'
import importlib
import sys

modules = [
    "RL_Framework",
    "RL_Framework.config",
    "RL_Framework.infra.cost_model.preflight_planner",
    "torch",
    "transformers",
    "vllm",
    "pyarrow",
]

for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "ok")
    print(f"{name}: {version}")

print("python:", sys.executable)
PY
```

Expected result: every import succeeds and `RL_Framework` resolves from the
checkout. If `RL_Framework` fails, rerun `python -m pip install -e "$PROJECT_DIR"`
inside the same virtual environment used by Slurm jobs.

## 3. Planner Preflight

This planner check uses synthetic request lengths and the in-repository
analytic cost model. It validates configuration loading, strict config fields,
planner search, and YAML output without requiring Sailor, Vidur, vLLM servers,
or GPUs.

```bash
mkdir -p "$PROJECT_DIR/logs/quick_start"

python - <<'PY'
from pathlib import Path

root = Path.cwd()
base = (root / "configs" / "r2e_gym_cmlfq_qwen3_4b.yaml").resolve()
out = root / "logs" / "quick_start" / "slurm_preflight.yaml"
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
  --config logs/quick_start/slurm_preflight.yaml \
  --output-config logs/quick_start/slurm_planned.yaml \
  --decision-json logs/quick_start/slurm_decision.json \
  --synthetic-requests 8 \
  --synthetic-input-len 512 \
  --synthetic-output-len 1024
```

Expected result: the command prints a line beginning with
`[PreflightPlanner] applied` and writes:

```text
logs/quick_start/slurm_planned.yaml
logs/quick_start/slurm_decision.json
```

## 4. Optional Single-Node NCCL Check

Run this inside a small one-node allocation before launching a multi-node job.
Adjust the partition, GPU count, and task syntax for your Slurm installation.

```bash
srun -p gpu -N 1 -G 4 --ntasks=4 --cpus-per-task=8 \
  --pty bash

cd "$PROJECT_DIR"
source "$VENV_PATH"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export NCCL_PREFLIGHT_ELEMENTS=1048576
export NCCL_PREFLIGHT_ITERATIONS=1

torchrun --nproc_per_node=4 scripts/nccl_allgather_preflight.py
exit
```

Expected result: every rank prints a line beginning with `[NCCL preflight]`.
Resolve NCCL failures before running Libra, because the training backend depends
on reliable collectives.

## 5. Data Checklist

The short R2E-Gym pilot expects the R2E-Gym index at the default location:

```bash
test -s "$PROJECT_DIR/data/r2e_gym_v1/index.jsonl"
test -s "$PROJECT_DIR/data/r2e_gym_v1/manifest.json"
python examples/test_r2e_gym.py
```

If these files are missing, prepare the dataset using
[Data preparation](data_preparation.md#r2e-gym-v1). R2E-Gym execution also
requires the container images referenced by the dataset to be available on the
compute nodes.

## 6. Short Libra Pilot

After the import, planner, data, model, and NCCL checks pass, submit a short
validation run:

```bash
cd "$PROJECT_DIR"
mkdir -p logs

PROJECT_DIR="$PROJECT_DIR" \
VENV_PATH="$VENV_PATH" \
MODEL_PATH="$MODEL_PATH" \
TOTAL_STEPS=2 \
MAX_MODEL_LEN=4096 \
MAX_NEW_TOKENS=256 \
R2E_MAX_TURNS=1 \
TRAIN_BATCH_SIZE=8 \
N_SAMPLES=2 \
MAX_CONCURRENT_ROLLOUTS=8 \
QUEUE_SIZE=32 \
EVAL_INTERVAL=999 \
SYNC_INTERVAL=999 \
WANDB_MODE=offline \
sbatch scripts/submit_libra_validation_pilot.slurm
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/libra_pilot_<jobid>.out
tail -f logs/libra_pilot_<jobid>.err
```

Replace `<jobid>` with the Slurm job ID returned by `sbatch`.

Successful startup should show:

- Python, Torch, Transformers, and vLLM versions;
- rollout vLLM health checks passing;
- R2E-Gym validation passing;
- `examples/r2e_gym_async_rl.py` launched through `torchrun`;
- planner and runtime logs under `logs/r2e_gym_cmlfq_24gpu/job_<jobid>/`.

## 7. Common Adjustments

Use these environment variables instead of editing scripts:

| Variable | Purpose |
| --- | --- |
| `PROJECT_DIR` | Remote Libra checkout |
| `VENV_PATH` | Virtual environment activation script |
| `MODEL_PATH` | Local Hugging Face model directory |
| `TOTAL_STEPS` | Number of training steps for the pilot |
| `MAX_MODEL_LEN` | vLLM and trainer context length |
| `MAX_NEW_TOKENS` | rollout generation cap |
| `R2E_MAX_TURNS` | maximum tool-interaction turns per task |
| `GRP_DECOUPLE_COMMUNICATION_DOMAINS` | keep elastic gradient exchange separate from core training collectives |
| `RUN_NCCL_PREFLIGHT` | enable launcher-level NCCL preflight when set to `1` |

Keep `GRP_DECOUPLE_COMMUNICATION_DOMAINS=1` when validating Libra's elastic
training-side changes. It exercises the decoupled communication domain used by
the current cluster-swap path.

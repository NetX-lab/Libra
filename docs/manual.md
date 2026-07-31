# Libra Cluster Manual

This manual describes the main switches used when running Libra on a cluster.
Most options can be set either in YAML files under `configs/` or as environment
variables consumed by the Slurm launchers. Commands below assume the repository
root is the current directory.

Libra defaults to the Megatron-Core training backend for cluster launches.
Recommended stability settings for large Qwen-style models are:

```bash
export MEGATRON_USE_PRECISION_AWARE_OPTIMIZER=1
export MEGATRON_OPTIMIZER_CPU_OFFLOAD=1
export MEGATRON_OPTIMIZER_OFFLOAD_FRACTION=1.0
```

For an initial memory-sensitive run, start with a shorter context:

```bash
export MAX_MODEL_LEN=4096
export MAX_NEW_TOKENS=128
export R2E_MAX_PROMPT_TOKENS=2048
export R2E_MAX_TURNS=2
```

## 1. Configure the Rollout Cluster

Libra can run a homogeneous rollout pool or a heterogeneous pool. For C-MLFQ,
heterogeneous rollout is the common setup.

```bash
export GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2
export GRP_REQUIRE_HETEROGENEOUS_ROLLOUT_TP=1
```

The corresponding YAML fields are:

```yaml
heterogeneous_rollout:
  enabled: true
  scheduling:
    scheduler_type: cmlfq
```

Useful scheduler choices:

| Scheduler | Meaning |
| --- | --- |
| `cmlfq` | Causality-aware multi-level feedback queue |
| `length_aware` | Length-predictor based routing |
| `la_mlfq` | Length-aware MLFQ baseline |
| `load_balance` | Simple load-aware routing |

## 2. Enable C-MLFQ

Use C-MLFQ when the workload has tool calls, variable rollout lengths, or
multi-turn trajectories.

```yaml
heterogeneous_rollout:
  scheduling:
    scheduler_type: cmlfq
```

Common C-MLFQ controls:

| Option | Purpose |
| --- | --- |
| `cmlfq_buckets` | TP bucket definitions and token limits |
| `cmlfq_rebuild_interval` | Completed trajectories between prefix-tree rebuilds |
| `cmlfq_tree_path` | Prefix-tree output path |
| `cmlfq_default_bucket` | Initial bucket when no prefix-tree match exists |

Runtime prefix-tree files are written under the job log directory when enabled.
For multi-rank jobs, Libra may write both rank-local files and a merged tree.

## 3. Enable the Global Resource Planner

The planner can be enabled without applying changes to the live runtime. This
is useful for calibration and dry runs:

```bash
export GRP_DYNAMIC_RECONFIG_ENABLED=0
export GRP_FORCE_RUNTIME_RECONFIGURE=1
export GRP_FORCE_TRAIN_GPUS=8
export GRP_FORCE_ROLLOUT_TP_LIST=4:2:2
```

To enable online planning and runtime reconfiguration:

```bash
export GRP_DYNAMIC_RECONFIG_ENABLED=1
export GRP_INTERVAL=1
export GRP_WARMUP_STEPS=0
export GRP_MIN_HISTORY=4
export GRP_MANAGE_ROLLOUT_PROCESSES=1
```

Important planner options:

| Option | Purpose |
| --- | --- |
| `global_resource_planner.enabled` | Enable planner decisions |
| `plan_interval` | Training steps between planner evaluations |
| `warmup_steps` | Steps before online replanning begins |
| `min_history_size` | Minimum observations required before planning |
| `min_gain_ratio` | Required projected gain before applying a plan |
| `reconfiguration_cost_s` | Cost charged to runtime transitions |
| `allowed_rollout_tp` | Candidate TP bucket sizes |
| `allowed_train_tp` | Candidate training TP sizes |
| `fixed_train_gpus` | Pin training GPU count if nonzero |

## 4. Enable the Elastic Hybrid Pool and Cluster Swap

Use the Elastic Hybrid Pool to model resources that can move between rollout
and training. Use cluster-swap when no extra spare GPU or node is reserved for
prewarming.

```bash
export GRP_RECONFIGURE_TRAINING=1
export GRP_TRAINING_POOL_PLAN_ONLY=0
export GRP_DECOUPLE_COMMUNICATION_DOMAINS=1
export GRP_CLUSTER_SWAP_ENABLED=1
export GRP_ROLLOUT_RECONFIGURE_STRATEGY=cluster_swap
export GRP_DRAIN_BEFORE_RECONFIGURE=0
```

With decoupled communication domains, Libra receives hybrid-worker gradient
payloads through the configured elastic side channel while leaving the core
training backend's DP/TP/PP process groups untouched. Set
`GRP_DECOUPLE_COMMUNICATION_DOMAINS=0` only when elastic gradient reduction
must reuse the training DP group.

To force a short cluster-swap run:

```bash
export GRP_FORCE_RUNTIME_RECONFIGURE=1
export GRP_FORCE_TRAIN_GPUS=12
export GRP_FORCE_ROLLOUT_TP_LIST=2:2
export GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2
export GRP_TRAINING_POOL_TARGET_GPUS=12
```

When `GRP_DRAIN_BEFORE_RECONFIGURE=0`, Libra skips dispatcher and peer-rank
drain before applying the cluster-swap plan. This is useful for validating
runtime control paths without waiting for long-tail rollout tasks.

## 5. Control Batch Collection and Long-Tail Rollouts

R2E-Gym and other tool-using workloads can produce long-tail trajectories. Use
finite collection timeouts so a job reports and recovers from stalled batches.

```bash
export GRP_BATCH_COLLECTION_TIMEOUT_S=300
export GRP_BATCH_COLLECTION_MAX_RETRIES=2
```

If a batch collection timeout occurs, Libra resets dispatcher-side rollout
state and retries. Increase the timeout for long-context experiments.

## 6. Configure Weight Synchronization

To debug the Megatron-Core training path without rollout weight refresh:

```bash
export ROLLOUT_WEIGHT_SYNC_MODE=none
export REQUIRE_ROLLOUT_WEIGHT_SYNC=0
```

For production experiments, configure a shared checkpoint path and enable the
desired rollout synchronization mode in the YAML file or launcher.

## 7. Start Training

### R2E-Gym with a Fixed Megatron-Core Layout

This is the recommended initial cluster run. It checks model loading,
Megatron-Core optimizer construction, rollout generation, GRPO updates, and
end-to-end training-loop progress.

```bash
export MODEL_PATH=/path/to/Qwen3-14B
export TRAIN_TP_SIZE=4
export TRAIN_PP_SIZE=1
export TRAIN_CP_SIZE=1
export TRAIN_GPUS=8
export TOTAL_STEPS=4
export MAX_MODEL_LEN=4096
export MAX_NEW_TOKENS=128
export R2E_MAX_PROMPT_TOKENS=2048
export R2E_MAX_TURNS=2
export MEGATRON_USE_PRECISION_AWARE_OPTIMIZER=1
export MEGATRON_OPTIMIZER_CPU_OFFLOAD=1
export MEGATRON_OPTIMIZER_OFFLOAD_FRACTION=1.0
export ROLLOUT_WEIGHT_SYNC_MODE=none
export REQUIRE_ROLLOUT_WEIGHT_SYNC=0
export GRP_DYNAMIC_RECONFIG_ENABLED=0
export GRP_FORCE_RUNTIME_RECONFIGURE=1
export GRP_FORCE_TRAIN_GPUS=8
export GRP_FORCE_ROLLOUT_TP_LIST=4:2:2
export GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2

sbatch --export=ALL \
  scripts/submit_r2e_gym_cmlfq_24gpu_qwen3_30b_a3b.slurm
```

### R2E-Gym with Dynamic Cluster Swap

```bash
export MODEL_PATH=/path/to/Qwen3-14B
export TRAIN_TP_SIZE=4
export TRAIN_PP_SIZE=1
export TRAIN_CP_SIZE=1
export TRAIN_GPUS=8
export TOTAL_STEPS=6
export MAX_MODEL_LEN=4096
export MAX_NEW_TOKENS=128
export ROLLOUT_WEIGHT_SYNC_MODE=none
export REQUIRE_ROLLOUT_WEIGHT_SYNC=0
export MEGATRON_USE_PRECISION_AWARE_OPTIMIZER=1
export MEGATRON_OPTIMIZER_CPU_OFFLOAD=1
export MEGATRON_OPTIMIZER_OFFLOAD_FRACTION=1.0
export GRP_DYNAMIC_RECONFIG_ENABLED=1
export GRP_INTERVAL=1
export GRP_WARMUP_STEPS=0
export GRP_MIN_HISTORY=4
export GRP_MANAGE_ROLLOUT_PROCESSES=1
export GRP_RECONFIGURE_TRAINING=1
export GRP_TRAINING_POOL_PLAN_ONLY=0
export GRP_DECOUPLE_COMMUNICATION_DOMAINS=1
export GRP_CLUSTER_SWAP_ENABLED=1
export GRP_ROLLOUT_RECONFIGURE_STRATEGY=cluster_swap
export GRP_DRAIN_BEFORE_RECONFIGURE=0
export GRP_FORCE_RUNTIME_RECONFIGURE=1
export GRP_FORCE_TRAIN_GPUS=12
export GRP_FORCE_ROLLOUT_TP_LIST=2:2
export GRP_INITIAL_ROLLOUT_TP_LIST=4:2:2
export GRP_TRAINING_POOL_TARGET_GPUS=12

sbatch --export=ALL \
  scripts/submit_r2e_gym_cmlfq_24gpu_qwen3_30b_a3b.slurm
```

### Search-R1 with C-MLFQ

```bash
export MODEL_PATH=/path/to/Qwen3-14B
export SEARXNG_URL=http://searxng-host:8080
sbatch --export=ALL scripts/submit_search_r1_searxng_train.slurm
```

### DAPO-Math-17K with C-MLFQ

```bash
export MODEL_PATH=/path/to/Qwen3-14B
export DAPO_MATH_PATH=/path/to/train.parquet
sbatch --export=ALL scripts/submit_dapo_math_cmlfq_train.slurm
```

## 8. Run Without Slurm

For a manually managed rollout cluster, start the configured OpenAI-compatible
vLLM endpoints first, then run an entrypoint directly:

```bash
python examples/search_r1_async_rl.py \
  --config configs/search_r1_cmlfq_qwen3_14b.yaml
```

Run `python -m pip install -e .` once from the repository root. The editable
installation exposes `RL_Framework` regardless of the checkout directory name;
the supplied launchers also prepend `PROJECT_DIR` to `PYTHONPATH`.

For the full list of configuration fields, see the
[configuration reference](configuration_reference.md).

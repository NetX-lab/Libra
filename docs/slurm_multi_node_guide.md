# Multi-Node vLLM Deployment with Slurm

This guide records the Slurm behavior observed on the original two-node test cluster. GPU resource syntax varies between clusters, so validate the commands on your installation before using the launchers at scale.

## Problem

Starting vLLM on a remote node from inside an allocation such as `#SBATCH -G 8 -N 2` repeatedly failed with:

```text
srun: error: Unable to create step for job XXXXX: Invalid generic resource (gres) specification
```

The following combinations were tested:

| Attempt | `srun` arguments | Result |
| --- | --- | --- |
| 1 | `--overlap` without an explicit GPU count | Failed with an invalid inherited GRES specification |
| 2 | `--gres=gpu:N` | Failed on the original cluster |
| 3 | `-G N` / `--gpus=N` | The first step started, but a second step on the same node reported that the node was busy |
| 4 | `-G N --overlap` | Succeeded for multiple concurrent steps on one node |

The practical conclusions are:

1. `-G N`, `--gres=gpu:N`, and `--gres=gpu:TYPE:N` are not interchangeable on every Slurm installation.
2. Multiple job steps sharing one allocated node generally need `--overlap`.
3. On the original cluster, both an explicit `-G N` and `--overlap` were required.

## Reference launch pattern

An allocation for two four-GPU nodes can use:

```bash
#SBATCH -N 2
#SBATCH -G 8
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH -p gpu
```

Launch one remote vLLM instance as a background Slurm step:

```bash
srun --nodelist="$INST_NODE" --nodes=1 --ntasks=1 \
    -G "$GPU_COUNT" --overlap \
    --export=ALL,CUDA_VISIBLE_DEVICES="$INST_GPUS",TOKENIZERS_PARALLELISM=false \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$INST_PORT" \
    --tensor-parallel-size "$INST_TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --trust-remote-code \
    >"$VLLM_LOG" 2>&1 &
PID=$!
```

## Operational notes

### GPU resource syntax

The original cluster accepted:

```bash
srun -G 2 --overlap ...
```

Without `--overlap`, a second step on the same node was rejected as busy. `--gres=gpu:2` was not accepted by that installation, and `--overlap` without `-G` inherited an invalid GRES request.

When moving to another cluster, start with a probe such as:

```bash
srun --nodelist="$NODE" -G 1 --overlap hostname
```

If that fails, consult the site documentation and try the cluster's supported `--gres` form.

### Environment propagation

Prefer explicit environment propagation:

```bash
srun --export=ALL,CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server ...
```

This preserves the active virtual environment, `PATH`, and `PYTHONPATH`, and avoids another layer of shell quoting. If the site disables environment export, source the environment explicitly inside the job step.

### Process lifetime

Run Python directly under `srun`. The local background PID then belongs to the Slurm step and remains valid for the lifetime of the remote vLLM process:

```bash
srun ... python3 -m vllm.entrypoints.openai.api_server ... >"$LOG" 2>&1 &
PID=$!
```

Avoid starting a detached process inside `bash -c`; the `srun` parent may exit immediately, making local PID tracking unreliable.

### Readiness checks

Use the local `srun` PID for process liveness and the service endpoint for functional readiness:

```bash
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: vLLM instance exited"
fi

curl -fsS "http://${INST_NODE}:${INST_PORT}/health"
```

There is no need to launch another remote step merely to check process state.

### Cleanup

Terminating the local `srun` step normally causes Slurm to terminate its remote child process. The project launchers also use a guarded pattern such as:

```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
```

Scope cleanup commands carefully on shared login nodes.

## Example topology

```text
Training node                         Rollout node
+-------------------------+           +-------------------------+
| GPU 0,1: training       |           | GPU 0,1: vLLM TP=2     |
| GPU 2:   vLLM TP=1      |  HTTP     | GPU 2,3: vLLM TP=2     |
| GPU 3:   vLLM TP=1      | <-------> |                         |
| local srun step A -------+---------->| vLLM :8002             |
| local srun step B -------+---------->| vLLM :8003             |
+-------------------------+           +-------------------------+
```

`kill -0 PID_A` checks the corresponding remote step, `kill PID_A` terminates it, and a job-scoped cleanup handler can stop all remaining vLLM steps.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `Invalid generic resource (gres) specification` | The job-step GPU syntax does not match the site configuration | Test `-G N` and the site-supported `--gres` form |
| `Requested nodes are busy` | Multiple steps are competing for one allocated node | Add `--overlap` if the site permits it |
| Remote vLLM log is empty | `srun` failed before Python started | Inspect the Slurm step error in the redirected log |
| Tracked PID exits immediately | A detached child was started inside another shell | Run Python directly under `srun` |
| Remote process cannot find Python | The virtual environment was not propagated | Use `--export=ALL` or source the environment explicitly |
| Cleanup cannot allocate another step | A cleanup `srun` encounters the same GRES issue | Terminate the existing local `srun` PID instead |

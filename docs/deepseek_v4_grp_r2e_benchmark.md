# DeepSeek-V4 R2E-Gym GRP benchmark flow

This runbook defines the repeatable 112-card GRP arm used to compare with the existing Baseline and C-MLFQ measurements.

## Fixed experiment setup

- Image: `libra-v4-cann91-unified:20261009`.
- Model: DeepSeek-V4-Flash-DSpark-BF16.
- Training: 64 Ascend 910B3 cards, MindSpeed, TP=8, PP=2, CP=1, DP=4, EP=32.
- Rollout: 48 cards, two already deployed vLLM APIs at `192.168.0.2:8010` and `192.168.0.109:8011`; each is an 8-way TP endpoint with its DP workers in its Ray cluster.
- Rollout context limit: `max_model_len=4096`; training sequence length 2048; max new tokens 64.
- Workload: R2E-Gym JSONL index, at most 2 turns per trajectory; 8 optimizer steps; discard step 0 as warmup.
- Keep the Baseline and C-MLFQ data, prompts, sampling, model, image, and endpoints unchanged.

## GRP startup and validation

The GRP config enables `initial_allocation_strategy: grp`, with 8-card allocation granularity and feasible 64-train/48-rollout limits. Rank 0 runs `scripts/deepseek_v4_grp_preflight.py` before torchrun. It must produce a planned config with `initial_allocation_applied: true`, preserve the two deployed rollout endpoints, and pass a YAML round trip. Other ranks wait for this artifact and refuse to launch if it is missing. The preflight artifact is written to `benchmark_r2e_112_20261010/grp/planned.yaml`; its planner decision is saved beside it as `planned.yaml.decision.json`.

Before launching training, check that both rollout health endpoints return HTTP 200 and that all eight training nodes each have eight free NPUs. Launch ranks 0–7 concurrently with:

```bash
bash scripts/launch_deepseek_v4_r2e_112_train.sh grp <rank>
```

Use one process per training node and do not reuse an old container or planned config. A run counts as successful only when all eight node exit files contain `0`, all eight `step_timings/step_*.json` files exist, and the completed-step list is exactly 0–7.

## Metrics and comparison

After a successful run, execute on a host with the benchmark image and shared benchmark directory:

```bash
bash scripts/collect_deepseek_v4_r2e_112_metrics.sh grp
```

The collector discards the warmup step and writes `grp/metrics.json`, including generated tokens/s, trajectories/s, mean/P50/P95/P99 TTFT, request end-to-end latency and steady-step latency, plus gains relative to `baseline/metrics.json`. Do not publish a gain when the run is incomplete or telemetry validation fails.

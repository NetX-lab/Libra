# Paper Tree-DP GRP experiment

`RUN_ONLY=paper` uses a separate, paper-faithful startup planner rather than
the legacy exhaustive `grp` optimizer.  It evaluates every training budget,
enumerates training layouts through the topology-aware `TP -> PP -> DP` tree,
and computes each rollout allocation with the length-sorted dynamic program.

Run the treatment plus the existing static 24/24 control on the same six idle
hosts:

```bash
cd /path/to/RL_Framework_npu
NODE_PASSWORD=... \
AVAILABLE_HOSTS='host1 host2 host3 host4 host5 host6' \
GRP_PROFILE_JSONL=/path/to/grp_startup_profile.jsonl \
RUN_ONLY=paper \
bash scripts/run_6node48_grp_vs_no_grp_equal.sh
```

Use `PREFLIGHT_ONLY=1` to validate the planning path without launching NPU
processes.  The output directory contains:

- `paper_tree_dp_trace.json`: every tree node, prune reason, DP budget, and
  final choice.  A non-empty `tree_nodes` array is direct proof that the
  decision tree executed.
- `device_placement.json`: the exact device-level train/rollout split consumed
  by the launcher.
- `effective_config.yaml`: the actual parallel topology passed to training.

The experiment defaults to `TP` communication ratio `<= 0.50`, pipeline-bubble
ratio `<= 0.30`, and at most 64 stratified historical requests in the exact
rollout DP.  Override them with `PAPER_TP_COMM_RATIO_LIMIT`,
`PAPER_PP_BUBBLE_RATIO_LIMIT`, and `PAPER_MAX_DP_REQUESTS` respectively.

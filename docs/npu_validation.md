# Qwen3-14B 6-node/48-NPU validation

## Result

The formal R2E-Gym experiment completed all 100 training steps (step 0 through
step 99) on August 6, 2026. Rank 0 reported a total training time of 7536.7
seconds. No fatal Python, HCCL, or NPU errors were observed.

## Configuration

- Model: Qwen3-14B
- Training backend: Megatron-Core with HCCL
- Workload: R2E-Gym
- Cluster: 6 nodes, 48 Ascend NPUs
- Training pool: 4 nodes / 32 NPUs
- Rollout pool: 2 nodes / 16 NPUs, 8 OpenAI-compatible endpoints
- Maximum generated tokens: 30,000
- Scheduler: C-MLFQ with TP=1, TP=2, and TP=4 buckets
- Resource management: Global Resource Planner and Elastic Hybrid Pool

## Resource-planning evidence

- The planner preserved all eight externally managed rollout endpoints.
- Elastic Hybrid Pool attached exactly one hybrid worker to the training domain.
- The effective training target reached 33 NPUs and subsequent events reported
  `training_gpus_unchanged:33:active_hybrids=1`, demonstrating that the worker
  was not added repeatedly.

## Reward and throughput

- Mean reward across 100 steps: 0.2866
- Mean reward over the first 10 steps: 0.2778
- Mean reward over the final 10 steps: 0.3829
- Final-step reward: 0.5030
- End-to-end throughput: 47.97 steps/hour and 191.86 trajectories/hour
- Steady-state throughput: 113.10 steps/hour and 452.40 trajectories/hour
- Estimated steady-state generated-token throughput: 313.72 tokens/second

The reward improved in the latter part of the run but remained noisy because
the rank-0 history contained one trajectory per step. Periodic checkpoint and
weight synchronization at steps 25, 50, and 75 consumed 32.83% of total time;
rollout generation accounted for 94.2% of steady-state step time.

## Validation status

**PASS** — the complete Megatron-Core, R2E-Gym, Global Resource Planner,
C-MLFQ, and Elastic Hybrid Pool path completed successfully on the six-node
Ascend cluster.

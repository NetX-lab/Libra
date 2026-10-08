# Observability

Each Slurm launcher writes a job directory under `logs/`. The files below
capture training progress, rollout state, planner decisions, and runtime
reconfiguration activity.

| File | Purpose |
| --- | --- |
| `training_rank*_*.log` | Training-rank logs |
| `vllm_*.log` | Initial rollout-instance logs |
| `runtime_rollout/*.log` | Runtime-created rollout-instance logs |
| `global_resource_rollout_manifest.json` | Planned and applied rollout layouts |
| `rollout_process_registry.jsonl` | Managed rollout-process registry |
| `runtime_reconfiguration/runtime_reconfiguration_events.jsonl` | Planner decisions and runtime actions |
| `cmlfq_tree_*.json` | C-MLFQ prefix-tree snapshots |
| `history/*.jsonl` | Per-step throughput, reward, and queue metrics |

History records can be collected and reused for planner calibration. See
[Runtime History Collection](history_data_collection.md) for the file format
and collection workflow.

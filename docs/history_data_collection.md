# Runtime History Collection

Libra can record per-step rollout lengths, timing measurements, resource configurations, and cost-model predictions. The history serves three purposes:

1. Validate cost-model accuracy against observed rollout and training time.
2. Compare throughput across resource and parallelism configurations.
3. Track how the sequence-length distribution changes as the policy evolves.

Enable collection in the experiment configuration:

```yaml
enable_history_collection: true
history_output_dir: ./logs/history
history_save_raw_lengths: false
history_flush_interval: 1
history_experiment_name: libra_run
```

`HistoryDataCollector` writes JSONL records during training and a summary at shutdown. Records include sequence-length statistics, rollout and training timing, current resource allocation, model version, and optional predicted times. Raw per-trajectory lengths can be retained with `history_save_raw_lengths: true`, but this substantially increases output size for large batches.

The relevant implementation is in `infra/observability/history_collector.py`; integration with the training loop is in `trainer/async_rl_trainer.py`. The examples in `examples/test_history_collector.py` demonstrate standalone collection, trend summaries, and cost-model comparison fields.

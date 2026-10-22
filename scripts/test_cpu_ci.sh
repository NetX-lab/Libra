#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q --strict-markers --junitxml=reports/cpu-tests.xml \
  tests/test_config_parsing.py \
  tests/test_qwen3_config.py \
  tests/test_planner_global_batch_workload.py \
  tests/test_async_task_runner.py \
  tests/test_sync_drain_prefetch.py \
  tests/test_rollout_credit.py \
  tests/test_phase_tracer.py \
  tests/test_restartable_vllm_server.py \
  tests/test_checked_prefix_cache_reset.py \
  tests/test_nccl_store_lifecycle.py \
  tests/test_nccl_store_timeout.py \
  tests/test_nccl_weight_sync.py \
  tests/test_publication_sessions.py \
  tests/test_serving_protocol.py \
  tests/test_collective_weight_export.py \
  tests/test_cmlfq_scheduler.py \
  tests/test_cmlfq_cost_scheduler.py \
  tests/test_cost_model_units.py \
  tests/test_cmlfq_rebind_state.py \
  tests/test_hetero_cmlfq_integration.py \
  tests/test_rollout_engine.py \
  tests/test_cpu_offload_backend.py \
  tests/test_r2e_patch_harness.py \
  tests/test_r2e_gym_workflow.py \
  tests/test_optimizer_state.py \
  tests/test_elastic_optimizer_state.py \
  tests/test_ehp_dp_resharding.py \
  tests/test_ehp_batch_partition.py \
  tests/test_fixed_batch_gradients.py \
  tests/test_grpo_sample_mean.py \
  tests/test_hybrid_sample_statistics_data.py \
  tests/test_elastic_gradient_idle_path.py \
  tests/test_hybrid_optimizer_cuda_cache.py \
  tests/test_torch_optimizer_memory.py \
  tests/test_megatron_core_train_engine.py \
  tests/test_hybrid_worker_batch_protocol.py

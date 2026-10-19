#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q --strict-markers --junitxml=reports/cpu-tests.xml \
  tests/test_config_parsing.py \
  tests/test_qwen3_config.py \
  tests/test_planner_global_batch_workload.py \
  tests/test_async_task_runner.py \
  tests/test_phase_tracer.py \
  tests/test_restartable_vllm_server.py \
  tests/test_cmlfq_scheduler.py \
  tests/test_cmlfq_cost_scheduler.py \
  tests/test_hetero_cmlfq_integration.py \
  tests/test_rollout_engine.py \
  tests/test_cpu_offload_backend.py \
  tests/test_r2e_patch_harness.py \
  tests/test_r2e_gym_workflow.py \
  tests/test_collective_weight_export.py \
  tests/test_core_reconfiguration_protocol.py \
  tests/test_optimizer_state.py \
  tests/test_ehp_batch_partition.py \
  tests/test_ehp_correctness.py \
  tests/test_torch_optimizer_memory.py \
  tests/test_ehp_membership_release.py \
  tests/test_ehp_readiness_stop.py \
  tests/test_elastic_gradient_idle_path.py \
  tests/test_elastic_hybrid_pool.py \
  tests/test_grp_decision_gates.py \
  tests/test_rollout_lending.py \
  tests/test_rollout_lifecycle.py \
  tests/test_rollout_process.py \
  tests/test_rollout_supply.py \
  tests/test_runtime_boundary_integration.py \
  tests/test_runtime_core_capabilities.py \
  tests/test_runtime_elastic_executor.py \
  tests/test_runtime_executor_shutdown.py \
  tests/test_runtime_rollout_ownership.py \
  tests/test_scheduling_audit.py \
  tests/test_sync_drain_prefetch.py

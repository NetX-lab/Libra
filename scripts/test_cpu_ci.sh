#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q --strict-markers --junitxml=reports/cpu-tests.xml \
  tests/test_config_parsing.py \
  tests/test_qwen3_config.py \
  tests/test_async_task_runner.py \
  tests/test_phase_tracer.py \
  tests/test_restartable_vllm_server.py \
  tests/test_cmlfq_scheduler.py \
  tests/test_cmlfq_cost_scheduler.py \
  tests/test_hetero_cmlfq_integration.py \
  tests/test_rollout_engine.py \
  tests/test_cpu_offload_backend.py

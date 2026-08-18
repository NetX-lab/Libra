"""Measure R2E-Gym component behavior without allocating model devices.

This is deliberately a control-plane benchmark: C-MLFQ routing and GRP
planning are CPU-side decisions, while EHP exercises its real join/release
state machine and local gradient aggregation path. It can run before a model
job to establish that the controller layer is healthy.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner
from RL_Framework.infra.elastic.hybrid_pool import (
    ElasticHybridPool,
    GradientPayload,
)
from RL_Framework.infra.scheduling.cmlfq_offline_profile import CMLFQOfflineProfiler
from RL_Framework.infra.scheduling.cmlfq_scheduler import CMLFQScheduler


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _measure_cmlfq(requests: int) -> dict:
    profiler = CMLFQOfflineProfiler()
    raw_history = [
        {
            "prompt_id": "r2e-component-benchmark",
            "total_output_tokens": 36000,
            "tool_returns": [
                {
                    "tool_type": "code_executor",
                    "output": "R2E pytest failure with a large repository trace.",
                    "status": "failure",
                    "payload_tokens": 6000,
                    "token_position": 1200,
                    "remaining_length": 34800,
                }
            ],
        }
    ]
    profiler.profile_from_raw_trajectories(raw_history)
    scheduler = CMLFQScheduler(
        buckets={
            "short": {"tp_degrees": [1], "max_tokens": 256},
            "medium": {"tp_degrees": [1], "max_tokens": 768},
            "long": {"tp_degrees": [2], "max_tokens": 1024},
        },
        prefix_tree=profiler.prefix_tree,
    )
    scheduler.register_instance(0, "short_tp1_0", 1)
    scheduler.register_instance(1, "short_tp1_1", 1)
    scheduler.register_instance(2, "long_tp2", 2)

    latencies_ms: list[float] = []
    initial_buckets: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    migrated_buckets: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    migration_latencies_ms: list[float] = []
    migrations = 0
    for index in range(requests):
        input_tokens = 1024
        start = time.perf_counter()
        route = scheduler.schedule(
            input_tokens=input_tokens,
            prompt_id="r2e-component-benchmark",
        )
        latencies_ms.append((time.perf_counter() - start) * 1000)
        initial_buckets[route.category] = initial_buckets.get(route.category, 0) + 1
        tool_start = time.perf_counter()
        decision = scheduler.on_tool_return(
            route.request_id,
            raw_history[0]["tool_returns"][0],
            generated_tokens=1200,
        )
        if decision.should_migrate:
            migrated = scheduler.execute_migration(route.request_id, decision)
            migrations += 1
            migrated_buckets[migrated.category] = (
                migrated_buckets.get(migrated.category, 0) + 1
            )
        migration_latencies_ms.append((time.perf_counter() - tool_start) * 1000)
        scheduler.finish_request(route.request_id, 36000)

    return {
        "requests": requests,
        "initial_route_buckets": initial_buckets,
        "migrated_route_buckets": migrated_buckets,
        "migrations": migrations,
        "route_latency_ms_mean": statistics.fmean(latencies_ms),
        "route_latency_ms_p95": _percentile(latencies_ms, 0.95),
        "migration_latency_ms_mean": statistics.fmean(migration_latencies_ms),
        "migration_latency_ms_p95": _percentile(migration_latencies_ms, 0.95),
        "throughput_requests_per_s": requests / max(sum(latencies_ms) / 1000, 1e-9),
    }


def _planner_config() -> AsyncRLConfig:
    config = AsyncRLConfig(
        model_path="/path/to/Qwen3-14B",
        train_gpus=4,
        rollout_gpus=4,
        train_tp_size=1,
        train_pp_size=1,
        train_dp_size=4,
        micro_batch_size=1,
        batch_size=4,
        n_total_gpus=8,
        max_seq_length=32768,
        max_new_tokens=32768,
    )
    planner = config.global_resource_planner
    planner.enabled = True
    planner.train_backend = "analytic"
    planner.rollout_backend = "analytic"
    planner.plan_interval = 1
    planner.warmup_steps = 0
    planner.min_history_size = 4
    planner.min_gain_ratio = 0.0
    planner.reconfiguration_cost_s = 0.0
    planner.allowed_train_tp = [1]
    planner.allowed_train_pp = [1]
    planner.fixed_train_gpus = 0
    planner.allowed_train_tp = [1, 2]
    planner.allowed_rollout_tp = [1, 2, 4]
    planner.require_heterogeneous_rollout_tp = True
    planner.micro_batch_sizes = [1]
    return config


def _measure_grp() -> dict:
    config = _planner_config()
    batch = [
        {
            "prompt_id": f"r2e-{index}",
            "input_len": 1024 + (index % 4) * 256,
            "output_len": 16000 + index * 1000,
        }
        for index in range(16)
    ]
    planner = GlobalResourcePlanner.from_config(config)
    start = time.perf_counter()
    decision = planner.plan_if_needed(step=1, config=config, batch=batch)
    elapsed_ms = (time.perf_counter() - start) * 1000
    candidate = decision.candidate_plan
    if candidate is None:
        raise RuntimeError(f"GRP produced no candidate: {decision.reason}")
    return {
        "requests": decision.num_requests,
        "decision_reason": decision.reason,
        "should_reconfigure": decision.should_reconfigure,
        "planning_latency_ms": elapsed_ms,
        "current_global_s": decision.current_plan.t_global if decision.current_plan else None,
        "candidate_global_s": candidate.t_global,
        "expected_gain_s": candidate.expected_gain_s,
        "candidate_train_gpus": candidate.train_config.n_gpus,
        "candidate_rollout_tp": candidate.rollout_tp_list,
    }


def _measure_ehp(iterations: int, tensor_elements: int) -> dict:
    pool = ElasticHybridPool(
        core_train_workers=["dp0", "dp1"],
        core_rollout_workers=["rollout0", "rollout1"],
        snapshot_fetcher=lambda _worker, _target: 7,
        zero_sync_steps=1,
    )
    join_latencies_ms: list[float] = []
    try:
        for index in range(iterations):
            worker_id = f"rollout{index % 2}"
            target = f"dp{index % 2}"
            start = time.perf_counter()
            worker = pool.join_training(worker_id, target).result(timeout=5)
            join_latencies_ms.append((time.perf_counter() - start) * 1000)
            if worker.state_version != 7:
                raise RuntimeError("EHP snapshot version was not propagated")
            pool.release_to_rollout(worker_id)

        core = {
            "dp0": (torch.ones(tensor_elements),),
            "dp1": (torch.ones(tensor_elements) * 2,),
        }
        pool.gradient_domain.request_join("rollout0", "dp0")
        pool.gradient_domain.mark_active("rollout0")
        payload = GradientPayload(
            replica_id="rollout0",
            target_core_id="dp0",
            tensors=(torch.ones(tensor_elements) * 3,),
        )
        start = time.perf_counter()
        reduced = pool.gradient_domain.reduce_core_gradients(
            core_gradients=core,
            hybrid_payloads=[payload],
        )
        aggregate_ms = (time.perf_counter() - start) * 1000
        expected = 3.0
        observed = float(reduced["dp0"][0][0])
        if abs(observed - expected) > 1e-5:
            raise RuntimeError(f"unexpected EHP aggregate: {observed} != {expected}")
        bytes_processed = tensor_elements * 4 * 3
        return {
            "join_release_iterations": iterations,
            "join_latency_ms_mean": statistics.fmean(join_latencies_ms),
            "join_latency_ms_p95": _percentile(join_latencies_ms, 0.95),
            "gradient_tensor_elements": tensor_elements,
            "gradient_aggregate_ms": aggregate_ms,
            "gradient_effective_gbps": bytes_processed / max(aggregate_ms, 1e-9) / 1e6,
            "gradient_value": observed,
            "communication_domains": pool.gradient_domain.communication_state(),
        }
    finally:
        pool.close()


def main() -> None:
    requests = int(os.environ.get("CORE_BENCH_REQUESTS", "1000"))
    iterations = int(os.environ.get("EHP_BENCH_ITERATIONS", "100"))
    tensor_elements = int(os.environ.get("EHP_TENSOR_ELEMENTS", "262144"))
    report = {
        "cmlfq": _measure_cmlfq(requests),
        "global_resource_planner": _measure_grp(),
        "elastic_hybrid_pool": _measure_ehp(iterations, tensor_elements),
    }
    output = Path(
        os.environ.get(
            "CORE_COMPONENT_REPORT",
            "logs/r2e_gym_qwen3_14b_npu_full/component_report.json",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"R2E_CORE_COMPONENTS_OK output={output}")


if __name__ == "__main__":
    main()

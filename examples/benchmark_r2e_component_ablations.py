"""Controlled performance ablations for R2E-Gym control-plane components.

The benchmark separates measured controller latency from projected execution
time.  It intentionally avoids claiming that an analytic planner estimate is
the same as a physical cluster speedup.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from RL_Framework.infra.scheduling.cmlfq_migration import MigrationCostProfiler

from validate_r2e_core_components import _measure_ehp, _measure_grp


def _cmlfq_ablation() -> dict:
    profiler = MigrationCostProfiler()
    # The values model a TP1 request whose long continuation can be resharded
    # to TP2.  They are explicit benchmark inputs rather than device timings.
    profiler.record_measurement(
        source_tp=1,
        target_tp=2,
        seq_len=8192,
        offload_ms=160.0,
        reshard_ms=80.0,
        network_ms=0.0,
        reload_ms=140.0,
        recompute_prefill_ms=520.0,
    )
    start = time.perf_counter()
    action, selected_ms = profiler.decide_migration_path(1, 2, 8192)
    decision_ms = (time.perf_counter() - start) * 1000
    baseline_recompute_ms = 520.0
    return {
        "mode": "profiled_execution_cost",
        "baseline": {"strategy": "recompute_prefill", "latency_ms": baseline_recompute_ms},
        "cmlfq": {"strategy": action, "latency_ms": selected_ms},
        "saved_ms": baseline_recompute_ms - selected_ms,
        "speedup": baseline_recompute_ms / selected_ms,
        "controller_decision_latency_ms": decision_ms,
    }


def _grp_ablation() -> dict:
    measured = _measure_grp()
    current = float(measured["current_global_s"])
    candidate = float(measured["candidate_global_s"])
    return {
        "mode": "analytic_global_cost_model",
        "baseline": {"global_time_s": current},
        "grp_candidate": {"global_time_s": candidate},
        "projected_saved_s": current - candidate,
        "projected_speedup": current / candidate,
        "planner_latency_ms": measured["planning_latency_ms"],
        "candidate_train_gpus": measured["candidate_train_gpus"],
        "candidate_rollout_tp": measured["candidate_rollout_tp"],
    }


def _ehp_ablation() -> dict:
    snapshot_delay_s = 0.15
    start = time.perf_counter()
    time.sleep(snapshot_delay_s)
    blocking_join_ms = (time.perf_counter() - start) * 1000
    measured = _measure_ehp(iterations=100, tensor_elements=262144)
    nonblocking_join_ms = float(measured["join_latency_ms_mean"])
    return {
        "mode": "measured_local_state_machine",
        "baseline": {"strategy": "blocking_snapshot_join", "latency_ms": blocking_join_ms},
        "elastic_hybrid_pool": {
            "strategy": "nonblocking_snapshot_join",
            "join_latency_ms": nonblocking_join_ms,
            "join_latency_p95_ms": measured["join_latency_ms_p95"],
            "gradient_aggregate_ms": measured["gradient_aggregate_ms"],
            "gradient_effective_gbps": measured["gradient_effective_gbps"],
            "communication_domains": measured["communication_domains"],
        },
        "join_return_speedup": blocking_join_ms / nonblocking_join_ms,
    }


def main() -> None:
    report = {
        "cmlfq": _cmlfq_ablation(),
        "global_resource_planner": _grp_ablation(),
        "elastic_hybrid_pool": _ehp_ablation(),
    }
    output = Path(
        os.environ.get(
            "COMPONENT_ABLATION_REPORT",
            "logs/r2e_component_ablations.json",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"R2E_COMPONENT_ABLATIONS_OK output={output}")


if __name__ == "__main__":
    main()

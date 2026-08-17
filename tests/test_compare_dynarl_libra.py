from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_dynarl_libra.py"
SPEC = importlib.util.spec_from_file_location("compare_dynarl_libra", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_rlinf_log_supports_scientific_notation(tmp_path: Path) -> None:
    log_path = tmp_path / "main.log"
    log_path.write_text(
        "\n".join(
            [
                "Global Step: 1/3 [step_time=24.5, sync_weights_time=8.69, "
                "variance_of_response_length=3.17e+3, actor/lr=2e-6]",
                "Global Step: 2/3 [step_time=12.9, sync_weights_time=2.85, "
                "advantages_mean=-1.07e-8]",
                "[ElasticMegatron-Perf] : nccl_build_cost=1.2e+0ms, "
                "transfer_cost=3.85e+2ms, update_model_cost=1e-2ms, "
                "transfer_bytes=4.97GB, bandwidth=1.29e+1 GB/s",
            ]
        ),
        encoding="utf-8",
    )

    result = MODULE.parse_rlinf_log(log_path)

    assert result["num_steps"] == 2
    assert result["steps"][0]["variance_of_response_length"] == 3170.0
    assert result["steps"][0]["actor/lr"] == 2e-6
    assert result["steps"][1]["advantages_mean"] == -1.07e-8
    assert result["warm_step_time_mean_s"] == 12.9
    assert result["warm_sync_time_mean_s"] == 2.85
    assert result["migrations"][0]["transfer_ms"] == 385.0
    assert result["migrations"][0]["bandwidth_gbps"] == 12.9


def test_parse_libra_result_uses_last_round_as_steady_state(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-4B",
                "outputs_stable": True,
                "rounds": [
                    {"send_seconds": 20.84, "payload_gib_per_second": 0.77},
                    {"send_seconds": 0.19, "payload_gib_per_second": 17.6},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.parse_libra_result(result_path)

    assert result["cold_reload_s"] == pytest.approx(20.84)
    assert result["steady_reload_s"] == pytest.approx(0.19)
    assert result["steady_payload_gib_per_second"] == pytest.approx(17.6)

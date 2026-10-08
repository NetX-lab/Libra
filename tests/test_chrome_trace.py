import json

import pytest

try:
    from RL_Framework.infra.observability.chrome_trace import ChromeTraceCollector
except ModuleNotFoundError:
    from infra.observability.chrome_trace import ChromeTraceCollector


def test_chrome_trace_writes_metadata_spans_and_instants(tmp_path):
    collector = ChromeTraceCollector(
        tmp_path,
        rank=2,
        local_rank=1,
        world_size=4,
        process_name="test",
    )
    collector.instant("rollout.enqueue", cat="rollout", args={"task_id": 7})
    with collector.span("grpo_update", cat="train", args={"step": 3}):
        pass
    with pytest.raises(ValueError):
        with collector.span("failed", cat="train"):
            raise ValueError("expected")
    collector.close()
    collector.close()

    files = list(tmp_path.glob("trace.rank_2.pid_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["displayTimeUnit"] == "ms"
    names = [event["name"] for event in payload["traceEvents"]]
    assert "process_name" in names
    assert "rollout.enqueue" in names
    assert "grpo_update" in names
    failed = next(event for event in payload["traceEvents"] if event["name"] == "failed")
    assert failed["ph"] == "X"
    assert failed["args"]["error"].startswith("ValueError")


def test_disabled_collector_is_a_noop(tmp_path):
    collector = ChromeTraceCollector(tmp_path, enabled=False)
    collector.instant("ignored")
    collector.close()
    assert list(tmp_path.iterdir()) == []

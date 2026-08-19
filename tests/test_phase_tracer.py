import json

from RL_Framework.infra.observability.phase_tracer import PhaseTracer


def test_phase_tracer_writes_spans_and_preserves_details(tmp_path):
    path = tmp_path / "phase_trace_rank0.jsonl"
    tracer = PhaseTracer(path, rank=0, world_size=2)
    tracer.start(3, "rollout", {"batch_size": 4})
    tracer.end(3, "rollout", {"requests": 4})
    tracer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["phase_start", "phase_span"]
    assert records[1]["phase"] == "rollout"
    assert records[1]["step"] == 3
    assert records[1]["details"] == {"batch_size": 4, "requests": 4}
    assert records[1]["duration_s"] >= 0

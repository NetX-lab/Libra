import json

from RL_Framework.infra.observability.history_collector import HistoryDataCollector


def test_history_collector_writes_runtime_length_profile(tmp_path):
    profile_path = tmp_path / "runtime_length_profile.jsonl"
    collector = HistoryDataCollector(
        output_dir=str(tmp_path / "history"),
        flush_interval=10,
        length_profile_path=str(profile_path),
    )
    collector.initialize()

    collector.record_step(
        step=3,
        sequence_records=[
            {
                "input_len": 11,
                "output_len": 22,
                "prompt_id": "case-1",
            }
        ],
    )
    collector.finalize()

    rows = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows == [
        {
            "step": 3,
            "timestamp": rows[0]["timestamp"],
            "input_len": 11,
            "output_len": 22,
            "profile_source": "runtime_online",
            "prompt_id": "case-1",
        }
    ]

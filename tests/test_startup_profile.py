from RL_Framework.infra.cost_model.startup_profile import (
    build_length_profile,
    load_length_profile_records,
    load_profile_summary,
    profile_jsonl_has_records,
    profile_metadata_matches,
    profile_summary_matches,
    sample_indices,
    startup_profile_metadata,
    summarize_length_profile,
    write_profile_summary,
)
from RL_Framework.workflow.r2e_gym_profile import build_r2e_gym_length_profile


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text or ""]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(f"{msg['role']}:{msg['content']}" for msg in messages)


def test_spread_sampling_covers_dataset_extremes():
    assert sample_indices(10, 4, strategy="spread") == [0, 3, 6, 9]


def test_build_length_profile_repeats_real_sample_records():
    rows = [
        {"prompt_id": "p0", "prompt": "aaa", "target_issue": "bb"},
        {"prompt_id": "p1", "prompt": "aaaaa", "target_issue": "bbbb"},
        {"prompt_id": "p2", "prompt": "aaaaaaa", "target_issue": "bbbbbb"},
    ]

    records = build_length_profile(
        rows,
        prompt_length_fn=lambda row: len(row["prompt"]),
        output_length_fn=lambda row: len(row["target_issue"]),
        sample_size=2,
        strategy="spread",
        samples_per_prompt=2,
    )

    assert [record["prompt_id"] for record in records] == ["p0", "p0", "p2", "p2"]
    assert [record["input_len"] for record in records] == [3, 3, 7, 7]
    assert [record["output_len"] for record in records] == [2, 2, 6, 6]
    assert {record["profile_source"] for record in records} == {"startup_real_sample"}


def test_profile_summary_reports_output_tail():
    summary = summarize_length_profile(
        [
            {"input_len": 10, "output_len": 100},
            {"input_len": 20, "output_len": 300},
        ]
    )

    assert summary["records"] == 2
    assert summary["output_len"]["max"] == 300.0
    assert summary["total_len"]["mean"] == 215.0


def test_profile_jsonl_has_records_detects_real_rows(tmp_path):
    path = tmp_path / "profile.jsonl"
    path.write_text('{"input_len": 10, "output_len": 20}\n', encoding="utf-8")

    assert profile_jsonl_has_records(path)


def test_profile_metadata_reuse_matches_expected_fields(tmp_path):
    expected = startup_profile_metadata(
        dataset_jsonl="/data/r2e/index.jsonl",
        model_path="/models/qwen3-4b",
        tokenizer_path="/models/qwen3-4b",
        sample_size=64,
        strategy="spread",
        seed=0,
        samples_per_prompt=1,
        max_new_tokens=128,
        max_seq_length=4096,
        r2e_max_turns=3,
        r2e_max_prompt_tokens=0,
    )
    summary_path = tmp_path / "profile_summary.json"
    write_profile_summary({**expected, "records": 2}, summary_path)

    summary = load_profile_summary(summary_path)

    assert profile_metadata_matches(summary, expected)
    assert profile_summary_matches(summary_path, expected)
    assert summary["dataset_jsonl"] == "/data/r2e/index.jsonl"


def test_load_length_profile_records_flattens_step_history_and_limits_tail(tmp_path):
    path = tmp_path / "runtime_profile.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"step": 0, "sequences": [{"input_len": 10, "output_len": 20}]}',
                '{"input_len": 30, "output_len": 40, "profile_source": "runtime_online"}',
                '{"step": 1, "raw_prompt_lengths": [50], "raw_gen_lengths": [60]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_length_profile_records(path, max_records=2)

    assert [(r["input_len"], r["output_len"]) for r in records] == [
        (30, 40),
        (50, 60),
    ]
    assert records[-1]["step"] == 1


def test_r2e_gym_length_profile_uses_real_prompt_builder_and_target_issue():
    rows = [
        {
            "repo_name": "repo",
            "commit_hash": "abc",
            "prompt": "Describe failure",
            "target_issue": "Title\nExpected\nActual",
        }
    ]

    records = build_r2e_gym_length_profile(
        rows,
        tokenizer=_Tokenizer(),
        sample_size=1,
        max_new_tokens=512,
        max_seq_length=2048,
    )

    assert len(records) == 1
    assert records[0]["prompt_id"] == "repo:abc"
    assert records[0]["input_len"] > len(rows[0]["prompt"])
    assert records[0]["output_len"] > len(rows[0]["target_issue"])

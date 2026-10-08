import pytest

from RL_Framework.engine.rollout_engine import VLLMRolloutEngine


def test_parse_response_accepts_missing_logprobs():
    engine = VLLMRolloutEngine()

    parsed = engine._parse_response(
        {
            "choices": [
                {
                    "text": "hello",
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert parsed == {
        "text": "hello",
        "tokens": [],
        "logprobs": [],
        "finish_reason": "stop",
    }


def test_parse_response_rejects_invalid_choice():
    engine = VLLMRolloutEngine()

    with pytest.raises(ValueError, match="Invalid choice type"):
        engine._parse_response({"choices": [None]})

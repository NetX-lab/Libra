import pytest
from unittest.mock import AsyncMock, MagicMock

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


def test_generate_forwards_stop_sequence_and_keeps_delimiter():
    engine = VLLMRolloutEngine(model_path="model")
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "text": "[ISSUE]done[/ISSUE]",
                "logprobs": {"tokens": [], "token_logprobs": []},
                "finish_reason": "stop",
            }
        ]
    }
    client = AsyncMock()
    client.is_closed = False
    client.post.return_value = response
    engine.http_client = client
    engine._ensure_client = AsyncMock()

    import asyncio

    result = asyncio.run(
        engine.generate(
            "prompt",
            stop=["[/ISSUE]"],
            include_stop_str_in_output=True,
        )
    )

    payload = client.post.await_args.kwargs["json"]
    assert payload["stop"] == ["[/ISSUE]"]
    assert payload["include_stop_str_in_output"] is True
    assert result["text"].endswith("[/ISSUE]")

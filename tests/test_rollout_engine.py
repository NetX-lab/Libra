import asyncio
import json

import httpx
import pytest

from RL_Framework.engine.rollout_engine import VLLMRolloutEngine
from RL_Framework.workflow.behavior import SamplingEvidenceError


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


class InterruptedBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"choices":'
        raise httpx.ReadError("completion response interrupted")


@pytest.mark.parametrize("initial_failure", ["read_error", "missing_response"])
def test_http_disconnect_retries_same_nonstreaming_request_once_without_relaxing_evidence(initial_failure):
    attempts = []

    async def handle(request):
        payload = json.loads(request.content)
        attempts.append((str(request.url), dict(request.headers), payload))
        if len(attempts) == 1:
            if initial_failure == "missing_response":
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return httpx.Response(200, stream=InterruptedBody())
        trace = dict(logprob_kind="sampler", prompt_token_ids=payload["prompt"],
            output_token_ids=[21], logprobs=[-.2],
            sampling_parameters={key: payload[key] for key in ("seed", "temperature", "top_p")})
        return httpx.Response(200, headers={"x-libra-optimizer-revision": "24"},
            json={"choices": [{"text": "answer", "libra_behavior": trace}]})

    async def run():
        engine = VLLMRolloutEngine()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine.http_client = client
            engine._http_client_loop = asyncio.get_running_loop()
            return await engine.generate("display", prompt_token_ids=[10, 11],
                capture_behavior=True, seed=12, temperature=.8, top_p=.95, max_new_tokens=4)

    result = asyncio.run(run())
    assert result["tokens"] == [21]
    assert result["logprobs"] == [-.2]
    assert result["sample_revision"] == 24
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert attempts[1][1]["x-libra-capture-behavior"] == "1"


@pytest.mark.parametrize(("initial_failure", "second_failure"), [
    ("read_error", "read_error"),
    ("read_error", "invalid_evidence"),
    ("read_error", "cancelled"),
    ("missing_response", "missing_response"),
    ("missing_response", "invalid_evidence"),
    ("missing_response", "cancelled"),
])
def test_http_disconnect_retry_keeps_original_failure_and_cancellation_boundaries(initial_failure, second_failure):
    attempts = []

    async def handle(request):
        attempts.append(request)
        if initial_failure == "missing_response" and (
            len(attempts) == 1 or second_failure == "missing_response"
        ):
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        if len(attempts) == 1 or second_failure == "read_error":
            return httpx.Response(200, stream=InterruptedBody())
        if second_failure == "cancelled":
            raise asyncio.CancelledError()
        return httpx.Response(200, headers={"x-libra-optimizer-revision": "24"},
            json={"choices": [{"text": "answer"}]})

    async def run():
        engine = VLLMRolloutEngine()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine.http_client = client
            engine._http_client_loop = asyncio.get_running_loop()
            return await engine.generate("display", prompt_token_ids=[10, 11],
                capture_behavior=True, seed=12, temperature=.8, top_p=.95, max_new_tokens=4)

    error = asyncio.CancelledError if second_failure == "cancelled" else SamplingEvidenceError
    with pytest.raises(error) as caught:
        asyncio.run(run())
    assert len(attempts) == 2
    if second_failure == "read_error":
        assert isinstance(caught.value.__cause__, httpx.ReadError)
    elif second_failure == "missing_response":
        assert isinstance(caught.value.__cause__, httpx.RemoteProtocolError)
    elif second_failure == "invalid_evidence":
        assert str(caught.value) == "Server omitted sampler token evidence"


def test_other_remote_protocol_errors_are_not_retried():
    attempts = []

    async def handle(request):
        attempts.append(request)
        raise httpx.RemoteProtocolError("illegal status line: bytearray(b'broken')")

    async def run():
        engine = VLLMRolloutEngine()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine.http_client = client
            engine._http_client_loop = asyncio.get_running_loop()
            return await engine.generate("display", prompt_token_ids=[10, 11],
                capture_behavior=True, seed=12, temperature=.8, top_p=.95, max_new_tokens=4)

    with pytest.raises(SamplingEvidenceError) as caught:
        asyncio.run(run())
    assert isinstance(caught.value.__cause__, httpx.RemoteProtocolError)
    assert len(attempts) == 1

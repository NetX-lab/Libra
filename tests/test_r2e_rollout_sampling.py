import asyncio

from RL_Framework.workflow.r2e_gym import R2EGymWorkflow


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(message["content"] for message in messages)


class _Engine:
    def __init__(self):
        self.requests = []

    async def generate(self, **kwargs):
        self.requests.append(kwargs)
        return {"text": "[ISSUE] sampled [/ISSUE]", "logprobs": []}


class _CMLFQEvalEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.tool_returns = []
        self.finished = []

    def begin_cmlfq_request(self, prompt_id, input_tokens):
        assert prompt_id == "eval-1"
        assert input_tokens > 0
        return "request-1"

    def route_cmlfq_tool_return(self, request_id, event, generated_tokens):
        self.tool_returns.append((request_id, event, generated_tokens))

    def finish_cmlfq_request(self, request_id, generated_tokens):
        self.finished.append((request_id, generated_tokens))

    def cancel_cmlfq_request(self, request_id):
        raise AssertionError(f"unexpected cancellation: {request_id}")


def test_r2e_rollout_index_produces_distinct_reproducible_sampling_seeds():
    workflow = R2EGymWorkflow(
        reward_fn=lambda **_: 0.5,
        tokenizer=_Tokenizer(),
        max_turns=1,
        max_new_tokens=32,
        max_seq_length=512,
        temperature=0.8,
        top_p=0.95,
    )
    data = {"prompt_id": "sample-42", "task_text": "Describe the failure."}
    engine = _Engine()

    asyncio.run(workflow.run_episode(engine, data, version=3, rollout_index=0))
    asyncio.run(workflow.run_episode(engine, data, version=3, rollout_index=1))
    asyncio.run(workflow.run_episode(engine, data, version=3, rollout_index=0))

    first, second, repeated = [request["seed"] for request in engine.requests]
    assert first != second
    assert first == repeated
    assert all(request["stop"] == ["[/ISSUE]"] for request in engine.requests)
    assert all(
        request["include_stop_str_in_output"] is True
        for request in engine.requests
    )


def test_r2e_eval_shares_episode_budget_and_uses_cmlfq_lifecycle():
    workflow = R2EGymWorkflow(
        reward_fn=lambda **_: 0.0,
        tokenizer=_Tokenizer(),
        max_turns=3,
        max_new_tokens=90,
        max_seq_length=512,
        stop_reward=1.0,
    )
    engine = _CMLFQEvalEngine()
    dataset = [{"prompt_id": "eval-1", "task_text": "Describe the failure."}]

    stats = asyncio.run(
        workflow.evaluate(
            engine,
            dataset,
            max_samples=1,
            concurrency=1,
            max_new_tokens=90,
        )
    )

    assert stats["eval_samples"] == 1
    assert engine.requests
    assert engine.requests[0]["max_new_tokens"] <= 30
    assert engine.requests[0]["request_id"] == "request-1"
    assert engine.tool_returns
    assert engine.finished and engine.finished[0][0] == "request-1"

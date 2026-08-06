import pytest

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


@pytest.mark.asyncio
async def test_r2e_rollout_index_produces_distinct_reproducible_sampling_seeds():
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

    await workflow.run_episode(engine, data, version=3, rollout_index=0)
    await workflow.run_episode(engine, data, version=3, rollout_index=1)
    await workflow.run_episode(engine, data, version=3, rollout_index=0)

    first, second, repeated = [request["seed"] for request in engine.requests]
    assert first != second
    assert first == repeated

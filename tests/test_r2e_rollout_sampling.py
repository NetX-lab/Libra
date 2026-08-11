import pytest

from RL_Framework.config import AsyncRLConfig
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


def test_r2e_stop_reward_loads_from_production_config():
    config = AsyncRLConfig.from_yaml(
        "configs/r2e_gym_qwen3_14b_mcore_gpu_6node48_production_ehp.yaml"
    )

    assert config.r2e_stop_reward == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_r2e_stop_reward_ends_multiturn_rollout(monkeypatch):
    monkeypatch.setattr(
        "RL_Framework.workflow.r2e_gym.evaluate_issue",
        lambda **_: {"reward": 0.6},
    )
    workflow = R2EGymWorkflow(
        reward_fn=lambda **_: 0.6,
        tokenizer=_Tokenizer(),
        max_turns=3,
        max_new_tokens=32,
        max_seq_length=512,
        stop_reward=0.5,
    )
    engine = _Engine()

    trajectory = await workflow.run_episode(
        engine,
        {"prompt_id": "stop-threshold", "task_text": "Describe the failure."},
    )

    assert len(engine.requests) == 1
    assert trajectory["n_turns"] == 1

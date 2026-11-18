import asyncio
import copy
import json

import httpx
import pytest

from RL_Framework.engine.rollout_engine import VLLMRolloutEngine
from RL_Framework.workflow.behavior import SamplingEvidenceError
from RL_Framework.workflow.r2e_gym import R2EGymWorkflow


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text in {"first", "second"}:
            raise AssertionError("Generated text must not be re-encoded")
        return [10, 11] if text == "prompt" else [20, 21, 22]

    def decode(self, tokens, skip_special_tokens=True):
        return "transport text"


class Engine(VLLMRolloutEngine):
    def __init__(self, corrupt=None, outputs=None, revisions=(2, 3)):
        super().__init__()
        self.calls, self.events = [], []
        self.corrupt, self.outputs, self.revisions = corrupt, outputs, revisions

    def begin_cmlfq_request(self, prompt_id, length):
        self.events.append(("begin", prompt_id, length))
        return "trajectory"

    def route_cmlfq_tool_return(self, request_id, event, length):
        self.events.append(("tool", request_id, length))

    def finish_cmlfq_request(self, request_id, length, *, generated_tokens=None):
        self.events.append(("finish", request_id, length))

    def cancel_cmlfq_request(self, request_id):
        self.events.append(("cancel", request_id))

    async def generate(self, **kwargs):
        turn = len(self.calls)
        self.calls.append(copy.deepcopy(kwargs))
        ids, text = (([301, 302], "first") if turn == 0 else ([303], "second")) \
            if self.outputs is None else self.outputs[turn]
        trace = dict(
            logprob_kind="sampler", prompt_token_ids=list(kwargs["prompt_token_ids"]),
            output_token_ids=ids, logprobs=[-0.2] * len(ids),
            sampling_parameters={key: kwargs[key] for key in ("seed", "temperature", "top_p")},
        )
        if self.corrupt:
            self.corrupt(trace)
        revision = self.revisions[turn % len(self.revisions)]
        headers = {} if revision is None else {"X-Libra-Optimizer-Revision": str(revision)}
        body = json.dumps({"choices": [{"text": text, "libra_behavior": trace}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body, headers=headers)
        )) as client:
            self.http_client = client
            self._http_client_loop = asyncio.get_running_loop()
            return await super().generate(**kwargs)


def workflow(monkeypatch, turns=2):
    result = R2EGymWorkflow(
        reward_fn=lambda **kwargs: 0.75, tokenizer=Tokenizer(),
        max_turns=turns, max_new_tokens=16, max_seq_length=512,
        temperature=0.8, top_p=0.95, grpo_objective="decoupled", run_seed=42,
    )
    result._build_initial_prompt = lambda data: "prompt"

    async def score(completion, data):
        return {"reward": 1.0 if completion == "second" else 0.0}

    result._score = score
    monkeypatch.setattr("RL_Framework.workflow.r2e_gym.format_validator_feedback", lambda metrics: "feedback")
    return result


def test_exact_multiturn_prefix_and_behavior_need_only_sampler_fields(monkeypatch):
    engine = Engine()
    row = asyncio.run(workflow(monkeypatch).run_episode(engine, {"prompt_id": "p"}, version=2))
    assert engine.calls[0]["prompt_token_ids"] == [10, 11]
    assert engine.calls[1]["prompt_token_ids"] == [10, 11, 301, 302, 20, 21, 22]
    assert all(call["capture_behavior"] for call in engine.calls)
    assert row["input_ids"].tolist() == [[10, 11, 301, 302, 20, 21, 22, 303]]
    assert row["loss_mask"].tolist() == [[0, 0, 1, 1, 0, 0, 0, 1]]
    assert row["behavior_logprobs"][0, [2, 3, 7]].tolist() == pytest.approx([-0.2] * 3)
    assert row["output_len"] == 3 and row["total_output_tokens"] == 6
    assert row["n_turns"] == 2 and row["rewards"].item() == 0.75
    assert engine.events[-1] == ("finish", "trajectory", 6)
    assert "logprobs" not in row and "behavior_turns" not in row


def test_multiturn_uses_oldest_actual_action_revision_not_submission(monkeypatch):
    engine = Engine(revisions=(3, 4))
    row = asyncio.run(workflow(monkeypatch).run_episode(engine, {"prompt_id": "p"}, version=99))
    assert row["sample_revision"] == 3
    assert type(row["sample_revision"]) is int


def test_workflow_reuses_client_validation_for_each_response(monkeypatch):
    from RL_Framework.workflow import behavior

    checked = []
    validator = behavior.validate_behavior_trace

    def validate(response, *args, **kwargs):
        checked.append(response["sample_revision"])
        return validator(response, *args, **kwargs)

    monkeypatch.setattr(behavior, "validate_behavior_trace", validate)
    engine = Engine(revisions=(2, 3))
    row = asyncio.run(workflow(monkeypatch).run_episode(engine, {"prompt_id": "p"}))
    assert checked == [2, 3]
    assert row["sample_revision"] == 2


@pytest.mark.parametrize("corrupt,reason", [
    (lambda trace: trace["prompt_token_ids"].append(99), "context differs"),
    (lambda trace: trace["sampling_parameters"].update(seed=-1), "sampler parameters"),
    (lambda trace: trace["logprobs"].__setitem__(0, float("nan")), "sampler logprob"),
])
def test_direct_engine_invalid_behavior_trace_is_rejected(monkeypatch, corrupt, reason):
    class DirectEngine:
        async def generate(self, **kwargs):
            trace = {
                "logprob_kind": "sampler",
                "prompt_token_ids": list(kwargs["prompt_token_ids"]),
                "output_token_ids": [301],
                "logprobs": [-0.2],
                "sampling_parameters": {key: kwargs[key] for key in ("seed", "temperature", "top_p")},
            }
            corrupt(trace)
            return {"text": "second", "behavior_trace": trace, "sample_revision": 2}

    model = workflow(monkeypatch)
    with pytest.raises(SamplingEvidenceError, match=reason):
        asyncio.run(model.run_episode(DirectEngine(), {"prompt_id": "p"}))


def test_action_with_missing_actual_revision_is_fatal(monkeypatch):
    engine = Engine(revisions=(None,))
    with pytest.raises(SamplingEvidenceError, match="revision"):
        asyncio.run(workflow(monkeypatch).run_episode(engine, {"prompt_id": "p"}))
    assert engine.events[-1] == ("cancel", "trajectory")


@pytest.mark.parametrize("corrupt", [
    lambda trace: trace["prompt_token_ids"].append(99),
    lambda trace: trace["logprobs"].pop(),
    lambda trace: trace["logprobs"].__setitem__(0, float("nan")),
    lambda trace: trace["sampling_parameters"].__setitem__("seed", -1),
    lambda trace: trace.update(logprob_kind="model"),
    lambda trace: trace.update(output_token_ids=[301] * 17, logprobs=[-0.2] * 17),
])
def test_invalid_sampler_response_cancels_managed_request(monkeypatch, corrupt):
    engine = Engine(corrupt)
    with pytest.raises(SamplingEvidenceError):
        asyncio.run(workflow(monkeypatch).run_episode(engine, {"prompt_id": "p"}))
    assert engine.events[-1] == ("cancel", "trajectory")
    assert not any(event[0] == "finish" for event in engine.events)


def test_transport_capture_receives_actual_revision_with_sampler_fields():
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append((request, payload))
        choice = dict(text="native", logprobs={"tokens": ["string token"], "token_logprobs": [-99.]})
        if request.headers.get("x-libra-capture-behavior") == "1":
            choice["libra_behavior"] = dict(
                logprob_kind="sampler", prompt_token_ids=payload["prompt"],
                output_token_ids=[21], logprobs=[-0.2],
                sampling_parameters={key: payload[key] for key in ("seed", "temperature", "top_p")},
            )
        return httpx.Response(200, json={"choices": [choice]},
            headers={"X-Libra-Optimizer-Revision": "7"})

    async def run():
        engine = VLLMRolloutEngine()
        engine.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        engine._http_client_loop = asyncio.get_running_loop()
        try:
            captured = await engine.generate("display text", prompt_token_ids=[10, 11],
                capture_behavior=True, seed=12, temperature=0.8, top_p=0.95, max_new_tokens=4)
            ordinary = await engine.generate("prefill", temperature=0.0,
                include_stop_str_in_output=True, ignore_eos=True)
            return captured, ordinary
        finally:
            await engine.close()

    captured, ordinary = asyncio.run(run())
    assert captured["sample_revision"] == ordinary["sample_revision"] == 7
    assert captured["tokens"] == [21] and captured["logprobs"] == [-0.2]
    assert ordinary["tokens"] == ["string token"]
    assert requests[0][1]["prompt"] == [10, 11]
    assert requests[0][1]["include_stop_str_in_output"]
    assert "x-libra-capture-behavior" not in requests[1][0].headers
    assert requests[1][1]["include_stop_str_in_output"] and requests[1][1]["ignore_eos"]


def test_legacy_workflow_does_not_require_sampler_fields(monkeypatch):
    result = workflow(monkeypatch, turns=1)
    result.configure_objective("legacy")
    result._encode = lambda text: [1, 2]

    class LegacyEngine:
        async def generate(self, **kwargs):
            assert "capture_behavior" not in kwargs
            return {"text": "ordinary text"}

    row = asyncio.run(result.run_episode(LegacyEngine(), {"prompt_id": "p"}))
    assert "behavior_logprobs" not in row
    assert row["loss_mask"].sum().item() == 2


@pytest.mark.parametrize('version,member,expected', [
    (0, 0, [189670238, 1626177510]),
    (100, 0, [744221753, 214662701]),
    (0, 1, [277167250, 2082093911]),
])
def test_frozen_decoupled_seed_contract(monkeypatch, version, member, expected):
    engine = Engine()
    asyncio.run(workflow(monkeypatch).run_episode(engine, {'prompt_id': 'task'},
        version=version, rollout_index=member))
    assert [call['seed'] for call in engine.calls] == expected


@pytest.mark.parametrize('outputs', [
    [([301], 'first'), ([], '')],
    [([301], 'first'), ([], ''), ([303], 'second')],
], ids=['empty-final', 'empty-middle'])
def test_decoupled_empty_turn_keeps_scoring_feedback_and_finish_order(monkeypatch, outputs):
    engine = Engine(outputs=outputs)
    model = workflow(monkeypatch, turns=len(outputs))

    async def score(text, data):
        engine.events.append(('score', text))
        return {'reward': 0.0}

    def feedback(metrics):
        engine.events.append(('feedback',))
        return 'feedback'

    def reward(**kwargs):
        engine.events.append(('reward', kwargs['completion']))
        return 0.75

    model._score, model._format_feedback, model.reward_fn = score, feedback, reward
    row = asyncio.run(model.run_episode(engine, {'prompt_id': 'p'}))
    expected_ids, expected_mask = [10, 11], [0, 0]
    expected_events = [('begin', 'p', 2)]
    for index, (ids, text) in enumerate(outputs):
        assert engine.calls[index]['prompt_token_ids'] == expected_ids
        expected_ids = expected_ids + ids
        expected_mask += [1] * len(ids)
        expected_events.append(('score', text))
        if index < len(outputs) - 1:
            expected_ids += [20, 21, 22]
            expected_mask += [0, 0, 0]
            expected_events += [('feedback',), ('tool', 'trajectory', len(expected_ids) - 2)]
    expected_events += [('reward', outputs[-1][1]), ('finish', 'trajectory', len(expected_ids) - 2)]
    assert engine.events == expected_events
    assert row['input_ids'].tolist() == [expected_ids]
    assert row['loss_mask'].tolist() == [expected_mask]
    assert row['output_len'] == sum(expected_mask) > 0
    assert row['n_turns'] == len(outputs)
    for index, event in enumerate(row['tool_returns']):
        assert event['payload_tokens'] == 3
        assert event['generated_token_position'] == sum(len(ids) for ids, _ in outputs[:index + 1])
        assert event['remaining_length'] == sum(len(ids) for ids, _ in outputs[index + 1:])


@pytest.mark.parametrize('turns', [1, 2])
def test_decoupled_all_empty_trajectory_is_scored_then_rejected_and_cancelled(monkeypatch, turns):
    engine = Engine(outputs=[([], '')] * turns)
    model = workflow(monkeypatch, turns=turns)
    scored = []

    async def score(text, data):
        scored.append(text)
        return {'reward': 0.0}

    model._score = score
    with pytest.raises(SamplingEvidenceError, match='action|mask/count'):
        asyncio.run(model.run_episode(engine, {'prompt_id': 'p'}))
    assert scored == [''] * turns
    assert engine.events[-1] == ('cancel', 'trajectory')
    assert not any(event[0] == 'finish' for event in engine.events)


@pytest.mark.parametrize('mutate', [
    lambda trace: trace.update(logprobs=[-0.2]),
    lambda trace: trace['prompt_token_ids'].pop(),
])
def test_decoupled_empty_turn_requires_aligned_sampler_fields(monkeypatch, mutate):
    def corrupt(trace):
        if not trace['output_token_ids']:
            mutate(trace)

    engine = Engine(corrupt, outputs=[([301], 'first'), ([], '')])
    with pytest.raises(SamplingEvidenceError):
        asyncio.run(workflow(monkeypatch).run_episode(engine, {'prompt_id': 'p'}))
    assert engine.events[-1] == ('cancel', 'trajectory')


def test_decoupled_feedback_overrun_preserves_sampled_prefix(monkeypatch):
    model = workflow(monkeypatch, turns=2)
    model._format_feedback = lambda metrics: 'oversized'
    original_encode = model._encode
    model._encode = lambda text: [9] * 500 if 'oversized' in text else original_encode(text)
    engine = Engine(outputs=[([301], 'first')])
    row = asyncio.run(model.run_episode(engine, {'prompt_id': 'p'}))
    assert row['input_ids'].tolist() == [[10, 11, 301]]
    assert len(engine.calls) == 1 and not row['tool_returns']
    assert engine.events[-1] == ('finish', 'trajectory', 1)


@pytest.mark.parametrize('sample_revision', [4, 1, None])
def test_trainer_admits_only_fresh_rollouts_with_an_actual_revision(sample_revision):
    from types import SimpleNamespace
    from RL_Framework.infra.execution.async_runner import AsyncTaskRunner
    from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
    from RL_Framework.infra.sync.staleness import StalenessManager
    from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer

    async def run_episode(*args, **kwargs):
        return {'sample_revision': sample_revision}

    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(grpo_objective='decoupled', max_head_offpolicyness=1)
    trainer.workflow = SimpleNamespace(run_episode=run_episode)
    trainer.train_engine = SimpleNamespace(optimizer_revision=5, get_version=lambda: 5)
    trainer.rollout_engine = object()
    trainer.is_main_process = False
    trainer.staleness_manager = StalenessManager(
        version_provider=trainer.train_engine, max_concurrent_rollouts=1, consumer_batch_size=1, max_staleness=1,
    )
    trainer.dispatcher = BatchTaskDispatcher(
        AsyncTaskRunner(max_queue_size=4), trainer.staleness_manager, trainer._create_rollout_task,
    )
    trainer.staleness_manager.on_rollout_enqueued()
    trainer.staleness_manager.on_rollout_submitted()
    task = trainer._create_rollout_task(TaskInput(0, {}, version=5))

    if sample_revision is None:
        with pytest.raises(SamplingEvidenceError, match='actual sampling revision'):
            asyncio.run(task())
        with pytest.raises(RuntimeError, match='actual sampling revision'):
            trainer.dispatcher.wait_results(count=1, timeout=.1)
    else:
        result = asyncio.run(task())
        assert (result is not None) == (sample_revision == 4)

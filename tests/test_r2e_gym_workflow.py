import asyncio

import pytest

from RL_Framework.workflow.r2e_gym import R2EGymWorkflow


class _Tokenizer:
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
    async def generate(self, **kwargs):
        del kwargs
        return {
            "text": (
                "[ISSUE] Title: parser error\nExpected behavior: parse input.\n"
                "Actual error: parser fails in parse_input. [/ISSUE]"
            )
        }


def test_r2e_evaluation_reports_dense_reward_components(monkeypatch):
    monkeypatch.setenv("R2E_EVAL_MULTI_TURN", "0")
    workflow = R2EGymWorkflow(
        reward_fn=lambda **kwargs: 0.0,
        tokenizer=_Tokenizer(),
        max_turns=1,
        max_new_tokens=256,
    )
    dataset = [
        {
            "prompt_id": "row0",
            "prompt": "Describe the parser error.",
            "target_issue": "parser error parse input",
            "expected_output_json": '{"test_parse_input": "failed"}',
            "modified_files": ["src/parser.py"],
        }
    ]

    stats = asyncio.run(
        workflow.evaluate(
            _Engine(),
            dataset,
            max_samples=1,
            accuracy_threshold=0.5,
        )
    )

    assert stats["eval_reward_mean"] > 0.0
    assert stats["eval_index_digest"]
    assert stats["eval_lexical_f1"] > 0.0
    assert stats["eval_test_coverage"] > 0.0
    assert stats["eval_file_coverage"] > 0.0
    assert stats["eval_format_score"] > 0.0


@pytest.mark.parametrize('objective,feedback_tokens', [('legacy', 8), ('decoupled', 10)])
def test_shared_episode_orders_generation_score_feedback_and_finish(objective, feedback_tokens):
    events = []

    class Engine:
        def begin_cmlfq_request(self, prompt_id, length):
            events.append(('begin', length))
            return 'request'

        async def generate(self, **kwargs):
            turn = sum(event[0] == 'generate' for event in events)
            events.append(('generate', turn))
            result = {'text': 'a' if turn == 0 else 'b', 'sample_revision': turn}
            if kwargs.get('capture_behavior'):
                result['behavior_trace'] = dict(logprob_kind='sampler',
                    prompt_token_ids=kwargs['prompt_token_ids'],
                    output_token_ids=[ord(result['text'])], logprobs=[-.2],
                    sampling_parameters={key: kwargs[key] for key in ('seed', 'temperature', 'top_p')})
            return result

        def record_cmlfq_tool_return(self, request_id, event, length):
            events.append(('tool', event['payload_tokens'], length))

        def finish_cmlfq_request(self, request_id, length, *, generated_tokens=None):
            events.append(('finish', length, generated_tokens))

    def reward(**kwargs):
        events.append(('reward', kwargs['completion']))
        return .75

    model = R2EGymWorkflow(reward, _Tokenizer(), max_turns=2,
        max_new_tokens=16, max_seq_length=512, temperature=.8, top_p=.95,
        grpo_objective=objective)
    model._build_initial_prompt = lambda data: 'P'

    async def score(text, data):
        events.append(('score', text))
        return {'reward': 0.0}

    def feedback(metrics):
        events.append(('feedback',))
        return 'feedback'

    model._score, model._format_feedback = score, feedback
    row = asyncio.run(model.run_episode(Engine(), {'prompt_id': 'p'}))
    assert events == [('begin', 1), ('generate', 0), ('score', 'a'),
        ('feedback',), ('tool', feedback_tokens, feedback_tokens + 1),
        ('generate', 1), ('score', 'b'), ('reward', 'b'), ('finish', feedback_tokens + 2, 2)]
    assert row['loss_mask'].tolist() == [[0, 1] + [0] * feedback_tokens + [1]]
    assert row['tool_returns'][0]['remaining_length'] == 1
    assert row['tool_returns'][0]['generated_token_position'] == 1
    assert row['output_len'] == 2


@pytest.mark.parametrize('objective', ['legacy', 'decoupled'])
def test_cancelled_episode_releases_cmlfq_request(objective):
    cancelled = []

    class Engine:
        def begin_cmlfq_request(self, *args):
            return 'request'

        async def generate(self, **kwargs):
            raise asyncio.CancelledError

        def cancel_cmlfq_request(self, request_id):
            cancelled.append(request_id)

    model = R2EGymWorkflow(lambda **kwargs: 0.0, _Tokenizer(),
        max_seq_length=512, grpo_objective=objective)
    model._build_initial_prompt = lambda data: 'P'
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(model.run_episode(Engine(), {'prompt_id': 'p'}))
    assert cancelled == ['request']


def test_legacy_training_and_evaluation_keep_native_stop_payload(monkeypatch):
    import httpx
    import json
    from RL_Framework.engine.rollout_engine import VLLMRolloutEngine

    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'text': 'answer'}]})

    model = R2EGymWorkflow(lambda **kwargs: 0.0, _Tokenizer(), max_turns=1, max_seq_length=512)
    model._build_initial_prompt = lambda data: 'P'
    async def score(text, data):
        return {'reward': 0.0}
    model._score = score
    monkeypatch.setenv('R2E_EVAL_MULTI_TURN', '0')

    async def run():
        engine = VLLMRolloutEngine()
        engine.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        engine._http_client_loop = asyncio.get_running_loop()
        try:
            await model.run_episode(engine, {'prompt_id': 'p'})
            await model.evaluate(engine, [{'prompt_id': 'p'}])
        finally:
            await engine.close()

    asyncio.run(run())
    assert [payload['stop'] for payload in requests] == [['[/ISSUE]'], ['[/ISSUE]']]
    assert all('include_stop_str_in_output' not in payload for payload in requests)


def test_legacy_oversized_initial_prompt_keeps_final_sequence_cap():
    completions = []
    model = R2EGymWorkflow(lambda **kwargs: completions.append(kwargs['completion']) or 0.0,
        _Tokenizer(), max_turns=1, max_seq_length=512)
    model._build_initial_prompt = lambda data: 'P' * 520
    async def score(text, data):
        return {'reward': 0.0}
    model._score = score
    row = asyncio.run(model.run_episode(_Engine(), {'prompt_id': 'p'}))
    assert row['input_ids'].shape == (1, 512)
    assert row['logprobs'].shape == row['loss_mask'].shape == (1, 512)
    assert completions == ['P' * 512]

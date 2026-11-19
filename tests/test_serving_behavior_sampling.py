"""Sampling probabilities and evidence transport without a GPU/vLLM install."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from RL_Framework.scripts.behavior_sampling import (
    BehaviorEvidenceError, install_behavior_response, install_behavior_sampler,
)


def native_masks(logits, k, p):
    sorted_logits, indices = logits.sort(dim=-1)
    if k is not None:
        threshold = sorted_logits.gather(1, (logits.shape[1] - k.long()).unsqueeze(1))
        sorted_logits.masked_fill_(sorted_logits < threshold, -torch.inf)
        k.sub_(1)  # Exercise the mutability present in the native top-k-only path.
    if p is not None:
        discard = sorted_logits.softmax(-1).cumsum(-1) <= 1 - p.unsqueeze(1)
        discard[:, -1] = False
        sorted_logits.masked_fill_(discard, -torch.inf)
    return sorted_logits.scatter(1, indices, sorted_logits)


def sampler_class():
    class TopSampler:
        def forward(self, logits, generators, k, p):
            probabilities = native_masks(logits, k, p).softmax(-1)
            noise = torch.empty_like(probabilities)
            for row, generator in generators.items():
                noise[row].exponential_(generator=generator)
            return probabilities.div_(noise).argmax(-1)

    class Sampler:
        def __init__(self):
            self.topk_topp_sampler = TopSampler()

        def gather_logprobs(self, logprobs, count, token_ids):
            selected = logprobs.gather(1, token_ids.unsqueeze(1))
            return SimpleNamespace(logprobs=torch.cat([selected, logprobs.topk(count).values], 1))

        def forward(self, logits, metadata):
            raw = logits.log_softmax(-1)
            processed = logits.float()
            if not metadata.no_penalties:
                processed[:, 0].sub_(metadata.presence_penalties)
            processed.div_(metadata.temperature.unsqueeze(1))
            chosen = self.topk_topp_sampler.forward(processed, metadata.generators,
                                                    metadata.top_k, metadata.top_p)
            return SimpleNamespace(sampled_token_ids=chosen.int().unsqueeze(1),
                                   logprobs_tensors=self.gather_logprobs(raw, 1, chosen.long()))

    return Sampler


def metadata(temperature=.8, top_p=.95, *, seed=42):
    return SimpleNamespace(all_random=True, generators={0: torch.Generator().manual_seed(seed)},
                           max_num_logprobs=1, temperature=torch.tensor([temperature]),
                           top_k=None, top_p=torch.tensor([top_p]),
                           no_penalties=True, presence_penalties=torch.tensor([0.0]),
                           libra_behavior_rows=(0,))


@pytest.mark.parametrize('top_k,top_p', [(None, None), (None, .95), (3, .6), (3, None)])
def test_sampler_install_keeps_native_ids_and_rng(top_k, top_p):
    cls = sampler_class()
    source = torch.tensor([[2.0, 1.5, 0.0, -0.5, -1.0, -2.0]])
    def params():
        result = metadata()
        result.top_k = None if top_k is None else torch.tensor([top_k])
        result.top_p = None if top_p is None else torch.tensor([top_p])
        return result
    original_params = params()
    original = cls().forward(source.clone(), original_params)
    original_rng = original_params.generators[0].get_state()
    install_behavior_sampler(cls, native_masks)
    install_behavior_sampler(cls, native_masks)
    captured_params = params()
    captured = cls().forward(source.clone(), captured_params)
    assert torch.equal(captured.sampled_token_ids, original.sampled_token_ids)
    assert torch.equal(captured_params.generators[0].get_state(), original_rng)


def test_processed_distribution_has_correct_temperature_penalty_and_top_p():
    cls = sampler_class()
    sampler = cls()
    install_behavior_sampler(cls, native_masks)
    meta = metadata(temperature=.5, top_p=.9)
    meta.no_penalties = False
    meta.presence_penalties[:] = 1.0
    logits = torch.tensor([[3.0, 2.0, 0.0]])
    result = sampler.forward(logits.clone(), meta)
    # Penalty makes first two logits equal, temperature doubles their gap to
    # the final token, and top-p drops the final token: each winner has p=1/2.
    assert result.sampled_token_ids.item() in (0, 1)
    assert result.logprobs_tensors.logprobs[0, 0].item() == pytest.approx(-0.69314718)


@pytest.mark.parametrize('change', [dict(temperature=torch.tensor([0.])), dict(generators={}),
                                   dict(max_num_logprobs=None)])
def test_unsupported_sampling_fails_before_original_random_draw(change):
    cls = sampler_class()
    install_behavior_sampler(cls, native_masks)
    meta = metadata()
    generator = meta.generators[0]
    initial = generator.get_state().clone()
    for key, value in change.items():
        setattr(meta, key, value)
    with pytest.raises(ValueError):
        cls().forward(torch.tensor([[1.0, 2.0, 3.0]]), meta)
    assert torch.equal(generator.get_state(), initial)


def test_top_sampler_restored_after_error():
    cls = sampler_class()
    sampler = cls()
    original = sampler.topk_topp_sampler.forward
    def broken(*_):
        raise RuntimeError('native failure')
    sampler.topk_topp_sampler.forward = broken
    install_behavior_sampler(cls, native_masks)
    with pytest.raises(RuntimeError, match='native failure'):
        sampler.forward(torch.tensor([[1.0, 2.0, 3.0]]), metadata())
    assert sampler.topk_topp_sampler.forward is broken
    sampler.topk_topp_sampler.forward = original


def response_classes():
    class Request:
        def __init__(self, seed=1, **changes):
            self.echo, self.n, self.stream = False, 1, False
            self.seed, self.temperature, self.top_p, self.logprobs = seed, .8, .95, 1
            self.prompt = [3, 4]
            self.__dict__.update(changes)

        def to_sampling_params(self, max_tokens, pattern, defaults):
            return SimpleNamespace(temperature=self.temperature, top_p=self.top_p,
                                   seed=self.seed, logprobs=self.logprobs, max_tokens=max_tokens,
                                   top_k=defaults['top_k'], min_p=defaults['min_p'],
                                   repetition_penalty=defaults['repetition_penalty'])

    class Serving:
        async def create_completion(self, request, raw_request=None):
            request.to_sampling_params(8, None, dict(top_k=20, min_p=.1, repetition_penalty=1.05))
            await asyncio.sleep(0)
            output = SimpleNamespace(token_ids=[99, 2],
                                     logprobs=[{99: SimpleNamespace(logprob=-10000.0)},
                                               {2: SimpleNamespace(logprob=-.5)}])
            final = SimpleNamespace(prompt_token_ids=[3, 4], outputs=[output])
            return self.request_output_to_completion_response([final], request)

        def request_output_to_completion_response(self, final, request):
            return SimpleNamespace(choices=[SimpleNamespace(text='visible')])

    return Serving, Request


def test_response_uses_actual_resolved_defaults_unclamped_probs_and_exact_ids():
    cls, request_cls = response_classes()
    install_behavior_response(cls, request_cls)
    async def run():
        return await asyncio.gather(*(cls().create_completion(request_cls(seed=i),
            raw_request=SimpleNamespace(headers={'x-libra-capture-behavior': '1'})) for i in (1, 2)))
    for seed, result in enumerate(asyncio.run(run()), 1):
        trace = result.choices[0].libra_behavior
        assert trace['output_token_ids'] == [99, 2] and trace['prompt_token_ids'] == [3, 4]
        assert trace['logprobs'] == [-10000.0, -.5]
        assert trace['sampling_parameters']['seed'] == seed
        assert trace['sampling_parameters']['top_k'] == 20
        assert trace['sampling_parameters']['min_p'] == .1
        assert trace['sampling_parameters']['repetition_penalty'] == 1.05


@pytest.mark.parametrize('changes', [dict(seed=None), dict(temperature=0), dict(temperature=float('nan')),
                                    dict(temperature=float('inf')), dict(stream=True),
                                    dict(n=2), dict(prompt='text'), dict(logprobs=None)])
def test_server_prevalidation_prevents_unsupported_worker_requests(changes):
    cls, request_cls = response_classes()
    install_behavior_response(cls, request_cls)
    with pytest.raises(BehaviorEvidenceError):
        asyncio.run(cls().create_completion(request_cls(**changes),
            raw_request=SimpleNamespace(headers={'x-libra-capture-behavior': '1'})))


def test_response_without_capture_context_keeps_ordinary_output():
    cls, request_cls = response_classes()
    install_behavior_response(cls, request_cls)
    result = cls().request_output_to_completion_response([], request_cls())
    assert not hasattr(result.choices[0], 'libra_behavior')

"""Request-scoped sampling evidence across mixed batches and reload identity."""

import asyncio
from types import SimpleNamespace as NS

import pytest
import torch

from RL_Framework.scripts.behavior_sampling import (
    BehaviorEvidenceError, CAPTURE_EXTRA_ARG, install_behavior_batch_metadata,
    install_behavior_response, install_behavior_sampler,
)
from RL_Framework.scripts.serving_protocol import GenerationMiddleware, ServingState


def native_masks(logits, k, p):
    ordered, indices = logits.sort(dim=-1)
    if k is not None:
        cutoff = ordered.gather(1, (logits.shape[1] - k.long()).unsqueeze(1))
        ordered.masked_fill_(ordered < cutoff, -torch.inf)
    if p is not None:
        discard = ordered.softmax(-1).cumsum(-1) <= 1 - p.unsqueeze(1)
        discard[:, -1] = False
        ordered.masked_fill_(discard, -torch.inf)
    return ordered.scatter(1, indices, ordered)


def sampler_class():
    class Top:
        def forward(self, logits, generators, k, p):
            probabilities = native_masks(logits, k, p).softmax(-1)
            noise = torch.empty_like(probabilities)
            if len(generators) != logits.shape[0]:
                noise.exponential_()
            for row, generator in generators.items():
                noise[row].exponential_(generator=generator)
            return probabilities.div_(noise).argmax(-1)

    class Sampler:
        def __init__(self):
            self.topk_topp_sampler = Top()

        def gather_logprobs(self, probabilities, count, token_ids):
            top_values, top_ids = probabilities.topk(count, dim=-1)
            chosen = probabilities.gather(1, token_ids.unsqueeze(1))
            return NS(logprob_token_ids=torch.cat((token_ids.unsqueeze(1), top_ids), 1),
                logprobs=torch.cat((chosen, top_values), 1),
                selected_token_ranks=(probabilities >= chosen).sum(-1))

        def forward(self, logits, metadata):
            raw = logits.log_softmax(-1)
            processed = logits.float().clone()
            processed[:, 0].sub_(metadata.presence_penalties)
            greedy = None if metadata.all_random else processed.argmax(-1)
            if metadata.all_greedy:
                selected = greedy
            else:
                processed.div_(metadata.temperature.unsqueeze(1))
                selected = self.topk_topp_sampler.forward(processed,
                    metadata.generators, metadata.top_k, metadata.top_p)
                if greedy is not None:
                    selected = torch.where(metadata.temperature < 1e-5, greedy, selected)
            logs = (None if metadata.max_num_logprobs is None else
                self.gather_logprobs(raw, metadata.max_num_logprobs, selected))
            return NS(sampled_token_ids=selected.int().unsqueeze(1), logprobs_tensors=logs)

    return Sampler


def mixed_metadata():
    return NS(all_random=False, all_greedy=False,
        generators={0: torch.Generator().manual_seed(42)},
        max_num_logprobs=2, temperature=torch.tensor([.5, -1., .8]),
        top_k=torch.tensor([2, 3, 3]), top_p=torch.tensor([.95, 1., 1.]),
        presence_penalties=torch.tensor([1., 0., .3]), libra_behavior_rows=(0,))


def test_mixed_batch_keeps_native_ids_rng_and_unmarked_raw_probabilities():
    cls = sampler_class()
    logits = torch.tensor([[3., 2., -1.], [0., 4., 1.], [1., 2., 3.]])
    initial_rng = torch.random.get_rng_state()
    baseline_meta = mixed_metadata()
    baseline = cls().forward(logits.clone(), baseline_meta)
    expected_global_rng = torch.random.get_rng_state()
    expected_request_rng = baseline_meta.generators[0].get_state()
    install_behavior_sampler(cls, native_masks)
    torch.random.set_rng_state(initial_rng)
    hooked_meta = mixed_metadata()
    actual = cls().forward(logits.clone(), hooked_meta)
    assert torch.equal(actual.sampled_token_ids, baseline.sampled_token_ids)
    assert torch.equal(torch.random.get_rng_state(), expected_global_rng)
    assert torch.equal(hooked_meta.generators[0].get_state(), expected_request_rng)
    assert torch.equal(actual.logprobs_tensors.logprobs[1:], baseline.logprobs_tensors.logprobs[1:])
    assert torch.equal(actual.logprobs_tensors.logprob_token_ids[1:], baseline.logprobs_tensors.logprob_token_ids[1:])
    assert actual.logprobs_tensors.logprobs[0, 0].item() == pytest.approx(-.69314718)
    assert actual.logprobs_tensors.logprobs[0, 0] != baseline.logprobs_tensors.logprobs[0, 0]


def test_ordinary_greedy_request_needs_neither_seed_nor_logprobs():
    cls = sampler_class()
    install_behavior_sampler(cls, native_masks)
    metadata = NS(all_greedy=True, all_random=False, max_num_logprobs=None,
        temperature=None, generators={}, presence_penalties=torch.tensor([0.]))
    result = cls().forward(torch.tensor([[1., 3., 2.]]), metadata)
    assert result.sampled_token_ids.item() == 1
    assert result.logprobs_tensors is None


def batch_class():
    class Batch:
        def __init__(self):
            self.req_id_to_index = {}
            self.is_spec_decode = False

        def add_request(self, request):
            index = len(self.req_id_to_index)
            self.req_id_to_index[request.req_id] = index
            return index

        def remove_request(self, request_id):
            return self.req_id_to_index.pop(request_id, None)

        def _make_sampling_metadata(self):
            return NS()
    return Batch


def test_capture_rows_follow_request_identity_through_moves_and_removal():
    cls = batch_class()
    batch = cls()
    install_behavior_batch_metadata(cls)
    batch.add_request(NS(req_id='capture', sampling_params=NS(extra_args={CAPTURE_EXTRA_ARG: True},
        seed=42, logprobs=1, temperature=.8)))
    batch.add_request(NS(req_id='ordinary', sampling_params=NS(extra_args=None)))
    assert batch._make_sampling_metadata().libra_behavior_rows == (0,)
    batch.req_id_to_index.update(capture=1, ordinary=0)
    assert batch._make_sampling_metadata().libra_behavior_rows == (1,)
    batch.remove_request('capture')
    assert batch._make_sampling_metadata().libra_behavior_rows == ()
    batch.add_request(NS(req_id='capture', sampling_params=NS(extra_args={})))
    assert batch._make_sampling_metadata().libra_behavior_rows == ()


def test_speculative_capture_is_rejected_before_batch_mutation():
    cls = batch_class()
    install_behavior_batch_metadata(cls)
    batch = cls()
    batch.is_spec_decode = True
    with pytest.raises(BehaviorEvidenceError, match='speculative'):
        batch.add_request(NS(req_id='capture', sampling_params=NS(extra_args={CAPTURE_EXTRA_ARG: True})))
    assert batch.req_id_to_index == {}


def response_classes():
    class Request:
        def __init__(self, *, seed=None, prompt='plain text'):
            self.seed, self.prompt = seed, prompt
            self.echo, self.n, self.stream = False, 1, False
            self.temperature, self.top_p, self.logprobs = .8, .95, 1

        def to_sampling_params(self):
            self.resolved = NS(seed=None if self.seed == -1 else self.seed,
                temperature=self.temperature, logprobs=self.logprobs,
                top_p=self.top_p, extra_args=getattr(self, 'extra_args', {'other_extension': 'preserved'}))
            return self.resolved

    class Serving:
        def __init__(self):
            self.engine_requests = []

        async def create_completion(self, request, raw_request=None):
            self.engine_requests.append(request.to_sampling_params())
            await asyncio.sleep(0)
            final = NS(prompt_token_ids=[7, 8], outputs=[NS(token_ids=[9],
                logprobs=[{9: NS(logprob=-.7)}])])
            return self.request_output_to_completion_response([final], request)

        def request_output_to_completion_response(self, finals, request):
            return NS(choices=[NS(text='response')])
    return Serving, Request


def test_http_opt_in_is_request_local_and_preserves_other_extra_args():
    serving, request_cls = response_classes()
    install_behavior_response(serving, request_cls)
    capture = request_cls(seed=42, prompt=[7, 8])
    ordinary = request_cls()
    async def run():
        return await asyncio.gather(
            serving().create_completion(capture, raw_request=NS(headers={'x-libra-capture-behavior': '1'})),
            serving().create_completion(ordinary, raw_request=NS(headers={})))
    captured, normal = asyncio.run(run())
    trace = captured.choices[0].libra_behavior
    assert trace['prompt_token_ids'] == [7, 8] and trace['output_token_ids'] == [9]
    assert trace['sampling_parameters']['seed'] == 42
    assert not hasattr(normal.choices[0], 'libra_behavior')
    assert capture.resolved.extra_args == {'other_extension': 'preserved', CAPTURE_EXTRA_ARG: True}
    assert ordinary.resolved.extra_args == {'other_extension': 'preserved'}


def test_client_extra_args_cannot_enable_capture_without_header():
    serving, request_cls = response_classes()
    install_behavior_response(serving, request_cls)
    ordinary = request_cls()
    ordinary.extra_args = {CAPTURE_EXTRA_ARG: True, 'other_extension': 'preserved'}
    result = asyncio.run(serving().create_completion(ordinary, raw_request=NS(headers={})))
    assert ordinary.resolved.extra_args == {'other_extension': 'preserved'}
    assert not hasattr(result.choices[0], 'libra_behavior')


def test_resolved_seed_sentinel_is_rejected_before_engine_submission():
    serving_cls, request_cls = response_classes()
    install_behavior_response(serving_cls, request_cls)
    serving = serving_cls()
    capture = request_cls(seed=-1, prompt=[7, 8])
    with pytest.raises(BehaviorEvidenceError, match='parameters'):
        asyncio.run(serving.create_completion(capture,
            raw_request=NS(headers={'x-libra-capture-behavior': '1'})))
    assert serving.engine_requests == []
    assert CAPTURE_EXTRA_ARG not in capture.resolved.extra_args
    normal = asyncio.run(serving.create_completion(request_cls(seed=-1),
        raw_request=NS(headers={})))
    assert len(serving.engine_requests) == 1
    assert serving.engine_requests[0].seed is None
    assert not hasattr(normal.choices[0], 'libra_behavior')


def test_invalid_capture_parameters_do_not_add_request_to_worker_batch():
    cls = batch_class()
    install_behavior_batch_metadata(cls)
    batch = cls()
    with pytest.raises(BehaviorEvidenceError, match='parameters'):
        batch.add_request(NS(req_id='capture', sampling_params=NS(extra_args={CAPTURE_EXTRA_ARG: True},
            seed=None, logprobs=1, temperature=.8)))
    assert not batch.req_id_to_index

"""Opt-in vLLM 0.9.2 hooks for the actual seeded sampling distribution.

The original sampler owns all random draws. These hooks only calculate the
probability of its chosen token after temperature, penalties and truncation.
The hooks are project extensions; installed dependency sources are untouched.
"""

from __future__ import annotations

import math
from contextvars import ContextVar


_SAMPLING_CAPTURE = ContextVar('libra_behavior_sampling', default=None)
CAPTURE_HEADER = 'x-libra-capture-behavior'
CAPTURE_EXTRA_ARG = 'libra_capture_behavior'


class BehaviorEvidenceError(ValueError):
    """Invalid or missing behavior evidence; never downgrade this to a sample."""


def _validate_capture_parameters(params):
    if (getattr(params, 'seed', None) is None
            or getattr(params, 'logprobs', None) is None
            or not isinstance(getattr(params, 'temperature', None), (int, float))
            or not math.isfinite(params.temperature) or params.temperature < 1e-5):
        raise BehaviorEvidenceError('Invalid request-scoped behavior sampling parameters')


def install_behavior_sampler(sampler_class=None, apply_masks=None):
    """Replace probabilities only for explicitly marked V1 batch rows.

    The original sampler still runs once for the entire, potentially mixed
    batch. Unmarked requests retain native logprobs and sampling semantics.
    Marked rows have request-local generators, so vLLM 0.9.2 selects its native
    top-k/top-p implementation even if FlashInfer is installed.
    """
    if sampler_class is None:
        import vllm
        if vllm.__version__.split("+", 1)[0] != "0.9.2":
            raise RuntimeError("Behavior capture supports vLLM 0.9.2")
        from vllm.v1.sample.sampler import Sampler
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        sampler_class, apply_masks = Sampler, apply_top_k_top_p
    if getattr(sampler_class, '_libra_behavior_installed', False):
        return
    if apply_masks is None:
        raise ValueError('An exact native top-k/top-p filter is required')
    original_forward = sampler_class.forward

    def forward(self, logits, sampling_metadata):
        rows = tuple(getattr(sampling_metadata, 'libra_behavior_rows', ()))
        if not rows:
            return original_forward(self, logits, sampling_metadata)
        if (len(rows) != len(set(rows)) or any(type(row) is not int
                or not 0 <= row < logits.shape[0] for row in rows)):
            raise BehaviorEvidenceError('Invalid behavior capture batch rows')
        temperature = getattr(sampling_metadata, 'temperature', None)
        if (getattr(sampling_metadata, 'all_greedy', False) or temperature is None
                or any(row not in sampling_metadata.generators for row in rows)):
            raise BehaviorEvidenceError('Behavior capture requires seeded stochastic request rows')
        import torch
        row_index = torch.tensor(rows, dtype=torch.long, device=logits.device)
        if not getattr(sampling_metadata, 'libra_behavior_rows_validated', False):
            selected_temperature = temperature.index_select(0, row_index)
            if (not torch.isfinite(selected_temperature).all()
                    or (selected_temperature < 1e-5).any()):
                raise BehaviorEvidenceError('Behavior capture requires finite stochastic temperatures')
        if sampling_metadata.max_num_logprobs is None:
            raise BehaviorEvidenceError('Behavior capture requires output logprobs')
        original_top = self.topk_topp_sampler.forward
        captured = []

        def top_forward(processed, generators, k, p):
            if processed.shape[0] != logits.shape[0]:
                raise BehaviorEvidenceError('Unsupported sampling row expansion during behavior capture')
            # index_select already owns the buffers that vLLM may mutate.
            filtered = apply_masks(
                processed.index_select(0, row_index),
                None if k is None else k.index_select(0, row_index),
                None if p is None else p.index_select(0, row_index),
            )
            chosen = original_top(processed, generators, k, p)
            captured.append(filtered.log_softmax(dim=-1, dtype=torch.float32))
            return chosen

        self.topk_topp_sampler.forward = top_forward
        try:
            result = original_forward(self, logits, sampling_metadata)
        finally:
            self.topk_topp_sampler.forward = original_top
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one native sampling call')
        corrected = self.gather_logprobs(
            captured[0], sampling_metadata.max_num_logprobs,
            token_ids=result.sampled_token_ids.reshape(-1).long().index_select(0, row_index))
        if result.logprobs_tensors is None:
            raise BehaviorEvidenceError('Native sampler omitted requested logprobs')
        for field in ('logprob_token_ids', 'logprobs', 'selected_token_ranks'):
            destination = getattr(result.logprobs_tensors, field, None)
            if destination is not None:
                destination.index_copy_(0, row_index, getattr(corrected, field))
        return result

    sampler_class.forward = forward
    sampler_class._libra_behavior_installed = True


def install_behavior_batch_metadata(batch_class=None):
    """Carry request opt-in through vLLM's persistent batch reordering."""
    if batch_class is None:
        from vllm.v1.worker.gpu_input_batch import InputBatch
        batch_class = InputBatch
    if getattr(batch_class, '_libra_behavior_metadata_installed', False):
        return
    original_add = batch_class.add_request
    original_remove = batch_class.remove_request
    original_metadata = batch_class._make_sampling_metadata

    def add(self, request, *args, **kwargs):
        params = request.sampling_params
        marked = bool(params and (getattr(params, 'extra_args', None) or {}).get(CAPTURE_EXTRA_ARG))
        if marked and getattr(self, 'is_spec_decode', False):
            raise BehaviorEvidenceError('Behavior capture does not support speculative decoding')
        if marked:
            _validate_capture_parameters(params)
        index = original_add(self, request, *args, **kwargs)
        requests = getattr(self, '_libra_behavior_requests', None)
        if requests is None:
            requests = self._libra_behavior_requests = set()
        if marked:
            requests.add(request.req_id)
        else:
            requests.discard(request.req_id)
        return index

    def remove(self, req_id, *args, **kwargs):
        result = original_remove(self, req_id, *args, **kwargs)
        getattr(self, '_libra_behavior_requests', set()).discard(req_id)
        return result

    def metadata(self, *args, **kwargs):
        result = original_metadata(self, *args, **kwargs)
        requests = getattr(self, '_libra_behavior_requests', ())
        result.libra_behavior_rows = tuple(sorted(self.req_id_to_index[req_id]
            for req_id in requests if req_id in self.req_id_to_index))
        # Parameters were validated on CPU at admission. Avoid a CUDA scalar
        # synchronization to recheck temperatures on every decode iteration.
        result.libra_behavior_rows_validated = True
        return result

    batch_class.add_request = add
    batch_class.remove_request = remove
    batch_class._make_sampling_metadata = metadata
    batch_class._libra_behavior_metadata_installed = True


def _sampling_parameters(params):
    """Capture the resolved SamplingParams, including model generation defaults."""
    fields = ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty',
              'presence_penalty', 'frequency_penalty', 'seed', 'min_tokens',
              'max_tokens', 'ignore_eos', 'stop', 'stop_token_ids',
              'include_stop_str_in_output', 'skip_special_tokens',
              'spaces_between_special_tokens')
    return {field: list(value) if isinstance(value, list) else value
            for field in fields if (value := getattr(params, field, None)) is not None}


def install_behavior_response(serving_class=None, request_class=None):
    """Expose exact IDs/probabilities and the params actually sent to the engine.

    The ContextVar follows one create_completion coroutine through its real
    to_sampling_params call. Reconstructing params at response time would miss
    processing/defaults and could re-run custom logits-processor constructors.
    """
    if serving_class is None:
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
        serving_class = OpenAIServingCompletion
    if request_class is None:
        from vllm.entrypoints.openai.protocol import CompletionRequest
        request_class = CompletionRequest
    if getattr(serving_class, '_libra_behavior_response_installed', False):
        return
    original = serving_class.request_output_to_completion_response
    original_create = serving_class.create_completion
    original_parameters = request_class.to_sampling_params

    def parameters(self, *args, **kwargs):
        params = original_parameters(self, *args, **kwargs)
        capture = _SAMPLING_CAPTURE.get()
        if capture is not None and capture['request'] is self:
            # vLLM resolves sentinel values here (for example seed=-1 -> None).
            # Reject them in the API before an invalid request reaches EngineCore.
            _validate_capture_parameters(params)
            params.extra_args = dict(getattr(params, 'extra_args', None) or {},
                                     **{CAPTURE_EXTRA_ARG: True})
            capture['parameters'].append(_sampling_parameters(params))
        elif CAPTURE_EXTRA_ARG in (getattr(params, 'extra_args', None) or {}):
            params.extra_args = dict(params.extra_args)
            del params.extra_args[CAPTURE_EXTRA_ARG]
        return params

    async def create(self, request, *args, **kwargs):
        raw_request = kwargs.get('raw_request', args[0] if args else None)
        headers = getattr(raw_request, 'headers', {})
        if headers.get(CAPTURE_HEADER) != '1':
            token = _SAMPLING_CAPTURE.set(None)
            try:
                return await original_create(self, request, *args, **kwargs)
            finally:
                _SAMPLING_CAPTURE.reset(token)
        if (request.echo or request.n != 1 or request.stream or
                getattr(request, 'use_beam_search', False) or
                getattr(request, 'best_of', None) not in (None, 1)):
            raise BehaviorEvidenceError('Behavior capture requires one nonstream stochastic completion')
        if (request.seed is None or request.temperature is None or
                not math.isfinite(request.temperature) or request.temperature < 1e-5):
            raise BehaviorEvidenceError('Behavior capture requires an explicit seed and stochastic temperature')
        if request.top_p is None or not 0 < request.top_p <= 1 or request.logprobs is None:
            raise BehaviorEvidenceError('Behavior capture requires explicit top_p and output logprobs')
        if (not isinstance(request.prompt, list) or not request.prompt or
                any(type(token) is not int or token < 0 for token in request.prompt)):
            raise BehaviorEvidenceError('Behavior capture requires one exact token-list prompt')
        capture = dict(request=request, parameters=[])
        token = _SAMPLING_CAPTURE.set(capture)
        try:
            return await original_create(self, request, *args, **kwargs)
        finally:
            _SAMPLING_CAPTURE.reset(token)

    def response(self, final_res_batch, request, *args, **kwargs):
        capture = _SAMPLING_CAPTURE.get()
        if capture is None:
            return original(self, final_res_batch, request, *args, **kwargs)
        if capture['request'] is not request or len(capture['parameters']) != 1:
            raise BehaviorEvidenceError('Missing resolved engine sampling parameters')
        if len(final_res_batch) != 1 or len(final_res_batch[0].outputs) != 1:
            raise BehaviorEvidenceError('Behavior capture requires exactly one final engine output')
        result = original(self, final_res_batch, request, *args, **kwargs)
        final = final_res_batch[0]
        output = final.outputs[0]
        if final.prompt_token_ids is None:
            raise ValueError('Behavior response omitted actual prompt token IDs')
        if output.logprobs is None or len(output.token_ids) != len(output.logprobs):
            raise BehaviorEvidenceError('Behavior response has unaligned output probabilities')
        try:
            probabilities = [float(step[token].logprob)
                             for token, step in zip(output.token_ids, output.logprobs)]
        except (KeyError, TypeError, AttributeError) as exc:
            raise BehaviorEvidenceError('Behavior response omitted a selected token probability') from exc
        if any(not math.isfinite(lp) or lp > 0 for lp in probabilities):
            raise ValueError('Behavior response has invalid selected sampling probabilities')
        trace = dict(prompt_token_ids=list(final.prompt_token_ids),
                     output_token_ids=list(output.token_ids),
                     logprobs=probabilities, logprob_kind='sampler',
                     sampling_parameters=capture['parameters'][0],
                     temperature=capture['parameters'][0]['temperature'],
                     top_p=capture['parameters'][0]['top_p'])
        if len(result.choices) != 1:
            raise ValueError('Behavior trace count differs from completion count')
        result.choices[0].libra_behavior = trace
        return result

    serving_class.request_output_to_completion_response = response
    serving_class.create_completion = create
    request_class.to_sampling_params = parameters
    serving_class._libra_behavior_response_installed = True

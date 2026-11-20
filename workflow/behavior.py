"""Selected sampler probabilities and the Decoupled training boundary."""

from __future__ import annotations

import math

import torch


class SamplingEvidenceError(ValueError):
    """A sampled action cannot be matched to its probability and context."""


class ValidatedBehaviorTrace(dict):
    """A sampler trace checked by the rollout client against its request."""


def is_decoupled(engine):
    return getattr(engine, "grpo_objective", "legacy") == "decoupled"


def validate_behavior_trace(response, prompt_ids, *, seed, temperature, top_p, output_limit):
    """Check the actual selected-token distribution before extending a prefix."""
    trace = response.get("behavior_trace")
    if not isinstance(trace, dict) or trace.get("logprob_kind") != "sampler":
        raise SamplingEvidenceError("Missing actual sampler behavior trace")
    if trace.get("prompt_token_ids") != prompt_ids:
        raise SamplingEvidenceError("Sampler context differs from the training prefix")
    ids, probabilities = trace.get("output_token_ids"), trace.get("logprobs")
    if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
            or not isinstance(probabilities, list) or len(ids) != len(probabilities)
            or len(ids) > output_limit):
        raise SamplingEvidenceError("Sampler IDs and logprobs are unaligned or exceed the budget")
    if any(type(lp) not in (int, float) or not math.isfinite(lp) or lp > 0 for lp in probabilities):
        raise SamplingEvidenceError("Invalid selected sampler logprob")
    parameters = trace.get("sampling_parameters")
    if (not isinstance(parameters, dict) or parameters.get("seed") != seed
            or parameters.get("temperature") != temperature or parameters.get("top_p") != top_p
            or type(seed) is not int or type(temperature) not in (int, float)
            or not math.isfinite(temperature) or temperature <= 0
            or type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise SamplingEvidenceError("Invalid or mismatched resolved sampler parameters")
    if not isinstance(response.get("text"), str):
        raise SamplingEvidenceError("Sampler response omitted scoring text")
    return trace


def _validate_probability(row, field, *, contents=True):
    values = row.get(field)
    ids, mask = row["input_ids"], row["loss_mask"]
    if (not isinstance(values, torch.Tensor) or values.ndim != 2
            or values.shape[0] != 1 or values.shape[1] > ids.shape[1]):
        raise SamplingEvidenceError(f"Decoupled PPO requires aligned {field}")
    if not contents:
        return
    if mask[:, values.shape[1]:].any():
        raise SamplingEvidenceError(f"An active token is missing its {field}")
    selected = values[mask[:, :values.shape[1]].bool()]
    if not torch.isfinite(selected).all() or (selected > 1e-6).any():
        raise FloatingPointError(f"Active {field} must be finite and nonpositive")


def validate_behavior_rows(trajectories):
    """Check received CPU rows once, before computing their update anchor."""
    for row in trajectories:
        if "anchor_logprobs" in row:
            raise SamplingEvidenceError("Runtime anchor_logprobs is obsolete; recompute the anchor in logprobs")
        ids, mask = row["input_ids"], row["loss_mask"]
        if (ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape
                or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()
                or mask[:, 0].any() or not mask[:, 1:].any()):
            raise SamplingEvidenceError("Decoupled trajectories require aligned IDs and an active binary action mask")
        _validate_probability(row, "behavior_logprobs")
        advantages = row.get("advantages", row.get("rewards"))
        if advantages is not None and not torch.isfinite(advantages).all():
            raise FloatingPointError("Decoupled advantages must be finite")


def validate_anchor_rows(trajectories, optimizer_revision):
    """A fixed anchor belongs to exactly one optimizer revision."""
    for row in trajectories:
        if "anchor_logprobs" in row:
            raise SamplingEvidenceError("Runtime anchor_logprobs is obsolete; recompute the anchor in logprobs")
        _validate_probability(row, "logprobs", contents=False)
        revision = row.get("anchor_optimizer_revision")
        if type(revision) is not int or revision != int(optimizer_revision):
            raise SamplingEvidenceError("Decoupled anchor optimizer revision differs from the update's initial state")

"""Fixed global-batch allocation in whole GRPO prompt groups."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence


def complete_group_sample_counts(global_batch_size: int, n_samples: int, replicas: int) -> list[int]:
    """Balance complete groups, allowing replicas with no assigned group."""
    if n_samples < 1 or replicas < 1 or global_batch_size < 1:
        raise ValueError("batch size, group size and replica count must be positive")
    if global_batch_size % n_samples:
        raise ValueError("global batch size must contain complete GRPO groups")
    groups, remainder = divmod(global_batch_size // n_samples, replicas)
    return [(groups + (index < remainder)) * n_samples for index in range(replicas)]


def partition_complete_grpo_groups(
    trajectories: list[dict[str, Any]], n_samples: int, participant_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Partition one global batch without expanding, duplicating or splitting it."""
    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError("training participant IDs must be unique")
    counts = complete_group_sample_counts(len(trajectories), n_samples, len(participant_ids))
    groups: dict[str, list[dict[str, Any]]] = OrderedDict()
    for trajectory in trajectories:
        groups.setdefault(str(trajectory["grpo_group_id"]), []).append(trajectory)
    if any(len(members) != n_samples for members in groups.values()):
        raise ValueError("global batch contains incomplete or duplicate GRPO groups")
    for members in groups.values():
        indices = [member.get("rollout_index") for member in members]
        if all(index is not None for index in indices) and len(set(indices)) != n_samples:
            raise ValueError("global batch contains duplicate GRPO members")
    ordered = [member for members in groups.values() for member in members]
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for participant, count in zip(participant_ids, counts):
        result[participant] = ordered[offset:offset + count]
        offset += count
    return result

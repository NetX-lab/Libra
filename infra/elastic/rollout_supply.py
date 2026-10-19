"""Inspect already generated GRPO supply without consuming or reordering it."""
from __future__ import annotations


def unique_group_members(members, n_samples=None):
    """Keep the first successful occurrence of each explicit member identity.

    Older workflows without member IDs retain their positional semantics; their
    distinct samples cannot safely be identified by text or tensor equality.
    """
    selected, seen = [], set()
    for member in members:
        if not isinstance(member, dict) or member.get("status") == "failed":
            continue
        index = member.get("grpo_rollout_index", member.get("rollout_index"))
        if index is not None:
            try:
                index = int(index)
            except (ValueError, TypeError):
                continue
            if n_samples is not None and not 0 <= index < n_samples:
                continue
        keys = [("member", index)] if index is not None else []
        if member.get("task_id") is not None:
            keys.append(("task", member["task_id"]))
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        selected.append(member)
    return selected


def ready_complete_samples(trainer) -> int:
    """Count whole groups immediately available across both consumer queues."""
    group_size = int(trainer.config.n_samples)
    if group_size < 1:
        raise ValueError("n_samples must be positive")
    consumed = getattr(trainer, "_consumed_grpo_group_ids", set())
    grouped = {str(group): list(members) for group, members in trainer._pending_grpo_groups.items()
               if str(group) not in consumed}
    dispatcher = getattr(trainer, "dispatcher", None)
    peek = getattr(dispatcher, "peek_ready_data", None)
    for member in peek() if callable(peek) else ():
        if not isinstance(member, dict) or member.get("status") == "failed":
            continue
        group = member.get("grpo_group_id")
        if group is None or str(group) in consumed:
            continue
        grouped.setdefault(str(group), []).append(member)
    return sum(group_size for members in grouped.values()
               if len(unique_group_members(members, group_size)) >= group_size)

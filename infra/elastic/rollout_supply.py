"""Inspect already generated GRPO supply without consuming or reordering it."""
from __future__ import annotations

from RL_Framework.infra.sync.staleness import can_fit_group_deadlines
from .batch_partition import unique_group_members


def ready_complete_samples(trainer) -> int:
    """Count complete groups that can supply the next update without generating."""
    dispatcher = getattr(trainer, "dispatcher", None)
    manager = getattr(dispatcher, "staleness_manager", None)
    if getattr(manager, "group_reservations_enabled", False):
        revision = int(trainer.train_engine.optimizer_revision)
        reservations = manager.get_group_reservations()
        group_count = manager.consumer_batch_size // manager.group_size
        ready = sorted((group for group in reservations if group["ready"]),
                       key=lambda group: group["deadline_revision"])
        chosen = ready[:group_count]
        ids = {group["group_id"] for group in chosen}
        remaining = [group["deadline_revision"] for group in reservations if group["group_id"] not in ids]
        if (len(chosen) < group_count
                or any(revision - group["sample_revision"] > manager.max_staleness for group in chosen)
                or not can_fit_group_deadlines(remaining, revision + 1, group_count)):
            return 0
        return manager.consumer_batch_size

    group_size = int(trainer.config.n_samples)
    if group_size < 1:
        raise ValueError("n_samples must be positive")
    consumed = getattr(trainer, "_consumed_grpo_group_ids", set())
    grouped = {str(group): list(members) for group, members in getattr(trainer, "_pending_grpo_groups", {}).items()
               if str(group) not in consumed}
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

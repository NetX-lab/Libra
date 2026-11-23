"""Support code for Staleness."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from threading import Lock
from typing import Protocol


class VersionProvider(Protocol):
    """Version provider implementation."""

    def get_version(self) -> int: ...


@dataclass
class RolloutStat:
    """Rollout stat implementation."""

    enqueued: int = 0
    running: int = 0
    accepted: int = 0
    rejected: int = 0
    cancelled: int = 0


@dataclass
class GroupReservation:
    group_id: str
    promised_revision: int
    deadline_revision: int
    members: dict[int, str]
    sample_revision: int | None = None
    selected: bool = False


def can_fit_group_deadlines(deadlines: Iterable[int], revision: int, groups_per_batch: int) -> bool:
    """Check if all reservations fit in the batches starting at revision."""
    return all(
        count <= (deadline - revision + 1) * groups_per_batch
        for count, deadline in enumerate(sorted(deadlines), start=1)
    )


class StalenessManager:
    """Staleness manager implementation."""

    def __init__(
        self,
        version_provider: VersionProvider,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
    ):
        self.version_provider = version_provider
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.runtime_max_concurrent_rollouts: int | None = None
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness

        self.lock = Lock()
        self.rollout_stat = RolloutStat()
        self.group_size: int | None = None
        self._groups: dict[str, GroupReservation] = {}
        self.published_revision = int(version_provider.get_version())
        self.optimizer_revision = self.published_revision
        self._admission_open = True

    def get_pending_limit(self) -> int:
        """Get pending limit."""
        return (self.max_staleness + 1) * self.consumer_batch_size

    def get_capacity(self) -> int:
        """Execution concurrency is independent of the group reservation window."""
        with self.lock:
            if self.group_size is None:
                stats = self.rollout_stat
                limit = self.runtime_max_concurrent_rollouts or self.max_concurrent_rollouts
                version = self.version_provider.get_version()
                credit = (self.max_staleness + version + 1) * max(1, self.consumer_batch_size)
                return min(max(1, limit) - stats.running, credit - stats.running - stats.accepted)
            running = sum(state == "running" for group in self._groups.values()
                          for state in group.members.values())
            return max(0, self.max_concurrent_rollouts - running)

    def set_runtime_max_concurrent_rollouts(self, limit: int | None) -> None:
        if limit is not None and limit < 1:
            raise ValueError(f"runtime rollout limit must be positive; got {limit}")
        with self.lock:
            self.runtime_max_concurrent_rollouts = limit

    def on_rollout_enqueued(self) -> None:
        with self.lock:
            self.rollout_stat.enqueued += 1

    def on_rollout_submitted(self) -> None:
        with self.lock:
            self.rollout_stat.enqueued -= 1
            self.rollout_stat.running += 1

    def on_rollout_finished(self, accepted: bool) -> None:
        with self.lock:
            self.rollout_stat.running -= 1
            if accepted:
                self.rollout_stat.accepted += 1
            else:
                self.rollout_stat.rejected += 1

    def on_rollout_accepted(self) -> None:
        self.on_rollout_finished(accepted=True)

    def on_rollout_rejected(self) -> None:
        self.on_rollout_finished(accepted=False)

    def on_rollouts_cancelled(self, *, enqueued: int = 0, running: int = 0) -> None:
        if enqueued < 0 or running < 0:
            raise ValueError("cancelled rollout counts cannot be negative")
        with self.lock:
            self.rollout_stat.enqueued = max(0, self.rollout_stat.enqueued - enqueued)
            self.rollout_stat.running = max(0, self.rollout_stat.running - running)
            self.rollout_stat.cancelled += enqueued + running

    def on_batch_consumed(self, count: int) -> None:
        with self.lock:
            self.rollout_stat.accepted -= count

    @property
    def group_reservations_enabled(self) -> bool:
        return self.group_size is not None

    def configure_groups(self, group_size: int) -> None:
        if group_size < 1 or self.consumer_batch_size % group_size:
            raise ValueError("consumer batch must contain complete groups")
        with self.lock:
            if self.group_size is not None and self.group_size != group_size:
                raise RuntimeError("cannot change the admitted group shape")
            self.group_size = group_size

    def _can_reserve_group(self) -> bool:
        if not self._admission_open:
            return False
        deadline = self.published_revision + self.max_staleness
        if deadline < self.optimizer_revision:
            return False
        groups_per_batch = self.consumer_batch_size // self.group_size
        # Every unretired group, including the currently training batch, owns
        # one slot. Check each deadline so ready younger groups cannot hide a
        # slow older group that must fit in an earlier update.
        deadlines = [group.deadline_revision for group in self._groups.values()] + [deadline]
        return can_fit_group_deadlines(deadlines, self.optimizer_revision, groups_per_batch)

    def can_reserve_group(self) -> bool:
        with self.lock:
            return self.group_size is not None and self._can_reserve_group()

    def reserve_group(self, group_id: str, task_ids: list[int]) -> int | None:
        with self.lock:
            if len(task_ids) != self.group_size or len(set(task_ids)) != len(task_ids):
                raise ValueError("reservation requires one complete group of unique tasks")
            if group_id in self._groups:
                raise ValueError(f"duplicate group reservation: {group_id}")
            if not self._can_reserve_group():
                return None
            promised = self.published_revision
            self._groups[group_id] = GroupReservation(
                group_id, promised, promised + self.max_staleness,
                {task_id: "enqueued" for task_id in task_ids})
            return promised

    def on_member_submitted(self, group_id: str, task_id: int) -> None:
        with self.lock:
            self._groups[group_id].members[task_id] = "running"

    def on_member_completed(self, group_id: str, task_id: int, sample_revision: int) -> None:
        with self.lock:
            group = self._groups.get(group_id)
            if group is None:
                return
            if sample_revision < group.promised_revision:
                raise ValueError("sample revision is below the group's publication promise")
            if sample_revision > self.optimizer_revision:
                raise ValueError("sample revision is ahead of the current optimizer revision")
            group.members[task_id] = "ready"
            group.sample_revision = (sample_revision if group.sample_revision is None
                                     else min(group.sample_revision, sample_revision))

    def terminate_group(self, group_id: str) -> list[int]:
        with self.lock:
            group = self._groups.pop(group_id, None)
            if group is None:
                return []
            self.rollout_stat.rejected += 1
            self.rollout_stat.cancelled += len(group.members) - 1
            return list(group.members)

    def select_groups(self, group_count: int, optimizer_revision: int) -> list[str] | None:
        with self.lock:
            if optimizer_revision != self.optimizer_revision:
                raise RuntimeError("collector revision differs from the reservation window")
            if any(group.selected for group in self._groups.values()):
                raise RuntimeError("previous training batch has not retired")
            ready = [group for group in self._groups.values()
                     if all(state == "ready" for state in group.members.values())]
            if len(ready) < group_count:
                return None
            chosen = sorted(ready, key=lambda group: group.deadline_revision)[:group_count]
            chosen_ids = {group.group_id for group in chosen}
            remaining = (group.deadline_revision for group in self._groups.values()
                         if group.group_id not in chosen_ids)
            if not can_fit_group_deadlines(remaining, optimizer_revision + 1, group_count):
                return None
            for group in chosen:
                if optimizer_revision - group.sample_revision > self.max_staleness:
                    raise ValueError("actual sampled group revision exceeds the configured staleness")
                group.selected = True
            return [group.group_id for group in chosen]

    def get_group_task_ids(self, group_id: str) -> list[int]:
        with self.lock:
            return list(self._groups[group_id].members)

    def retire_batch(self, group_ids: set[str], optimizer_revision: int) -> None:
        with self.lock:
            selected = {group.group_id for group in self._groups.values() if group.selected}
            if group_ids != selected or not selected:
                raise RuntimeError("successful update must retire exactly its selected group reservations")
            if optimizer_revision != self.optimizer_revision + 1:
                raise RuntimeError("successful update must advance optimizer revision once")
            for group_id in selected:
                del self._groups[group_id]
            self.optimizer_revision = optimizer_revision
            self._admission_open = False

    def on_publication(self, revision: int) -> None:
        with self.lock:
            if revision != self.optimizer_revision:
                raise RuntimeError("publication must install the current optimizer revision")
            self.published_revision = revision
            self._admission_open = True

    def get_group_reservations(self) -> list[dict]:
        with self.lock:
            return [{"group_id": group.group_id,
                     "deadline_revision": group.deadline_revision,
                     "ready": all(state == "ready" for state in group.members.values()),
                     "members_total": len(group.members),
                     "members_ready": sum(state == "ready" for state in group.members.values()),
                     "members_running": sum(state == "running" for state in group.members.values()),
                     "sample_revision": (group.sample_revision if group.sample_revision is not None
                                         else group.promised_revision)}
                    for group in self._groups.values()]

    def get_stats(self) -> RolloutStat:
        with self.lock:
            if self.group_size is None:
                return replace(self.rollout_stat)
            states = [state for group in self._groups.values() for state in group.members.values()]
            return RolloutStat(enqueued=states.count("enqueued"), running=states.count("running"),
                               accepted=states.count("ready"), rejected=self.rollout_stat.rejected,
                               cancelled=self.rollout_stat.cancelled)

    def reset_runtime_state(self) -> None:
        """Clear queue accounting after an explicit rollout runtime cutover."""
        with self.lock:
            self._groups.clear()
            self.rollout_stat = RolloutStat(rejected=self.rollout_stat.rejected)

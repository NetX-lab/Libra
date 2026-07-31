"""Support code for Staleness."""

from dataclasses import dataclass, field
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
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness

        self.lock = Lock()
        self.rollout_stat = RolloutStat()

    def get_pending_limit(self) -> int:
        """Get pending limit."""
        return (self.max_staleness + 1) * self.consumer_batch_size

    def get_capacity(self) -> int:
        """Get capacity."""
        with self.lock:
            current_version = self.version_provider.get_version()


            max_concurrent = max(1, self.max_concurrent_rollouts)
            concurrency_capacity = max_concurrent - self.rollout_stat.running


            ofp = self.max_staleness
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            consumer_bs = max(1, self.consumer_batch_size)
            staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

            return min(concurrency_capacity, staleness_capacity)

    def on_rollout_enqueued(self) -> None:
        """On rollout enqueued."""
        with self.lock:
            self.rollout_stat.enqueued += 1

    def on_rollout_submitted(self) -> None:
        """On rollout submitted."""
        with self.lock:
            self.rollout_stat.enqueued -= 1
            self.rollout_stat.running += 1

    def on_rollout_accepted(self) -> None:
        """On rollout accepted."""
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.accepted += 1

    def on_rollout_rejected(self) -> None:
        """On rollout rejected."""
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.rejected += 1

    def on_batch_consumed(self, count: int) -> None:
        """On batch consumed."""
        with self.lock:
            self.rollout_stat.accepted -= count

    def get_stats(self) -> RolloutStat:
        """Return runtime statistics."""
        with self.lock:
            return RolloutStat(
                enqueued=self.rollout_stat.enqueued,
                running=self.rollout_stat.running,
                accepted=self.rollout_stat.accepted,
                rejected=self.rollout_stat.rejected,
            )

    def reset_runtime_state(self) -> None:
        """Clear queue accounting after an explicit rollout runtime cutover."""
        with self.lock:
            self.rollout_stat.enqueued = 0
            self.rollout_stat.running = 0
            self.rollout_stat.accepted = 0

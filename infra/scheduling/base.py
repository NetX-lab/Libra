"""Support code for Base."""

import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """Routing rule implementation."""
    category: str                       # "short" / "medium" / "long" / "extra_long"
    preferred_tp_degrees: list[int]
    fallback_tp_degrees: list[int] = field(default_factory=list)


@dataclass
class SchedulingResult:
    """Scheduling result implementation."""
    instance_index: int
    tp_degree: int
    category: str
    is_fallback: bool
    reason: str = ""
    pending: bool = False
    prompt_id: str = ""
    request_id: str = ""


@dataclass
class SchedulerStats:
    """Scheduler stats implementation."""
    total_requests: int = 0
    preferred_routes: int = 0
    fallback_routes: int = 0
    failed_routes: int = 0
    pending_routes: int = 0
    migrated_routes: int = 0
    category_counts: dict = field(default_factory=lambda: defaultdict(int))
    category_tp_counts: dict = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.preferred_routes + self.fallback_routes) / self.total_requests

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "preferred_routes": self.preferred_routes,
            "fallback_routes": self.fallback_routes,
            "failed_routes": self.failed_routes,
            "pending_routes": self.pending_routes,
            "migrated_routes": self.migrated_routes,
            "success_rate": self.success_rate,
            "category_counts": dict(self.category_counts),
            "category_tp_counts": {
                cat: dict(tp_counts)
                for cat, tp_counts in self.category_tp_counts.items()
            },
        }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class InstanceHandle:
    """Instance handle implementation."""
    index: int
    instance_id: str
    tp_degree: int
    is_ready: bool = True
    active_requests: int = 0

    def inc_active(self):
        self.active_requests += 1

    def dec_active(self):
        self.active_requests = max(0, self.active_requests - 1)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class BaseScheduler(ABC):
    """Base scheduler implementation."""

    def __init__(self, name: str = "base"):
        self.name = name
        self._instances: list[InstanceHandle] = []
        self._instances_by_tp: dict[int, list[InstanceHandle]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stats = SchedulerStats()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def register_instance(self, index: int, instance_id: str, tp_degree: int):
        """Register instance."""
        handle = InstanceHandle(
            index=index,
            instance_id=instance_id,
            tp_degree=tp_degree,
        )
        self._instances.append(handle)
        self._instances_by_tp[tp_degree].append(handle)
        logger.info(
            f"[{self.name}] registered instance {instance_id}: index={index}, TP={tp_degree} "
            f"({len(self._instances)} instances total)"
        )

    def get_instance_handle(self, index: int) -> Optional[InstanceHandle]:
        """Get instance handle."""
        for h in self._instances:
            if h.index == index:
                return h
        return None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @abstractmethod
    def schedule(
        self,
        input_tokens: int,
        prompt_id: str = "",
        n_samples: int = 1,
        epoch: int = -1,
    ) -> SchedulingResult:
        """Schedule."""
        ...

    @abstractmethod
    def on_request_done(
        self,
        instance_index: int,
        prompt_id: str = "",
        final_bucket: str = "",
        output_tokens: int = 0,
    ):
        """On request done."""
        ...

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        """On epoch start."""
        pass

    def on_epoch_end(self, epoch: int):
        """On epoch end."""
        pass

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def get_stats(self) -> SchedulerStats:
        return self._stats

    def reset_stats(self):
        self._stats = SchedulerStats()

    def print_stats(self):
        """Print stats."""
        s = self._stats
        print("\n" + "=" * 60)
        print(f"{self.name} scheduler statistics")
        print("=" * 60)
        print(f"  total requests: {s.total_requests}")
        print(f"  preferred routes: {s.preferred_routes}")
        print(f"  fallback routes: {s.fallback_routes}")
        print(f"  failed: {s.failed_routes}")
        print(f"  pending: {s.pending_routes}")
        print(f"  migrated: {s.migrated_routes}")
        print(f"  success rate: {s.success_rate:.2%}")
        print(f"\n  category distribution:")
        for cat, cnt in sorted(s.category_counts.items()):
            print(f"    {cat}: {cnt}")
        print(f"\n  category-to-TP distribution:")
        for cat, tp_counts in sorted(s.category_tp_counts.items()):
            for tp, cnt in sorted(tp_counts.items()):
                print(f"    {cat} -> TP={tp}: {cnt}")
        print(f"\n  instance status:")
        for h in self._instances:
            print(
                f"    {h.instance_id}: TP={h.tp_degree}, "
                f"active={h.active_requests}, ready={h.is_ready}"
            )
        print("=" * 60)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "BaseScheduler":
        """From config."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_config(), "
            "use SchedulerFactory.create() or a subclass implementation of from_config()"
        )

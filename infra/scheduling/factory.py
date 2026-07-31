"""Support code for Factory."""

import logging
from typing import Any

from RL_Framework.infra.scheduling.base import BaseScheduler

logger = logging.getLogger(__name__)


_SCHEDULER_REGISTRY: dict[str, type[BaseScheduler]] = {}


def register_scheduler(name: str, cls: type[BaseScheduler]):
    """Register scheduler."""
    _SCHEDULER_REGISTRY[name] = cls


def _ensure_registry():
    """Ensure registry."""
    if _SCHEDULER_REGISTRY:
        return

    from RL_Framework.infra.scheduling.length_aware import LengthAwareScheduler
    from RL_Framework.infra.scheduling.la_mlfq import LAMLFQScheduler
    from RL_Framework.infra.scheduling.cmlfq_scheduler import CMLFQScheduler
    from RL_Framework.infra.scheduling.load_balance import LoadBalanceScheduler

    register_scheduler("length_aware", LengthAwareScheduler)
    register_scheduler("la_mlfq", LAMLFQScheduler)
    register_scheduler("cmlfq", CMLFQScheduler)
    register_scheduler("load_balance", LoadBalanceScheduler)


class SchedulerFactory:
    """Scheduler factory implementation."""

    @staticmethod
    def create(
        scheduler_type: str,
        hetero_config: Any,
    ) -> BaseScheduler:
        """Create."""
        _ensure_registry()

        cls = _SCHEDULER_REGISTRY.get(scheduler_type)
        if cls is None:
            available = ", ".join(sorted(_SCHEDULER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown scheduler type: '{scheduler_type}'. "
                f"Available types: {available}"
            )

        scheduler = cls.from_config(hetero_config)
        logger.info(
            f"[SchedulerFactory] Creating scheduler: type={scheduler_type}, "
            f"class={cls.__name__}"
        )
        return scheduler

    @staticmethod
    def available_types() -> list[str]:
        """Available types."""
        _ensure_registry()
        return sorted(_SCHEDULER_REGISTRY.keys())

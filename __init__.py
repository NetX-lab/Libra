"""Support code for Init."""

__version__ = "0.2.0"

from RL_Framework.config import (
    AsyncRLConfig,
    HeterogeneousRolloutConfig,
    HeterogeneousInstanceConfig,
    SchedulingConfig,
    load_config,
    parse_args_and_load_config,
)

__all__ = [
    "AsyncRLConfig",
    "HeterogeneousRolloutConfig",
    "HeterogeneousInstanceConfig",
    "SchedulingConfig",
    "AsyncRLTrainer",
    "HeterogeneousRolloutEngine",
    "load_config",
    "parse_args_and_load_config",
]


def __getattr__(name):
    """Getattr."""
    if name == "AsyncRLTrainer":
        from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
        return AsyncRLTrainer
    if name == "HeterogeneousRolloutEngine":
        from RL_Framework.engine.heterogeneous_engine import (
            HeterogeneousRolloutEngine,
        )
        return HeterogeneousRolloutEngine
    raise AttributeError(name)

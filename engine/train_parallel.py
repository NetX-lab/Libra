"""Support code for Train parallel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch.distributed as dist
except ModuleNotFoundError:
    dist = None


def _validate_positive(name: str, value: int):
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero; got {value}")


@dataclass(frozen=True)
class ParallelCoordinates:
    """Parallel coordinates implementation."""

    dp_rank: int
    pp_rank: int
    tp_rank: int


class TrainParallelRuntime:
    """Train parallel runtime implementation."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
    ):
        _validate_positive("tensor_parallel_size", tensor_parallel_size)
        _validate_positive("pipeline_parallel_size", pipeline_parallel_size)
        _validate_positive("data_parallel_size", data_parallel_size)

        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = (
            self.tensor_parallel_size * self.pipeline_parallel_size
        )

        expected_world = (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
        )
        if expected_world != self.world_size:
            raise ValueError(
                "The 3D parallel topology does not match world_size: "
                f"tp={self.tensor_parallel_size}, "
                f"pp={self.pipeline_parallel_size}, "
                f"dp={self.data_parallel_size}, "
                f"world_size={self.world_size}"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid rank={self.rank}, world_size={self.world_size}")

        self.coords = self.rank_to_coords(
            rank=self.rank,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
        )

        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
        self.model_replica_group = None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @staticmethod
    def rank_to_coords(
        *,
        rank: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> ParallelCoordinates:
        """Rank to coords."""
        mp_size = tensor_parallel_size * pipeline_parallel_size
        dp_rank = rank // mp_size
        intra = rank % mp_size
        pp_rank = intra // tensor_parallel_size
        tp_rank = intra % tensor_parallel_size
        return ParallelCoordinates(
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
        )

    @staticmethod
    def coords_to_rank(
        *,
        dp_rank: int,
        pp_rank: int,
        tp_rank: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> int:
        """Coords to rank."""
        return (
            dp_rank * pipeline_parallel_size * tensor_parallel_size
            + pp_rank * tensor_parallel_size
            + tp_rank
        )

    def get_model_replica_ranks(self, dp_rank: int | None = None) -> list[int]:
        """Get model replica ranks."""
        dp_rank = self.coords.dp_rank if dp_rank is None else dp_rank
        start = dp_rank * self.model_parallel_size
        return list(range(start, start + self.model_parallel_size))

    def get_tp_ranks(
        self,
        *,
        dp_rank: int | None = None,
        pp_rank: int | None = None,
    ) -> list[int]:
        """Get tp ranks."""
        dp_rank = self.coords.dp_rank if dp_rank is None else dp_rank
        pp_rank = self.coords.pp_rank if pp_rank is None else pp_rank
        return [
            self.coords_to_rank(
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
            )
            for tp_rank in range(self.tensor_parallel_size)
        ]

    def get_pp_ranks(
        self,
        *,
        dp_rank: int | None = None,
        tp_rank: int | None = None,
    ) -> list[int]:
        """Get pp ranks."""
        dp_rank = self.coords.dp_rank if dp_rank is None else dp_rank
        tp_rank = self.coords.tp_rank if tp_rank is None else tp_rank
        return [
            self.coords_to_rank(
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
            )
            for pp_rank in range(self.pipeline_parallel_size)
        ]

    def get_dp_ranks(
        self,
        *,
        pp_rank: int | None = None,
        tp_rank: int | None = None,
    ) -> list[int]:
        """Get dp ranks."""
        pp_rank = self.coords.pp_rank if pp_rank is None else pp_rank
        tp_rank = self.coords.tp_rank if tp_rank is None else tp_rank
        return [
            self.coords_to_rank(
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
            )
            for dp_rank in range(self.data_parallel_size)
        ]

    def get_pipeline_rank(
        self,
        *,
        dp_rank: int | None = None,
        pp_rank: int | None = None,
        tp_rank: int | None = None,
    ) -> int:
        """Get pipeline rank."""
        return self.coords_to_rank(
            dp_rank=self.coords.dp_rank if dp_rank is None else dp_rank,
            pp_rank=self.coords.pp_rank if pp_rank is None else pp_rank,
            tp_rank=self.coords.tp_rank if tp_rank is None else tp_rank,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
        )

    def get_prev_pipeline_rank(self) -> int | None:
        if self.is_pipeline_first_stage:
            return None
        return self.get_pipeline_rank(pp_rank=self.pipeline_parallel_rank - 1)

    def get_next_pipeline_rank(self) -> int | None:
        if self.is_pipeline_last_stage:
            return None
        return self.get_pipeline_rank(pp_rank=self.pipeline_parallel_rank + 1)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def initialize_process_groups(self):
        """Initialize process groups."""
        if dist is None or not dist.is_initialized():
            return self

        tp_groups = []
        pp_groups = []
        dp_groups = []
        model_replica_groups = []

        for dp_rank in range(self.data_parallel_size):
            model_replica_groups.append(self.get_model_replica_ranks(dp_rank))

            for pp_rank in range(self.pipeline_parallel_size):
                tp_groups.append(self.get_tp_ranks(dp_rank=dp_rank, pp_rank=pp_rank))

            for tp_rank in range(self.tensor_parallel_size):
                pp_groups.append(self.get_pp_ranks(dp_rank=dp_rank, tp_rank=tp_rank))

        for pp_rank in range(self.pipeline_parallel_size):
            for tp_rank in range(self.tensor_parallel_size):
                dp_groups.append(self.get_dp_ranks(pp_rank=pp_rank, tp_rank=tp_rank))

        self.tp_group = self._create_member_group(tp_groups)
        self.pp_group = self._create_member_group(pp_groups)
        self.dp_group = self._create_member_group(dp_groups)
        self.model_replica_group = self._create_member_group(model_replica_groups)
        return self

    def _create_member_group(self, rank_groups: list[list[int]]):
        """Create member group."""
        current = None
        for ranks in rank_groups:
            if dist is None:
                raise RuntimeError("torch.distributed is unavailable; cannot create a process group")
            group = dist.new_group(ranks=ranks)
            if self.rank in ranks:
                current = group
        return current

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @property
    def data_parallel_rank(self) -> int:
        return self.coords.dp_rank

    @property
    def pipeline_parallel_rank(self) -> int:
        return self.coords.pp_rank

    @property
    def tensor_parallel_rank(self) -> int:
        return self.coords.tp_rank

    @property
    def is_pipeline_first_stage(self) -> bool:
        return self.pipeline_parallel_rank == 0

    @property
    def is_pipeline_last_stage(self) -> bool:
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1

    @property
    def is_tensor_parallel_source(self) -> bool:
        return self.tensor_parallel_rank == 0

    @property
    def is_data_parallel_coordinator(self) -> bool:
        return self.is_pipeline_first_stage and self.is_tensor_parallel_source

    @property
    def is_global_coordinator(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a dictionary."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "train_tp": self.tensor_parallel_size,
            "train_pp": self.pipeline_parallel_size,
            "train_dp": self.data_parallel_size,
            "coords": {
                "dp_rank": self.data_parallel_rank,
                "pp_rank": self.pipeline_parallel_rank,
                "tp_rank": self.tensor_parallel_rank,
            },
            "is_pipeline_first_stage": self.is_pipeline_first_stage,
            "is_pipeline_last_stage": self.is_pipeline_last_stage,
            "is_batch_source": self.is_data_parallel_coordinator,
        }

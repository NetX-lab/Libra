"""Distributed checkpoint and streaming rollout export for Megatron-Core."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist


@dataclass(frozen=True)
class MegatronCheckpointManifest:
    version: int
    checkpoint_format: str
    distributed_checkpoint: str
    rollout_export: str
    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    expert_parallel_size: int
    data_parallel_size: int
    includes_optimizer: bool
    completed_at: float


class MegatronDistributedCheckpointManager:
    """Own MCore sharded checkpoints and memory-bounded HF exports."""

    def __init__(
        self,
        *,
        sync_path: str,
        checkpoint_format: str = "torch_dist",
        fully_parallel_save: bool = True,
        async_save: bool = False,
    ):
        if checkpoint_format != "torch_dist":
            raise ValueError("Only Megatron-Core torch_dist checkpoints are supported")
        self.sync_path = Path(sync_path)
        self.checkpoint_format = checkpoint_format
        self.fully_parallel_save = fully_parallel_save
        self.async_save = async_save
        self._save_strategy = None
        self._load_strategy = None

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def save(
        self,
        *,
        model: list[Any],
        optimizer: Any,
        bridge: Any,
        version: int,
        topology: dict[str, int],
        export_for_rollout: bool = True,
    ) -> Path:
        version_dir = self.sync_path / f"v{version}"
        checkpoint_dir = version_dir / "megatron_dist"
        rollout_export_only = (
            os.environ.get("MEGATRON_ROLLOUT_EXPORT_ONLY", "0") == "1"
        )
        if not rollout_export_only:
            self._mkdir_collective(checkpoint_dir)
        else:
            self._mkdir_collective(version_dir)
        checkpoint_model = self._unwrap_model(model)

        if rollout_export_only:
            self._record_phase(version_dir, version, "dist_checkpoint_skipped")
        else:
            self._record_phase(version_dir, version, "before_sharded_state_dict")
            state_dict = self._build_sharded_state_dict(
                model=checkpoint_model,
                optimizer=optimizer,
                version=version,
            )
            self._record_phase(version_dir, version, "before_dist_checkpoint_save")

            dist_checkpointing = self._dist_checkpointing()
            sharded_strategy = self._get_save_strategy(dist_checkpointing)
            async_request = dist_checkpointing.save(
                sharded_state_dict=state_dict,
                checkpoint_dir=str(checkpoint_dir),
                sharded_strategy=sharded_strategy,
                async_sharded_save=self.async_save,
            )
            if async_request is not None:
                async_request.execute_sync()
            self._barrier()
            self._record_phase(version_dir, version, "after_dist_checkpoint_save")

        if export_for_rollout:
            self._record_phase(version_dir, version, "before_streaming_hf_export")
            bridge.save_hf_pretrained(
                model,
                version_dir,
                show_progress=self.rank == 0,
            )
            self._barrier()
            self._record_phase(version_dir, version, "after_streaming_hf_export")

        manifest = MegatronCheckpointManifest(
            version=version,
            checkpoint_format=self.checkpoint_format,
            distributed_checkpoint=(
                "" if rollout_export_only else str(checkpoint_dir)
            ),
            rollout_export=str(version_dir) if export_for_rollout else "",
            world_size=self.world_size,
            tensor_parallel_size=int(topology.get("train_tp", 1)),
            pipeline_parallel_size=int(topology.get("train_pp", 1)),
            context_parallel_size=int(topology.get("train_cp", 1)),
            expert_parallel_size=int(topology.get("train_ep", 1)),
            data_parallel_size=int(topology.get("train_dp", 1)),
            includes_optimizer=(optimizer is not None and not rollout_export_only),
            completed_at=time.time(),
        )
        if self.rank == 0:
            self._write_json_atomic(
                version_dir / "megatron_checkpoint_manifest.json",
                asdict(manifest),
            )
        self._barrier()
        self._record_phase(version_dir, version, "complete")
        return version_dir

    def load(
        self,
        *,
        model: list[Any],
        optimizer: Any,
        version: int,
    ) -> None:
        version_dir = self.sync_path / f"v{version}"
        checkpoint_dir = version_dir / "megatron_dist"
        manifest_path = version_dir / "megatron_checkpoint_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Incomplete Megatron checkpoint: {manifest_path} does not exist"
            )

        template = self._build_sharded_state_dict(
            model=self._unwrap_model(model),
            optimizer=optimizer,
            version=version,
            is_loading=True,
        )
        dist_checkpointing = self._dist_checkpointing()
        loaded = dist_checkpointing.load(
            sharded_state_dict=template,
            checkpoint_dir=str(checkpoint_dir),
            sharded_strategy=self._get_load_strategy(
                dist_checkpointing,
                checkpoint_dir,
            ),
        )
        self._load_model_state(self._unwrap_model(model), loaded)
        if optimizer is not None and "optimizer" in loaded:
            optimizer.load_state_dict(loaded["optimizer"])
        self._barrier()

    def _build_sharded_state_dict(
        self,
        *,
        model: list[Any],
        optimizer: Any,
        version: int,
        is_loading: bool = False,
    ) -> dict[str, Any]:
        state_dict: dict[str, Any] = {
            "checkpoint_version": 3.0,
            "iteration": version,
        }
        if len(model) == 1:
            state_dict["model"] = model[0].sharded_state_dict()
        else:
            for index, model_chunk in enumerate(model):
                state_dict[f"model{index}"] = model_chunk.sharded_state_dict()

        if optimizer is not None:
            state_dict["optimizer"] = optimizer.sharded_state_dict(
                state_dict,
                is_loading=is_loading,
            )
        return state_dict

    @staticmethod
    def _load_model_state(model: list[Any], state_dict: dict[str, Any]) -> None:
        if len(model) == 1:
            model[0].load_state_dict(state_dict["model"])
            return
        for index, model_chunk in enumerate(model):
            model_chunk.load_state_dict(state_dict[f"model{index}"])

    def _record_phase(
        self,
        version_dir: Path,
        version: int,
        phase: str,
    ) -> None:
        progress_dir = version_dir / "sync_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            progress_dir / f"rank_{self.rank}.json",
            {
                "rank": self.rank,
                "world_size": self.world_size,
                "version": version,
                "phase": phase,
                "timestamp": time.time(),
            },
        )
        print(
            f"[Megatron checkpoint] rank={self.rank} "
            f"version={version} phase={phase}",
            flush=True,
        )

    def _mkdir_collective(self, path: Path) -> None:
        if self.rank == 0:
            path.mkdir(parents=True, exist_ok=True)
        self._barrier()

    @staticmethod
    def _barrier() -> None:
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _dist_checkpointing():
        try:
            from megatron.core import dist_checkpointing
        except ImportError as exc:
            raise RuntimeError(
                "Megatron-Core distributed checkpointing is unavailable. "
                "Install the pinned Megatron dependencies from requirements-megatron.txt."
            ) from exc
        return dist_checkpointing

    def _get_save_strategy(self, dist_checkpointing):
        if not self.fully_parallel_save:
            return None
        if self._save_strategy is None:
            from megatron.core.dist_checkpointing.strategies.fully_parallel import (
                FullyParallelSaveStrategyWrapper,
            )

            base_strategy = (
                dist_checkpointing.serialization.get_default_save_sharded_strategy(
                    self.checkpoint_format,
                    1,
                )
            )
            self._save_strategy = FullyParallelSaveStrategyWrapper(
                base_strategy,
                do_cache_distribution=True,
            )
        return self._save_strategy

    def _get_load_strategy(self, dist_checkpointing, checkpoint_dir: Path):
        if not self.fully_parallel_save:
            return None
        from megatron.core.dist_checkpointing.strategies.fully_parallel import (
            FullyParallelLoadStrategyWrapper,
        )

        base_strategy = (
            dist_checkpointing.serialization.get_default_load_sharded_strategy(
                str(checkpoint_dir)
            )
        )
        self._load_strategy = FullyParallelLoadStrategyWrapper(base_strategy)
        return self._load_strategy

    @staticmethod
    def _unwrap_model(model: list[Any]) -> list[Any]:
        try:
            from megatron.core.utils import unwrap_model
        except ImportError:
            return model
        return unwrap_model(model)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

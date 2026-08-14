"""In-place vLLM weight reload support for Libra rollout workers."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any


class InplaceReloadWorkerExtension:
    """Extend a vLLM GPU worker with checkpoint-based weight replacement."""

    def reload_weights(self, checkpoint_path: str) -> dict[str, Any]:
        """Load a compatible Hugging Face checkpoint into the resident model."""
        import torch

        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(checkpoint / "config.json")
        weight_files = (
            list(checkpoint.glob("*.safetensors"))
            + list(checkpoint.glob("pytorch_model*.bin"))
            + list(checkpoint.glob("*.safetensors.index.json"))
            + list(checkpoint.glob("pytorch_model*.bin.index.json"))
        )
        if not weight_files:
            raise FileNotFoundError(
                f"No Hugging Face weights found in checkpoint: {checkpoint}"
            )

        model_runner = self.model_runner
        previous_path = str(model_runner.model_config.model)
        model_runner.model_config.model = str(checkpoint)
        self.model_config.model = str(checkpoint)
        self.vllm_config.model_config.model = str(checkpoint)

        started_at = time.time()
        started = time.perf_counter()
        try:
            # vLLM's V1 runner reloads weights in place when the model exists.
            model_runner.load_model()
            torch.cuda.synchronize()
        except Exception:
            model_runner.model_config.model = previous_path
            self.model_config.model = previous_path
            self.vllm_config.model_config.model = previous_path
            raise
        finished_at = time.time()
        return {
            "rank": int(getattr(self, "rank", 0)),
            "checkpoint_path": str(checkpoint),
            "previous_path": previous_path,
            "started_at": started_at,
            "finished_at": finished_at,
            "load_seconds": time.perf_counter() - started,
        }

    def reload_weights_nccl(
        self,
        host: str,
        port: int,
        world_size: int,
        rank_offset: int,
        timeout_seconds: float = 1200.0,
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Load full HF-format weights from a Megatron NCCL broadcast."""
        import torch

        from RL_Framework.infra.sync.nccl_weight_sync import (
            NcclReloadSpec,
            receive_weights,
        )

        local_rank = int(getattr(self, "rank", 0))
        spec = NcclReloadSpec(
            host=str(host),
            port=int(port),
            world_size=int(world_size),
            rank=1 + int(rank_offset) + local_rank,
            device=torch.cuda.current_device(),
            timeout_s=float(timeout_seconds),
            chunk_bytes=int(chunk_bytes),
        )
        started = time.perf_counter()
        loaded = self.model_runner.model.load_weights(receive_weights(spec))
        torch.cuda.synchronize()
        return {
            "rank": local_rank,
            "nccl_rank": spec.rank,
            "loaded_parameters": len(loaded) if loaded is not None else 0,
            "load_seconds": time.perf_counter() - started,
        }

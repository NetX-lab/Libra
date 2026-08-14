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

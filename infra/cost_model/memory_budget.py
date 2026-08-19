"""Memory budget helpers shared by GRP planning and runtime preflight."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecomputeLogprobsMemoryEstimate:
    """Conservative per-device temporary-memory estimate."""

    batch_size: int
    sequence_length: int
    local_vocab_size: int
    logits_dtype_bytes: int
    workspace_factor: float
    logits_bytes: int
    total_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "local_vocab_size": self.local_vocab_size,
            "logits_dtype_bytes": self.logits_dtype_bytes,
            "workspace_factor": self.workspace_factor,
            "logits_bytes": self.logits_bytes,
            "total_bytes": self.total_bytes,
        }


def estimate_recompute_logprobs_memory(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    tensor_parallel_size: int,
    logits_dtype_bytes: int = 4,
    workspace_factor: float = 1.5,
) -> RecomputeLogprobsMemoryEstimate:
    """Estimate temporary logits memory for ``recompute_logprobs``.

    MCore computes the normalization in FP32 even when model weights/logits are
    BF16. Tensor parallelism gives each rank only a local vocabulary shard. The
    workspace factor accounts for the FP32 cast and transient shifted tensors;
    the runtime dry-run remains the authoritative check.
    """
    batch = max(1, int(batch_size))
    length = max(1, int(sequence_length) - 1)
    vocab = max(1, int(vocab_size))
    tp = max(1, int(tensor_parallel_size))
    dtype_bytes = max(1, int(logits_dtype_bytes))
    factor = max(1.0, float(workspace_factor))
    local_vocab = (vocab + tp - 1) // tp
    logits_bytes = batch * length * local_vocab * dtype_bytes
    total_bytes = int(logits_bytes * factor)
    return RecomputeLogprobsMemoryEstimate(
        batch_size=batch,
        sequence_length=max(1, int(sequence_length)),
        local_vocab_size=local_vocab,
        logits_dtype_bytes=dtype_bytes,
        workspace_factor=factor,
        logits_bytes=logits_bytes,
        total_bytes=total_bytes,
    )

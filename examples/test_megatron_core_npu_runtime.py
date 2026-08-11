"""Single-rank Ascend runtime preflight for the Megatron-Core backend."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--training-step",
        action="store_true",
        help="also run logprob recomputation, backward, and one optimizer step",
    )
    args = parser.parse_args()

    os.environ.setdefault("DEVICE_BACKEND", "npu")
    os.environ.setdefault("DIST_BACKEND", "hccl")
    engine = MegatronCoreTrainEngine(
        model_path=args.model,
        train_tp_size=1,
        train_pp_size=1,
        train_cp_size=1,
        train_ep_size=1,
        # The production 14B topology uses MCore's distributed optimizer.  It
        # also avoids the non-distributed wrapper's CUDA tensor-name check,
        # which is not portable to torch-npu in MCore 0.14.
        use_distributed_optimizer=args.training_step,
        streaming_export=False,
        use_transformer_engine=False,
        device_backend="npu",
    )
    engine.initialize(
        max_seq_length=args.max_seq_length,
        initialize_optimizer=args.training_step,
    )

    encoded = engine.tokenizer(
        "Ascend Megatron-Core runtime preflight.",
        return_tensors="pt",
    )["input_ids"]
    micro_batch = {
        "input_ids": encoded.to(engine._device()),
    }
    engine._set_train_mode(False)
    with torch.no_grad():
        logits = engine._forward(micro_batch)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Megatron-Core NPU forward produced non-finite logits")
    print(
        "MCORE_NPU_PREFLIGHT_OK "
        f"device={engine._device()} shape={tuple(logits.shape)} "
        f"dtype={logits.dtype}",
        flush=True,
    )
    if args.training_step:
        loss_mask = torch.zeros_like(encoded, dtype=torch.long)
        loss_mask[:, max(1, encoded.shape[1] // 2) :] = 1
        trajectory = {
            "input_ids": encoded,
            "attention_mask": torch.ones_like(encoded),
            "logprobs": torch.zeros_like(encoded, dtype=torch.float32),
            "loss_mask": loss_mask,
            "rewards": torch.tensor([1.0], dtype=torch.float32),
            "advantages": torch.tensor([0.5], dtype=torch.float32),
            "versions": torch.tensor([0], dtype=torch.long),
            "input_len": int((loss_mask == 0).sum()),
            "output_len": int(loss_mask.sum()),
        }
        engine.recompute_logprobs([trajectory])
        stats = engine.grpo_update([trajectory], ppo_epochs=1)
        numeric_stats = {
            name: value
            for name, value in stats.items()
            if isinstance(value, (int, float))
        }
        if not numeric_stats or not all(
            torch.isfinite(torch.tensor(value))
            for value in numeric_stats.values()
        ):
            raise RuntimeError(
                f"Megatron-Core NPU optimizer produced invalid stats: {stats}"
            )
        print(f"MCORE_NPU_TRAIN_STEP_OK stats={stats}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

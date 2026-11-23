"""Verify an actual full-weight Libra GRPO update on the CUDA V4 path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from RL_Framework.config import load_config
from RL_Framework.engine.train_factory import create_train_engine


def run_gate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_path = Path(args.model).resolve(strict=True)
    topology = (
        {"train_cp_size": args.cp_size, "train_dp_size": 0}
        if args.cp_size is not None else {}
    )
    config = replace(config, model_path=str(model_path), tokenizer_path=str(model_path), **topology)
    if int(os.environ.get("WORLD_SIZE", "1")) != config.train_gpus:
        raise ValueError(f"Launch with torchrun on {config.train_gpus} training ranks")

    engine = create_train_engine(config)
    try:
        engine.initialize(max_seq_length=config.max_seq_length)
        local_batch = engine.get_local_batch_size(config.batch_size)
        generator = torch.Generator().manual_seed(42 + engine.get_data_parallel_rank())
        vocab_size = engine.provider.vocab_size
        trajectories = []
        for index in range(local_batch):
            ids = torch.randint(0, vocab_size, (1, config.max_seq_length), generator=generator)
            loss_mask = torch.zeros_like(ids)
            loss_mask[:, -32:] = 1
            value = torch.tensor([1.0 if index % 2 == 0 else -1.0])
            trajectories.append({
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "loss_mask": loss_mask,
                "rewards": value,
                "advantages": value,
            })

        engine.recompute_logprobs(trajectories)
        if any(not torch.isfinite(row["logprobs"]).all() for row in trajectories):
            raise FloatingPointError("Nonfinite recomputed logprobs")
        logprobs = torch.cat([row["logprobs"] for row in trajectories])
        digest = hashlib.sha256(logprobs.contiguous().numpy().tobytes()).hexdigest()
        parallel_state = engine._parallel_state()
        if config.train_cp_size > 1:
            digests = [None] * config.train_cp_size
            dist.all_gather_object(digests, digest, group=parallel_state.get_context_parallel_group())
            if len(set(digests)) != 1:
                raise RuntimeError("Context ranks disagree on the reconstructed logprobs")
        probes = []
        for parameter in engine.model[0].parameters():
            if parameter.requires_grad and parameter.is_contiguous() and parameter.numel():
                view = parameter.detach().view(-1)
                sample = view[::max(1, view.numel() // 64)][:64]
                probes.append((sample, sample.clone()))
        metrics = engine.grpo_update(trajectories)
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("Nonfinite training metrics")
        if metrics["n_samples"] != config.batch_size:
            raise RuntimeError("Each global trajectory must be counted once")
        if engine.optimizer_revision != 1:
            raise RuntimeError("Expected exactly one optimizer update")
        changed = torch.zeros((), dtype=torch.int32, device=engine._device())
        for current, before in probes:
            changed.add_((current != before).any())
        dist.all_reduce(changed, op=dist.ReduceOp.SUM)
        if changed.item() == 0:
            raise RuntimeError("No sampled parameter changed after the optimizer step")

        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        receipt = {
            "rank": dist.get_rank(),
            "cp_size": config.train_cp_size,
            "cp_rank": parallel_state.get_context_parallel_rank(),
            "dp_rank": engine.get_data_parallel_rank(),
            "logprobs_sha256": digest,
            "optimizer_revision": engine.optimizer_revision,
            "changed_parameter_probes": changed.item(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "metrics": metrics,
        }
        (output / f"rank_{dist.get_rank():04d}.json").write_text(json.dumps(receipt) + "\n")
        dist.barrier()
        if engine.is_main_process:
            print(f"LIBRA_DEEPSEEK_V4_CUDA_GATE_PASSED ranks={dist.get_world_size()}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cp-size", type=int, help="Override CP and derive DP from the training allocation")
    parser.add_argument("--model", required=True, help="Local Hugging Face checkpoint directory")
    parser.add_argument("--output", required=True, help="Run-specific shared receipt directory")
    run_gate(parser.parse_args())

"""Fixed offline workload for the DeepSeek V4 128-NPU throughput experiment."""

import json
import os
from pathlib import Path

# Sparse Flash MLA opens a per-rank local listener during model execution.
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HCCL_IF_BASE_PORT"] = str(60000 + local_rank * 100)
os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"

import torch
from transformers import AutoTokenizer

from RL_Framework import AsyncRLTrainer, parse_args_and_load_config
from RL_Framework.workflow.rlvr import RLVRWorkflow


def reward_fn(prompt: str, completion: str, **kwargs) -> float:
    # A deterministic outcome reward keeps the performance test independent of
    # external graders while retaining the full RLVR/GRPO update path.
    return float(sum(map(ord, completion[-16:])) % 2)


def main() -> None:
    config = parse_args_and_load_config()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        n_samples=config.n_samples,
    )
    # Same prompt distribution and order for every policy. Different context
    # lengths exercise the rollout scheduler without reading remote datasets.
    dataset = [
        {
            "question": ("Compute 37 + 58 and give the final integer. " * (1 + (i % 8) * 8)).strip(),
            "answer": "95",
        }
        for i in range(256)
    ]
    trainer = AsyncRLTrainer(config)
    measured_steps = []
    if int(os.environ.get("RANK", "0")) == 0:
        record_step = trainer._record_history_step

        def capture_step(step, batch, stats, *timings):
            measured_steps.append({
                "step": step,
                "step_time_s": float(stats["step_time"]),
                "rollout_time_s": float(stats["rollout_time"]),
                "train_time_s": float(stats["train_time"]),
                "global_trajectories": config.batch_size,
            })
            return record_step(step, batch, stats, *timings)

        trainer._record_history_step = capture_step
    trainer.train(workflow=workflow, dataset=dataset)
    if int(os.environ.get("RANK", "0")) == 0:
        stats = dict(trainer.stats)
        # Exclude the first step's compilation/cache warmup. Planner and EHP
        # decisions can still run during all steps and affect measured steps.
        steady_steps = measured_steps[1:]
        if not steady_steps:
            raise RuntimeError("throughput benchmark needs at least two completed steps")
        steady_time = sum(row["step_time_s"] for row in steady_steps)
        steady_trajectories = sum(row["global_trajectories"] for row in steady_steps)
        rank_ready = Path(config.log_dir) / "rank_ready" / "job_local"
        source_ranks = range(0, config.train_gpus, config.train_tp_size * config.train_pp_size * config.train_cp_size)
        for row in measured_steps:
            output_tokens = 0
            counted_trajectories = 0
            for rank in source_ranks:
                record = json.loads((rank_ready / f"step_{row['step']}" / f"rank_{rank}.json").read_text())
                output_tokens += sum(int(length) for length in record["output_lengths"])
                counted_trajectories += int(record["batch_size"])
            if counted_trajectories != config.batch_size:
                raise RuntimeError(f"step {row['step']} counted {counted_trajectories} trajectories, expected {config.batch_size}")
            row["generated_tokens"] = output_tokens
        steady_tokens = sum(row["generated_tokens"] for row in steady_steps)
        payload = {
            "experiment": os.environ.get("LIBRA_BENCH_ARM", "unknown"),
            "train_gpus": config.train_gpus,
            "rollout_gpus": config.rollout_gpus,
            "global_batch_size": config.batch_size,
            "n_samples": config.n_samples,
            "warmup_steps": 1,
            "measured_steps": len(steady_steps),
            "measured_trajectories": steady_trajectories,
            "measured_generated_tokens": steady_tokens,
            "measured_time_s": steady_time,
            "trajectories_per_s": steady_trajectories / steady_time,
            "generated_tokens_per_s": steady_tokens / steady_time,
            "per_step": measured_steps,
            "rollout_time_s": float(stats["rollout_time"]),
            "train_time_s": float(stats["train_time"]),
            "stats": stats,
        }
        output = Path(os.environ["LIBRA_BENCH_RESULT"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print("LIBRA_128_BENCH_RESULT " + json.dumps(payload, default=str), flush=True)


if __name__ == "__main__":
    main()

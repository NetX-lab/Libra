"""Serve or verify a fixed-checkpoint DeepSeek V4 speculative rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import time


def speculative_config(model: Path, method: str) -> dict:
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "deepseek_v4":
        raise ValueError("Expected a DeepSeek V4 checkpoint")
    block_size = config.get("dspark_block_size")
    if method == "dspark":
        if type(block_size) is not int or block_size < 2:
            raise ValueError("DSpark requires a DeepSeek V4 checkpoint with a DSpark draft")
        return {"method": method, "num_speculative_tokens": max(7, block_size),
                "draft_sample_method": "greedy"}
    if block_size or not config.get("num_nextn_predict_layers"):
        raise ValueError("MTP requires the DeepSeek V4 checkpoint with a classic MTP draft")
    return {"method": "mtp", "num_speculative_tokens": 1}


def serve(args: argparse.Namespace) -> None:
    model = args.model.resolve(strict=True)
    spec = speculative_config(model, args.method)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    missing = sorted({name for name in index["weight_map"].values() if not (model / name).is_file()})
    if missing:
        raise ValueError(f"Incomplete DeepSeek V4 checkpoint, missing shards {missing}")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
        "--served-model-name", args.model_name, "--host", args.host, "--port", str(args.port),
        "--distributed-executor-backend", "mp",
        "--tensor-parallel-size", str(args.tensor_parallel_size), "--enable-expert-parallel",
        "--kv-cache-dtype", "fp8", "--block-size", "256", "--tokenizer-mode", "deepseek_v4",
        "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--generation-config", "vllm", "--speculative-config", json.dumps(spec),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    print(json.dumps({"checkpoint": str(model), "speculative_config": spec,
                      "tensor_parallel_size": args.tensor_parallel_size}), flush=True)
    os.execv(sys.executable, command)


async def speculative_counters(client, url: str) -> dict[str, float]:
    response = await client.get(f"{url}/metrics")
    response.raise_for_status()
    prefix = "vllm:spec_decode_num_"
    counters = dict.fromkeys(("drafts", "draft_tokens", "accepted_tokens"), 0.0)
    for line in response.text.splitlines():
        if not line.startswith(prefix):
            continue
        metric, value, *_ = line.split()
        name = metric.split("{", 1)[0].removeprefix(prefix).removesuffix("_total")
        if name in counters:
            counters[name] += float(value)
    return counters


async def verify(args: argparse.Namespace) -> dict:
    import httpx
    from RL_Framework.engine.rollout_engine import VLLMRolloutEngine

    engine = VLLMRolloutEngine(host=args.host, port=args.port, model_path=args.model_name)
    prompts = (
        "Continue the sequence with the next thirty integers separated by commas. 1, 2, 3, 4,",
        "Continue this Python program with a docstring and the rest of the implementation.\n"
        "def fibonacci(n: int) -> list[int]:\n",
        "The days of the week in order are Monday, Tuesday,",
    )
    async with httpx.AsyncClient(timeout=10) as client:
        before = await speculative_counters(client, engine.base_url)
        started = time.monotonic()
        try:
            rows = await asyncio.gather(*(engine.generate(
                prompt, max_new_tokens=64, temperature=0.7, top_p=0.95,
                seed=seed, ignore_eos=True,
            ) for seed, prompt in enumerate(prompts)))
        finally:
            await engine.close()
        elapsed = time.monotonic() - started
        for row in rows:
            if (not row["text"].strip() or len(row["tokens"]) != 64 or len(row["logprobs"]) != 64
                    or not all(isinstance(lp, (float, int)) and math.isfinite(lp) and lp <= 0
                               for lp in row["logprobs"])):
                raise RuntimeError("Rollout did not return 64 tokens with finite logprobs")

        # Prometheus may collect engine counters after the HTTP reply is sent.
        for _ in range(15):
            after = await speculative_counters(client, engine.base_url)
            delta = {name: after[name] - before[name] for name in before}
            if 0 < delta["accepted_tokens"] <= delta["draft_tokens"] and delta["drafts"] > 0:
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError(f"No accepted speculative tokens recorded by {engine.base_url}: {delta}")

    return {"endpoint": engine.base_url, "model": args.model_name, "requests": len(rows),
            "output_tokens": sum(len(row["tokens"]) for row in rows), "elapsed_seconds": elapsed,
            **delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--method", choices=("mtp", "dspark"), required=True)
    server.add_argument("--model", type=Path, required=True)
    server.add_argument("--tensor-parallel-size", type=int, default=8)
    server.add_argument("--max-model-len", type=int, default=4096)
    server.add_argument("--max-num-seqs", type=int, default=16)
    server.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    server.add_argument("--enforce-eager", action="store_true")
    probe = commands.add_parser("verify")
    for command in (server, probe):
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8018)
        command.add_argument("--model-name", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
    else:
        print(json.dumps(asyncio.run(verify(args)), allow_nan=False))


if __name__ == "__main__":
    main()

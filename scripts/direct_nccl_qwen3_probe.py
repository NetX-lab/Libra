#!/usr/bin/env python3
"""End-to-end Megatron Bridge to two vLLM NCCL reload probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import torch
import torch.distributed as dist

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.infra.sync.nccl_weight_sync import NcclReloadSpec, send_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--served-model-name", default="qwen3-probe")
    parser.add_argument("--transport-host", required=True)
    parser.add_argument("--transport-port", type=int, default=29620)
    parser.add_argument("--train-tp", type=int, default=1)
    parser.add_argument("--train-ep", type=int, default=1)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}: {detail}") from exc


def completion(endpoint: str, served_model_name: str) -> dict:
    body = post_json(
        f"{endpoint}/v1/completions",
        {
            "model": served_model_name,
            "prompt": "Compute 17 * 19. The answer is",
            "max_tokens": 8,
            "temperature": 0.0,
        },
        120.0,
    )
    text = body["choices"][0]["text"]
    return {
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    endpoints = [value.rstrip("/") for value in args.endpoints.split(",")]
    if len(endpoints) != 2:
        raise ValueError("The probe requires exactly two vLLM endpoints")
    if world_size != args.train_tp:
        raise ValueError("torchrun world size must equal --train-tp")

    engine = MegatronCoreTrainEngine(
        model_path=args.model,
        train_tp_size=args.train_tp,
        train_ep_size=args.train_ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=args.train_tp > 1,
        use_distributed_optimizer=False,
        streaming_export=True,
        use_transformer_engine=False,
        use_cpu_initialization=False,
    )
    engine.initialize(max_seq_length=512, initialize_optimizer=False)
    if dist.is_initialized():
        dist.barrier()

    before = None
    futures = None
    executor = None
    if rank == 0:
        before = {
            endpoint: completion(endpoint, args.served_model_name)
            for endpoint in endpoints
        }
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        futures = {
            endpoint: executor.submit(
                post_json,
                f"{endpoint}/reload_weights_nccl",
                {
                    "host": args.transport_host,
                    "port": args.transport_port,
                    "world_size": 3,
                    "rank_offset": index,
                    "timeout_seconds": 1800.0,
                    "chunk_bytes": args.chunk_mb * 1024 * 1024,
                },
                1810.0,
            )
            for index, endpoint in enumerate(endpoints)
        }

    if dist.is_initialized():
        dist.barrier()
    weights = engine.stream_rollout_weights()
    started = time.perf_counter()
    tensor_count = 0
    transferred_bytes = 0
    transport_keepalive: list[object] = []
    if rank == 0:
        def record_tensor(_name: str, size_bytes: int) -> None:
            nonlocal transferred_bytes
            transferred_bytes += size_bytes

        tensor_count = send_weights(
            weights,
            NcclReloadSpec(
                host=args.transport_host,
                port=args.transport_port,
                world_size=3,
                rank=0,
                device=torch.cuda.current_device(),
                timeout_s=1800.0,
                chunk_bytes=args.chunk_mb * 1024 * 1024,
            ),
            on_tensor=record_tensor,
            keepalive=transport_keepalive,
        )
    else:
        for _name, tensor in weights:
            del tensor
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        send_finished = time.perf_counter()
        responses = {
            endpoint: future.result()
            for endpoint, future in futures.items()
        }
        executor.shutdown(wait=True)
        transport_keepalive.clear()
        total_seconds = time.perf_counter() - started
        after = {
            endpoint: completion(endpoint, args.served_model_name)
            for endpoint in endpoints
        }
        payload = {
            "model": args.model,
            "train_tp": args.train_tp,
            "train_ep": args.train_ep,
            "tensor_count": tensor_count,
            "transferred_bytes": transferred_bytes,
            "fanout_bytes": transferred_bytes * 2,
            "total_seconds": total_seconds,
            "send_seconds": send_finished - started,
            "payload_gib_per_second": (
                transferred_bytes / (1024**3) / total_seconds
            ),
            "before": before,
            "after": after,
            "outputs_stable": before == after,
            "responses": responses,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

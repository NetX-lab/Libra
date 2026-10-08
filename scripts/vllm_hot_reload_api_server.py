#!/usr/bin/env python3
"""Run vLLM's OpenAI server with a guarded in-place reload endpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from fastapi import HTTPException, Request
from pydantic import BaseModel
import uvloop

from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser


class ReloadWeightsRequest(BaseModel):
    """Payload accepted by the private rollout weight reload endpoint."""

    checkpoint_path: str
    timeout_seconds: float = 300.0


class ReloadWeightsNcclRequest(BaseModel):
    """Payload for a direct Megatron-to-vLLM NCCL reload."""

    host: str
    port: int
    world_size: int
    rank_offset: int
    timeout_seconds: float = 1200.0
    chunk_bytes: int = 268435456


_reload_lock = asyncio.Lock()


@api_server.router.post("/reload_weights")
async def reload_weights(payload: ReloadWeightsRequest, request: Request):
    """Replace resident worker weights after Libra has drained rollout work."""
    checkpoint = Path(payload.checkpoint_path).resolve()
    if not checkpoint.is_dir():
        raise HTTPException(status_code=400, detail=f"Missing checkpoint: {checkpoint}")

    async with _reload_lock:
        started = time.perf_counter()
        client = api_server.engine_client(request)
        try:
            results = await client.collective_rpc(
                "reload_weights",
                timeout=payload.timeout_seconds,
                args=(str(checkpoint),),
            )
            # Prefix-cache entries encode outputs from the previous parameters.
            await client.reset_prefix_cache()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "checkpoint_path": str(checkpoint),
            "total_seconds": time.perf_counter() - started,
            "workers": results,
        }


@api_server.router.post("/reload_weights_nccl")
async def reload_weights_nccl(
    payload: ReloadWeightsNcclRequest,
    request: Request,
):
    """Receive the next model directly from the Megatron GPU stream."""
    async with _reload_lock:
        started = time.perf_counter()
        client = api_server.engine_client(request)
        try:
            results = await client.collective_rpc(
                "reload_weights_nccl",
                timeout=payload.timeout_seconds,
                args=(
                    payload.host,
                    payload.port,
                    payload.world_size,
                    payload.rank_offset,
                    payload.timeout_seconds,
                    payload.chunk_bytes,
                ),
            )
            await client.reset_prefix_cache()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "total_seconds": time.perf_counter() - started,
            "workers": results,
        }


if __name__ == "__main__":
    api_server.cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI server with Libra hot weight reload support."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))

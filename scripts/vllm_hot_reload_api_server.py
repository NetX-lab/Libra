#!/usr/bin/env python3
"""Run vLLM's OpenAI server with a guarded in-place reload endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Request
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
_custom_router = APIRouter()


def _kv_directory(key: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise HTTPException(status_code=400, detail="invalid KV key")
    return Path(os.environ.get("LIBRA_KV_STORAGE_PATH", "/dev/shm/libra-kv")) / key


@_custom_router.get("/libra/kv/{key}")
async def cpu_offload_status(key: str, source_tp: int = 1):
    directory = _kv_directory(key)
    if not 1 <= source_tp <= 64:
        raise HTTPException(status_code=400, detail="invalid source TP")
    if (directory / "released").exists():
        raise HTTPException(status_code=410, detail="KV released")
    manifests = []
    for rank in range(source_tp):
        error = directory / f"rank_{rank}.error.json"
        if error.exists():
            raise HTTPException(status_code=500, detail=error.read_text())
        path = directory / f"rank_{rank}.json"
        if not path.exists():
            return {"ready": False}
        manifests.append(json.loads(path.read_text()))
    return {"ready": True, "workers": [
        {k: v for k, v in manifest.items() if k != "tokens"} for manifest in manifests
    ]}


@_custom_router.delete("/libra/kv/{key}")
async def release_cpu_offload(key: str):
    directory = _kv_directory(key)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "released").touch()
    for path in directory.glob("rank_*"):
        path.unlink(missing_ok=True)
    return {"released": True}


@_custom_router.post("/reload_weights")
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


@_custom_router.post("/reload_weights_nccl")
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


# vLLM 0.9.x constructs a fresh FastAPI app in build_app() and includes its
# module router there.  Adding routes to api_server.router alone is fragile
# across V1 initialization; explicitly attach our private router to the
# concrete app after vLLM has built it.
_build_app = api_server.build_app


def build_app_with_reload_routes(args):
    app = _build_app(args)
    app.include_router(_custom_router)
    return app


api_server.build_app = build_app_with_reload_routes


if __name__ == "__main__":
    api_server.cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI server with Libra hot weight reload support."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))

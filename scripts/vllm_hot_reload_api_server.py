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
from RL_Framework.infra.sync.prefix_cache_reset import (
    install_checked_prefix_cache_reset, reset_prefix_cache_verified,
)
from RL_Framework.scripts.serving_protocol import ServingState, GenerationMiddleware

# EngineCore subprocesses import this project entrypoint before utility RPCs.
install_checked_prefix_cache_reset()


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
    snapshot_id: str | None = None
    optimizer_revision: int | None = None
    expected_tensor_count: int | None = None


_reload_lock = asyncio.Lock()
_custom_router = APIRouter()


@_custom_router.get('/publication/status')
@_custom_router.get('/libra/kv/identity')
async def publication_status(request: Request):
    return request.app.state.serving_state.status()


@_custom_router.post('/libra/kv/lease')
async def acquire_kv_lease(request: Request):
    body = await request.body()
    lease_id = json.loads(body).get('lease_id') if body else None
    if lease_id is not None and (not isinstance(lease_id, str) or not re.fullmatch('[a-f0-9]{32}', lease_id)):
        raise HTTPException(status_code=400, detail='Invalid KV lease ID')
    try:
        return await request.app.state.serving_state.acquire_kv_lease(lease_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@_custom_router.delete('/libra/kv/lease/{lease_id}')
async def release_kv_lease(lease_id: str, request: Request):
    await request.app.state.serving_state.release_kv_lease(lease_id)
    return {'released': True}


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
        state = request.app.state.serving_state
        await state.begin_reload(payload.timeout_seconds)
        client = api_server.engine_client(request)
        try:
            results = await client.collective_rpc(
                "reload_weights", timeout=payload.timeout_seconds if payload.timeout_seconds > 0 else None,
                args=(str(checkpoint),),
            )
            cleared = await reset_prefix_cache_verified(client,
                require_verified=os.environ.get('LIBRA_CHECK_PREFIX_CACHE') == '1')
            state.install(snapshot_id=None, optimizer_revision=None, workers=results)
            await state.reopen()
        except BaseException as exc:
            await state.fail(repr(exc))
            raise
        return dict(checkpoint_path=str(checkpoint), total_seconds=time.perf_counter()-started,
                    workers=results, prefix_cache_cleared=cleared is True,
                    service_generation=state.server_generation, service_instance_id=state.server_instance_id)


@_custom_router.post("/reload_weights_nccl")
async def reload_weights_nccl(
    payload: ReloadWeightsNcclRequest,
    request: Request,
):
    """Receive the next model directly from the Megatron GPU stream."""
    async with _reload_lock:
        started = time.perf_counter()
        state = request.app.state.serving_state
        if ((payload.snapshot_id is None) != (payload.optimizer_revision is None)
                or payload.optimizer_revision is not None and payload.optimizer_revision < state.optimizer_revision):
            raise HTTPException(status_code=409, detail='Missing or regressing snapshot identity')
        await state.begin_reload(payload.timeout_seconds)
        client = api_server.engine_client(request)
        try:
            results = await client.collective_rpc(
                "reload_weights_nccl", timeout=payload.timeout_seconds if payload.timeout_seconds > 0 else None,
                args=(payload.host, payload.port, payload.world_size, payload.rank_offset,
                      payload.timeout_seconds, payload.chunk_bytes)
                     + ((payload.expected_tensor_count,) if payload.expected_tensor_count is not None else ()),
            )
            cleared = await reset_prefix_cache_verified(client,
                require_verified=bool(payload.snapshot_id) or os.environ.get('LIBRA_CHECK_PREFIX_CACHE') == '1')
            state.install(snapshot_id=payload.snapshot_id, optimizer_revision=payload.optimizer_revision,
                          workers=results, rank_offset=payload.rank_offset,
                          expected_tensor_count=payload.expected_tensor_count)
            await state.reopen()
        except BaseException as exc:
            await state.fail(repr(exc))
            raise
        return dict(total_seconds=time.perf_counter()-started, workers=results,
                    prefix_cache_cleared=cleared is True, service_generation=state.server_generation,
                    service_instance_id=state.server_instance_id, snapshot_id=state.snapshot_id,
                    optimizer_revision=state.optimizer_revision)


# vLLM 0.9.x constructs a fresh FastAPI app in build_app() and includes its
# module router there.  Adding routes to api_server.router alone is fragile
# across V1 initialization; explicitly attach our private router to the
# concrete app after vLLM has built it.
_build_app = api_server.build_app


def make_serving_state(args):
    """Build the service state shared by generation and weight reload."""
    return ServingState(args.tensor_parallel_size, args.model)


def build_app_with_reload_routes(args):
    app = _build_app(args)
    app.state.serving_state = make_serving_state(args)
    app.include_router(_custom_router)
    app.add_middleware(GenerationMiddleware, state=app.state.serving_state)
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

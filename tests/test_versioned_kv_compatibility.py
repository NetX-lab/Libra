"""Version/incarnation fencing through real KV clients and the native reload gate."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import httpx
from fastapi import FastAPI, HTTPException, Request
import pytest
import torch

from RL_Framework.engine.cmlfq_backend import CMLFQGenerationBackend
from RL_Framework.engine.rollout_engine import VLLMRolloutEngine
from RL_Framework.infra.scheduling.kv_identity import same_snapshot, same_incarnation
from RL_Framework.scripts.serving_protocol import ServingState, GenerationMiddleware
from test_cpu_offload_backend import FakeCPUBackend


def peer(revision=7, before_completion=None):
    state = ServingState(1, "same-model")
    state.snapshot_id, state.optimizer_revision = f"snapshot-{revision}", revision
    app, calls = FastAPI(), []

    @app.post("/libra/kv/lease")
    async def acquire(request: Request):
        try:
            return await state.acquire_kv_lease((await request.json()).get("lease_id"))
        except RuntimeError as error:
            raise HTTPException(409, str(error))

    @app.delete("/libra/kv/lease/{lease_id}")
    async def release(lease_id: str):
        await state.release_kv_lease(lease_id)
        return {"released": True}

    @app.get("/libra/kv/identity")
    async def identity():
        return state.status()

    @app.post("/v1/completions")
    async def complete(request: Request):
        calls.append(await request.json())
        if before_completion:
            await before_completion()
        return dict(choices=[dict(text="answer", logprobs={})], kv_transfer_params=dict(
            remote_engine_id="source", remote_request_id="request", remote_host="host",
            remote_port=5555, remote_block_ids=[[1, 2]]))

    app.add_middleware(GenerationMiddleware, state=state)
    engine = VLLMRolloutEngine(host="test")
    engine.http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    engine._http_client_loop = asyncio.get_running_loop()
    return engine, state, calls


def test_identity_distinguishes_shared_weights_and_service_incarnations():
    identity = dict(snapshot_id="shared", optimizer_revision=7, server_instance_id="a", server_generation=3)
    other = dict(identity, server_instance_id="b")
    assert same_snapshot(identity, other)
    assert not same_incarnation(identity, other)
    assert not same_incarnation(identity, dict(identity, server_generation=4))
    assert not same_snapshot(identity, dict(identity, optimizer_revision=8))
    assert not same_snapshot({}, {})


@pytest.mark.parametrize("target_revision", [7, 8])
def test_nixl_same_snapshot_transfers_but_new_revision_recomputes(target_revision):
    async def run():
        source, source_state, source_calls = peer()
        target, target_state, target_calls = peer(target_revision)
        backend = CMLFQGenerationBackend("nixl", [(1, 4)])
        output = await backend.generate(source, target, "full prompt", "transfer", prompt_token_ids=[1, 2, 3])
        assert output.path == ("transfer" if target_revision == 7 else "recompute")
        assert bool(source_calls) == (target_revision == 7)
        assert target_calls[-1]["prompt"] == [1, 2, 3]
        assert bool(target_calls[-1].get("kv_transfer_params")) == (target_revision == 7)
        assert not source_state.kv_leases and not target_state.kv_leases
        await source.close()
        await target.close()
    asyncio.run(run())


@pytest.mark.parametrize("change,reason", [("revision", "snapshot_mismatch"), ("generation", "incarnation_mismatch"),
                                           ("restart", "incarnation_mismatch")])
def test_cpu_old_cache_recomputes_after_reload_or_restart(change, reason):
    async def run():
        source, target = FakeCPUBackend(), FakeCPUBackend()
        source.ready.set()
        target.ready.set()
        backend = CMLFQGenerationBackend("cpu_offload", [(1, 4)])
        await backend.generate(source, source, "prompt", "stay", "request")
        if change == "revision":
            target.revision += 1
        elif change == "generation":
            source.generation += 1
        else:
            source.instance_id = "restarted"
        output = await backend.generate(source, target, "prompt+tool", "transfer", "request")
        assert output.path == "recompute" and output.fallback_reason == reason
        assert "libra_load_key" not in target.calls[-1]
        assert not source.leases and not target.leases
        await backend.close()
    asyncio.run(run())


def test_cancelled_cpu_wait_releases_both_source_and_target_leases():
    async def run():
        source, target = FakeCPUBackend(), FakeCPUBackend()
        backend = CMLFQGenerationBackend("cpu_offload", [(1, 4)])
        await backend.generate(source, source, "prompt", "stay", "request")
        task = asyncio.create_task(backend.generate(source, target, "prompt+tool", "transfer", "request"))
        await asyncio.sleep(0)
        assert source.leases and target.leases
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not source.leases and not target.leases
        await backend.close()
    asyncio.run(run())


def load_connector(monkeypatch):
    name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    base = ModuleType(name)
    base.KVConnectorBase_V1, base.KVConnectorMetadata = object, object
    base.KVConnectorRole = SimpleNamespace(WORKER="worker")
    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, name, base)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    path = Path(__file__).parents[1] / "infra/scheduling/vllm_cpu_offload.py"
    module_name = "RL_Framework.infra.scheduling.kv_compat_connector_fixture"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def connector_fixture(monkeypatch, tmp_path, source_tp=2, target_tp=4, rank=0):
    module = load_connector(monkeypatch)
    connector = module.LibraCPUOffloadConnector.__new__(module.LibraCPUOffloadConnector)
    connector.root, connector.tp, connector.rank = str(tmp_path), target_tp, rank
    connector.block_size, connector.num_heads = 2, 8
    connector.matches, connector.imported = {}, {}
    key = "a" * 32
    directory = tmp_path / key
    directory.mkdir(exist_ok=True)
    source_identity = dict(snapshot_id="snapshot", optimizer_revision=7, server_instance_id="source", server_generation=3)
    target_identity = dict(source_identity, server_instance_id="target", server_generation=8)
    manifest = dict(tokens=list(range(4)), source_tp=source_tp, block_size=2, libra_kv_identity=source_identity)
    for source_rank in range(source_tp):
        (directory / f"rank_{source_rank}.json").write_text(json.dumps(manifest))
    request = SimpleNamespace(kv_transfer_params=dict(libra_load_key=key, libra_kv_identity=target_identity,
        libra_kv_source_identity=source_identity), request_id="request", prompt_token_ids=list(range(5)))
    return module, connector, directory, manifest, request


def test_cpu_connector_checks_all_tp_source_incarnations(monkeypatch, tmp_path):
    _, connector, directory, manifest, request = connector_fixture(monkeypatch, tmp_path)
    assert connector.get_num_new_matched_tokens(request, 0) == (4, False)
    changed = dict(manifest, libra_kv_identity=dict(manifest["libra_kv_identity"], server_generation=4))
    (directory / "rank_1.json").write_text(json.dumps(changed))
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)


@pytest.mark.parametrize("source_tp,target_tp", [(1, 2), (2, 1), (2, 4), (4, 2)])
def test_cpu_real_tensor_head_resharding_keeps_identity_checks(monkeypatch, tmp_path, source_tp, target_tp):
    module, connector, directory, _, request = connector_fixture(monkeypatch, tmp_path, source_tp, target_tp)
    full = torch.arange(2 * 2 * 2 * 8 * 4, dtype=torch.float32).reshape(2, 2, 2, 8, 4)
    for rank, shard in enumerate(full.chunk(source_tp, dim=-2)):
        torch.save({"layer": shard.contiguous()}, directory / f"rank_{rank}.pt")
    # Exercise the production file loading/head slicing/index-copy code on CPU;
    # CUDA DMA and pinned transfers remain covered by the GPU regression.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value: value)
    module.torch = SimpleNamespace(tensor=lambda values, **kwargs: torch.tensor(values, dtype=kwargs.get("dtype")),
                                   long=torch.long, load=torch.load, cat=torch.cat)
    for rank in range(target_tp):
        connector.rank = rank
        destination = torch.zeros(2, 3, 2, 8 // target_tp, 4)
        connector.caches = {"layer": destination}
        connector._reload(dict(key=directory.name, source_tp=source_tp, num_tokens=4, block_ids=[2, 0],
                               **request.kv_transfer_params))
        torch.testing.assert_close(destination[:, [2, 0]], full.chunk(target_tp, dim=-2)[rank])


@pytest.mark.parametrize("unknown", [None, [], "snapshot", {}, {"snapshot_id": "snapshot", "optimizer_revision": True}])
def test_unknown_kv_identities_are_rejected_without_crashing(unknown):
    identity = dict(snapshot_id="snapshot", optimizer_revision=1, server_instance_id="s", server_generation=0)
    assert not same_snapshot(unknown, identity)
    assert not same_snapshot(identity, unknown)
    assert not same_incarnation(unknown, identity)


def test_worker_fences_changed_incarnation_before_touching_cuda(monkeypatch, tmp_path):
    _, connector, directory, manifest, request = connector_fixture(monkeypatch, tmp_path)
    assert connector.get_num_new_matched_tokens(request, 0) == (4, False)
    changed = dict(manifest, libra_kv_identity=dict(manifest["libra_kv_identity"], server_instance_id="restarted"))
    (directory / "rank_1.json").write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="identity changed"):
        connector._reload(dict(key=directory.name, source_tp=2, num_tokens=4, block_ids=[0, 1],
                               **request.kv_transfer_params))


def test_cancelled_lease_response_withdraws_server_reservation():
    async def run():
        state = ServingState(1)
        async def handler(request):
            if request.method == "POST":
                await state.acquire_kv_lease(json.loads(request.content)["lease_id"])
                raise asyncio.CancelledError()
            await state.release_kv_lease(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={"released": True})
        engine = VLLMRolloutEngine(host="test")
        engine.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        engine._http_client_loop = asyncio.get_running_loop()
        with pytest.raises(asyncio.CancelledError):
            await engine.acquire_kv_lease()
        assert not state.kv_leases
        await engine.close()
    asyncio.run(run())

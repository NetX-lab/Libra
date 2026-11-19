import asyncio
import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
from starlette.middleware.cors import CORSMiddleware

from RL_Framework.infra.scheduling import vllm_cpu_offload as cpu_offload
from RL_Framework.infra.scheduling.vllm_cpu_offload import LibraCPUOffloadConnector
from RL_Framework.scripts import vllm_hot_reload_api_server as server
from RL_Framework.scripts.serving_protocol import GenerationMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "libra-test-model"


def bare_connector():
    connector = object.__new__(LibraCPUOffloadConnector)
    connector.block_size = 16
    connector.num_heads = 8
    connector.tp = 1
    return connector


def install_layer(connector, layer):
    connector._vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
    )


def test_unknown_vllm_version_uses_backend_capabilities(monkeypatch):
    class Backend:
        @classmethod
        def get_kv_cache_shape(cls, num_blocks, block_size, num_heads, head_size):
            return (2, num_blocks, block_size, num_heads, head_size)

    connector = bare_connector()
    install_layer(connector, SimpleNamespace(attn_backend=Backend))
    monkeypatch.setattr(vllm, "__version__", "99.0.0-capability-check", raising=False)
    monkeypatch.setattr(cpu_offload.torch.cuda, "Stream", lambda: object())

    native = torch.empty(2, 2, 16, 8, 4)
    connector.register_kv_caches({"layer": native})

    assert connector.caches["layer"].stride() == native.stride()
    assert connector.caches["layer"].data_ptr() == native.data_ptr()


def test_legacy_selector_resolves_layer_without_attn_backend(monkeypatch):
    class Impl:
        pass

    class Backend:
        @classmethod
        def get_impl_cls(cls):
            return Impl

        @classmethod
        def get_kv_cache_shape(cls, num_blocks, block_size, num_heads, head_size):
            return (num_blocks, 2, block_size, num_heads, head_size)

    calls = []

    def get_attn_backend(*args, **kwargs):
        calls.append((args, kwargs))
        return Backend

    attention_module = ModuleType("vllm.attention")
    selector_module = ModuleType("vllm.attention.selector")
    selector_module.get_attn_backend = get_attn_backend
    attention_module.selector = selector_module
    monkeypatch.setitem(sys.modules, "vllm.attention", attention_module)
    monkeypatch.setitem(sys.modules, "vllm.attention.selector", selector_module)
    monkeypatch.setattr(vllm, "attention", attention_module, raising=False)
    monkeypatch.setattr(cpu_offload.torch.cuda, "Stream", lambda: object())

    connector = bare_connector()
    install_layer(connector, SimpleNamespace(
        head_size=4, dtype=torch.float16, kv_cache_dtype="auto", impl=Impl(),
    ))
    native = torch.empty(2, 2, 16, 8, 4)
    connector.register_kv_caches({"layer": native})

    assert calls == [((4, torch.float16, "auto", 16), {"is_attention_free": False})]
    assert connector.caches["layer"].stride() == (
        native.stride(1), native.stride(0), *native.stride()[2:],
    )


def test_legacy_selector_rejects_mismatched_implementation(monkeypatch):
    class ExpectedImpl:
        pass

    class OtherImpl:
        pass

    class Backend:
        @classmethod
        def get_impl_cls(cls):
            return ExpectedImpl

    attention_module = ModuleType("vllm.attention")
    selector_module = ModuleType("vllm.attention.selector")
    selector_module.get_attn_backend = lambda *args, **kwargs: Backend
    attention_module.selector = selector_module
    monkeypatch.setitem(sys.modules, "vllm.attention", attention_module)
    monkeypatch.setitem(sys.modules, "vllm.attention.selector", selector_module)
    monkeypatch.setattr(vllm, "attention", attention_module, raising=False)

    connector = bare_connector()
    install_layer(connector, SimpleNamespace(
        head_size=4, dtype=torch.float16, kv_cache_dtype="auto", impl=OtherImpl(),
    ))
    with pytest.raises(ValueError, match="backend implementation mismatch") as raised:
        connector.register_kv_caches({"layer": torch.empty(2, 2, 16, 8, 4)})
    assert isinstance(raised.value.__cause__, TypeError)
    assert "layer=layer" in str(raised.value)
    assert "backend=Backend" in str(raised.value)
    assert "vLLM=" in str(raised.value)


def test_backend_shape_probe_failure_has_layer_context_and_cause(monkeypatch):
    class Backend:
        @classmethod
        def get_kv_cache_shape(cls, num_blocks, block_size, num_heads, head_size):
            return (num_blocks, block_size, num_heads, head_size)

    connector = bare_connector()
    install_layer(connector, SimpleNamespace(attn_backend=Backend))
    with pytest.raises(ValueError, match="unsupported backend KV shape probe") as raised:
        connector.register_kv_caches({"layer": torch.empty(2, 2, 16, 8, 4)})
    assert isinstance(raised.value.__cause__, TypeError)
    for detail in ("layer=layer", "backend=Backend", "shape=(2, 2, 16, 8, 4)", "vLLM="):
        assert detail in str(raised.value)


def _flash_attention_backend():
    failures = []
    for module_name in (
        "vllm.v1.attention.backends.flash_attn",
        "vllm.attention.backends.flash_attn",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(exc)
            continue
        backend = getattr(module, "FlashAttentionBackend", None)
        if backend is not None:
            return backend
    raise AssertionError(f"FlashAttentionBackend import failed: {failures!r}")


def test_real_flash_attention_backend_shape_probe_and_cuda_registration():
    backend = _flash_attention_backend()
    block_size, num_heads, head_size = 16, 8, 128
    shape2 = tuple(backend.get_kv_cache_shape(2, block_size, num_heads, head_size))
    shape3 = tuple(backend.get_kv_cache_shape(3, block_size, num_heads, head_size))
    old_layout = (2, 3, block_size, num_heads, head_size)
    new_layout = (3, 2, block_size, num_heads, head_size)
    if shape3 == old_layout:
        layout = "old"
        assert shape2 == (2, 2, block_size, num_heads, head_size)
    else:
        assert shape3 == new_layout
        layout = "new"
        assert shape2 == (2, 2, block_size, num_heads, head_size)

    device = f"cuda:{torch.cuda.current_device()}"
    for num_blocks, native_shape in ((2, shape2), (3, shape3)):
        connector = bare_connector()
        connector._vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={
                "layer": SimpleNamespace(attn_backend=backend),
            }),
        )
        native = torch.empty(native_shape, dtype=torch.bfloat16, device=device)
        connector.register_kv_caches({"layer": native})

        assert connector.caches["layer"].shape == (
            2, num_blocks, block_size, num_heads, head_size,
        )
        assert connector.caches["layer"].data_ptr() == native.data_ptr()
        if layout == "old":
            assert connector.caches["layer"].stride() == native.stride()
        else:
            assert connector.caches["layer"].stride() == (
                native.stride(1), native.stride(0), *native.stride()[2:],
            )


def _serving_app():
    parser = server.make_arg_parser(server.FlexibleArgumentParser())
    args = parser.parse_args(["--model", MODEL_PATH])
    args.tensor_parallel_size = 1
    build_parameters = inspect.signature(server._build_app).parameters
    kwargs = {"supported_tasks": ("generate",)} if "supported_tasks" in build_parameters else {}
    return server.build_app_with_reload_routes(args, **kwargs)


def test_server_builds_private_routes_and_reports_publication_status():
    import httpx

    app = _serving_app()
    routes = app.openapi()["paths"].keys()
    assert {
        "/publication/status", "/reload_weights", "/reload_weights_nccl",
        "/libra/kv/{key}",
    } <= routes

    middleware = {item.cls for item in app.user_middleware}
    assert GenerationMiddleware in middleware
    assert CORSMiddleware in middleware
    scaling_middleware = getattr(server.api_server, "ScalingMiddleware", None)
    if scaling_middleware is not None:
        assert scaling_middleware in middleware

    async def fetch_status():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            return await client.get("/publication/status")

    response = asyncio.run(fetch_status())
    assert response.status_code == 200
    assert response.json()["tp_size"] == 1
    assert response.json()["server_generation_verified"] is True


def test_checked_prefix_reset_is_asserted_only_for_installed_092():
    if vllm.__version__.split("+")[0] == "0.9.2":
        engine_core = importlib.import_module("vllm.v1.engine.core")
        assert getattr(engine_core.EngineCore, "_libra_checked_prefix_reset", False)


def test_server_cli_help_exits_successfully():
    script = PROJECT_ROOT / "scripts" / "vllm_hot_reload_api_server.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

"""Run in the supported vLLM environment; GPU tests use real CUDA tensors."""

import json
import tempfile
import uuid
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from RL_Framework.infra.scheduling import vllm_cpu_offload as cpu_offload
from RL_Framework.infra.scheduling.vllm_cpu_offload import LibraCPUOffloadConnector, cache_directory


def bare_connector(root, tp=1, rank=0):
    connector = object.__new__(LibraCPUOffloadConnector)
    connector.root, connector.tp, connector.rank = root, tp, rank
    connector.block_size, connector.num_heads = 16, 8
    connector.matches, connector.imported = {}, {}
    connector.pending = {}
    connector.loads = []
    connector.verify_reload = False
    return connector


def test_prefix_identity_and_complete_block_boundaries():
    with tempfile.TemporaryDirectory() as root:
        key = uuid.uuid4().hex
        directory = cache_directory(root, key)
        directory.mkdir()
        (directory / "rank_0.json").write_text(json.dumps({
            "tokens": list(range(64)), "source_tp": 1, "block_size": 16,
        }))
        connector = bare_connector(root)
        request = SimpleNamespace(kv_transfer_params={"libra_load_key": key},
                                  request_id="r", prompt_token_ids=list(range(65)))
        assert connector.get_num_new_matched_tokens(request, 16) == (48, False)
        request.prompt_token_ids[1] = 999
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        request.prompt_token_ids = list(range(64))
        # Must recompute at least one token for the next logits.
        assert connector.get_num_new_matched_tokens(request, 0) == (48, False)


def test_largest_matching_complete_block_prefix():
    with tempfile.TemporaryDirectory() as root:
        key = uuid.uuid4().hex
        directory = cache_directory(root, key)
        directory.mkdir()
        (directory / "rank_0.json").write_text(json.dumps({
            "tokens": list(range(64)), "source_tp": 1, "block_size": 16,
        }))
        connector = bare_connector(root)
        prompt_tokens = list(range(65))
        prompt_tokens[20] = 999
        request = SimpleNamespace(
            kv_transfer_params={"libra_load_key": key},
            request_id="r", prompt_token_ids=prompt_tokens,
        )
        assert connector.get_num_new_matched_tokens(request, 0) == (16, False)
        assert connector.matches["r"]["num_tokens"] == 16


def test_allocation_keeps_local_prefix_offset():
    connector = bare_connector("/tmp", tp=1, rank=0)
    connector.matches["r"] = {
        "key": "k", "source_tp": 1, "num_tokens": 48,
        "num_local_tokens": 16, "request_id": "r",
    }
    blocks = SimpleNamespace(get_block_ids=lambda: ([10, 11, 12],))
    connector.update_state_after_alloc(
        SimpleNamespace(request_id="r"), blocks, num_external_tokens=32,
    )
    assert connector.loads[0]["num_local_tokens"] == 16
    assert connector.loads[0]["block_ids"] == [10, 11, 12]


def test_constructor_supports_optional_cache_config_and_rejects_multiple_groups(
    monkeypatch, tmp_path,
):
    base_calls = []

    def base_init(self, *args):
        base_calls.append(args)

    monkeypatch.setattr(cpu_offload.KVConnectorBase_V1, "__init__", base_init)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda name, default=None:
                str(tmp_path) if name == "storage_path" else default,
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_key_value_heads=8)),
    )
    role = object()

    LibraCPUOffloadConnector(config, role)
    assert base_calls[-1] == (config, role)

    one_group = SimpleNamespace(kv_cache_groups=[object()])
    LibraCPUOffloadConnector(config, role, one_group)
    assert base_calls[-1] == (config, role, one_group)

    multiple_groups = SimpleNamespace(kv_cache_groups=[object(), object()])
    with pytest.raises(ValueError, match="exactly one KV cache group"):
        LibraCPUOffloadConnector(config, role, multiple_groups)


@pytest.mark.parametrize("layout", ["old", "new"])
def test_register_accepts_native_layout_as_zero_copy_view(monkeypatch, layout):
    connector = bare_connector("/tmp")
    native = native_strided_cache(torch.empty(2, 3, 16, 8, 4), layout)
    stream = object()
    monkeypatch.setattr(cpu_offload.torch.cuda, "Stream", lambda: stream)

    connector.register_kv_caches({"layer": native})

    canonical = connector.caches["layer"]
    assert canonical.shape == (2, 3, 16, 8, 4)
    assert canonical.data_ptr() == native.data_ptr()
    assert not canonical.is_contiguous()
    assert connector.stream is stream


def test_allocation_rejects_external_token_count_mismatch():
    connector = bare_connector("/tmp")
    connector.matches["r"] = {
        "key": "k", "source_tp": 1, "num_tokens": 48,
        "num_local_tokens": 16, "request_id": "r",
    }
    blocks = SimpleNamespace(get_block_ids=lambda: ([10, 11],))

    with pytest.raises(RuntimeError, match="does not match the available prefix"):
        connector.update_state_after_alloc(
            SimpleNamespace(request_id="r"), blocks, num_external_tokens=16,
        )
    assert connector.loads == []
    assert "r" not in connector.matches


@pytest.mark.parametrize(
    "shape, message",
    [
        ((3, 4, 16, 8, 4), "first two axes"),
        ((2, 3, 15, 8, 4), "block_size"),
        ((2, 3, 16, 7, 4), "local_heads"),
        ((2, 3, 16, 8), "five-dimensional"),
    ],
)
def test_register_rejects_invalid_cache_shape(shape, message):
    connector = bare_connector("/tmp")
    with pytest.raises(ValueError, match=message) as raised:
        connector.register_kv_caches({"layer": torch.empty(shape)})
    detail = str(raised.value)
    assert "layer=layer" in detail and "backend=" in detail
    assert f"shape={shape}" in detail and "vLLM=" in detail


def test_register_rejects_ambiguous_layout_without_layer_metadata():
    connector = bare_connector("/tmp")
    with pytest.raises(ValueError, match="ambiguous leading axes") as raised:
        connector.register_kv_caches({"layer": torch.empty(2, 2, 16, 8, 4)})
    assert isinstance(raised.value.__cause__, AttributeError)
    for detail in ("layer=layer", "backend=unresolved", "shape=(2, 2, 16, 8, 4)", "vLLM="):
        assert detail in str(raised.value)


def native_strided_cache(canonical, layout="new"):
    native_shape = canonical if layout == "old" else canonical.movedim(0, 1)
    backing = torch.empty(
        (*native_shape.shape[:-1], native_shape.shape[-1] * 2),
        dtype=canonical.dtype, device=canonical.device,
    )
    native = backing[..., ::2]
    native.copy_(native_shape)
    return native


def native_cache_block(native, layout, block):
    return native[:, block] if layout == "old" else native[block]


def install_backend_metadata(connector, layout):
    class Impl:
        pass

    class Backend:
        @classmethod
        def get_impl_cls(cls):
            return Impl

        @classmethod
        def get_kv_cache_shape(cls, num_blocks, block_size, num_heads, head_size):
            if layout == "old":
                return (2, num_blocks, block_size, num_heads, head_size)
            return (num_blocks, 2, block_size, num_heads, head_size)

    layer = SimpleNamespace(attn_backend=Backend, impl=Impl())
    connector._vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
    )


@pytest.mark.parametrize("layout", ["old", "new"])
@pytest.mark.parametrize("source_tp,target_tp", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_real_cuda_native_layout_offload_and_reload(
    layout, source_tp, target_tp, caplog, monkeypatch,
):
    if torch.cuda.device_count() < max(source_tp, target_tp):
        pytest.skip("needs one CUDA GPU per source or target TP rank")
    caplog.set_level("INFO", logger="RL_Framework.infra.scheduling.vllm_cpu_offload")
    with tempfile.TemporaryDirectory() as root:
        key = uuid.uuid4().hex
        directory = cache_directory(root, key)
        directory.mkdir()
        full = torch.arange(2 * 2 * 16 * 8 * 4, dtype=torch.float32).reshape(
            2, 2, 16, 8, 4,
        )
        save_item = {
            "request_id": "native-offload", "key": key, "source_tp": source_tp,
            "block_ids": [1, 0], "tokens": list(range(32)),
        }

        # Two blocks deliberately make native axis 0 equal the KV size.
        # Distinct values and non-contiguous cache strides expose axis swaps.
        for rank in range(source_tp):
            with torch.cuda.device(rank):
                source = full.chunk(source_tp, dim=-2)[rank].to(f"cuda:{rank}")
                native = native_strided_cache(source, layout)
                connector = bare_connector(root, source_tp, rank)
                install_backend_metadata(connector, layout)
                connector.register_kv_caches({"layer": native})
                assert native.shape[:2] == (2, 2)
                assert not connector.caches["layer"].is_contiguous()
                connector._start_offload(save_item)
                event, _gathered, host, _item, started = connector.pending["native-offload"]
                event.synchronize()
                connector._publish(host, save_item, started, started)

        for rank in range(target_tp):
            with torch.cuda.device(rank):
                seed = torch.full((2, 4, 16, 8 // target_tp, 4), -7.0, device=f"cuda:{rank}")
                native = native_strided_cache(seed, layout)
                connector = bare_connector(root, target_tp, rank)
                install_backend_metadata(connector, layout)
                connector.register_kv_caches({"layer": native})
                connector.verify_reload = True
                request_id = f"native-{source_tp}-to-{target_tp}-rank-{rank}"
                # One local prefix block is preserved; the single external
                # block is written to destination block 2.
                connector._reload({
                    "key": key, "source_tp": source_tp, "num_tokens": 32,
                    "num_local_tokens": 16, "block_ids": [0, 2],
                    "request_id": request_id,
                })
                expected = full[:, 0:1].chunk(target_tp, dim=-2)[rank]
                indices = torch.tensor([2], device=f"cuda:{rank}")
                observed = connector.caches["layer"].index_select(1, indices).cpu()
                assert torch.equal(observed, expected)
                # The engine-owned native tensor sees the restored values.
                assert torch.equal(native_cache_block(native, layout, 2).cpu(), expected[:, 0])
                for untouched in (0, 1, 3):
                    actual = native_cache_block(native, layout, untouched).cpu()
                    assert torch.equal(actual, torch.full_like(actual, -7.0))

                record = next(
                    record for record in reversed(caplog.records)
                    if "CPU_KV_RELOAD_VERIFIED " in record.getMessage()
                )
                event = json.loads(record.getMessage().split("CPU_KV_RELOAD_VERIFIED ", 1)[1])
                assert event["event"] == "CPU_KV_RELOAD_VERIFIED" and event["verified"] is True
                assert event["request_id"] == request_id
                assert event["layers_verified"] == 1
                assert event["bytes"] > 0 and event["rank"] == rank
                # Disabled verification still restores the KV and emits no verification record.
                false_request_id = request_id + "-verification-disabled"
                with monkeypatch.context() as patcher:
                    patcher.setattr(
                        cpu_offload.torch, "equal",
                        lambda *_args, **_kwargs: pytest.fail("verification ran while disabled"),
                    )
                    connector.verify_reload = False
                    connector._reload({
                        "key": key, "source_tp": source_tp, "num_tokens": 32,
                        "num_local_tokens": 16, "block_ids": [0, 3],
                        "request_id": false_request_id,
                    })
                assert torch.equal(native_cache_block(native, layout, 3).cpu(), expected[:, 0])
                assert not any(
                    false_request_id in record.getMessage() for record in caplog.records
                )

from __future__ import annotations

from dataclasses import dataclass

import pytest

from RL_Framework.infra.sync.hccl_weight_transfer import (
    HCCLWeightMetadata,
    OfficialHCCLWeightTransfer,
    build_hccl_rollout_endpoints,
    collect_weight_metadata,
)


@dataclass
class _FakeDType:
    name: str

    def __str__(self) -> str:
        return f"torch.{self.name}"


class _FakeTensor:
    def __init__(self, shape, dtype="bfloat16", element_size=2):
        self.shape = shape
        self.dtype = _FakeDType(dtype)
        self._element_size = element_size

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self):
        return self._element_size


def test_build_hccl_topology_assigns_contiguous_rank_offsets():
    endpoints, world_size = build_hccl_rollout_endpoints(
        [
            ("short_tp1", "http://host-a:8000/", 1),
            ("medium_tp2", "http://host-b:8001", 2),
            ("long_tp4", "http://host-c:8002", 4),
        ]
    )

    assert [endpoint.rank_offset for endpoint in endpoints] == [1, 2, 4]
    assert [endpoint.worker_world_size for endpoint in endpoints] == [1, 2, 4]
    assert endpoints[0].base_url == "http://host-a:8000"
    assert world_size == 8


def test_collect_weight_metadata_keeps_layout_without_tensors():
    metadata = collect_weight_metadata(
        iter(
            [
                ("model.embed.weight", _FakeTensor((8, 16))),
                ("model.norm.weight", _FakeTensor((16,), dtype="float32", element_size=4)),
            ]
        )
    )

    assert metadata.names == ("model.embed.weight", "model.norm.weight")
    assert metadata.dtype_names == ("bfloat16", "float32")
    assert metadata.shapes == ((8, 16), (16,))
    assert metadata.max_tensor_bytes == 256


def test_prepare_and_finish_use_official_lifecycle_endpoints(monkeypatch):
    endpoints, _ = build_hccl_rollout_endpoints(
        [("instance_0", "http://127.0.0.1:8000", 1)]
    )
    transfer = OfficialHCCLWeightTransfer(
        master_address="10.0.0.1",
        master_port=29620,
        timeout_s=30,
        packed=True,
        packed_buffer_size_bytes=1024,
        packed_num_buffers=2,
        checkpoint_format=True,
    )
    transfer._group = object()
    calls = []

    def record_post(endpoint, path, payload):
        calls.append((endpoint.instance_id, path, payload))
        return {}

    monkeypatch.setattr(transfer, "_post", record_post)
    metadata = HCCLWeightMetadata(
        names=("model.weight",),
        dtype_names=("bfloat16",),
        shapes=((4, 4),),
        max_tensor_bytes=32,
    )

    session = transfer.prepare(endpoints, metadata)
    transfer.finish(session, endpoints)

    assert [path for _instance, path, _payload in calls] == [
        "/pause",
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
        "/resume",
    ]
    update_payload = calls[2][2]["update_info"]
    assert update_payload["names"] == ["model.weight"]
    assert update_payload["packed"] is True
    assert update_payload["packed_num_buffers"] == 2


def test_initialized_communicator_rejects_rollout_address_change():
    endpoints, world_size = build_hccl_rollout_endpoints(
        [("instance_0", "http://host-a:8000", 1)]
    )
    transfer = OfficialHCCLWeightTransfer(
        master_address="10.0.0.1",
        master_port=29620,
        timeout_s=30,
        packed=True,
        packed_buffer_size_bytes=1024,
        packed_num_buffers=2,
        checkpoint_format=True,
    )
    transfer._group = object()
    transfer._topology_signature = (
        ("instance_0", "http://host-a:8000", 1, 1),
    )
    transfer._world_size = world_size
    changed, _ = build_hccl_rollout_endpoints(
        [("instance_0", "http://host-b:8000", 1)]
    )

    with pytest.raises(RuntimeError, match="topology changed"):
        transfer.initialize(changed, world_size)

    transfer.initialize(endpoints, world_size)

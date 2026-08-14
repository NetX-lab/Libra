import sys
from types import SimpleNamespace
from types import ModuleType

import torch

from RL_Framework.infra.sync import nccl_weight_sync
from RL_Framework.infra.sync.nccl_weight_sync import NcclReloadSpec


class FakeGroup:
    def __init__(self, rank, metadata):
        self.rank = rank
        self.metadata = metadata

    def broadcast_obj(self, value, src):
        assert src == 0
        if self.rank == 0:
            self.metadata.append(value)
            return value
        return self.metadata.pop(0)


class FakeCommunicator:
    def __init__(self, rank, metadata, tensors):
        self.rank = rank
        self.group = FakeGroup(rank, metadata)
        self.tensors = tensors

    def broadcast(self, tensor, src):
        assert src == 0
        if self.rank == 0:
            self.tensors.append(tensor.clone())
        else:
            tensor.copy_(self.tensors.pop(0))


def test_nccl_stream_preserves_metadata_and_end_marker(monkeypatch):
    metadata = []
    tensors = []
    monkeypatch.setattr(
        nccl_weight_sync,
        "_communicator",
        lambda spec: FakeCommunicator(spec.rank, metadata, tensors),
    )
    monkeypatch.setattr(torch.Tensor, "to", lambda self, **_kwargs: self)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_args, **_kwargs: SimpleNamespace(synchronize=lambda: None),
    )
    sender = NcclReloadSpec("127.0.0.1", 29620, 2, 0, "cuda:0")
    receiver = NcclReloadSpec("127.0.0.1", 29620, 2, 1, "cpu")
    source = [
        ("model.embed_tokens.weight", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)),
        ("lm_head.weight", torch.ones(1, dtype=torch.float32)),
    ]

    assert nccl_weight_sync.send_weights(source, sender) == 2
    received = list(nccl_weight_sync.receive_weights(receiver))

    assert [name for name, _tensor in received] == [name for name, _tensor in source]
    assert torch.equal(received[0][1], source[0][1])
    assert received[0][1].dtype == torch.bfloat16
    assert metadata == []
    assert tensors == []


def test_send_weights_can_keep_transport_alive(monkeypatch):
    metadata = []
    tensors = []
    communicator = FakeCommunicator(0, metadata, tensors)
    monkeypatch.setattr(nccl_weight_sync, "_communicator", lambda _spec: communicator)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, **_kwargs: self)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_args, **_kwargs: SimpleNamespace(synchronize=lambda: None),
    )
    keepalive = []

    nccl_weight_sync.send_weights(
        [("weight", torch.ones(2, dtype=torch.float32))],
        NcclReloadSpec("127.0.0.1", 29620, 2, 0, "cuda:0"),
        keepalive=keepalive,
    )

    assert keepalive == [communicator]


def test_nccl_dtype_rejects_unknown_values():
    try:
        nccl_weight_sync._dtype_from_name("not_a_dtype")
    except ValueError as exc:
        assert "Unsupported NCCL weight dtype" in str(exc)
    else:
        raise AssertionError("unknown dtype must be rejected")


def test_communicator_retains_vllm_constructor_warmup_output(monkeypatch):
    warmup_output = object()

    class FakePyNcclCommunicator:
        def __init__(self, group, device):
            self.group = group
            self.device = device
            result = self.all_reduce("warmup")
            assert result is warmup_output
            assert self._constructor_warmup_output is warmup_output

        def all_reduce(self, _tensor, op=None, stream=None):
            assert op is None
            assert stream is None
            return warmup_output

    pynccl_module = ModuleType(
        "vllm.distributed.device_communicators.pynccl"
    )
    pynccl_module.PyNcclCommunicator = FakePyNcclCommunicator
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.pynccl",
        pynccl_module,
    )
    monkeypatch.setattr(nccl_weight_sync, "_group", lambda _spec: object())

    communicator = nccl_weight_sync._communicator(
        NcclReloadSpec("127.0.0.1", 29620, 3, 0, "cuda:0")
    )

    assert communicator._constructor_warmup_output is None

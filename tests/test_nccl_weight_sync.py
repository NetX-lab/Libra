import sys
import ctypes
from dataclasses import replace
from types import SimpleNamespace
from types import ModuleType

import torch
import pytest

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


def test_communicator_is_cached_by_transport_identity(monkeypatch):
    class FakePyNcclCommunicator:
        init_count = 0

        def __init__(self, group, device):
            type(self).init_count += 1
            self.group = group
            self.device = device
            self.all_reduce("warmup")

        def all_reduce(self, _tensor, op=None, stream=None):
            return object()

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
    nccl_weight_sync.clear_communicator_cache()
    spec = NcclReloadSpec("127.0.0.1", 29620, 3, 0, "cuda:0")

    first = nccl_weight_sync._communicator(spec)
    second = nccl_weight_sync._communicator(spec)

    assert first is second
    assert first.__class__.init_count == 1
    nccl_weight_sync.clear_communicator_cache()


@pytest.mark.parametrize('direction', ['send', 'receive'])
def test_nccl_side_stream_waits_for_each_tensor_producer_before_broadcast(monkeypatch, direction):
    events = []

    class Stream:
        def wait_stream(self, other):
            assert self is transfer and other is producer
            events.append('wait_producer')

        def wait_event(self, event):
            assert self is producer
            events.append('wait_receive')

    class Event:
        def record(self, stream):
            assert stream is transfer
            events.append('record_event')

        def synchronize(self):
            events.append('finish_send')

    producer, transfer = Stream(), Stream()
    monkeypatch.setattr(nccl_weight_sync, '_cuda_stream', lambda device: transfer)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda **kwargs: producer)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'synchronize',
        lambda *args, **kwargs: pytest.fail('Transport dependencies must not synchronize the device'))

    def broadcast(tensor, src, stream):
        assert src == 0 and stream is transfer
        events.append('broadcast')

    if direction == 'send':
        communicator = SimpleNamespace(group=FakeGroup(0, []), broadcast=broadcast)
        contiguous = torch.Tensor.contiguous

        def copy_to_device(tensor, **kwargs):
            assert kwargs['non_blocking'] is True
            events.append('copy')
            return tensor

        def prepare_contiguous(tensor):
            result = contiguous(tensor)
            events.append('contiguous')
            return result

        monkeypatch.setattr(torch.Tensor, 'to', copy_to_device)
        monkeypatch.setattr(torch.Tensor, 'contiguous', prepare_contiguous)
        monkeypatch.setattr(nccl_weight_sync, '_communicator', lambda spec: communicator)
        source = [(str(i), torch.arange(4).reshape(2, 2).t()) for i in range(2)]
        spec = NcclReloadSpec('host', 29620, 2, 0, 'cuda:0', chunk_bytes=16)
        assert nccl_weight_sync.send_weights(source, spec) == 2
        per_tensor = ['copy', 'contiguous', 'wait_producer', 'broadcast', 'broadcast', 'record_event']
        assert events == per_tensor * 2 + ['finish_send', 'finish_send']
    else:
        metadata = [dict(name=str(i), shape=[4], dtype='float32', chunk_numel=2, done=False)
                    for i in range(2)] + [dict(done=True)]
        communicator = SimpleNamespace(group=FakeGroup(1, metadata), broadcast=broadcast)
        allocate = torch.empty

        def allocate_on_current_stream(shape, **kwargs):
            events.append('allocate')
            return allocate(shape, dtype=kwargs['dtype'])

        def record_consumer_stream(tensor, stream):
            assert stream is producer
            events.append('record_consumer')

        monkeypatch.setattr(torch, 'empty', allocate_on_current_stream)
        monkeypatch.setattr(torch.Tensor, 'record_stream', record_consumer_stream)
        monkeypatch.setattr(nccl_weight_sync, '_communicator', lambda spec: communicator)
        spec = NcclReloadSpec('host', 29620, 2, 1, 'cuda:0')
        assert len(list(nccl_weight_sync.receive_weights(spec))) == 2
        per_tensor = ['allocate', 'wait_producer', 'broadcast', 'broadcast',
                      'record_event', 'wait_receive', 'record_consumer']
        assert events == per_tensor * 2


@pytest.fixture
def session_transport(monkeypatch):
    events = []

    class FakePyNcclCommunicator:
        def __init__(self, group, device):
            session = group.session_id
            events.append(('build', session))
            self.group, self.device = group, device
            self.comm = ctypes.c_void_p(len(events))

            def abort(handle):
                assert handle is self.comm
                events.append(('abort', session))
                return 0

            def check(result):
                assert result == 0

            self.nccl = SimpleNamespace(lib=SimpleNamespace(ncclCommAbort=abort), NCCL_CHECK=check)
            self.all_reduce('warmup')
            events.append(('ready', session))

        def all_reduce(self, _tensor, op=None, stream=None):
            return object()

    module = ModuleType('vllm.distributed.device_communicators.pynccl')
    module.PyNcclCommunicator = FakePyNcclCommunicator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(nccl_weight_sync, '_group', lambda spec: SimpleNamespace(session_id=spec.session_id))
    nccl_weight_sync.clear_communicator_cache()
    yield SimpleNamespace(events=events,
        spec=NcclReloadSpec('127.0.0.1', 29620, 3, 0, 'cpu', session_id='a'))
    nccl_weight_sync.clear_communicator_cache()


def test_new_physical_session_rebuilds_then_releases_old_group(session_transport):
    fixture = session_transport
    first = nccl_weight_sync._communicator(fixture.spec)
    assert nccl_weight_sync._communicator(fixture.spec) is first
    other = nccl_weight_sync._communicator(replace(fixture.spec, port=29621, session_id='other'))
    second = nccl_weight_sync._communicator(replace(fixture.spec, session_id='new-members'))
    assert second is not first
    assert fixture.events.index(('ready', 'new-members')) < fixture.events.index(('abort', 'a'))
    assert first.comm is None and first.group is None
    assert set(nccl_weight_sync._COMMUNICATOR_CACHE.values()) == {second, other}
    assert ('abort', 'other') not in fixture.events
    nccl_weight_sync.clear_communicator_cache()
    assert second.comm is other.comm is None

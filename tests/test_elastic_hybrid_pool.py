import importlib
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from RL_Framework.infra.elastic import (
    ElasticHybridPool,
    GradientCommunicationDomains,
    GradientPayload,
    InterReplicaGradientDomain,
    JoinState,
    ReplicaRole,
    TorchDistributedRDMATransport,
)


def test_joining_worker_contributes_zero_gradient_placeholder():
    domain = InterReplicaGradientDomain(core_replica_ids=["core0", "core1"])
    domain.request_join("hybrid0", "core0")

    core = {
        "core0": (torch.tensor([1.0, 3.0]),),
        "core1": (torch.tensor([5.0, 7.0]),),
    }
    payload = GradientPayload(
        replica_id="hybrid0",
        target_core_id="core0",
        tensors=(torch.tensor([100.0, 100.0]),),
    )

    reduced = domain.reduce_core_gradients(
        core_gradients=core,
        hybrid_payloads=[payload],
    )

    expected = torch.tensor([3.0, 5.0])
    assert torch.allclose(reduced["core0"][0], expected)
    assert torch.allclose(reduced["core1"][0], expected)


def test_active_hybrid_gradient_is_routed_to_target_core_before_reduce():
    domain = InterReplicaGradientDomain(core_replica_ids=["core0", "core1"])
    domain.request_join("hybrid0", "core0")
    domain.mark_active("hybrid0")

    core = {
        "core0": (torch.tensor([1.0, 3.0]),),
        "core1": (torch.tensor([5.0, 7.0]),),
    }
    payload = GradientPayload(
        replica_id="hybrid0",
        target_core_id="core0",
        tensors=(torch.tensor([2.0, 4.0]),),
    )

    reduced = domain.reduce_core_gradients(
        core_gradients=core,
        hybrid_payloads=[payload],
    )

    expected = torch.tensor([4.0, 7.0])
    assert torch.allclose(reduced["core0"][0], expected)
    assert torch.allclose(reduced["core1"][0], expected)


def test_non_blocking_join_returns_before_snapshot_fetch_finishes():
    def slow_fetch(worker_id, target_core_id):
        del worker_id, target_core_id
        time.sleep(0.15)
        return 12

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=slow_fetch,
        zero_sync_steps=1,
    )
    started = time.perf_counter()
    handle = pool.join_training("rollout0", "core0")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert not handle.done()

    snapshot = pool.snapshot()
    assert snapshot["rollout0"].role == ReplicaRole.HYBRID_JOINING
    assert snapshot["rollout0"].join_state in {
        JoinState.REQUESTED,
        JoinState.FETCHING_STATE,
        JoinState.ZERO_SYNC,
    }

    worker = handle.result(timeout=2.0)
    assert worker.role == ReplicaRole.HYBRID_TRAINING
    assert worker.join_state == JoinState.ACTIVE
    assert worker.state_version == 12
    assert pool.gradient_domain.active_hybrid_ids() == ("rollout0",)
    pool.close()


def test_join_activation_waits_for_real_core_training_boundary():
    prepared = []
    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=lambda _worker, _core: 9,
        join_preparer=lambda worker, core, version: prepared.append(
            (worker, core, version)
        ),
        zero_sync_steps=1,
        require_training_boundary=True,
    )

    handle = pool.join_training("rollout0", "core0")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        worker = pool.snapshot()["rollout0"]
        if worker.join_state == JoinState.ZERO_SYNC:
            break
        time.sleep(0.01)

    assert prepared == [("rollout0", "core0", 9)]
    assert not handle.done()
    assert pool.gradient_domain.active_hybrid_ids() == ()

    pool.advance_training_boundary(0, state_version=10)
    worker = handle.result(timeout=2.0)

    assert worker.role == ReplicaRole.HYBRID_TRAINING
    assert pool.gradient_domain.active_hybrid_ids() == ("rollout0",)
    pool.close()


def test_release_cancels_inflight_join_without_worker_resurrection():
    fetch_started = threading.Event()
    allow_fetch = threading.Event()

    def fetch(_worker, _core):
        fetch_started.set()
        allow_fetch.wait(timeout=2.0)
        return 4

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=fetch,
        require_training_boundary=True,
    )
    handle = pool.join_training("rollout0", "core0")
    assert fetch_started.wait(timeout=1.0)

    pool.release_to_rollout("rollout0")
    allow_fetch.set()
    with pytest.raises(RuntimeError, match="join cancelled"):
        handle.result(timeout=2.0)

    assert pool.snapshot()["rollout0"].role == ReplicaRole.HYBRID_ROLLOUT
    assert pool.gradient_domain.active_hybrid_ids() == ()
    pool.close()


def test_join_failure_is_persisted_in_worker_state():
    def fail_fetch(_worker, _core):
        raise TimeoutError("state source unavailable")

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=fail_fetch,
    )
    handle = pool.join_training("rollout0", "core0")

    with pytest.raises(TimeoutError, match="state source unavailable"):
        handle.result(timeout=2.0)

    worker = pool.snapshot()["rollout0"]
    assert worker.join_state == JoinState.FAILED
    assert worker.last_error == "TimeoutError: state source unavailable"
    pool.close()


def test_join_publishes_snapshot_lease_before_worker_preparation():
    leased_versions = []
    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=lambda _worker, _core: 9,
        join_preparer=lambda _worker, _core, _version: leased_versions.append(
            pool.snapshot()["rollout0"].state_version
        ),
        zero_sync_steps=0,
    )

    worker = pool.join_training("rollout0", "core0").result(timeout=2.0)

    assert leased_versions == [9]
    assert worker.state_version == 9
    pool.close()


def test_local_accumulation_rejects_stale_step_and_membership_epoch():
    domain = InterReplicaGradientDomain(core_replica_ids=["core0"])
    domain.request_join("hybrid0", "core0", membership_epoch=3)
    domain.mark_active("hybrid0")
    base = (torch.tensor([1.0]),)
    payloads = [
        GradientPayload("hybrid0", "core0", (torch.tensor([2.0]),), step=7, membership_epoch=3),
        GradientPayload("hybrid0", "core0", (torch.tensor([100.0]),), step=6, membership_epoch=3),
        GradientPayload("hybrid0", "core0", (torch.tensor([100.0]),), step=7, membership_epoch=2),
    ]

    accumulated = domain.accumulate_local_gradients(
        core_replica_id="core0",
        core_gradients=base,
        hybrid_payloads=payloads,
        step=7,
    )

    assert torch.equal(accumulated[0], torch.tensor([3.0]))


def test_attached_hybrid_ids_include_joining_and_active_workers():
    domain = InterReplicaGradientDomain(core_replica_ids=["core0"])
    domain.request_join("joining", "core0", membership_epoch=1)
    domain.request_join("active", "core0", membership_epoch=2)
    domain.mark_active("active")

    assert set(domain.attached_hybrid_ids()) == {"joining", "active"}
    assert domain.active_hybrid_ids() == ("active",)


def test_transport_falls_back_to_local_clone_without_distributed_init():
    transport = TorchDistributedRDMATransport(chunk_bytes=8)
    tensor = torch.arange(16, dtype=torch.float32)

    exchanged = transport.exchange(tensor, group=None, average=True)

    assert torch.equal(exchanged, tensor)
    assert exchanged.data_ptr() != tensor.data_ptr()


def test_decoupled_domain_does_not_use_core_process_group(monkeypatch):
    hybrid_pool = importlib.import_module("RL_Framework.infra.elastic.hybrid_pool")

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

    class RecordingTransport:
        def __init__(self):
            self.calls = []

        def exchange(self, local, *, group, average):
            self.calls.append((group, average))
            return local.clone()

    monkeypatch.setattr(hybrid_pool, "dist", FakeDist)
    transport = RecordingTransport()
    domain = InterReplicaGradientDomain(
        core_replica_ids=["core0"],
        transport=transport,
        communication_domains=GradientCommunicationDomains(
            core_process_group="core-dp",
            hybrid_process_group=None,
            decoupled=True,
        ),
    )
    domain.request_join("hybrid0", "core0")
    domain.mark_active("hybrid0")

    reduced = domain.reduce_core_gradients(
        core_gradients={"core0": (torch.tensor([1.0]),)},
        hybrid_payloads=[
            GradientPayload(
                replica_id="hybrid0",
                target_core_id="core0",
                tensors=(torch.tensor([2.0]),),
            )
        ],
    )

    assert torch.equal(reduced["core0"][0], torch.tensor([3.0]))
    assert transport.calls == []
    assert domain.communication_state() == {
        "decoupled": True,
        "has_core_process_group": True,
        "has_hybrid_process_group": False,
        "uses_core_group_for_elastic_reduce": False,
    }


def test_coupled_domain_uses_core_process_group(monkeypatch):
    hybrid_pool = importlib.import_module("RL_Framework.infra.elastic.hybrid_pool")

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            del group
            return 1

    class RecordingTransport:
        def __init__(self):
            self.calls = []

        def exchange(self, local, *, group, average):
            self.calls.append((group, average))
            return local.clone()

    monkeypatch.setattr(hybrid_pool, "dist", FakeDist)
    transport = RecordingTransport()
    domain = InterReplicaGradientDomain(
        core_replica_ids=["core0"],
        transport=transport,
        process_group="core-dp",
        decouple_communication_domains=False,
    )

    reduced = domain.reduce_core_gradients(
        core_gradients={"core0": (torch.tensor([1.0]),)},
    )

    assert torch.equal(reduced["core0"][0], torch.tensor([1.0]))
    assert transport.calls == [("core-dp", True)]
    assert domain.communication_state() == {
        "decoupled": False,
        "has_core_process_group": True,
        "has_hybrid_process_group": True,
        "uses_core_group_for_elastic_reduce": True,
    }

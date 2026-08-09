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
    JoinCancelledError,
    ReplicaJoinHandle,
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


def test_complete_replica_payload_is_consumed_by_matching_model_parallel_lane():
    domain = InterReplicaGradientDomain(
        core_replica_ids=["core0"],
        local_replica_rank=1,
        replica_world_size=2,
        average_core_replicas=False,
    )
    domain.request_join("hybrid_dp0", "core0")
    domain.mark_active("hybrid_dp0")
    payloads = [
        GradientPayload(
            replica_id="hybrid_dp0",
            target_core_id="core0",
            tensors=(torch.tensor([100.0]),),
            replica_rank=0,
            replica_world_size=2,
        ),
        GradientPayload(
            replica_id="hybrid_dp0",
            target_core_id="core0",
            tensors=(torch.tensor([3.0]),),
            replica_rank=1,
            replica_world_size=2,
        ),
    ]

    reduced = domain.reduce_core_gradients(
        core_gradients={"core0": (torch.tensor([2.0]),)},
        hybrid_payloads=payloads,
    )

    assert torch.equal(reduced["core0"][0], torch.tensor([5.0]))


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
        "isolated_group_verified": False,
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
        "isolated_group_verified": False,
    }


def test_activation_barrier_keeps_join_non_blocking_until_worker_ready():
    release = threading.Event()

    def activation_barrier(worker_id, target_core_id, version):
        assert (worker_id, target_core_id, version) == ("rollout0", "core0", 5)
        assert release.wait(timeout=2.0)

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=lambda *_args: 5,
        zero_sync_steps=0,
    )
    started = time.perf_counter()
    handle = pool.join_training(
        "rollout0",
        "core0",
        activation_barrier=activation_barrier,
        timeout_s=2.0,
    )
    assert time.perf_counter() - started < 0.05

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if pool.snapshot()["rollout0"].join_state == JoinState.ACTIVATING:
            break
        time.sleep(0.01)
    assert pool.snapshot()["rollout0"].join_state == JoinState.ACTIVATING
    assert not handle.done()
    release.set()
    assert handle.result(timeout=1.0).role == ReplicaRole.HYBRID_TRAINING
    pool.close()


def test_complete_dp_replica_rejoins_non_blocking_and_activates_atomically():
    release = threading.Event()

    def fetch(worker_id, _target_core_id):
        assert worker_id in {"rollout0", "rollout1"}
        assert release.wait(timeout=2.0)
        return 23

    pool = ElasticHybridPool(
        core_train_workers=["core0", "core1"],
        core_rollout_workers=["rollout0", "rollout1"],
        snapshot_fetcher=fetch,
        zero_sync_steps=0,
    )
    started = time.perf_counter()
    handle = pool.join_replica(
        "hybrid_dp0",
        ["rollout0", "rollout1"],
        "core1",
        timeout_s=2.0,
    )
    assert isinstance(handle, ReplicaJoinHandle)
    assert time.perf_counter() - started < 0.05
    assert not handle.done()
    assert pool.gradient_domain.is_joining("hybrid_dp0")
    assert {
        pool.snapshot()[worker_id].role
        for worker_id in ("rollout0", "rollout1")
    } == {ReplicaRole.HYBRID_JOINING}

    release.set()
    workers = handle.result(timeout=2.0)
    assert {worker.role for worker in workers} == {ReplicaRole.HYBRID_TRAINING}
    assert {worker.state_version for worker in workers} == {23}
    assert pool.gradient_domain.active_hybrid_ids() == ("hybrid_dp0",)
    pool.close()


def test_dp_replica_join_failure_rolls_back_every_member():
    def fetch(worker_id, _target_core_id):
        if worker_id == "rollout1":
            raise RuntimeError("rank snapshot failed")
        return 7

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0", "rollout1"],
        snapshot_fetcher=fetch,
        zero_sync_steps=0,
    )
    handle = pool.join_replica(
        "hybrid_dp0", ["rollout0", "rollout1"], "core0"
    )
    with pytest.raises(RuntimeError, match="rank snapshot failed"):
        handle.result(timeout=1.0)
    workers = pool.snapshot()
    assert {workers[wid].role for wid in ("rollout0", "rollout1")} == {
        ReplicaRole.HYBRID_ROLLOUT
    }
    assert {workers[wid].join_state for wid in ("rollout0", "rollout1")} == {
        JoinState.FAILED
    }
    assert pool.gradient_domain.active_hybrid_ids() == ()
    assert pool.replica_snapshot() == {}
    pool.close()


def test_released_dp_replica_can_rejoin_with_a_new_generation():
    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0", "rollout1"],
        snapshot_fetcher=lambda *_args: 4,
        zero_sync_steps=0,
    )
    first = pool.join_replica(
        "hybrid_dp0", ["rollout0", "rollout1"], "core0"
    )
    first.result(timeout=1.0)
    pool.release_replica_to_rollout("hybrid_dp0")

    second = pool.join_replica(
        "hybrid_dp0", ["rollout0", "rollout1"], "core0"
    )
    assert second.generation > first.generation
    assert {worker.role for worker in second.result(timeout=1.0)} == {
        ReplicaRole.HYBRID_TRAINING
    }
    pool.close()


def test_cancelled_join_cannot_become_active_after_grp_reversal():
    release = threading.Event()

    def fetch(*_args):
        assert release.wait(timeout=2.0)
        return 9

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=fetch,
        zero_sync_steps=0,
    )
    handle = pool.join_training("rollout0", "core0", timeout_s=2.0)
    handle.cancel()
    release.set()
    with pytest.raises(JoinCancelledError):
        handle.result(timeout=1.0)
    worker = pool.snapshot()["rollout0"]
    assert worker.role == ReplicaRole.HYBRID_ROLLOUT
    assert worker.join_state == JoinState.CANCELLED
    assert pool.gradient_domain.active_hybrid_ids() == ()
    pool.close()


def test_failed_join_rolls_worker_back_to_rollout():
    def fetch(*_args):
        raise RuntimeError("snapshot unavailable")

    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0"],
        snapshot_fetcher=fetch,
        zero_sync_steps=0,
    )
    handle = pool.join_training("rollout0", "core0", timeout_s=1.0)
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        handle.result(timeout=1.0)
    worker = pool.snapshot()["rollout0"]
    assert worker.role == ReplicaRole.HYBRID_ROLLOUT
    assert worker.join_state == JoinState.FAILED
    assert worker.target_core_id is None
    pool.close()


def test_required_isolated_domain_rejects_missing_elastic_group(monkeypatch):
    hybrid_pool = importlib.import_module("RL_Framework.infra.elastic.hybrid_pool")

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setattr(hybrid_pool, "dist", FakeDist)
    domain = InterReplicaGradientDomain(
        core_replica_ids=["core0"],
        communication_domains=GradientCommunicationDomains(
            core_process_group="core-dp",
            hybrid_process_group=None,
            decoupled=True,
        ),
        require_isolated_process_group=True,
    )
    with pytest.raises(RuntimeError, match="isolated elastic CCL"):
        domain.reduce_core_gradients(
            core_gradients={"core0": (torch.tensor([1.0]),)},
        )


def test_decoupled_domain_rejects_core_group_alias():
    domain = InterReplicaGradientDomain(
        core_replica_ids=["core0"],
        communication_domains=GradientCommunicationDomains(
            core_process_group="core-dp",
            hybrid_process_group=None,
            decoupled=True,
        ),
    )
    with pytest.raises(ValueError, match="cannot reuse the core process group"):
        domain.bind_hybrid_process_group("core-dp")


def test_decoupled_domain_membership_changes_do_not_mutate_core_domain():
    domains = GradientCommunicationDomains(
        core_process_group="fixed-megatron-dp",
        hybrid_process_group="elastic-side-channel",
        decoupled=True,
    )
    domain = InterReplicaGradientDomain(
        core_replica_ids=["core0", "core1"],
        communication_domains=domains,
    )
    domain.request_join("hybrid_dp0", "core1")
    requested = domain.membership_state()
    domain.mark_active("hybrid_dp0")
    active = domain.membership_state()
    domain.detach("hybrid_dp0")
    detached = domain.membership_state()

    assert requested["joining_replica_ids"] == ("hybrid_dp0",)
    assert active["active_replica_ids"] == ("hybrid_dp0",)
    assert detached["active_replica_ids"] == ()
    assert [
        requested["membership_epoch"],
        active["membership_epoch"],
        detached["membership_epoch"],
    ] == [1, 2, 3]
    assert domains.core_process_group == "fixed-megatron-dp"
    assert domains.hybrid_process_group == "elastic-side-channel"

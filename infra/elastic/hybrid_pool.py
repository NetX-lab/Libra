"""Elastic Hybrid Pool and inter-replica gradient exchange.

This module implements the Libra paper's elastic execution contract in a
backend-neutral way:

* the core TP/PP training topology stays fixed;
* hybrid workers join/leave only in the inter-replica gradient domain;
* joining workers first contribute zero-gradient placeholders, so core training
  can keep stepping while their state is fetched and aligned;
* gradient payloads are exchanged through a chunked RDMA-style data plane, with
  a torch.distributed implementation for CUDA/NCCL or CPU/Gloo tests.
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Protocol

import torch

try:
    import torch.distributed as dist
except ModuleNotFoundError:  # pragma: no cover - torch is a project dependency
    dist = None


DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024


class ReplicaRole(str, enum.Enum):
    """Execution role for a worker in the hybrid pool."""

    CORE_TRAINING = "core_training"
    CORE_ROLLOUT = "core_rollout"
    HYBRID_ROLLOUT = "hybrid_rollout"
    HYBRID_JOINING = "hybrid_joining"
    HYBRID_TRAINING = "hybrid_training"


class JoinState(str, enum.Enum):
    """Lifecycle state for a non-blocking training join."""

    REQUESTED = "requested"
    FETCHING_STATE = "fetching_state"
    ZERO_SYNC = "zero_sync"
    ACTIVE = "active"
    FAILED = "failed"


@dataclass(frozen=True)
class GradientPayload:
    """Flattened gradient payload routed across the inter-replica domain."""

    replica_id: str
    target_core_id: str
    tensors: tuple[torch.Tensor, ...]
    zero_placeholder: bool = False

    @staticmethod
    def zeros_like(
        *,
        replica_id: str,
        target_core_id: str,
        reference: Iterable[torch.Tensor],
    ) -> "GradientPayload":
        return GradientPayload(
            replica_id=replica_id,
            target_core_id=target_core_id,
            tensors=tuple(torch.zeros_like(t) for t in reference),
            zero_placeholder=True,
        )


@dataclass
class ElasticWorker:
    """Worker metadata owned by :class:`ElasticHybridPool`."""

    worker_id: str
    role: ReplicaRole
    target_core_id: str | None = None
    join_state: JoinState | None = None
    state_version: int = 0
    last_transition_ts: float = field(default_factory=time.time)

    def transition(
        self,
        role: ReplicaRole,
        *,
        target_core_id: str | None = None,
        join_state: JoinState | None = None,
        state_version: int | None = None,
    ):
        self.role = role
        self.target_core_id = target_core_id
        self.join_state = join_state
        if state_version is not None:
            self.state_version = state_version
        self.last_transition_ts = time.time()


@dataclass
class JoinHandle:
    """Handle returned immediately by a non-blocking join request."""

    worker_id: str
    target_core_id: str
    future: Future

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: float | None = None) -> ElasticWorker:
        return self.future.result(timeout=timeout)


class GradientTransport(Protocol):
    """Transport interface for RDMA-style gradient payload exchange."""

    def exchange(
        self,
        local: torch.Tensor,
        *,
        group,
        average: bool,
    ) -> torch.Tensor:
        ...


@dataclass
class GradientCommunicationDomains:
    """Communication domains used by the elastic gradient path.

    ``core_process_group`` belongs to the fixed training backend, while
    ``hybrid_process_group`` is reserved for elastic side-channel traffic. In
    decoupled mode the hybrid domain never drives collectives on the core
    training process group.
    """

    core_process_group: object | None = None
    hybrid_process_group: object | None = None
    decoupled: bool = True

    @property
    def gradient_process_group(self):
        if self.decoupled:
            return self.hybrid_process_group
        return self.hybrid_process_group or self.core_process_group


class TorchDistributedRDMATransport:
    """Chunked torch.distributed transport used as the RDMA data plane.

    On CUDA with NCCL this maps to GPUDirect-capable collectives when the
    cluster is configured for it. On CPU/Gloo it provides the same semantics for
    CI and smoke tests.
    """

    def __init__(self, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.chunk_bytes = chunk_bytes

    def exchange(
        self,
        local: torch.Tensor,
        *,
        group=None,
        average: bool,
    ) -> torch.Tensor:
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return local.clone()

        result = local.clone()
        elem_size = max(result.element_size(), 1)
        chunk_elems = max(self.chunk_bytes // elem_size, 1)
        flat = result.view(-1)

        for start in range(0, flat.numel(), chunk_elems):
            chunk = flat[start:start + chunk_elems]
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=group)

        if average:
            world = dist.get_world_size(group=group)
            if world > 1:
                result.div_(world)
        return result


class InterReplicaGradientDomain:
    """Dynamic inter-replica gradient domain for core and hybrid replicas.

    The domain tracks core replicas separately from joining/active hybrid
    replicas. ``reduce_core_gradients`` implements Appendix D's aggregation
    rule: hybrid gradients are routed to their target core replica and
    accumulated before the core DP all-reduce. Joining workers may be present as
    zero placeholders, preserving collective shape without changing the update.
    """

    def __init__(
        self,
        *,
        core_replica_ids: Iterable[str],
        transport: GradientTransport | None = None,
        process_group=None,
        communication_domains: GradientCommunicationDomains | None = None,
        decouple_communication_domains: bool = True,
        average_core_replicas: bool = True,
    ):
        core_ids = list(core_replica_ids)
        if not core_ids:
            raise ValueError("core_replica_ids cannot be empty")
        if len(core_ids) != len(set(core_ids)):
            raise ValueError("core_replica_ids must be unique")

        self.core_replica_ids = tuple(core_ids)
        self.transport = transport or TorchDistributedRDMATransport()
        self.communication_domains = communication_domains or GradientCommunicationDomains(
            core_process_group=process_group,
            hybrid_process_group=None if decouple_communication_domains else process_group,
            decoupled=decouple_communication_domains,
        )
        self.average_core_replicas = average_core_replicas
        self._hybrid_targets: dict[str, str] = {}
        self._joining: set[str] = set()
        self._lock = threading.RLock()

    @property
    def process_group(self):
        """Compatibility accessor for the active elastic gradient group."""
        return self.communication_domains.gradient_process_group

    @process_group.setter
    def process_group(self, value) -> None:
        if self.communication_domains.decoupled:
            self.communication_domains.hybrid_process_group = value
        else:
            self.communication_domains.core_process_group = value
            self.communication_domains.hybrid_process_group = value

    @property
    def decoupled_communication_domains(self) -> bool:
        return bool(self.communication_domains.decoupled)

    def communication_state(self) -> dict[str, bool]:
        return {
            "decoupled": self.decoupled_communication_domains,
            "has_core_process_group": (
                self.communication_domains.core_process_group is not None
            ),
            "has_hybrid_process_group": (
                self.communication_domains.hybrid_process_group is not None
            ),
            "uses_core_group_for_elastic_reduce": (
                not self.decoupled_communication_domains
                and self.communication_domains.gradient_process_group
                is self.communication_domains.core_process_group
            ),
        }

    def request_join(self, hybrid_replica_id: str, target_core_id: str):
        self._validate_core(target_core_id)
        with self._lock:
            self._hybrid_targets[hybrid_replica_id] = target_core_id
            self._joining.add(hybrid_replica_id)

    def mark_active(self, hybrid_replica_id: str):
        with self._lock:
            if hybrid_replica_id not in self._hybrid_targets:
                raise KeyError(f"unknown hybrid replica: {hybrid_replica_id}")
            self._joining.discard(hybrid_replica_id)

    def detach(self, hybrid_replica_id: str):
        with self._lock:
            self._joining.discard(hybrid_replica_id)
            self._hybrid_targets.pop(hybrid_replica_id, None)

    def is_joining(self, hybrid_replica_id: str) -> bool:
        with self._lock:
            return hybrid_replica_id in self._joining

    def active_hybrid_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                rid for rid in self._hybrid_targets if rid not in self._joining
            )

    def zero_payload_for(
        self,
        *,
        hybrid_replica_id: str,
        reference: Iterable[torch.Tensor],
    ) -> GradientPayload:
        with self._lock:
            target = self._hybrid_targets[hybrid_replica_id]
        return GradientPayload.zeros_like(
            replica_id=hybrid_replica_id,
            target_core_id=target,
            reference=reference,
        )

    def reduce_core_gradients(
        self,
        *,
        core_gradients: Mapping[str, Iterable[torch.Tensor]],
        hybrid_payloads: Iterable[GradientPayload] = (),
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """Return synchronized gradients for each core replica.

        ``core_gradients`` must contain all fixed core replicas. Hybrid payloads
        whose workers are still joining are treated as zero placeholders even if
        a caller accidentally supplies non-zero tensors.
        """
        distributed = (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
        )
        provided_core_ids = tuple(rid for rid in self.core_replica_ids if rid in core_gradients)
        missing = set(self.core_replica_ids) - set(core_gradients)
        if missing and not distributed:
            raise ValueError(f"missing core gradients for: {sorted(missing)}")
        if not provided_core_ids:
            raise ValueError("no local core gradients were provided")

        effective = {
            rid: tuple(t.clone() for t in core_gradients[rid])
            for rid in provided_core_ids
        }
        self._validate_same_structure(effective.values())

        with self._lock:
            targets = dict(self._hybrid_targets)
            joining = set(self._joining)

        for payload in hybrid_payloads:
            if payload.replica_id not in targets:
                continue
            target = targets[payload.replica_id]
            if target not in effective:
                continue
            if target != payload.target_core_id:
                raise ValueError(
                    f"payload target mismatch for {payload.replica_id}: "
                    f"{payload.target_core_id} != {target}"
                )
            if payload.replica_id in joining or payload.zero_placeholder:
                continue
            if len(payload.tensors) != len(effective[target]):
                raise ValueError("hybrid payload tensor count mismatch")
            effective[target] = tuple(
                base + extra.to(device=base.device, dtype=base.dtype)
                for base, extra in zip(effective[target], payload.tensors)
            )

        # In a real distributed launch each rank owns one core replica lane. The
        # transport call performs the RDMA/NCCL all-reduce. For local tests we
        # compute the same reduction explicitly across the in-memory core map.
        if distributed and self.process_group is not None:
            return {
                rid: tuple(
                    self.transport.exchange(
                        tensor,
                        group=self.process_group,
                        average=self.average_core_replicas,
                    )
                    for tensor in tensors
                )
                for rid, tensors in effective.items()
            }

        reduced = []
        for tensors_at_idx in zip(*(effective[rid] for rid in provided_core_ids)):
            total = torch.stack(
                [t.to(tensors_at_idx[0].device) for t in tensors_at_idx]
            ).sum(dim=0)
            if self.average_core_replicas:
                total = total / len(provided_core_ids)
            reduced.append(total)
        return {
            rid: tuple(t.clone() for t in reduced)
            for rid in provided_core_ids
        }

    def _validate_core(self, core_id: str):
        if core_id not in self.core_replica_ids:
            raise KeyError(f"unknown core replica: {core_id}")

    @staticmethod
    def _validate_same_structure(
        tensors_by_replica: Iterable[tuple[torch.Tensor, ...]]
    ):
        iterator = iter(tensors_by_replica)
        try:
            reference = next(iterator)
        except StopIteration:
            return
        ref_shapes = tuple(t.shape for t in reference)
        for tensors in iterator:
            if tuple(t.shape for t in tensors) != ref_shapes:
                raise ValueError("core gradient tensor shapes do not match")


class ElasticHybridPool:
    """State machine for borrowing workers between rollout and training."""

    def __init__(
        self,
        *,
        core_train_workers: Iterable[str],
        core_rollout_workers: Iterable[str] = (),
        hybrid_workers: Iterable[str] = (),
        gradient_domain: InterReplicaGradientDomain | None = None,
        snapshot_fetcher: Callable[[str, str], int] | None = None,
        zero_sync_steps: int = 1,
        max_background_workers: int = 4,
    ):
        if zero_sync_steps < 0:
            raise ValueError("zero_sync_steps cannot be negative")
        self.zero_sync_steps = zero_sync_steps
        self.snapshot_fetcher = snapshot_fetcher or self._default_snapshot_fetcher
        self.gradient_domain = gradient_domain or InterReplicaGradientDomain(
            core_replica_ids=core_train_workers
        )
        self._workers: dict[str, ElasticWorker] = {}
        for wid in core_train_workers:
            self._workers[wid] = ElasticWorker(wid, ReplicaRole.CORE_TRAINING)
        for wid in core_rollout_workers:
            self._workers[wid] = ElasticWorker(wid, ReplicaRole.CORE_ROLLOUT)
        for wid in hybrid_workers:
            self._workers[wid] = ElasticWorker(wid, ReplicaRole.HYBRID_ROLLOUT)
        if not self._workers:
            raise ValueError("ElasticHybridPool requires at least one worker")

        self._executor = ThreadPoolExecutor(max_workers=max_background_workers)
        self._lock = threading.RLock()
        self._closed = False

    def snapshot(self) -> dict[str, ElasticWorker]:
        with self._lock:
            return {
                wid: ElasticWorker(
                    worker_id=w.worker_id,
                    role=w.role,
                    target_core_id=w.target_core_id,
                    join_state=w.join_state,
                    state_version=w.state_version,
                    last_transition_ts=w.last_transition_ts,
                )
                for wid, w in self._workers.items()
            }

    def borrow_rollout_worker(self, worker_id: str | None = None) -> ElasticWorker:
        """Convert an idle rollout worker into a hybrid rollout worker."""
        with self._lock:
            worker = self._select_worker(
                worker_id,
                allowed={ReplicaRole.CORE_ROLLOUT, ReplicaRole.HYBRID_ROLLOUT},
            )
            worker.transition(ReplicaRole.HYBRID_ROLLOUT)
            return worker

    def join_training(self, worker_id: str, target_core_id: str) -> JoinHandle:
        """Start rollout->training transition and return immediately."""
        with self._lock:
            self._ensure_open()
            worker = self._select_worker(
                worker_id,
                allowed={ReplicaRole.CORE_ROLLOUT, ReplicaRole.HYBRID_ROLLOUT},
            )
            worker.transition(
                ReplicaRole.HYBRID_JOINING,
                target_core_id=target_core_id,
                join_state=JoinState.REQUESTED,
            )
            self.gradient_domain.request_join(worker.worker_id, target_core_id)
            future = self._executor.submit(self._finish_join, worker.worker_id, target_core_id)
            return JoinHandle(worker.worker_id, target_core_id, future)

    def release_to_rollout(self, worker_id: str):
        """Detach a hybrid training worker and return it to rollout service."""
        with self._lock:
            worker = self._select_worker(
                worker_id,
                allowed={ReplicaRole.HYBRID_TRAINING, ReplicaRole.HYBRID_JOINING},
            )
            self.gradient_domain.detach(worker.worker_id)
            worker.transition(ReplicaRole.HYBRID_ROLLOUT)

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)

    def _finish_join(self, worker_id: str, target_core_id: str) -> ElasticWorker:
        try:
            with self._lock:
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.FETCHING_STATE,
                )
            version = self.snapshot_fetcher(worker_id, target_core_id)

            with self._lock:
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.ZERO_SYNC,
                    state_version=version,
                )

            for _ in range(self.zero_sync_steps):
                time.sleep(0)

            self.gradient_domain.mark_active(worker_id)
            with self._lock:
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_TRAINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.ACTIVE,
                    state_version=version,
                )
                return worker
        except Exception:
            with self._lock:
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.FAILED,
                )
            self.gradient_domain.detach(worker_id)
            raise

    def _select_worker(
        self,
        worker_id: str | None,
        *,
        allowed: set[ReplicaRole],
    ) -> ElasticWorker:
        if worker_id is not None:
            if worker_id not in self._workers:
                raise KeyError(f"unknown worker: {worker_id}")
            worker = self._workers[worker_id]
            if worker.role not in allowed:
                raise RuntimeError(
                    f"worker {worker_id} is {worker.role}, expected one of {allowed}"
                )
            return worker
        for worker in self._workers.values():
            if worker.role in allowed:
                return worker
        raise RuntimeError(f"no worker available in roles: {allowed}")

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("ElasticHybridPool is closed")

    @staticmethod
    def _default_snapshot_fetcher(worker_id: str, target_core_id: str) -> int:
        del worker_id, target_core_id
        return 0

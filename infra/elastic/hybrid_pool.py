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
    ACTIVATING = "activating"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JoinCancelledError(RuntimeError):
    """Raised inside a stale background join after GRP changed its target."""


@dataclass(frozen=True)
class GradientPayload:
    """Flattened gradient payload routed across the inter-replica domain."""

    replica_id: str
    target_core_id: str
    tensors: tuple[torch.Tensor, ...]
    zero_placeholder: bool = False
    replica_rank: int = 0
    replica_world_size: int = 1

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
            replica_rank=0,
            replica_world_size=1,
        )


@dataclass
class ElasticWorker:
    """Worker metadata owned by :class:`ElasticHybridPool`."""

    worker_id: str
    role: ReplicaRole
    target_core_id: str | None = None
    join_state: JoinState | None = None
    state_version: int = 0
    transition_generation: int = 0
    last_transition_ts: float = field(default_factory=time.time)

    def transition(
        self,
        role: ReplicaRole,
        *,
        target_core_id: str | None = None,
        join_state: JoinState | None = None,
        state_version: int | None = None,
        transition_generation: int | None = None,
    ):
        self.role = role
        self.target_core_id = target_core_id
        self.join_state = join_state
        if state_version is not None:
            self.state_version = state_version
        if transition_generation is not None:
            self.transition_generation = transition_generation
        self.last_transition_ts = time.time()


@dataclass
class JoinHandle:
    """Handle returned immediately by a non-blocking join request."""

    worker_id: str
    target_core_id: str
    future: Future
    generation: int = 0
    cancel_callback: Callable[[str, int], None] | None = None

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: float | None = None) -> ElasticWorker:
        return self.future.result(timeout=timeout)

    def cancel(self) -> None:
        if self.cancel_callback is not None:
            self.cancel_callback(self.worker_id, self.generation)


@dataclass
class ReplicaJoinHandle:
    """Non-blocking, atomic join handle for one complete DP replica."""

    replica_id: str
    member_worker_ids: tuple[str, ...]
    target_core_id: str
    future: Future
    generation: int = 0
    cancel_callback: Callable[[str, int], None] | None = None

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: float | None = None) -> tuple[ElasticWorker, ...]:
        return self.future.result(timeout=timeout)

    def cancel(self) -> None:
        if self.cancel_callback is not None:
            self.cancel_callback(self.replica_id, self.generation)


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
        require_isolated_process_group: bool = False,
        local_replica_rank: int = 0,
        replica_world_size: int = 1,
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
        self.require_isolated_process_group = bool(require_isolated_process_group)
        self.replica_world_size = max(1, int(replica_world_size))
        self.local_replica_rank = int(local_replica_rank)
        if not 0 <= self.local_replica_rank < self.replica_world_size:
            raise ValueError("local_replica_rank must be inside replica_world_size")
        self._hybrid_targets: dict[str, str] = {}
        self._joining: set[str] = set()
        self._membership_epoch = 0
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
            "isolated_group_verified": (
                self.communication_domains.hybrid_process_group is not None
                and self.communication_domains.hybrid_process_group
                is not self.communication_domains.core_process_group
            ),
        }

    def validate_communication_isolation(self) -> None:
        """Fail closed when the elastic collective can alias Megatron DP."""
        if not self.decoupled_communication_domains:
            return
        hybrid_group = self.communication_domains.hybrid_process_group
        core_group = self.communication_domains.core_process_group
        if hybrid_group is not None and hybrid_group is core_group:
            raise RuntimeError(
                "elastic gradient process group aliases the Megatron core group"
            )
        if self.require_isolated_process_group and hybrid_group is None:
            raise RuntimeError(
                "isolated elastic CCL process group is required but not initialized"
            )

    def bind_hybrid_process_group(self, process_group) -> None:
        if (
            self.decoupled_communication_domains
            and process_group is not None
            and process_group is self.communication_domains.core_process_group
        ):
            raise ValueError(
                "decoupled elastic gradients cannot reuse the core process group"
            )
        self.communication_domains.hybrid_process_group = process_group

    def membership_state(self) -> dict[str, object]:
        """Describe the independently versioned elastic domain membership."""
        with self._lock:
            targets = dict(self._hybrid_targets)
            joining = tuple(sorted(self._joining))
            return {
                "membership_epoch": self._membership_epoch,
                "core_replica_ids": self.core_replica_ids,
                "hybrid_targets": targets,
                "joining_replica_ids": joining,
                "active_replica_ids": tuple(
                    sorted(replica_id for replica_id in targets if replica_id not in self._joining)
                ),
                "decoupled": self.decoupled_communication_domains,
            }

    def request_join(self, hybrid_replica_id: str, target_core_id: str):
        self._validate_core(target_core_id)
        with self._lock:
            self._hybrid_targets[hybrid_replica_id] = target_core_id
            self._joining.add(hybrid_replica_id)
            self._membership_epoch += 1

    def mark_active(self, hybrid_replica_id: str):
        with self._lock:
            if hybrid_replica_id not in self._hybrid_targets:
                raise KeyError(f"unknown hybrid replica: {hybrid_replica_id}")
            self._joining.discard(hybrid_replica_id)
            self._membership_epoch += 1

    def detach(self, hybrid_replica_id: str):
        with self._lock:
            self._joining.discard(hybrid_replica_id)
            if self._hybrid_targets.pop(hybrid_replica_id, None) is not None:
                self._membership_epoch += 1

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
        if distributed and self.decoupled_communication_domains:
            self.validate_communication_isolation()
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
            if payload.replica_world_size != self.replica_world_size:
                raise ValueError(
                    f"payload replica world size mismatch for {payload.replica_id}: "
                    f"{payload.replica_world_size} != {self.replica_world_size}"
                )
            if payload.replica_rank != self.local_replica_rank:
                # Each fixed TP/PP/CP lane consumes only the matching rank from
                # the complete hybrid DP replica.
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
        max_background_workers: int = 0,
    ):
        if zero_sync_steps < 0:
            raise ValueError("zero_sync_steps cannot be negative")
        core_train_workers = tuple(core_train_workers)
        core_rollout_workers = tuple(core_rollout_workers)
        hybrid_workers = tuple(hybrid_workers)
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

        background_capacity = (
            int(max_background_workers)
            if int(max_background_workers) > 0
            else max(1, len(core_rollout_workers) + len(hybrid_workers))
        )
        self._executor = ThreadPoolExecutor(max_workers=background_capacity)
        self._lock = threading.RLock()
        self._closed = False
        self._join_generations: dict[str, int] = {}
        self._replica_generations: dict[str, int] = {}
        self._replica_members: dict[str, tuple[str, ...]] = {}

    def snapshot(self) -> dict[str, ElasticWorker]:
        with self._lock:
            return {
                wid: ElasticWorker(
                    worker_id=w.worker_id,
                    role=w.role,
                    target_core_id=w.target_core_id,
                    join_state=w.join_state,
                    state_version=w.state_version,
                    transition_generation=w.transition_generation,
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

    def join_training(
        self,
        worker_id: str,
        target_core_id: str,
        *,
        activation_barrier: Callable[[str, str, int], None] | None = None,
        timeout_s: float | None = None,
    ) -> JoinHandle:
        """Start rollout->training transition and return immediately."""
        with self._lock:
            self._ensure_open()
            worker = self._select_worker(
                worker_id,
                allowed={ReplicaRole.CORE_ROLLOUT, ReplicaRole.HYBRID_ROLLOUT},
            )
            generation = self._join_generations.get(worker.worker_id, 0) + 1
            self._join_generations[worker.worker_id] = generation
            worker.transition(
                ReplicaRole.HYBRID_JOINING,
                target_core_id=target_core_id,
                join_state=JoinState.REQUESTED,
                transition_generation=generation,
            )
            self.gradient_domain.request_join(worker.worker_id, target_core_id)
            future = self._executor.submit(
                self._finish_join,
                worker.worker_id,
                target_core_id,
                generation,
                activation_barrier,
                timeout_s,
            )
            return JoinHandle(
                worker.worker_id,
                target_core_id,
                future,
                generation=generation,
                cancel_callback=self.cancel_join,
            )

    def join_replica(
        self,
        replica_id: str,
        member_worker_ids: Iterable[str],
        target_core_id: str,
        *,
        activation_barrier: Callable[[str, str, int], None] | None = None,
        timeout_s: float | None = None,
    ) -> ReplicaJoinHandle:
        """Join a complete DP replica atomically and return immediately.

        Member ranks fetch and activate in the background.  The logical
        replica enters the elastic gradient domain only after *all* members are
        ready; cancellation or a single-rank failure rolls every member back.
        """
        members = tuple(str(wid) for wid in member_worker_ids)
        if not members:
            raise ValueError("a DP replica must contain at least one worker")
        if len(members) != len(set(members)):
            raise ValueError("DP replica member_worker_ids must be unique")
        with self._lock:
            self._ensure_open()
            if replica_id in self._replica_members:
                raise RuntimeError(f"replica is already registered: {replica_id}")
            workers = [
                self._select_worker(
                    wid,
                    allowed={ReplicaRole.CORE_ROLLOUT, ReplicaRole.HYBRID_ROLLOUT},
                )
                for wid in members
            ]
            generation = self._replica_generations.get(replica_id, 0) + 1
            self._replica_generations[replica_id] = generation
            self._replica_members[replica_id] = members
            for worker in workers:
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.REQUESTED,
                    transition_generation=generation,
                )
            self.gradient_domain.request_join(replica_id, target_core_id)
            future = self._executor.submit(
                self._finish_replica_join,
                replica_id,
                members,
                target_core_id,
                generation,
                activation_barrier,
                timeout_s,
            )
            return ReplicaJoinHandle(
                replica_id=replica_id,
                member_worker_ids=members,
                target_core_id=target_core_id,
                future=future,
                generation=generation,
                cancel_callback=self.cancel_replica_join,
            )

    def cancel_replica_join(self, replica_id: str, generation: int) -> None:
        with self._lock:
            if self._replica_generations.get(replica_id) != generation:
                return
            self._replica_generations[replica_id] = generation + 1
            self.gradient_domain.detach(replica_id)
            for worker_id in self._replica_members.get(replica_id, ()):
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_ROLLOUT,
                    target_core_id=None,
                    join_state=JoinState.CANCELLED,
                    transition_generation=generation + 1,
                )
            self._replica_members.pop(replica_id, None)

    def release_replica_to_rollout(self, replica_id: str) -> None:
        """Atomically detach all ranks of an active/joining DP replica."""
        with self._lock:
            members = self._replica_members.get(replica_id)
            if not members:
                raise KeyError(f"unknown hybrid replica: {replica_id}")
            generation = self._replica_generations.get(replica_id, 0) + 1
            self._replica_generations[replica_id] = generation
            self.gradient_domain.detach(replica_id)
            for worker_id in members:
                worker = self._workers[worker_id]
                if worker.role not in {
                    ReplicaRole.HYBRID_TRAINING,
                    ReplicaRole.HYBRID_JOINING,
                }:
                    raise RuntimeError(
                        f"replica {replica_id} member {worker_id} is {worker.role}"
                    )
                worker.transition(ReplicaRole.HYBRID_ROLLOUT)
            self._replica_members.pop(replica_id, None)

    def replica_snapshot(self) -> dict[str, tuple[ElasticWorker, ...]]:
        """Return logical replica membership without exposing mutable state."""
        workers = self.snapshot()
        with self._lock:
            memberships = dict(self._replica_members)
        return {
            replica_id: tuple(workers[wid] for wid in members)
            for replica_id, members in memberships.items()
        }

    def cancel_join(self, worker_id: str, generation: int) -> None:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None or self._join_generations.get(worker_id) != generation:
                return
            self._join_generations[worker_id] = generation + 1
            self.gradient_domain.detach(worker_id)
            worker.transition(
                ReplicaRole.HYBRID_ROLLOUT,
                join_state=JoinState.CANCELLED,
                transition_generation=generation + 1,
            )

    def release_to_rollout(self, worker_id: str):
        """Detach a hybrid training worker and return it to rollout service."""
        with self._lock:
            worker = self._select_worker(
                worker_id,
                allowed={ReplicaRole.HYBRID_TRAINING, ReplicaRole.HYBRID_JOINING},
            )
            self._join_generations[worker.worker_id] = (
                self._join_generations.get(worker.worker_id, 0) + 1
            )
            self.gradient_domain.detach(worker.worker_id)
            worker.transition(ReplicaRole.HYBRID_ROLLOUT)

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _finish_join(
        self,
        worker_id: str,
        target_core_id: str,
        generation: int,
        activation_barrier: Callable[[str, str, int], None] | None,
        timeout_s: float | None,
    ) -> ElasticWorker:
        deadline = (
            time.monotonic() + max(float(timeout_s), 0.0)
            if timeout_s is not None
            else None
        )
        try:
            self._assert_join_current(worker_id, generation, deadline)
            with self._lock:
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.FETCHING_STATE,
                )
            version = self.snapshot_fetcher(worker_id, target_core_id)
            self._assert_join_current(worker_id, generation, deadline)

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

            self._assert_join_current(worker_id, generation, deadline)
            if activation_barrier is not None:
                with self._lock:
                    worker = self._workers[worker_id]
                    worker.transition(
                        ReplicaRole.HYBRID_JOINING,
                        target_core_id=target_core_id,
                        join_state=JoinState.ACTIVATING,
                        state_version=version,
                        transition_generation=generation,
                    )
                activation_barrier(worker_id, target_core_id, version)
                self._assert_join_current(worker_id, generation, deadline)

            self.gradient_domain.mark_active(worker_id)
            with self._lock:
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_TRAINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.ACTIVE,
                    state_version=version,
                    transition_generation=generation,
                )
                return worker
        except JoinCancelledError:
            # Cancellation may race with the background state transition.  If
            # this is still the cancelled generation, make the rollback
            # visible before the future is observed by the planner.
            with self._lock:
                worker = self._workers.get(worker_id)
                if worker is not None and self._join_generations.get(worker_id) == generation + 1:
                    self.gradient_domain.detach(worker_id)
                    worker.transition(
                        ReplicaRole.HYBRID_ROLLOUT,
                        target_core_id=None,
                        join_state=JoinState.CANCELLED,
                        transition_generation=generation + 1,
                    )
            raise
        except Exception:
            with self._lock:
                worker = self._workers[worker_id]
                if self._join_generations.get(worker_id) == generation:
                    worker.transition(
                        ReplicaRole.HYBRID_ROLLOUT,
                        target_core_id=None,
                        join_state=JoinState.FAILED,
                        transition_generation=generation,
                    )
                    self.gradient_domain.detach(worker_id)
            raise

    def _finish_replica_join(
        self,
        replica_id: str,
        members: tuple[str, ...],
        target_core_id: str,
        generation: int,
        activation_barrier: Callable[[str, str, int], None] | None,
        timeout_s: float | None,
    ) -> tuple[ElasticWorker, ...]:
        deadline = (
            time.monotonic() + max(float(timeout_s), 0.0)
            if timeout_s is not None
            else None
        )
        try:
            self._assert_replica_join_current(replica_id, generation, deadline)
            with self._lock:
                for worker_id in members:
                    self._workers[worker_id].transition(
                        ReplicaRole.HYBRID_JOINING,
                        target_core_id=target_core_id,
                        join_state=JoinState.FETCHING_STATE,
                    )
            versions = tuple(
                int(self.snapshot_fetcher(worker_id, target_core_id))
                for worker_id in members
            )
            self._assert_replica_join_current(replica_id, generation, deadline)
            if len(set(versions)) != 1:
                raise RuntimeError(
                    f"DP replica {replica_id} fetched inconsistent state versions: {versions}"
                )
            version = versions[0]
            with self._lock:
                for worker_id in members:
                    self._workers[worker_id].transition(
                        ReplicaRole.HYBRID_JOINING,
                        target_core_id=target_core_id,
                        join_state=JoinState.ZERO_SYNC,
                        state_version=version,
                    )
            for _ in range(self.zero_sync_steps):
                time.sleep(0)

            self._assert_replica_join_current(replica_id, generation, deadline)
            if activation_barrier is not None:
                with self._lock:
                    for worker_id in members:
                        self._workers[worker_id].transition(
                            ReplicaRole.HYBRID_JOINING,
                            target_core_id=target_core_id,
                            join_state=JoinState.ACTIVATING,
                            state_version=version,
                            transition_generation=generation,
                        )
                for worker_id in members:
                    activation_barrier(worker_id, target_core_id, version)
                    self._assert_replica_join_current(
                        replica_id, generation, deadline
                    )

            self.gradient_domain.mark_active(replica_id)
            with self._lock:
                active = []
                for worker_id in members:
                    worker = self._workers[worker_id]
                    worker.transition(
                        ReplicaRole.HYBRID_TRAINING,
                        target_core_id=target_core_id,
                        join_state=JoinState.ACTIVE,
                        state_version=version,
                        transition_generation=generation,
                    )
                    active.append(worker)
                return tuple(active)
        except JoinCancelledError:
            with self._lock:
                if self._replica_generations.get(replica_id) == generation + 1:
                    self.gradient_domain.detach(replica_id)
                    for worker_id in members:
                        self._workers[worker_id].transition(
                            ReplicaRole.HYBRID_ROLLOUT,
                            target_core_id=None,
                            join_state=JoinState.CANCELLED,
                            transition_generation=generation + 1,
                        )
            raise
        except Exception:
            with self._lock:
                if self._replica_generations.get(replica_id) == generation:
                    self.gradient_domain.detach(replica_id)
                    for worker_id in members:
                        self._workers[worker_id].transition(
                            ReplicaRole.HYBRID_ROLLOUT,
                            target_core_id=None,
                            join_state=JoinState.FAILED,
                            transition_generation=generation,
                        )
                    self._replica_members.pop(replica_id, None)
            raise

    def _assert_replica_join_current(
        self,
        replica_id: str,
        generation: int,
        deadline: float | None,
    ) -> None:
        with self._lock:
            if self._closed:
                raise JoinCancelledError(
                    f"join cancelled because pool is closed: {replica_id}"
                )
            if self._replica_generations.get(replica_id) != generation:
                raise JoinCancelledError(
                    f"stale join generation for replica {replica_id}"
                )
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"hybrid replica join timed out for {replica_id}")

    def _assert_join_current(
        self,
        worker_id: str,
        generation: int,
        deadline: float | None,
    ) -> None:
        with self._lock:
            if self._closed:
                raise JoinCancelledError(f"join cancelled because pool is closed: {worker_id}")
            if self._join_generations.get(worker_id) != generation:
                raise JoinCancelledError(f"stale join generation for {worker_id}")
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"hybrid join timed out for {worker_id}")

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

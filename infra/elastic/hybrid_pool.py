"""Elastic Hybrid Pool and inter-replica gradient exchange.

This module implements the Libra paper's elastic execution contract in a
backend-neutral way:

* the core TP/PP training topology stays fixed;
* hybrid workers join/leave only in the inter-replica gradient domain;
* joining workers contribute no samples until a state-installation ACK proves
  that their parameters and optimizer are caught up;
* gradient payloads are exchanged through a chunked RDMA-style data plane, with
  a torch.distributed implementation for CUDA/NCCL or CPU/Gloo tests.
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
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
    step: int = -1
    state_version: int = -1
    membership_epoch: int = 0
    sample_count: int = 1
    global_sample_count: int = 0

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
            step=-1,
            state_version=-1,
            membership_epoch=0,
        )


@dataclass
class ElasticWorker:
    """Worker metadata owned by :class:`ElasticHybridPool`."""

    worker_id: str
    role: ReplicaRole
    target_core_id: str | None = None
    join_state: JoinState | None = None
    state_version: int = 0
    membership_epoch: int = 0
    transition_generation: int = 0
    last_transition_ts: float = field(default_factory=time.time)

    def transition(
        self,
        role: ReplicaRole,
        *,
        target_core_id: str | None = None,
        join_state: JoinState | None = None,
        state_version: int | None = None,
        membership_epoch: int | None = None,
        transition_generation: int | None = None,
    ):
        self.role = role
        self.target_core_id = target_core_id
        self.join_state = join_state
        if state_version is not None:
            self.state_version = state_version
        if membership_epoch is not None:
            self.membership_epoch = int(membership_epoch)
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
        self._membership_epochs: dict[str, int] = {}
        self._joining: set[str] = set()
        self._membership_epoch = 0
        self._consumed_payloads: set[tuple[str, int, int, int, int]] = set()
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
                "membership_epochs": dict(self._membership_epochs),
                "core_replica_ids": self.core_replica_ids,
                "hybrid_targets": targets,
                "joining_replica_ids": joining,
                "active_replica_ids": tuple(
                    sorted(replica_id for replica_id in targets if replica_id not in self._joining)
                ),
                "decoupled": self.decoupled_communication_domains,
            }

    def request_join(
        self,
        hybrid_replica_id: str,
        target_core_id: str,
        *,
        membership_epoch: int | None = None,
    ):
        self._validate_core(target_core_id)
        with self._lock:
            self._hybrid_targets[hybrid_replica_id] = target_core_id
            self._joining.add(hybrid_replica_id)
            if membership_epoch is None:
                self._membership_epoch += 1
                membership_epoch = self._membership_epoch
            else:
                self._membership_epoch = max(
                    self._membership_epoch, int(membership_epoch)
                )
            self._membership_epochs[hybrid_replica_id] = int(membership_epoch)

    def mark_active(self, hybrid_replica_id: str, *, membership_epoch: int | None = None):
        with self._lock:
            if hybrid_replica_id not in self._hybrid_targets:
                raise KeyError(f"unknown hybrid replica: {hybrid_replica_id}")
            self._joining.discard(hybrid_replica_id)
            if membership_epoch is None:
                self._membership_epoch += 1
                membership_epoch = self._membership_epoch
            else:
                self._membership_epoch = max(self._membership_epoch, int(membership_epoch))
            self._membership_epochs[hybrid_replica_id] = int(membership_epoch)

    def detach(self, hybrid_replica_id: str, *, membership_epoch: int | None = None):
        with self._lock:
            self._joining.discard(hybrid_replica_id)
            attached = self._hybrid_targets.pop(hybrid_replica_id, None) is not None
            if attached or membership_epoch is not None:
                if membership_epoch is None:
                    self._membership_epoch += 1
                    membership_epoch = self._membership_epoch
                else:
                    self._membership_epoch = max(self._membership_epoch, int(membership_epoch))
                # Keep the exit version so the publisher can emit an explicit
                # rollout record even after the replica leaves the pool.
                self._membership_epochs[hybrid_replica_id] = int(membership_epoch)

    def is_joining(self, hybrid_replica_id: str) -> bool:
        with self._lock:
            return hybrid_replica_id in self._joining

    def active_hybrid_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                rid for rid in self._hybrid_targets if rid not in self._joining
            )

    def attached_hybrid_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._hybrid_targets)

    def active_hybrid_ids_for_core(self, core_replica_id: str) -> tuple[str, ...]:
        self._validate_core(core_replica_id)
        with self._lock:
            return tuple(
                replica_id
                for replica_id, target in self._hybrid_targets.items()
                if target == core_replica_id and replica_id not in self._joining
            )

    def membership_epoch(self, hybrid_replica_id: str) -> int:
        with self._lock:
            return int(self._membership_epochs.get(hybrid_replica_id, 0))

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

    def accumulate_local_gradients(
        self,
        *,
        core_replica_id: str,
        core_gradients: Iterable[torch.Tensor],
        hybrid_payloads: Iterable[GradientPayload] = (),
        step: int = -1,
        state_version: int = -1,
        core_sample_count: int = 1,
        global_sample_count: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        """Accumulate matching external gradients before the core DP reduce."""
        self._validate_core(core_replica_id)
        effective = tuple(t.clone() for t in core_gradients)
        with self._lock:
            targets = dict(self._hybrid_targets)
            joining = set(self._joining)
            epochs = dict(self._membership_epochs)
        for payload in hybrid_payloads:
            if targets.get(payload.replica_id) != core_replica_id:
                continue
            if payload.replica_id in joining or payload.zero_placeholder:
                continue
            if payload.membership_epoch != epochs.get(payload.replica_id, 0):
                continue
            if payload.step >= 0 and step >= 0 and payload.step != step:
                continue
            if (
                payload.state_version >= 0
                and state_version >= 0
                and payload.state_version != state_version
            ):
                continue
            if payload.replica_world_size != self.replica_world_size:
                raise ValueError(
                    f"payload replica world size mismatch for {payload.replica_id}: "
                    f"{payload.replica_world_size} != {self.replica_world_size}"
                )
            if payload.replica_rank != self.local_replica_rank:
                continue
            if len(payload.tensors) != len(effective):
                raise ValueError("hybrid payload tensor count mismatch")
            effective = tuple(
                base + extra.to(device=base.device, dtype=base.dtype)
                for base, extra in zip(effective, payload.tensors)
            )
        return effective

    def reduce_core_gradients(
        self,
        *,
        core_gradients: Mapping[str, Iterable[torch.Tensor]],
        hybrid_payloads: Iterable[GradientPayload] = (),
        core_sample_counts: Mapping[str, int] | None = None,
        global_sample_count: int = 0,
        step: int = -1,
        state_version: int = -1,
        in_place: bool = False,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """Reduce local sample means into the fixed global batch gradient.

        With sample counts, each local mean contributes ``n_local / B`` and the
        core collective sums these contributions. Joining replicas have zero
        weight. Every core lane participates, including lanes with no Hybrid.
        Unweighted calls retain the legacy backend-neutral simulation API.
        ``in_place`` explicitly transfers ownership of one local core lane's
        gradient buffers to this call. It bounds temporary storage to transport
        chunks, rather than duplicating every FP32 model gradient. The default
        retains independent output buffers and leaves its inputs untouched.
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
            rid: tuple(core_gradients[rid])
            for rid in provided_core_ids
        }
        self._validate_same_structure(effective.values())
        if in_place and len(provided_core_ids) != 1:
            raise ValueError("in-place reduction requires exactly one local core lane")

        weighted = global_sample_count > 0
        if weighted:
            if core_sample_counts is None:
                raise ValueError("sample counts are required for a fixed global batch")
            scales = {rid: int(core_sample_counts[rid]) / global_sample_count
                      for rid in provided_core_ids}
        else:
            scales = {rid: 1.0 for rid in provided_core_ids}
        with self._lock:
            targets = dict(self._hybrid_targets)
            joining = set(self._joining)
            epochs = dict(self._membership_epochs)
        seen = set()
        accepted = []
        for payload in hybrid_payloads:
            target = targets.get(payload.replica_id)
            if target not in effective or payload.replica_id in joining:
                continue
            if payload.zero_placeholder:
                if step >= 0:
                    raise ValueError("active hybrid gradient cannot be a zero placeholder")
                continue
            if target != payload.target_core_id:
                raise ValueError(f"payload target mismatch for {payload.replica_id}")
            if payload.replica_world_size != self.replica_world_size:
                raise ValueError(f"payload replica world size mismatch for {payload.replica_id}")
            if payload.replica_rank != self.local_replica_rank:
                continue
            if (step >= 0 or weighted) and payload.membership_epoch != epochs.get(payload.replica_id, 0):
                raise ValueError("stale hybrid membership epoch")
            if step >= 0 and payload.step != step:
                raise ValueError("stale hybrid training step")
            if state_version >= 0 and payload.state_version != state_version:
                raise ValueError("stale hybrid optimizer revision")
            key = (payload.replica_id, payload.replica_rank, payload.step,
                   payload.state_version, payload.membership_epoch)
            if key in seen or (payload.step >= 0 and key in self._consumed_payloads):
                raise ValueError("duplicate hybrid gradient payload")
            seen.add(key)
            if len(payload.tensors) != len(effective[target]):
                raise ValueError("hybrid payload tensor count mismatch")
            if any(a.shape != b.shape for a, b in zip(effective[target], payload.tensors)):
                raise ValueError("hybrid payload tensor shape mismatch")
            if weighted and payload.global_sample_count != global_sample_count:
                raise ValueError("hybrid global batch size mismatch")
            if payload.sample_count < 0:
                raise ValueError("negative hybrid sample count")
            scale = payload.sample_count / global_sample_count if weighted else 1.0
            accepted.append((target, payload, scale))
        # Validate the complete payload set before mutating an owned buffer or
        # marking a payload consumed, including failures in a later payload.
        if in_place:
            all_tensors = [t for tensors in effective.values() for t in tensors]
            all_tensors += [t for _, payload, _ in accepted for t in payload.tensors]
            if any(not t.is_contiguous() for t in all_tensors):
                raise ValueError("in-place reduction requires contiguous gradients")
            with torch.no_grad():
                for rid, tensors in effective.items():
                    if scales[rid] != 1.0:
                        for tensor in tensors:
                            tensor.mul_(scales[rid])
                for target, payload, scale in accepted:
                    for base, extra in zip(effective[target], payload.tensors):
                        width = max(DEFAULT_CHUNK_BYTES // max(base.element_size(), 1), 1)
                        destination, source = base.view(-1), extra.view(-1)
                        for start in range(0, base.numel(), width):
                            stop = start + width
                            # Preserve multiply-then-add rounding; add_(alpha=)
                            # can fuse them and change the numerical update.
                            contribution = source[start:stop].to(
                                device=base.device, dtype=base.dtype, copy=True)
                            contribution.mul_(scale)
                            destination[start:stop].add_(contribution)
                            del contribution
        else:
            effective = {rid: tuple(t.clone() for t in tensors)
                         for rid, tensors in effective.items()}
            if weighted:
                effective = {rid: tuple(t * scales[rid] for t in tensors)
                             for rid, tensors in effective.items()}
            for target, payload, scale in accepted:
                effective[target] = tuple(
                    base + extra.to(device=base.device, dtype=base.dtype) * scale
                    for base, extra in zip(effective[target], payload.tensors)
                )
        with self._lock:
            self._consumed_payloads.update(k for k in seen if k[2] >= 0)
            if step >= 0:
                self._consumed_payloads = {k for k in self._consumed_payloads if k[2] >= step - 1}

        # In a real distributed launch each rank owns one core replica lane. The
        # transport call performs the RDMA/NCCL all-reduce. For local tests we
        # compute the same reduction explicitly across the in-memory core map.
        if distributed and self.process_group is not None:
            if in_place:
                with torch.no_grad():
                    for tensors in effective.values():
                        for tensor in tensors:
                            width = max(DEFAULT_CHUNK_BYTES // max(tensor.element_size(), 1), 1)
                            flat = tensor.view(-1)
                            for start in range(0, flat.numel(), width):
                                chunk = flat[start:start + width]
                                reduced = self.transport.exchange(
                                    chunk, group=self.process_group,
                                    average=self.average_core_replicas and not weighted,
                                )
                                chunk.copy_(reduced)
                                del reduced
                return effective
            return {
                rid: tuple(
                    self.transport.exchange(
                        tensor,
                        group=self.process_group,
                        average=self.average_core_replicas and not weighted,
                    )
                    for tensor in tensors
                )
                for rid, tensors in effective.items()
            }

        if in_place:
            # There is exactly one lane here. Avoid stack/sum/clone of that
            # entire lane when no collective is needed (notably Core DP1).
            return effective

        reduced = []
        for tensors_at_idx in zip(*(effective[rid] for rid in provided_core_ids)):
            total = torch.stack(
                [t.to(tensors_at_idx[0].device) for t in tensors_at_idx]
            ).sum(dim=0)
            if self.average_core_replicas and not weighted:
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
        require_state_ack: bool = True,
    ):
        if zero_sync_steps < 0:
            raise ValueError("zero_sync_steps cannot be negative")
        core_train_workers = tuple(core_train_workers)
        core_rollout_workers = tuple(core_rollout_workers)
        hybrid_workers = tuple(hybrid_workers)
        self.zero_sync_steps = zero_sync_steps
        self.require_state_ack = bool(require_state_ack)
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
        self._activation_context = threading.local()
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
        replica_activation_barrier: Callable[[str, tuple[str, ...], str, int], None] | None = None,
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
                replica_activation_barrier,
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

    def membership_snapshot(self) -> dict[str, dict[str, object]]:
        """Snapshot attached replicas and versioned exits under the pool lock."""
        with self._lock:
            state = self.gradient_domain.membership_state()
            targets = state["hybrid_targets"]
            joining = set(state["joining_replica_ids"])
            records = {}
            for replica_id, epoch in state["membership_epochs"].items():
                members = self._replica_members.get(
                    replica_id, (replica_id,) if replica_id in self._workers else (),
                )
                role = ReplicaRole.HYBRID_ROLLOUT
                if replica_id in targets:
                    role = (ReplicaRole.HYBRID_JOINING if replica_id in joining
                            else ReplicaRole.HYBRID_TRAINING)
                records[replica_id] = {
                    "worker_id": replica_id,
                    "target_core_id": targets.get(replica_id, ""),
                    "role": role.value,
                    "membership_epoch": epoch,
                    "members": list(members),
                    "replica_world_size": len(members) or self.gradient_domain.replica_world_size,
                }
            return records

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
            if self._closed:
                return
            self._closed = True
            # Finish local rollback here. Late callbacks must not mutate pool
            # state after close returns, including their exception handlers.
            for replica_id, generation in list(self._replica_generations.items()):
                self._replica_generations[replica_id] = generation + 1
            for worker_id, generation in list(self._join_generations.items()):
                self._join_generations[worker_id] = generation + 1
            for replica_id in self.gradient_domain.attached_hybrid_ids():
                self.gradient_domain.detach(replica_id)
            self._replica_members.clear()
            for worker in self._workers.values():
                if worker.role in {ReplicaRole.HYBRID_JOINING, ReplicaRole.HYBRID_TRAINING}:
                    worker.transition(ReplicaRole.HYBRID_ROLLOUT, join_state=JoinState.CANCELLED)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _invoke_activation(self, callback, args, *, identity, generation, replica, deadline):
        # Preserve the callback API while carrying the original join generation
        # to the runtime launch gate, even if cancellation/rejoin happens during
        # command preparation. Reading the latest worker generation is unsafe.
        previous = getattr(self._activation_context, "join", None)
        self._activation_context.join = (identity, generation, replica, deadline)
        try:
            receipt = callback(*args)
            check = self._assert_replica_join_current if replica else self._assert_join_current
            check(identity, generation, deadline)
            if self.require_state_ack:
                if not isinstance(receipt, Mapping) or receipt.get("acknowledged") is not True:
                    raise RuntimeError("Hybrid activation requires an actual state installation ACK")
                if int(receipt.get("state_version", -1)) < int(args[-1]):
                    raise RuntimeError("Hybrid activation ACK is older than the fetched state")
                if int(receipt.get("optimizer_revision", -1)) != int(receipt["state_version"]):
                    raise RuntimeError("Hybrid activation optimizer revision ACK mismatch")
            return receipt
        finally:
            self._activation_context.join = previous

    @contextmanager
    def activation_guard(self, handle=None):
        """Serialize a launch commit with cancellation, never readiness waits.

        Lock order: runtime lifecycle lock -> pool lock -> gradient-domain lock.
        Pool callbacks themselves always run outside the pool lock.
        """
        with self._lock:
            if self._closed:
                raise JoinCancelledError("activation cancelled because pool is closed")
            join = getattr(self._activation_context, "join", None)
            if handle is not None:
                replica = hasattr(handle, "replica_id")
                identity = handle.replica_id if replica else handle.worker_id
                join = (identity, handle.generation, replica, None)
            if join is not None:
                identity, generation, replica, deadline = join
                check = self._assert_replica_join_current if replica else self._assert_join_current
                check(identity, generation, deadline)
            yield

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
            if timeout_s is not None and float(timeout_s) > 0
            else None
        )
        try:
            self._assert_join_current(worker_id, generation, deadline)
            with self._lock:
                self._assert_join_current(worker_id, generation, deadline)
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.FETCHING_STATE,
                )
            version = self.snapshot_fetcher(worker_id, target_core_id)
            self._assert_join_current(worker_id, generation, deadline)

            with self._lock:
                self._assert_join_current(worker_id, generation, deadline)
                worker = self._workers[worker_id]
                worker.transition(
                    ReplicaRole.HYBRID_JOINING,
                    target_core_id=target_core_id,
                    join_state=JoinState.ZERO_SYNC,
                    state_version=version,
                )

            # State installation is acknowledged by the activation callback;
            # yielding the host thread does not synchronize optimizer updates.

            self._assert_join_current(worker_id, generation, deadline)
            if activation_barrier is None and self.require_state_ack:
                raise RuntimeError("Hybrid activation requires a state installation barrier")
            if activation_barrier is not None:
                with self._lock:
                    self._assert_join_current(worker_id, generation, deadline)
                    worker = self._workers[worker_id]
                    worker.transition(
                        ReplicaRole.HYBRID_JOINING,
                        target_core_id=target_core_id,
                        join_state=JoinState.ACTIVATING,
                        state_version=version,
                        transition_generation=generation,
                    )
                receipt = self._invoke_activation(
                    activation_barrier, (worker_id, target_core_id, version),
                    identity=worker_id, generation=generation, replica=False, deadline=deadline,
                )
                if isinstance(receipt, Mapping):
                    version = int(receipt["state_version"])
                self._assert_join_current(worker_id, generation, deadline)

            with self._lock:
                # Cancellation and activation must not interleave between the
                # final generation check and the membership/role update.
                self._assert_join_current(worker_id, generation, deadline)
                self.gradient_domain.mark_active(worker_id)
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
                if not self._closed and worker is not None and self._join_generations.get(worker_id) == generation + 1:
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
                if not self._closed and self._join_generations.get(worker_id) == generation:
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
        replica_activation_barrier: Callable[[str, tuple[str, ...], str, int], None] | None,
        timeout_s: float | None,
    ) -> tuple[ElasticWorker, ...]:
        deadline = (
            time.monotonic() + max(float(timeout_s), 0.0)
            if timeout_s is not None and float(timeout_s) > 0
            else None
        )
        try:
            self._assert_replica_join_current(replica_id, generation, deadline)
            with self._lock:
                self._assert_replica_join_current(replica_id, generation, deadline)
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
                self._assert_replica_join_current(replica_id, generation, deadline)
                for worker_id in members:
                    self._workers[worker_id].transition(
                        ReplicaRole.HYBRID_JOINING,
                        target_core_id=target_core_id,
                        join_state=JoinState.ZERO_SYNC,
                        state_version=version,
                    )
            # State installation is acknowledged by the activation callback;
            # yielding the host thread does not synchronize optimizer updates.

            self._assert_replica_join_current(replica_id, generation, deadline)
            if replica_activation_barrier is None and activation_barrier is None and self.require_state_ack:
                raise RuntimeError("Hybrid activation requires a state installation barrier")
            if replica_activation_barrier is not None:
                with self._lock:
                    self._assert_replica_join_current(replica_id, generation, deadline)
                    for worker_id in members:
                        self._workers[worker_id].transition(
                            ReplicaRole.HYBRID_JOINING,
                            target_core_id=target_core_id,
                            join_state=JoinState.ACTIVATING,
                            state_version=version,
                            transition_generation=generation,
                        )
                receipt = self._invoke_activation(
                    replica_activation_barrier, (replica_id, members, target_core_id, version),
                    identity=replica_id, generation=generation, replica=True, deadline=deadline,
                )
                if isinstance(receipt, Mapping):
                    version = int(receipt["state_version"])
                self._assert_replica_join_current(replica_id, generation, deadline)
            elif activation_barrier is not None:
                with self._lock:
                    self._assert_replica_join_current(replica_id, generation, deadline)
                    for worker_id in members:
                        self._workers[worker_id].transition(
                            ReplicaRole.HYBRID_JOINING,
                            target_core_id=target_core_id,
                            join_state=JoinState.ACTIVATING,
                            state_version=version,
                            transition_generation=generation,
                        )
                installed_versions = []
                for worker_id in members:
                    receipt = self._invoke_activation(
                        activation_barrier, (worker_id, target_core_id, version),
                        identity=replica_id, generation=generation, replica=True, deadline=deadline,
                    )
                    if isinstance(receipt, Mapping):
                        installed_versions.append(int(receipt["state_version"]))
                    self._assert_replica_join_current(replica_id, generation, deadline)
                if installed_versions:
                    if len(set(installed_versions)) != 1:
                        raise RuntimeError("Hybrid replica members acknowledged different optimizer revisions")
                    version = installed_versions[0]

            with self._lock:
                self._assert_replica_join_current(replica_id, generation, deadline)
                self.gradient_domain.mark_active(replica_id)
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
                if not self._closed and self._replica_generations.get(replica_id) == generation + 1:
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
                if not self._closed and self._replica_generations.get(replica_id) == generation:
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

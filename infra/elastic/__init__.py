"""Elastic execution primitives for Libra-style hybrid workers."""

try:
    from .hybrid_pool import (
        ElasticHybridPool,
        ElasticWorker,
        GradientCommunicationDomains,
        GradientPayload,
        InterReplicaGradientDomain,
        JoinHandle,
        ReplicaJoinHandle,
        JoinCancelledError,
        JoinState,
        ReplicaRole,
        TorchDistributedRDMATransport,
    )
    _HYBRID_POOL_EXPORTS = [
        "ElasticHybridPool",
        "ElasticWorker",
        "GradientCommunicationDomains",
        "GradientPayload",
        "InterReplicaGradientDomain",
        "JoinHandle",
        "ReplicaJoinHandle",
        "JoinCancelledError",
        "JoinState",
        "ReplicaRole",
        "TorchDistributedRDMATransport",
    ]
except ModuleNotFoundError as exc:  # pragma: no cover - local dev without torch
    if exc.name != "torch":
        raise
    _HYBRID_POOL_EXPORTS = []

from .runtime_executor import (
    ManagedRolloutProcess,
    RuntimeElasticExecutor,
    RuntimeReconfigurationResult,
)
try:
    from .gradient_ipc import (
        ElasticGradientClient,
        ElasticGradientServer,
        GradientEndpoint,
        GradientUpdate,
    )
    _GRADIENT_IPC_EXPORTS = [
        "ElasticGradientClient",
        "ElasticGradientServer",
        "GradientEndpoint",
        "GradientUpdate",
    ]
except ModuleNotFoundError as exc:  # pragma: no cover - local dev without torch
    if exc.name != "torch":
        raise
    _GRADIENT_IPC_EXPORTS = []
try:
    from .native_rdma import (
        NativeRDMAConfig,
        NativeRDMAError,
        NativeRDMAFileTransfer,
        NativeRDMAGradientClient,
        NativeRDMAGradientServer,
    )
    _NATIVE_RDMA_EXPORTS = [
        "NativeRDMAConfig",
        "NativeRDMAError",
        "NativeRDMAFileTransfer",
        "NativeRDMAGradientClient",
        "NativeRDMAGradientServer",
    ]
except ModuleNotFoundError as exc:  # pragma: no cover - local dev without torch
    if exc.name != "torch":
        raise
    _NATIVE_RDMA_EXPORTS = []

__all__ = _HYBRID_POOL_EXPORTS + _GRADIENT_IPC_EXPORTS + [
    "ManagedRolloutProcess",
    "RuntimeElasticExecutor",
    "RuntimeReconfigurationResult",
] + _NATIVE_RDMA_EXPORTS

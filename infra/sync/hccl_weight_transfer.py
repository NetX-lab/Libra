"""Official vLLM Ascend HCCL weight-transfer integration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import importlib.metadata
import json
import time
from typing import Any, Iterable, Iterator
import urllib.request


@dataclass(frozen=True)
class HCCLRolloutEndpoint:
    """One vLLM server and its worker ranks in the transfer group."""

    instance_id: str
    base_url: str
    rank_offset: int
    worker_world_size: int


@dataclass(frozen=True)
class HCCLWeightMetadata:
    """Stable HF tensor layout sent through the vLLM control plane."""

    names: tuple[str, ...]
    dtype_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    max_tensor_bytes: int


@dataclass
class HCCLTransferSession:
    """Outstanding worker-side receive requests for one update."""

    executor: ThreadPoolExecutor
    futures: list[Future]
    started_at: float


def build_hccl_rollout_endpoints(
    instances: Iterable[tuple[str, str, int]],
) -> tuple[list[HCCLRolloutEndpoint], int]:
    """Assign contiguous HCCL ranks after trainer rank zero."""
    endpoints: list[HCCLRolloutEndpoint] = []
    rank_offset = 1
    for instance_id, base_url, worker_world_size in instances:
        worker_world_size = max(1, int(worker_world_size))
        endpoints.append(
            HCCLRolloutEndpoint(
                instance_id=str(instance_id),
                base_url=str(base_url).rstrip("/"),
                rank_offset=rank_offset,
                worker_world_size=worker_world_size,
            )
        )
        rank_offset += worker_world_size
    return endpoints, rank_offset


def collect_weight_metadata(
    weights: Iterator[tuple[str, Any]],
) -> HCCLWeightMetadata:
    """Consume one export pass and retain only tensor layout metadata."""
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    max_tensor_bytes = 0
    for name, tensor in weights:
        names.append(str(name))
        dtype_names.append(str(tensor.dtype).removeprefix("torch."))
        shapes.append(tuple(int(value) for value in tensor.shape))
        max_tensor_bytes = max(
            max_tensor_bytes,
            int(tensor.numel()) * int(tensor.element_size()),
        )
        del tensor
    if not names:
        raise RuntimeError("Megatron Bridge exported no rollout weights")
    return HCCLWeightMetadata(
        names=tuple(names),
        dtype_names=tuple(dtype_names),
        shapes=tuple(shapes),
        max_tensor_bytes=max_tensor_bytes,
    )


class OfficialHCCLWeightTransfer:
    """Drive vLLM Ascend's official HCCL weight-transfer lifecycle."""

    def __init__(
        self,
        *,
        master_address: str,
        master_port: int,
        timeout_s: float,
        packed: bool,
        packed_buffer_size_bytes: int,
        packed_num_buffers: int,
        checkpoint_format: bool,
    ) -> None:
        self.master_address = str(master_address)
        self.master_port = int(master_port)
        self.timeout_s = max(1.0, float(timeout_s))
        self.packed = bool(packed)
        self.packed_buffer_size_bytes = max(1, int(packed_buffer_size_bytes))
        self.packed_num_buffers = max(1, int(packed_num_buffers))
        self.checkpoint_format = bool(checkpoint_format)
        self._group = None
        self._topology_signature: tuple[tuple[str, str, int, int], ...] | None = None
        self._world_size: int | None = None

    @staticmethod
    def validate_runtime() -> None:
        """Fail before collectives when the official HCCL stack is unavailable."""
        from packaging.version import Version

        required = {
            "vllm": Version("0.22.1"),
            "vllm-ascend": Version("0.22.1rc1"),
            "torch": Version("2.10.0"),
            "torch-npu": Version("2.10.0"),
        }
        installed: dict[str, Version] = {}
        for package, minimum in required.items():
            try:
                installed[package] = Version(importlib.metadata.version(package))
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Official HCCL weight transfer requires {package}>={minimum}"
                ) from exc
            if installed[package] < minimum:
                raise RuntimeError(
                    f"Official HCCL weight transfer requires {package}>={minimum}; "
                    f"found {installed[package]}"
                )

        import torch

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("Official HCCL weight transfer requires an available NPU")

    def initialize(
        self,
        endpoints: list[HCCLRolloutEndpoint],
        world_size: int,
    ) -> None:
        """Create one persistent stateless HCCL communicator."""
        signature = tuple(
            (
                endpoint.instance_id,
                endpoint.base_url,
                endpoint.rank_offset,
                endpoint.worker_world_size,
            )
            for endpoint in endpoints
        )
        if self._group is not None:
            if signature != self._topology_signature or int(world_size) != self._world_size:
                raise RuntimeError(
                    "Rollout topology changed after HCCL communicator initialization"
                )
            return

        self.validate_runtime()
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            HCCLWeightTransferEngine,
        )

        executor = ThreadPoolExecutor(max_workers=max(1, len(endpoints)))
        futures = [
            executor.submit(
                self._post,
                endpoint,
                "/init_weight_transfer_engine",
                {
                    "init_info": {
                        "master_address": self.master_address,
                        "master_port": self.master_port,
                        "rank_offset": endpoint.rank_offset,
                        "world_size": int(world_size),
                    }
                },
            )
            for endpoint in endpoints
        ]
        try:
            self._group = HCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": self.master_address,
                    "master_port": self.master_port,
                    "world_size": int(world_size),
                }
            )
            self._wait_all(futures)
        finally:
            executor.shutdown(wait=True)
        self._topology_signature = signature
        self._world_size = int(world_size)

    def prepare(
        self,
        endpoints: list[HCCLRolloutEndpoint],
        metadata: HCCLWeightMetadata,
    ) -> HCCLTransferSession:
        """Pause generation and start worker-side blocking receives."""
        if self._group is None:
            raise RuntimeError("HCCL communicator is not initialized")
        self._parallel_post(endpoints, "/pause", None)
        self._parallel_post(
            endpoints,
            "/start_weight_update",
            {"is_checkpoint_format": self.checkpoint_format},
        )
        update_info = {
            "names": list(metadata.names),
            "dtype_names": list(metadata.dtype_names),
            "shapes": [list(shape) for shape in metadata.shapes],
            "packed": self.packed,
            "packed_buffer_size_bytes": self.packed_buffer_size_bytes,
            "packed_num_buffers": self.packed_num_buffers,
        }
        executor = ThreadPoolExecutor(max_workers=max(1, len(endpoints)))
        futures = [
            executor.submit(
                self._post,
                endpoint,
                "/update_weights",
                {"update_info": update_info},
            )
            for endpoint in endpoints
        ]
        return HCCLTransferSession(
            executor=executor,
            futures=futures,
            started_at=time.perf_counter(),
        )

    def send(self, weights: Iterator[tuple[str, Any]]) -> None:
        """Broadcast a Megatron Bridge HF export through official HCCL."""
        if self._group is None:
            raise RuntimeError("HCCL communicator is not initialized")
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            HCCLTrainerSendWeightsArgs,
            HCCLWeightTransferEngine,
        )

        HCCLWeightTransferEngine.trainer_send_weights(
            iterator=weights,
            trainer_args=HCCLTrainerSendWeightsArgs(
                group=self._group,
                packed=self.packed,
                packed_buffer_size_bytes=self.packed_buffer_size_bytes,
                packed_num_buffers=self.packed_num_buffers,
            ),
        )

    def finish(
        self,
        session: HCCLTransferSession,
        endpoints: list[HCCLRolloutEndpoint],
    ) -> float:
        """Wait for loads, finalize model state, and resume generation."""
        try:
            self._wait_all(session.futures)
        finally:
            session.executor.shutdown(wait=True)
        self._parallel_post(endpoints, "/finish_weight_update", None)
        self._parallel_post(endpoints, "/resume", None)
        return time.perf_counter() - session.started_at

    def close(self) -> None:
        """Release the trainer-side communicator reference."""
        self._group = None
        self._topology_signature = None
        self._world_size = None

    def _parallel_post(
        self,
        endpoints: list[HCCLRolloutEndpoint],
        path: str,
        payload: dict[str, Any] | None,
    ) -> None:
        with ThreadPoolExecutor(max_workers=max(1, len(endpoints))) as executor:
            futures = [
                executor.submit(self._post, endpoint, path, payload)
                for endpoint in endpoints
            ]
            self._wait_all(futures)

    def _post(
        self,
        endpoint: HCCLRolloutEndpoint,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{endpoint.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"vLLM HCCL control request failed: "
                    f"instance={endpoint.instance_id} path={path} status={response.status}"
                )
            body = response.read()
        return json.loads(body.decode("utf-8")) if body else {}

    @staticmethod
    def _wait_all(futures: Iterable[Future]) -> None:
        for future in futures:
            future.result()

"""Support code for Cmlfq migration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class KVCacheShard:
    """K v cache shard implementation."""
    gpu_id: int
    tensor: "torch.Tensor"
    tp_rank: int


@dataclass
class MigrationLatencyProfile:
    """Migration latency profile implementation."""
    source_tp: int
    target_tp: int
    seq_len: int

    offload_ms: float
    reshard_ms: float
    network_ms: float
    reload_ms: float
    total_migration_ms: float
    recompute_prefill_ms: float


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class KVCacheManager:
    """K v cache manager implementation."""

    def __init__(self):
        # request_id -> {layer_idx -> (k_cache_cpu, v_cache_cpu)}
        self._cpu_cache: dict[str, dict] = {}

    def offload(
        self,
        request_id: str,
        kv_cache: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
    ) -> dict[int, tuple["torch.Tensor", "torch.Tensor"]]:
        """Offload."""
        cpu_cache = {}
        for layer_idx, (k_cache, v_cache) in kv_cache.items():

            k_cpu = k_cache.detach().cpu().pin_memory()
            v_cpu = v_cache.detach().cpu().pin_memory()
            cpu_cache[layer_idx] = (k_cpu, v_cpu)

        self._cpu_cache[request_id] = cpu_cache
        logger.debug(
            f"[KVCacheManager] Offload request={request_id}: "
            f"{len(cpu_cache)} layers"
        )
        return cpu_cache

    def reshard_kv_cache(
        self,
        request_id: str,
        source_tp: int,
        target_tp: int,
    ) -> dict[int, dict[int, tuple["torch.Tensor", "torch.Tensor"]]]:
        """Reshard kv cache."""
        cpu_cache = self._cpu_cache.get(request_id)
        if cpu_cache is None:
            raise ValueError(f"Request {request_id} KV cache not found in CPU memory")

        resharded = {}
        for layer_idx, (k_cpu, v_cpu) in cpu_cache.items():

            k_shards = self._reshard_tensor(k_cpu, source_tp, target_tp, dim=2)
            v_shards = self._reshard_tensor(v_cpu, source_tp, target_tp, dim=2)

            layer_shards = {}
            for rank in range(target_tp):
                layer_shards[rank] = (k_shards[rank], v_shards[rank])
            resharded[layer_idx] = layer_shards

        logger.debug(
            f"[KVCacheManager] Reshard request={request_id}: "
            f"TP {source_tp} -> {target_tp}, {len(resharded)} layers"
        )
        return resharded

    @staticmethod
    def _reshard_tensor(
        tensor: "torch.Tensor",
        source_tp: int,
        target_tp: int,
        dim: int = 2,
    ) -> list["torch.Tensor"]:
        """Reshard tensor."""
        import torch

        if source_tp <= 0 or target_tp <= 0:
            raise ValueError("source_tp and target_tp must be positive")
        if (
            source_tp % target_tp != 0
            and target_tp % source_tp != 0
        ):
            raise ValueError(
                "source_tp and target_tp must be integer multiples"
            )

        total_size = tensor.shape[dim]
        if total_size % source_tp != 0 or total_size % target_tp != 0:
            raise ValueError(
                f"dimension size {total_size} must be divisible by "
                f"source_tp={source_tp} and target_tp={target_tp}"
            )
        source_shard_size = total_size // source_tp


        if source_tp > 1:


            pass


        if target_tp > source_tp:

            ratio = target_tp // source_tp
            shards = []
            for s in range(source_tp):
                start = s * source_shard_size
                end = start + source_shard_size
                source_slice = tensor.narrow(dim, start, source_shard_size)
                sub_shard_size = source_shard_size // ratio
                for r in range(ratio):
                    sub_start = r * sub_shard_size
                    sub_shard = source_slice.narrow(dim, sub_start, sub_shard_size)
                    shards.append(sub_shard.contiguous())
            return shards
        elif target_tp < source_tp:

            ratio = source_tp // target_tp
            shards = []
            for t in range(target_tp):
                start_source = t * ratio
                slices = []
                for s in range(start_source, start_source + ratio):
                    slice_start = s * source_shard_size
                    slices.append(tensor.narrow(dim, slice_start, source_shard_size))
                merged = torch.cat(slices, dim=dim)
                shards.append(merged.contiguous())
            return shards
        else:

            shards = []
            for s in range(source_tp):
                start = s * source_shard_size
                shards.append(tensor.narrow(dim, start, source_shard_size).contiguous())
            return shards

    def reload(
        self,
        request_id: str,
        target_gpu_id: int,
        layer_shards: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
    ) -> dict[int, tuple["torch.Tensor", "torch.Tensor"]]:
        """Reload."""
        import torch

        gpu_cache = {}
        device = torch.device(f"cuda:{target_gpu_id}")
        for layer_idx, (k_shard, v_shard) in layer_shards.items():
            k_gpu = k_shard.to(device, non_blocking=True)
            v_gpu = v_shard.to(device, non_blocking=True)
            gpu_cache[layer_idx] = (k_gpu, v_gpu)

        logger.debug(
            f"[KVCacheManager] Reload request={request_id} to GPU {target_gpu_id}: "
            f"{len(gpu_cache)} layers"
        )
        return gpu_cache

    def release(self, request_id: str):
        """Release."""
        if request_id in self._cpu_cache:
            del self._cpu_cache[request_id]

    def clear(self):
        """Clear."""
        self._cpu_cache.clear()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class MigrationCostProfiler:
    """Migration cost profiler implementation."""

    def __init__(self, profile_path: str = ""):
        self._profile: dict[tuple[int, int, int], MigrationLatencyProfile] = {}
        self._profile_path = profile_path
        if profile_path and os.path.exists(profile_path):
            self.load_profile(profile_path)

    def record_measurement(
        self,
        source_tp: int,
        target_tp: int,
        seq_len: int,
        offload_ms: float,
        reshard_ms: float,
        network_ms: float,
        reload_ms: float,
        recompute_prefill_ms: float,
    ):
        """Record measurement."""
        key = (source_tp, target_tp, seq_len)
        self._profile[key] = MigrationLatencyProfile(
            source_tp=source_tp,
            target_tp=target_tp,
            seq_len=seq_len,
            offload_ms=offload_ms,
            reshard_ms=reshard_ms,
            network_ms=network_ms,
            reload_ms=reload_ms,
            total_migration_ms=offload_ms + reshard_ms + network_ms + reload_ms,
            recompute_prefill_ms=recompute_prefill_ms,
        )

    def decide_migration_path(
        self,
        source_tp: int,
        target_tp: int,
        seq_len: int,
        is_cross_node: bool = False,
    ) -> tuple[str, float]:
        """Decide migration path."""
        profile = self._lookup_profile(source_tp, target_tp, seq_len)

        if profile is None:



            return self._heuristic_decision(source_tp, target_tp, seq_len, is_cross_node)

        migration_cost = profile.total_migration_ms
        if not is_cross_node:

            migration_cost = profile.offload_ms + profile.reshard_ms + profile.reload_ms

        recompute_cost = profile.recompute_prefill_ms

        if migration_cost <= recompute_cost:
            return "migrate", migration_cost
        else:
            return "recompute", recompute_cost

    def _lookup_profile(
        self,
        source_tp: int,
        target_tp: int,
        seq_len: int,
    ) -> MigrationLatencyProfile | None:
        """Lookup profile."""
        exact = self._profile.get((source_tp, target_tp, seq_len))
        if exact is not None:
            return exact
        candidates = [
            profile
            for (src, dst, _), profile in self._profile.items()
            if src == source_tp and dst == target_tp
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item.seq_len - seq_len))

    def _heuristic_decision(
        self,
        source_tp: int,
        target_tp: int,
        seq_len: int,
        is_cross_node: bool,
    ) -> tuple[str, float]:
        """Heuristic decision."""

        crossover = 4096
        if is_cross_node:
            crossover = 2048

        if seq_len <= crossover:
            return "recompute", float(seq_len) * 0.5
        else:
            return "migrate", float(seq_len) * 0.3

    def save_profile(self, path: str):
        """Save profile."""
        data = []
        for prof in self._profile.values():
            data.append({
                "source_tp": prof.source_tp,
                "target_tp": prof.target_tp,
                "seq_len": prof.seq_len,
                "offload_ms": prof.offload_ms,
                "reshard_ms": prof.reshard_ms,
                "network_ms": prof.network_ms,
                "reload_ms": prof.reload_ms,
                "total_migration_ms": prof.total_migration_ms,
                "recompute_prefill_ms": prof.recompute_prefill_ms,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_profile(self, path: str):
        """Load profile."""
        with open(path, "r") as f:
            data = json.load(f)
        for entry in data:
            key = (entry["source_tp"], entry["target_tp"], entry["seq_len"])
            self._profile[key] = MigrationLatencyProfile(**entry)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class CMLFQMigrationCoordinator:
    """C m l f q migration coordinator implementation."""

    def __init__(
        self,
        kv_cache_manager: Optional[KVCacheManager] = None,
        cost_profiler: Optional[MigrationCostProfiler] = None,
    ):
        self.kv_manager = kv_cache_manager or KVCacheManager()
        self.profiler = cost_profiler or MigrationCostProfiler()

    def execute_full_migration(
        self,
        request_id: str,
        source_instance_index: int,
        source_tp: int,
        target_instance_index: int,
        target_tp: int,
        kv_cache: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
        seq_len: int,
        is_cross_node: bool = False,
    ) -> dict[int, tuple["torch.Tensor", "torch.Tensor"]]:
        """Execute full migration."""
        # Step 1: Offload
        self.kv_manager.offload(request_id, kv_cache)

        # Step 2: Reshard
        resharded = self.kv_manager.reshard_kv_cache(
            request_id, source_tp, target_tp
        )



        target_rank = target_instance_index % max(target_tp, 1)
        target_shards = {
            layer_idx: shards[target_rank]
            for layer_idx, shards in resharded.items()
        }


        target_gpu_id = target_instance_index
        result = self.kv_manager.reload(request_id, target_gpu_id, target_shards)


        self.kv_manager.release(request_id)

        logger.info(
            f"[MigrationCoordinator] Migration complete: {request_id}, "
            f"inst {source_instance_index}(TP{source_tp}) -> "
            f"inst {target_instance_index}(TP{target_tp}), "
            f"seq_len={seq_len}"
        )
        return result

    def decide_and_execute(
        self,
        request_id: str,
        source_instance_index: int,
        source_tp: int,
        target_instance_index: int,
        target_tp: int,
        kv_cache: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
        seq_len: int,
        is_cross_node: bool = False,
    ) -> tuple[str, Optional[dict]]:
        """Decide and execute."""
        decision, latency = self.profiler.decide_migration_path(
            source_tp, target_tp, seq_len, is_cross_node
        )

        if decision == "migrate":
            target_kv = self.execute_full_migration(
                request_id, source_instance_index, source_tp,
                target_instance_index, target_tp,
                kv_cache, seq_len, is_cross_node,
            )
            return "migrate", target_kv
        else:
            return "recompute", None

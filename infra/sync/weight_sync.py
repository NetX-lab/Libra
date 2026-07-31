"""Support code for Weight sync."""

import os
import time
from typing import Any

import torch
import torch.distributed as dist


class DiskWeightSync:
    """Disk weight sync implementation."""

    def __init__(self, sync_path: str):
        self.sync_path = sync_path
        self.current_version = 0
        os.makedirs(sync_path, exist_ok=True)

    def save_weights(
        self,
        model: Any,
        version: int,
        is_main_process: bool = True,
        is_distributed: bool = False,
    ):
        """Save weights."""
        save_path = os.path.join(self.sync_path, f"v{version}")
        os.makedirs(save_path, exist_ok=True)

        if is_main_process:
            start_time = time.time()

            if is_distributed:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.state_dict_type(model, "FULL_STATE_DICT"):
                    state_dict = model.state_dict()
                    torch.save(state_dict, os.path.join(save_path, "model.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

            meta = {
                "version": version,
                "timestamp": time.time(),
                "save_time": time.time() - start_time,
            }
            torch.save(meta, os.path.join(save_path, "meta.pt"))

            print(f"Weights saved: {save_path} ({meta['save_time']:.2f}s)")

        if is_distributed:
            dist.barrier()

        self.current_version = version

    def load_weights(
        self,
        model: Any,
        version: int | None = None,
        is_main_process: bool = True,
        is_distributed: bool = False,
    ) -> int:
        """Load weights."""
        if version is None:
            version = self._get_latest_version()

        load_path = os.path.join(self.sync_path, f"v{version}")

        if not os.path.exists(load_path):
            print(f"WARNING: Weight path does not exist: {load_path}")
            return self.current_version

        if is_main_process:
            start_time = time.time()
            state_dict = torch.load(
                os.path.join(load_path, "model.pt"),
                map_location="cpu",
                weights_only=True,
            )


            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if is_distributed:
                with FSDP.state_dict_type(model, "FULL_STATE_DICT"):
                    model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

            load_time = time.time() - start_time
            print(f"Weights loaded: {load_path} ({load_time:.2f}s)")

        if is_distributed:
            dist.barrier()

        self.current_version = version
        return version

    def _get_latest_version(self) -> int:
        """Get latest version."""
        if not os.path.exists(self.sync_path):
            return 0

        versions = []
        for name in os.listdir(self.sync_path):
            if name.startswith("v") and os.path.isdir(os.path.join(self.sync_path, name)):
                try:
                    version = int(name[1:])
                    versions.append(version)
                except ValueError:
                    continue

        return max(versions) if versions else 0

    def cleanup_old_versions(self, keep_last_n: int = 3):
        """Cleanup old versions."""
        if not os.path.exists(self.sync_path):
            return

        versions = []
        for name in os.listdir(self.sync_path):
            if name.startswith("v") and os.path.isdir(os.path.join(self.sync_path, name)):
                try:
                    version = int(name[1:])
                    versions.append((version, os.path.join(self.sync_path, name)))
                except ValueError:
                    continue


        versions.sort(key=lambda x: x[0], reverse=True)


        for version, path in versions[keep_last_n:]:
            import shutil
            print(f"Removed old version: {path}")
            shutil.rmtree(path, ignore_errors=True)


class NCCLWeightSync:
    """N c c l weight sync implementation."""

    def __init__(
        self,
        train_world_size: int,
        rollout_world_size: int,
    ):
        self.train_world_size = train_world_size
        self.rollout_world_size = rollout_world_size
        self.current_version = 0


        self.sync_group = None
        self.sync_group_ranks = None

    def initialize_sync_group(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
    ):
        """Initialize sync group."""
        print(f"Initializing NCCL synchronization group: {master_addr}:{master_port}, rank={rank}, world_size={world_size}")


        orig_master_addr = os.environ.get("MASTER_ADDR")
        orig_master_port = os.environ.get("MASTER_PORT")
        orig_rank = os.environ.get("RANK")
        orig_world_size = os.environ.get("WORLD_SIZE")


        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        try:



            self.sync_group = dist.new_group(
                ranks=list(range(world_size)),
                backend="nccl",
            )
            self.sync_group_ranks = list(range(world_size))
            print(f"NCCL synchronization group initialized")
        finally:

            if orig_master_addr is not None:
                os.environ["MASTER_ADDR"] = orig_master_addr
            if orig_master_port is not None:
                os.environ["MASTER_PORT"] = orig_master_port
            if orig_rank is not None:
                os.environ["RANK"] = orig_rank
            if orig_world_size is not None:
                os.environ["WORLD_SIZE"] = orig_world_size

    def broadcast_weights(
        self,
        model: Any,
        src_rank: int = 0,
    ):
        """Broadcast weights."""
        if self.sync_group is None:
            raise RuntimeError("NCCL synchronization group is not initialized")

        start_time = time.time()


        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(model, "FULL_STATE_DICT"):
            state_dict = model.state_dict()


        for key, param in state_dict.items():
            param.data = param.data.cuda()
            dist.broadcast(
                param.data,
                src=src_rank,
                group=self.sync_group,
            )

        dist.barrier(group=self.sync_group)

        broadcast_time = time.time() - start_time
        print(f"NCCL weight broadcast complete: {broadcast_time:.2f}s")

        self.current_version += 1

    def sync_and_update(
        self,
        train_model: Any,
        rollout_model: Any,
        train_rank: int = 0,
    ):
        """Sync and update."""
        if self.sync_group is None:
            raise RuntimeError("NCCL synchronization group is not initialized")


        with FSDP.state_dict_type(train_model, "FULL_STATE_DICT"):
            train_state = train_model.state_dict()


        for key in train_state.keys():
            if key in rollout_model.state_dict():
                dist.broadcast(
                    train_state[key],
                    src=train_rank,
                    group=self.sync_group,
                )


        with FSDP.state_dict_type(rollout_model, "FULL_STATE_DICT"):
            rollout_model.load_state_dict(train_state)

        dist.barrier(group=self.sync_group)
        self.current_version += 1
        print(f"NCCL weight synchronization complete: version={self.current_version}")


class WeightSyncFactory:
    """Weight sync factory implementation."""

    @staticmethod
    def create_sync(
        mode: str,
        sync_path: str = "./logs/async_rl_weights",
        train_world_size: int = 4,
        rollout_world_size: int = 4,
    ):
        """Create sync."""
        if mode == "disk":
            return DiskWeightSync(sync_path=sync_path)
        elif mode == "nccl":
            return NCCLWeightSync(
                train_world_size=train_world_size,
                rollout_world_size=rollout_world_size,
            )
        else:
            raise ValueError(f"Unsupported weight synchronization mode: {mode}")

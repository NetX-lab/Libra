"""Support code for Async rl trainer."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import glob
import itertools
import os
import shutil
import socket
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.engine.rollout_engine import MockRolloutEngine, MultiInstanceRolloutEngine
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.engine.train_engine import TrainEngine
from RL_Framework.engine.train_factory import create_train_engine
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor
from RL_Framework.infra.execution.async_runner import AsyncTaskRunner
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
from RL_Framework.infra.observability.history_collector import HistoryDataCollector, ResourceConfig
from RL_Framework.infra.observability.phase_tracer import PhaseTracer
from RL_Framework.infra.sync.staleness import StalenessManager
from RL_Framework.infra.sync.hccl_weight_transfer import (
    HCCLWeightMetadata,
    OfficialHCCLWeightTransfer,
    build_hccl_rollout_endpoints,
    collect_weight_metadata,
)
from RL_Framework.infra.sync.weight_sync import WeightSyncFactory


@dataclass
class RolloutTaskInput:
    """Rollout task input implementation."""

    task_id: int
    data: dict[str, Any]
    version: int


class AsyncRLTrainer:
    """Async r l trainer implementation."""

    def __init__(self, config: AsyncRLConfig):
        self.config = config


        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_main_process = self.rank == 0


        self.train_engine: TrainEngine | None = None
        self.rollout_engine: MultiInstanceRolloutEngine | HeterogeneousRolloutEngine | None = None
        self._use_heterogeneous = False
        self.weight_sync = None
        self.async_runner: AsyncTaskRunner | None = None
        self.staleness_manager: StalenessManager | None = None
        self.dispatcher: BatchTaskDispatcher | None = None


        self.dataloader: DataLoader | None = None


        self.global_step = 0
        self.stats = {}


        self.workflow = None


        self._task_counter = 0
        self._pending_grpo_groups: dict[str, list[dict[str, Any]]] = {}

        # WandB
        self.wandb_run = None


        self.history_collector: HistoryDataCollector | None = None
        self._resource_config_snapshot: ResourceConfig | None = None
        self.global_resource_planner: GlobalResourcePlanner | None = None
        self.runtime_elastic_executor: RuntimeElasticExecutor | None = None
        self._grp_executor: ThreadPoolExecutor | None = None
        self._grp_future: Future | None = None
        self._grp_future_step: int | None = None
        self._runtime_reconfiguration_coord_id: str = ""
        self._runtime_follow_stop = threading.Event()
        self._runtime_follow_thread: threading.Thread | None = None
        self._rollout_engine_rebind_lock = threading.RLock()
        # External restartable vLLM workers keep their launch-time IDs even if
        # GRP later rebuilds the in-process routing topology.  Preserve those
        # physical IDs so checkpoint reload acknowledgements cannot be changed
        # underneath an active set of external workers.
        self._external_rollout_reload_instance_ids: list[str] = []
        self._hccl_weight_transfer: OfficialHCCLWeightTransfer | None = None
        self._hccl_weight_metadata: HCCLWeightMetadata | None = None
        self._hccl_metadata_initialized = False
        self._trace_train_phases = os.environ.get("RL_TRAIN_PHASE_TRACE", "0") == "1"
        trace_enabled = self._trace_train_phases or bool(
            getattr(config, "phase_trace_enabled", False)
        )
        trace_dir = (
            os.environ.get("RL_TRAIN_PHASE_TRACE_DIR", "")
            or getattr(config, "phase_trace_dir", "")
            or os.path.join(getattr(config, "log_dir", "./logs"), "phase_trace")
        )
        self._phase_tracer = (
            PhaseTracer(
                Path(trace_dir) / f"phase_trace_rank{self.rank}.jsonl",
                rank=self.rank,
                world_size=self.world_size,
            )
            if trace_enabled
            else None
        )

    def setup(self, workflow):
        """Setup."""
        self.workflow = workflow

        if self.is_main_process:
            print("=" * 60)
            print("Initializing the asynchronous RL trainer (pipeline mode)")
            print("=" * 60)


        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        if self.is_main_process:
            print("Initializing the training engine...")
        self.train_engine = create_train_engine(self.config)
        self.train_engine.initialize(max_seq_length=self.config.max_seq_length)


        if self.is_main_process and self.train_engine.is_batch_source():
            print("Connecting to vLLM inference services...")


        if self.train_engine.is_batch_source():
            hetero_cfg = getattr(self.config, "heterogeneous_rollout", None)
            if hetero_cfg and hetero_cfg.enabled and hetero_cfg.instances:
                self._use_heterogeneous = True
                self._setup_heterogeneous_engine(hetero_cfg)
            else:
                self._use_heterogeneous = False
                self._setup_homogeneous_engine()

        if self.is_main_process and self.train_engine.is_batch_source():
            self.rollout_engine.wait_for_ready(timeout=300)
            n_inst = self.rollout_engine.num_instances
            print(f"vLLM inference services are ready ({n_inst} instances)")
            print(f"  Instance URLs: {self.rollout_engine.instance_urls}")
            if self._use_heterogeneous:
                print(f"  Heterogeneous mode: TP layout={self.rollout_engine.tp_list}")
                self._external_rollout_reload_instance_ids = [
                    str(cfg["instance_id"])
                    for cfg in self.rollout_engine.instance_configs
                ]
        if dist.is_initialized():
            dist.barrier()


        if self.config.weight_sync_mode == "hccl":
            self.weight_sync = None
        else:
            self.weight_sync = WeightSyncFactory.create_sync(
                mode=self.config.weight_sync_mode,
                sync_path=self.config.sync_path,
                train_world_size=self.config.train_gpus,
                rollout_world_size=self.config.rollout_gpus,
            )


        if self.train_engine.is_batch_source():
            self.staleness_manager = StalenessManager(
                version_provider=self.train_engine,
                max_concurrent_rollouts=self.config.max_concurrent_rollouts,
                consumer_batch_size=self.train_engine.get_local_batch_size(
                    self.config.batch_size
                ),
                max_staleness=self.config.max_head_offpolicyness,
            )


            self.async_runner = AsyncTaskRunner(
                max_queue_size=self.config.queue_size,
                enable_tracing=self.config.enable_rollout_tracing,
            )
            self.async_runner.initialize()


            self.dispatcher = BatchTaskDispatcher(
                async_runner=self.async_runner,
                staleness_manager=self.staleness_manager,
                task_factory=self._create_rollout_task,
                enable_tracing=self.config.enable_rollout_tracing,
            )
            self.dispatcher.initialize()

        if self.is_main_process:
            parallel_state = self.train_engine.get_parallel_state()
            print(f"The asynchronous pipeline is initialized:")
            print(
                f"  train_backend={parallel_state.get('backend')} "
                f"(tp={parallel_state.get('train_tp')}, "
                f"pp={parallel_state.get('train_pp')}, "
                f"dp={parallel_state.get('train_dp')})"
            )
            print(f"  max_concurrent_rollouts={self.config.max_concurrent_rollouts}")
            print(f"  max_head_offpolicyness={self.config.max_head_offpolicyness}")
            print(f"  queue_size={self.config.queue_size}")
            print(f"  recompute_logprobs={self.config.recompute_logprobs}")
            print("=" * 60)


        self._init_wandb()


        self._init_history_collector()


        self._init_global_resource_planner()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _setup_heterogeneous_engine(self, hetero_cfg):
        """Setup heterogeneous engine."""
        self.rollout_engine = HeterogeneousRolloutEngine.from_config(self.config)
        if self.is_main_process:
            scheduler_type = getattr(hetero_cfg.scheduling, "scheduler_type", "length_aware")
            print(f"Using the heterogeneous rollout engine:")
            print(f"  Scheduler type: {scheduler_type}")
            print(f"  Number of instances: {self.rollout_engine.num_instances}")
            print(f"  TP layout: {self.rollout_engine.tp_list}")
            for cfg in self.rollout_engine.instance_configs:
                print(f"    {cfg['instance_id']}: TP={cfg['tp_degree']}, "
                      f"GPU={cfg['gpu_ids']}, port={cfg['port']}")

    def _setup_homogeneous_engine(self):
        """Setup homogeneous engine."""
        rollout_backend = str(getattr(self.config, "rollout_backend", "vllm")).lower()
        if rollout_backend == "mock":
            self.rollout_engine = MockRolloutEngine(model_path=self.config.model_path)
            if self.is_main_process:
                print("Using mock rollout engine for training-loop smoke testing")
            return

        vllm_endpoints_str = getattr(self.config, "vllm_endpoints", "") or ""
        if not vllm_endpoints_str:
            vllm_endpoints_str = os.environ.get("VLLM_ENDPOINTS", "")

        if vllm_endpoints_str:
            endpoints = [ep.strip() for ep in vllm_endpoints_str.split(",") if ep.strip()]
            self.rollout_engine = MultiInstanceRolloutEngine(
                model_path=self.config.model_path,
                endpoints=endpoints,
            )
            if self.is_main_process:
                print(f"Using multi-host vLLM endpoints: {endpoints}")
        else:
            num_instances = self.config.vllm_num_instances
            if num_instances <= 0:
                num_instances = max(
                    1, self.config.rollout_gpus // max(1, self.config.vllm_tp_size)
                )
            self.rollout_engine = MultiInstanceRolloutEngine(
                host=self.config.vllm_host,
                base_port=self.config.vllm_port,
                num_instances=num_instances,
                model_path=self.config.model_path,
            )

    # ----------------------------------------------------------------
    # Task Factory
    # ----------------------------------------------------------------

    def _create_rollout_task(self, task_input: TaskInput):
        """Create rollout task."""
        workflow = self.workflow
        staleness_manager = self.staleness_manager
        train_engine = self.train_engine
        max_staleness = self.config.max_head_offpolicyness

        async def _execute():
            try:
                self._refresh_rollout_engine_from_manifest_for_task()
                rollout_engine = self.rollout_engine
                if rollout_engine is None:
                    raise RuntimeError("rollout engine is not initialized")

                trajectory = await workflow.run_episode(
                    rollout_engine,
                    task_input.data,
                    version=task_input.version,
                    rollout_index=task_input.rollout_index,
                )
                trajectory["grpo_group_id"] = (
                    task_input.group_id or f"task:{task_input.task_id}"
                )
                trajectory["grpo_rollout_index"] = task_input.rollout_index


                current_version = train_engine.get_version()
                traj_version = trajectory.get("versions", torch.tensor([task_input.version]))
                version_val = traj_version.min().item()
                staleness = current_version - version_val

                if staleness <= max_staleness:
                    staleness_manager.on_rollout_accepted()
                    return trajectory
                else:
                    staleness_manager.on_rollout_rejected()
                    if self.is_main_process and self.config.enable_rollout_tracing:
                        print(
                            f"REJECT rollout task_id={task_input.task_id}: "
                            f"version={version_val}, current={current_version}, "
                            f"staleness={staleness} > {max_staleness}"
                        )
                    return None

            except Exception as e:
                staleness_manager.on_rollout_rejected()
                if self.is_main_process:
                    print(f"ERROR: Rollout task_id={task_input.task_id} failed: {e}")
                return None

        return _execute

    def _create_data_generator(self, dataset, workflow):
        """Create data generator."""
        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.train_engine.get_data_parallel_world_size(),
            rank=self.train_engine.get_data_parallel_rank(),
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=lambda batch: batch,
        )

        n_samples = int(self.config.n_samples)
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive; got {n_samples}")

        for epoch in itertools.count():
            sampler.set_epoch(epoch)
            for sample_index, batch in enumerate(dataloader):
                for item in batch:
                    group_id = (
                        f"dp{self.train_engine.get_data_parallel_rank()}:"
                        f"epoch{epoch}:sample{sample_index}"
                    )
                    for rollout_index in range(n_samples):
                        task_id = self._task_counter
                        self._task_counter += 1
                        if not isinstance(item, dict):
                            raise TypeError(
                                "GRPO items must be dictionaries so that a stable group ID can be attached"
                            )
                        yield TaskInput(
                            task_id=task_id,
                            data=item,
                            version=self.train_engine.get_version(),
                            group_id=group_id,
                            rollout_index=rollout_index,
                        )

    def _collect_complete_grpo_batch(self, data_gen, batch_size: int):
        """Collect complete prompt groups even when individual rollouts are rejected."""
        n_samples = int(self.config.n_samples)
        selected: list[dict[str, Any]] = []
        planner_cfg = getattr(self.config, "global_resource_planner", None)
        collection_timeout = float(
            getattr(planner_cfg, "runtime_batch_collection_timeout_s", 0.0)
            if planner_cfg is not None
            else 0.0
        )
        max_retries = int(
            getattr(planner_cfg, "runtime_batch_collection_max_retries", 0)
            if planner_cfg is not None
            else 0
        )
        retries = 0
        is_main_process = getattr(self, "is_main_process", True)
        while len(selected) < batch_size:
            if not is_main_process:
                self._follow_applied_runtime_reconfiguration_if_available(
                    self._runtime_reconfiguration_coord_dir()
                )
            try:
                raw_batch = self.dispatcher.active_submit_and_wait(
                    input_generator=data_gen,
                    batch_size=batch_size,
                    timeout=collection_timeout if collection_timeout > 0 else None,
                )
            except TimeoutError as exc:
                retries += 1
                if is_main_process:
                    print(
                        "[BatchCollection] timeout "
                        f"retry={retries}/{max_retries} "
                        f"error={exc}"
                    )
                self._reset_rollout_pipeline_after_reconfigure()
                if retries > max_retries:
                    raise
                continue
            if self.staleness_manager is not None:
                self.staleness_manager.on_batch_consumed(len(raw_batch))
            for trajectory in raw_batch:
                group_id = str(trajectory["grpo_group_id"])
                self._pending_grpo_groups.setdefault(group_id, []).append(trajectory)

            for group_id in list(self._pending_grpo_groups):
                members = self._pending_grpo_groups[group_id]
                if len(members) < n_samples:
                    continue
                remaining = batch_size - len(selected)
                if remaining < n_samples:
                    break
                selected.extend(members[:n_samples])
                del self._pending_grpo_groups[group_id]

            # Rejected tasks can leave incomplete groups forever. Bound their memory.
            max_pending_groups = max(64, batch_size * 8)
            while len(self._pending_grpo_groups) > max_pending_groups:
                oldest_group = next(iter(self._pending_grpo_groups))
                del self._pending_grpo_groups[oldest_group]
        return selected

    def _compute_advantages(self, trajectories):
        """Compute advantages."""
        if not trajectories:
            return trajectories

        grouped: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(list)
        for index, traj in enumerate(trajectories):
            rewards = traj["rewards"].reshape(-1).float()
            if rewards.numel() != 1:
                raise ValueError(
                    "The current GRPO implementation requires exactly one outcome reward per trajectory"
                )
            group_id = str(traj.get("grpo_group_id", f"singleton:{index}"))
            grouped[group_id].append((traj, rewards[0]))

        zero_variance_groups = 0
        singleton_groups = 0
        for members in grouped.values():
            scores = torch.stack([score for _, score in members])
            if len(members) == 1:
                # Match verl's singleton fallback: no group baseline is available.
                normalized = scores
                singleton_groups += 1
            else:
                mean = scores.mean()
                std = scores.std()
                if not torch.isfinite(std) or std <= 1e-6:
                    normalized = torch.zeros_like(scores)
                    zero_variance_groups += 1
                else:
                    normalized = (scores - mean) / (std + 1e-6)
            for (traj, _), advantage in zip(members, normalized):
                traj["advantages"] = advantage.reshape(1)

        self._advantage_stats = {
            "grpo_num_groups": len(grouped),
            "grpo_mean_group_size": len(trajectories) / max(len(grouped), 1),
            "grpo_singleton_groups": singleton_groups,
            "grpo_zero_variance_groups": zero_variance_groups,
        }

        return trajectories

    def _wait_for_all_ranks_after_rollout(
        self,
        step: int,
        trajectories: list[dict[str, Any]],
    ) -> None:
        """Wait outside NCCL until all ranks finish long rollout collection."""
        if not dist.is_initialized() or self.world_size <= 1:
            return

        sync_root = Path(
            os.environ.get(
                "ROLLOUT_RANK_READY_DIR",
                os.path.join(self.config.log_dir, "rank_ready"),
            )
        )
        job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "local"))
        step_dir = sync_root / f"job_{job_id}" / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        ready_path = step_dir / f"rank_{self.rank}.json"
        self._write_json_atomic(
            ready_path,
            {
                "rank": int(self.rank),
                "local_rank": int(self.local_rank),
                "world_size": int(self.world_size),
                "batch_size": len(trajectories),
                "sequence_lengths": [
                    int(traj["input_ids"].shape[-1]) for traj in trajectories
                ],
                "output_lengths": [
                    int(traj.get("output_len", 0)) for traj in trajectories
                ],
                "turn_counts": [
                    int(traj.get("n_turns", 0)) for traj in trajectories
                ],
                "ready_at": time.time(),
            },
        )

        timeout = float(os.environ.get("ROLLOUT_RANK_READY_TIMEOUT", "86400"))
        poll_interval = float(os.environ.get("ROLLOUT_RANK_READY_POLL_INTERVAL", "2"))
        deadline = time.monotonic() + timeout
        expected = [step_dir / f"rank_{rank}.json" for rank in range(self.world_size)]
        while True:
            missing = [path.name for path in expected if not path.exists()]
            if not missing:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for rollout completion at step {step}; "
                    f"missing={missing[:8]}"
                )
            time.sleep(poll_interval)

        planner_cfg = getattr(self.config, "global_resource_planner", None)
        use_barrier = bool(
            getattr(planner_cfg, "runtime_use_nccl_barrier_after_rollout", False)
            if planner_cfg is not None
            else False
        )
        if use_barrier:
            dist.barrier()

    def _wait_for_all_ranks_before_weight_sync(self, step: int) -> None:
        """Wait outside NCCL until all ranks are ready to checkpoint weights."""
        if not dist.is_initialized() or self.world_size <= 1:
            return

        sync_root = Path(
            os.environ.get(
                "WEIGHT_SYNC_RANK_READY_DIR",
                os.path.join(self.config.log_dir, "weight_sync_rank_ready"),
            )
        )
        job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "local"))
        step_dir = sync_root / f"job_{job_id}" / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        ready_path = step_dir / f"rank_{self.rank}.json"
        self._write_json_atomic(
            ready_path,
            {
                "rank": int(self.rank),
                "local_rank": int(self.local_rank),
                "world_size": int(self.world_size),
                "is_batch_source": bool(self.train_engine.is_batch_source()),
                "ready_at": time.time(),
            },
        )

        timeout = float(
            os.environ.get(
                "WEIGHT_SYNC_RANK_READY_TIMEOUT",
                os.environ.get("ROLLOUT_RANK_READY_TIMEOUT", "86400"),
            )
        )
        poll_interval = float(
            os.environ.get("WEIGHT_SYNC_RANK_READY_POLL_INTERVAL", "2")
        )
        deadline = time.monotonic() + timeout
        expected = [step_dir / f"rank_{rank}.json" for rank in range(self.world_size)]
        while True:
            missing = [path.name for path in expected if not path.exists()]
            if not missing:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for weight-sync readiness at step {step}; "
                    f"missing={missing[:8]}"
                )
            time.sleep(poll_interval)

        planner_cfg = getattr(self.config, "global_resource_planner", None)
        use_barrier = bool(
            getattr(planner_cfg, "runtime_use_nccl_barrier_before_weight_sync", False)
            if planner_cfg is not None
            else False
        )
        if use_barrier:
            dist.barrier()

    def _trace_train_phase(self, step: int, phase: str, **details: Any) -> None:
        phase = str(phase)
        if self._phase_tracer is not None:
            if phase.endswith("_start"):
                self._phase_tracer.start(step, phase[:-6], details)
            elif phase.endswith("_done"):
                self._phase_tracer.end(step, phase[:-5], details)
            else:
                self._phase_tracer.end(step, phase, details)
        if not self._trace_train_phases:
            return
        payload = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
        if payload:
            payload = " " + payload
        print(
            f"[TrainPhase] rank={self.rank} step={step} phase={phase}{payload}",
            flush=True,
        )

    def train(self, workflow, dataset=None):
        """Train."""
        self.setup(workflow)

        if dataset is None:
            raise ValueError("dataset cannot be None")


        data_gen = self._create_data_generator(dataset, workflow)

        local_batch_size = self.train_engine.get_local_batch_size(self.config.batch_size)
        if local_batch_size % int(self.config.n_samples) != 0:
            raise ValueError(
                "Each data-parallel replica's local batch size must be divisible by n_samples: "
                f"local_batch_size={local_batch_size}, n_samples={self.config.n_samples}"
            )

        if self.is_main_process:
            print("\nStarting asynchronous pipeline training")
            print(f"Total steps: {self.config.total_steps}")
            print(f"global_batch_size: {self.config.batch_size}")
            print(f"local_batch_size_per_dp_replica: {local_batch_size}")
            print("=" * 60)

        start_time = time.time()

        for step in range(self.config.total_steps):
            self.global_step = step
            step_start = time.time()
            self._trace_train_phase(step, "step_start")

            if not self.is_main_process:
                self._trace_train_phase(step, "follow_reconfig_start")
                self._follow_runtime_reconfiguration_if_requested()
                self._trace_train_phase(step, "follow_reconfig_done")

            if self._use_heterogeneous and hasattr(self.rollout_engine, "notify_epoch_start"):
                self.rollout_engine.notify_epoch_start(epoch=step)


            batch = None
            if self.train_engine.is_batch_source():
                self._trace_train_phase(step, "collect_batch_start")
                batch = self._collect_complete_grpo_batch(
                    data_gen=data_gen,
                    batch_size=local_batch_size,
                )
                self._trace_train_phase(
                    step,
                    "collect_batch_done",
                    batch_size=len(batch),
                )
            self._trace_train_phase(step, "distribute_start")
            batch = self.train_engine.distribute_trajectories(batch)
            self._trace_train_phase(
                step,
                "distribute_done",
                batch_size=len(batch) if batch is not None else 0,
            )
            self._trace_train_phase(step, "rollout_rank_wait_start")
            self._wait_for_all_ranks_after_rollout(step, batch)
            self._trace_train_phase(step, "rollout_rank_wait_done")

            if not batch:
                if self.is_main_process:
                    print(f"Step {step}: received no valid trajectories; skipping")
                continue

            align_trajectories = getattr(
                self.train_engine,
                "align_distributed_trajectories",
                None,
            )
            if callable(align_trajectories):
                self._trace_train_phase(step, "align_start")
                batch = align_trajectories(batch)
                self._trace_train_phase(step, "align_done")

            rollout_time = time.time() - step_start


            adv_start = time.time()
            self._trace_train_phase(step, "advantage_start")
            batch = self._compute_advantages(batch)
            advantage_time = time.time() - adv_start
            self._trace_train_phase(step, "advantage_done")


            recompute_time = 0.0
            if self.config.recompute_logprobs:
                recompute_start = time.time()
                self._trace_train_phase(step, "recompute_logprobs_start")
                self.train_engine.recompute_logprobs(batch)
                recompute_time = time.time() - recompute_start
                self._trace_train_phase(
                    step,
                    "recompute_logprobs_done",
                    seconds=f"{recompute_time:.3f}",
                )


            train_start = time.time()
            self._trace_train_phase(step, "grpo_update_start")
            stats = self.train_engine.grpo_update(
                batch,
                ppo_epochs=self.config.ppo_epochs,
            )
            reward_start = time.time()
            self._trace_train_phase(step, "reward_compute_start")
            reward_values = torch.cat(
                [traj["rewards"].reshape(-1).float() for traj in batch]
            )
            stats.update(
                {
                    "reward_mean": float(reward_values.mean()),
                    "reward_std": float(reward_values.std(unbiased=False)),
                    "reward_min": float(reward_values.min()),
                    "reward_max": float(reward_values.max()),
                }
            )
            self._trace_train_phase(
                step,
                "reward_compute_done",
                seconds=f"{time.time() - reward_start:.6f}",
                reward_count=int(reward_values.numel()),
            )
            train_time = time.time() - train_start
            self._trace_train_phase(
                step,
                "grpo_update_done",
                seconds=f"{train_time:.3f}",
            )


            weight_sync_time = 0.0
            is_final_step = step == self.config.total_steps - 1
            if (
                self.config.sync_interval > 0
                and step > 0
                and (step % self.config.sync_interval == 0 or is_final_step)
            ):
                self._trace_train_phase(step, "weight_sync_start")
                if self.dispatcher is not None:
                    self.dispatcher.pause()
                    self.dispatcher.wait_until_idle(
                        timeout=float(
                            getattr(
                                self.config,
                                "rollout_weight_sync_timeout_s",
                                1200.0,
                            )
                        )
                    )

                self._wait_for_all_ranks_before_weight_sync(step)
                sync_start = time.time()
                skip_rollout_sync = (
                    str(getattr(self.config, "rollout_weight_sync_mode", "")).lower()
                    == "none"
                    and not bool(
                        getattr(self.config, "require_rollout_weight_sync", True)
                    )
                )
                if skip_rollout_sync:
                    self._record_weight_sync_phase("sync_skipped")
                else:
                    self._sync_weights(target_version=step + 1)
                weight_sync_time = time.time() - sync_start

                new_version = step + 1
                self.train_engine.set_version(new_version)
                self._rebind_rollout_engine_from_config(reason="after_weight_sync")

                if self.dispatcher is not None:
                    self.dispatcher.resume()
                self._trace_train_phase(step, "weight_sync_done")

                if self.is_main_process and self.config.enable_rollout_tracing:
                    sm_stats = (
                        self.staleness_manager.get_stats()
                        if self.staleness_manager is not None
                        else None
                    )
                    print(
                        f"Weight synchronization complete: version={new_version}, "
                        f"running={getattr(sm_stats, 'running', 0)}, "
                        f"accepted={getattr(sm_stats, 'accepted', 0)}"
                    )

            step_total_time = time.time() - step_start


            stats["n_trajectories"] = len(batch)
            stats["rollout_time"] = rollout_time
            stats["train_time"] = train_time
            stats["weight_sync_time"] = weight_sync_time
            stats["advantage_time"] = advantage_time
            stats["recompute_logprob_time"] = recompute_time
            stats["step_time"] = step_total_time
            stats["version"] = self.train_engine.get_version()
            stats.update(getattr(self, "_advantage_stats", {}))
            self.stats = stats


            self._log_to_wandb(stats, step)


            if self.is_main_process:
                self._record_history_step(step, batch, stats,
                                          rollout_time, train_time,
                                          weight_sync_time, advantage_time,
                                          recompute_time, step_total_time)
                self._trace_train_phase(step, "resource_planner_start")
                self._run_global_resource_planner(step, batch, stats)
                self._trace_train_phase(step, "resource_planner_done")
            else:
                self._trace_train_phase(step, "follow_reconfig_after_step_start")
                self._follow_runtime_reconfiguration_if_requested()
                self._trace_train_phase(step, "follow_reconfig_after_step_done")


            self._run_periodic_evaluation(workflow, dataset, step)


            if (
                self.is_main_process
                and (
                    step % 10 == 0
                    or self.config.total_steps <= 10
                    or step == self.config.total_steps - 1
                )
            ):
                elapsed = time.time() - start_time
                sm_stats = (
                    self.staleness_manager.get_stats()
                    if self.staleness_manager is not None
                    else None
                )
                print(f"\nStep {step}/{self.config.total_steps}")
                print(f"  Time: {elapsed:.1f}s (rollout={rollout_time:.1f}s, train={train_time:.1f}s)")
                print(f"  Loss: {stats.get('loss', 0):.4f}")
                print(f"  PG Loss: {stats.get('pg_loss', 0):.4f}")
                print(f"  KL: {stats.get('kl', 0):.4f}")
                print(f"  Reward: {stats.get('reward_mean', 0):.4f}")
                print(f"  Trajectories: {len(batch)}")
                print(f"  Version: {self.train_engine.get_version()}")
                print(
                    "  Pipeline: "
                    f"running={getattr(sm_stats, 'running', 0)}, "
                    f"accepted={getattr(sm_stats, 'accepted', 0)}"
                )


            if self._use_heterogeneous and hasattr(self.rollout_engine, "notify_epoch_end"):
                self.rollout_engine.notify_epoch_end(epoch=step)


            if (
                self.config.save_interval > 0
                and step > 0
                and (step % self.config.save_interval == 0 or is_final_step)
            ):
                self._save_checkpoint()


        if self.is_main_process:
            total_time = time.time() - start_time
            print("\n" + "=" * 60)
            print(f"Training complete. Total time: {total_time:.1f}s")
            print("=" * 60)

        self._cleanup()


    def _run_periodic_evaluation(self, workflow, dataset, step: int):
        """Run rank-0 evaluation while other ranks wait outside NCCL collectives."""
        interval = int(getattr(self.config, "eval_interval", 0) or 0)
        is_final_step = step == self.config.total_steps - 1
        if (
            interval <= 0
            or step <= 0
            or (step % interval != 0 and not is_final_step)
        ):
            return
        if not hasattr(workflow, "evaluate") or self.rollout_engine is None:
            return

        if self.dispatcher is not None:
            self.dispatcher.pause()

        eval_dir = os.path.join(self.config.log_dir, "eval")
        done_path = os.path.join(eval_dir, f"eval_step_{step}.done")
        error_path = os.path.join(eval_dir, f"eval_step_{step}.error")
        if self.is_main_process:
            os.makedirs(eval_dir, exist_ok=True)
            for path in (done_path, error_path):
                if os.path.exists(path):
                    os.remove(path)

        # Use a file sentinel instead of a long NCCL barrier. Full R2E eval can
        # exceed the default NCCL watchdog timeout while non-zero ranks wait.
        try:
            if self.is_main_process:
                max_samples = int(os.environ.get("EVAL_MAX_SAMPLES", os.environ.get("R2E_EVAL_MAX_SAMPLES", "0")))
                concurrency = int(os.environ.get("EVAL_CONCURRENCY", os.environ.get("R2E_EVAL_CONCURRENCY", "16")))
                threshold = float(os.environ.get("EVAL_ACCURACY_THRESHOLD", os.environ.get("R2E_EVAL_ACCURACY_THRESHOLD", "0.5")))
                eval_max_new = int(
                    os.environ.get(
                        "EVAL_MAX_NEW_TOKENS",
                        os.environ.get(
                            "R2E_EVAL_MAX_NEW_TOKENS",
                            str(getattr(self.config, "max_new_tokens", 1024)),
                        ),
                    )
                )
                print(
                    f"\n[Eval] step={step} samples="
                    f"{'all' if max_samples <= 0 else max_samples} "
                    f"threshold={threshold} max_new_tokens={eval_max_new}"
                )
                try:
                    eval_stats = asyncio.run(
                        workflow.evaluate(
                            self.rollout_engine,
                            dataset,
                            max_samples=max_samples,
                            concurrency=concurrency,
                            accuracy_threshold=threshold,
                            max_new_tokens=eval_max_new,
                        )
                    )
                    eval_stats["step"] = step
                    eval_records = eval_stats.pop("eval_records", [])
                    self.stats.update(eval_stats)
                    print(
                        "[Eval] "
                        f"accuracy={eval_stats['eval_accuracy']:.4f} "
                        f"threshold={eval_stats.get('eval_accuracy_threshold', threshold):.2f} "
                        f"mode={eval_stats.get('eval_mode', 'unknown')} "
                        f"reward_mean={eval_stats['eval_reward_mean']:.4f} "
                        f"reward_min={eval_stats['eval_reward_min']:.4f} "
                        f"reward_max={eval_stats['eval_reward_max']:.4f} "
                        f"failures={eval_stats['eval_failures']} "
                        f"samples={eval_stats['eval_samples']}"
                    )
                    metrics_file = os.environ.get("EVAL_METRICS_FILE", "r2e_eval_metrics.jsonl")
                    with open(os.path.join(eval_dir, metrics_file), "a", encoding="utf-8") as f:
                        import json

                        f.write(json.dumps(eval_stats, ensure_ascii=False) + "\n")
                    if eval_records:
                        samples_file = os.environ.get(
                            "EVAL_SAMPLES_FILE",
                            f"r2e_eval_samples_step_{step}.jsonl",
                        )
                        with open(os.path.join(eval_dir, samples_file), "w", encoding="utf-8") as f:
                            for record in eval_records:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if self.wandb_run is not None:
                        self.wandb_run.log(eval_stats, step=step)
                    Path(done_path).write_text("ok\n", encoding="utf-8")
                except Exception as exc:
                    Path(error_path).write_text(str(exc), encoding="utf-8")
                    raise
            else:
                wait_timeout = int(os.environ.get("EVAL_WAIT_TIMEOUT", os.environ.get("R2E_EVAL_WAIT_TIMEOUT", "86400")))
                start = time.time()
                while not os.path.exists(done_path) and not os.path.exists(error_path):
                    if time.time() - start > wait_timeout:
                        raise TimeoutError(f"Timed out waiting for eval step {step}")
                    time.sleep(10.0)
                if os.path.exists(error_path):
                    raise RuntimeError(Path(error_path).read_text(encoding="utf-8"))
        finally:
            if self.dispatcher is not None:
                self.dispatcher.resume()

    def _sync_weights(self, target_version: int):
        """Sync weights."""
        if self.config.weight_sync_mode == "hccl":
            self._sync_weights_hccl(target_version)
            return
        if self.config.weight_sync_mode == "disk":
            if self.is_main_process:
                print(f"Synchronizing weights (step={self.global_step})")

            mode = str(getattr(self.config, "rollout_weight_sync_mode", "none"))
            with self._rollout_export_only_context(mode):
                self.train_engine.save_weights(self.config.sync_path, self.global_step)
            self._record_weight_sync_phase("checkpoint_ready")
            if mode == "restart":
                self._sync_rollout_restart_with_sentinel(target_version)
            else:
                if self.is_main_process:
                    self._cleanup_old_weights()
                    checkpoint_path = os.path.abspath(
                        os.path.join(self.config.sync_path, f"v{self.global_step}")
                    )
                    self._reload_rollout_from_checkpoint(
                        checkpoint_path=checkpoint_path,
                        target_version=target_version,
                    )
                if dist.is_initialized():
                    dist.barrier()

        elif self.config.weight_sync_mode == "nccl":
            if self.is_main_process:
                print("WARNING: NCCL weight synchronization requires additional distributed setup")

    def _hccl_rollout_topology(self):
        if self.rollout_engine is None:
            raise RuntimeError("Rollout engine is unavailable for HCCL weight sync")
        urls = list(self.rollout_engine.instance_urls)
        if self._use_heterogeneous:
            configs = list(self.rollout_engine.instance_configs)
            if len(configs) != len(urls):
                raise RuntimeError("Heterogeneous rollout instance metadata is inconsistent")
            instances = [
                (
                    str(config["instance_id"]),
                    str(url),
                    int(config.get("tp_degree", config.get("tp_size", 1))),
                )
                for config, url in zip(configs, urls)
            ]
        else:
            tp_size = max(1, int(getattr(self.config, "vllm_tp_size", 1)))
            instances = [
                (f"instance_{index}", str(url), tp_size)
                for index, url in enumerate(urls)
            ]
        return build_hccl_rollout_endpoints(instances)

    @staticmethod
    def _consume_rollout_weight_export(weights) -> None:
        for _name, tensor in weights:
            del tensor

    def _wait_for_weight_sync_file(
        self,
        success_path: Path,
        error_path: Path,
    ) -> None:
        timeout = float(getattr(self.config, "rollout_weight_sync_timeout_s", 1200.0))
        poll_interval = float(
            getattr(self.config, "rollout_weight_sync_poll_interval_s", 0.05)
        )
        deadline = time.monotonic() + timeout
        while not success_path.exists() and not error_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {success_path.name}")
            time.sleep(poll_interval)
        if error_path.exists():
            raise RuntimeError(error_path.read_text(encoding="utf-8"))

    def _sync_weights_hccl(self, target_version: int) -> None:
        """Stream Megatron HF weights through vLLM Ascend's official HCCL backend."""
        if str(getattr(self.train_engine, "train_backend", "")) != "megatron_core":
            raise ValueError("Official HCCL sync requires train_backend=megatron_core")
        stream_weights = getattr(self.train_engine, "stream_rollout_weights", None)
        if not callable(stream_weights):
            raise RuntimeError("Training engine does not expose rollout weight streaming")

        control_dir_value = str(
            getattr(self.config, "rollout_weight_sync_control_dir", "") or ""
        )
        if not control_dir_value:
            raise ValueError("HCCL weight synchronization requires a control directory")
        control_dir = Path(control_dir_value).resolve()
        control_dir.mkdir(parents=True, exist_ok=True)
        ready_path = control_dir / f"hccl_ready_{target_version}.json"
        done_path = control_dir / f"hccl_done_{target_version}.json"
        error_path = control_dir / f"hccl_error_{target_version}.json"
        if self.is_main_process:
            ready_path.unlink(missing_ok=True)
            done_path.unlink(missing_ok=True)
            error_path.unlink(missing_ok=True)

        # Megatron Bridge export contains TP/EP collectives. Every training rank
        # participates; only rank zero retains the stable metadata cache and
        # owns the rollout control plane.
        if not self._hccl_metadata_initialized:
            try:
                metadata_export = stream_weights()
                if self.is_main_process:
                    self._hccl_weight_metadata = collect_weight_metadata(
                        metadata_export
                    )
                else:
                    self._consume_rollout_weight_export(metadata_export)
                self._hccl_metadata_initialized = True
            except Exception as exc:
                if self.is_main_process:
                    self._write_json_atomic(
                        error_path,
                        {
                            "version": int(target_version),
                            "error": str(exc),
                            "phase": "metadata_export",
                            "failed_at": time.time(),
                        },
                    )
                raise

        endpoints = []
        world_size = 0
        session = None
        if self.is_main_process:
            if self._hccl_weight_metadata is None:
                raise RuntimeError("HCCL rollout weight metadata is unavailable")
            try:
                endpoints, world_size = self._hccl_rollout_topology()
                host = (
                    str(getattr(self.config, "rollout_hccl_host", "") or "")
                    or os.environ.get("MASTER_ADDR", "")
                    or socket.gethostbyname(socket.gethostname())
                )
                minimum_buffer = (
                    int(getattr(self.config, "rollout_hccl_packed_buffer_mb", 1024))
                    * 1024
                    * 1024
                )
                headroom = (
                    int(getattr(self.config, "rollout_hccl_buffer_headroom_mb", 128))
                    * 1024
                    * 1024
                )
                packed_buffer_size = max(
                    minimum_buffer,
                    self._hccl_weight_metadata.max_tensor_bytes + headroom,
                )
                if self._hccl_weight_transfer is None:
                    self._hccl_weight_transfer = OfficialHCCLWeightTransfer(
                        master_address=host,
                        master_port=int(getattr(self.config, "rollout_hccl_port", 29620)),
                        timeout_s=float(
                            getattr(self.config, "rollout_weight_sync_timeout_s", 1200.0)
                        ),
                        packed=bool(getattr(self.config, "rollout_hccl_packed", True)),
                        packed_buffer_size_bytes=packed_buffer_size,
                        packed_num_buffers=int(
                            getattr(self.config, "rollout_hccl_num_buffers", 2)
                        ),
                        checkpoint_format=bool(
                            getattr(self.config, "rollout_hccl_checkpoint_format", True)
                        ),
                    )
                self._hccl_weight_transfer.initialize(endpoints, world_size)
                session = self._hccl_weight_transfer.prepare(
                    endpoints,
                    self._hccl_weight_metadata,
                )
                self._write_json_atomic(
                    ready_path,
                    {
                        "version": int(target_version),
                        "world_size": int(world_size),
                        "instances": [endpoint.instance_id for endpoint in endpoints],
                        "packed_buffer_size_bytes": int(packed_buffer_size),
                        "ready_at": time.time(),
                    },
                )
            except Exception as exc:
                self._write_json_atomic(
                    error_path,
                    {
                        "version": int(target_version),
                        "error": str(exc),
                        "failed_at": time.time(),
                    },
                )
                raise
        else:
            self._wait_for_weight_sync_file(ready_path, error_path)

        weight_export = stream_weights()
        if self.is_main_process:
            assert self._hccl_weight_transfer is not None
            assert session is not None
            try:
                self._hccl_weight_transfer.send(weight_export)
                transfer_seconds = self._hccl_weight_transfer.finish(
                    session,
                    endpoints,
                )
                self._record_rollout_weight_version(
                    target_version=target_version,
                    checkpoint_path=(
                        f"hccl://{self._hccl_weight_transfer.master_address}:"
                        f"{self._hccl_weight_transfer.master_port}"
                    ),
                    instance_ids=[endpoint.instance_id for endpoint in endpoints],
                )
                self._write_json_atomic(
                    done_path,
                    {
                        "version": int(target_version),
                        "world_size": int(world_size),
                        "transfer_seconds": float(transfer_seconds),
                        "completed_at": time.time(),
                    },
                )
                print(
                    "Official HCCL rollout refresh complete: "
                    f"version={target_version}, world_size={world_size}, "
                    f"seconds={transfer_seconds:.3f}",
                    flush=True,
                )
            except Exception as exc:
                self._write_json_atomic(
                    error_path,
                    {
                        "version": int(target_version),
                        "error": str(exc),
                        "failed_at": time.time(),
                    },
                )
                raise
        else:
            self._consume_rollout_weight_export(weight_export)
            self._wait_for_weight_sync_file(done_path, error_path)
        self._record_weight_sync_phase("sync_complete")

    @contextmanager
    def _rollout_export_only_context(self, mode: str):
        enabled = (
            mode == "restart"
            and str(getattr(self.train_engine, "train_backend", "")) == "megatron_core"
            and bool(getattr(self.config, "rollout_weight_sync_export_only", True))
        )
        if not enabled:
            yield
            return

        old_value = os.environ.get("MEGATRON_ROLLOUT_EXPORT_ONLY")
        os.environ["MEGATRON_ROLLOUT_EXPORT_ONLY"] = "1"
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop("MEGATRON_ROLLOUT_EXPORT_ONLY", None)
            else:
                os.environ["MEGATRON_ROLLOUT_EXPORT_ONLY"] = old_value

    def _sync_rollout_restart_with_sentinel(self, target_version: int) -> None:
        """Reload rollout servers without parking non-zero ranks in NCCL."""
        control_dir_value = str(
            getattr(self.config, "rollout_weight_sync_control_dir", "") or ""
        )
        if not control_dir_value:
            raise ValueError("restart weight synchronization requires rollout_weight_sync_control_dir")
        control_dir = Path(control_dir_value).resolve()
        if self.is_main_process:
            control_dir.mkdir(parents=True, exist_ok=True)

        done_path = control_dir / f"reload_done_{target_version}.json"
        error_path = control_dir / f"reload_error_{target_version}.json"
        if self.is_main_process:
            done_path.unlink(missing_ok=True)
            error_path.unlink(missing_ok=True)
            checkpoint_path = os.path.abspath(
                os.path.join(self.config.sync_path, f"v{self.global_step}")
            )
            try:
                self._record_weight_sync_phase("before_rollout_reload")
                self._cleanup_old_weights()
                self._reload_rollout_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    target_version=target_version,
                )
                self._record_weight_sync_phase("after_rollout_reload")
                self._write_json_atomic(
                    done_path,
                    {
                        "version": int(target_version),
                        "global_step": int(self.global_step),
                        "checkpoint_path": checkpoint_path,
                        "completed_at": time.time(),
                    },
                )
                self._record_rollout_weight_version(
                    target_version=target_version,
                    checkpoint_path=checkpoint_path,
                    instance_ids=self._current_rollout_instance_ids(),
                )
            except Exception as exc:
                self._write_json_atomic(
                    error_path,
                    {
                        "version": int(target_version),
                        "global_step": int(self.global_step),
                        "error": str(exc),
                        "failed_at": time.time(),
                    },
                )
                raise
        else:
            self._record_weight_sync_phase("waiting_for_rollout_reload")
            timeout = float(
                getattr(self.config, "rollout_weight_sync_timeout_s", 1200.0)
            )
            deadline = time.monotonic() + timeout + 300.0
            while not done_path.exists() and not error_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for rank 0 to finish reloading rolloutversion={target_version}timed out"
                    )
                time.sleep(2.0)
            if error_path.exists():
                raise RuntimeError(error_path.read_text(encoding="utf-8"))
            self._record_weight_sync_phase("rollout_reload_observed")

        # Keep a short collective after the file handshake so ranks resume the
        # training loop together without spending the reload window in NCCL.
        if dist.is_initialized():
            dist.barrier()
        self._record_weight_sync_phase("sync_complete")

    def _record_weight_sync_phase(self, phase: str) -> None:
        recorder = getattr(self.train_engine, "_record_weight_sync_phase", None)
        if not callable(recorder):
            return
        weight_file = os.path.abspath(
            os.path.join(
                self.config.sync_path,
                f"v{self.global_step}",
                "pytorch_model.bin",
            )
        )
        recorder(self.global_step, phase, weight_file)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _reload_rollout_from_checkpoint(
        self,
        checkpoint_path: str,
        target_version: int,
    ) -> None:
        mode = str(getattr(self.config, "rollout_weight_sync_mode", "none"))
        required = bool(getattr(self.config, "require_rollout_weight_sync", False))
        if mode == "none":
            if required:
                raise RuntimeError(
                    "Training weights were saved, but rollout_weight_sync_mode=none; "
                    "continuing would leave vLLM on the initial model"
                )
            return
        if mode != "restart":
            raise ValueError(f"Unsupported rollout_weight_sync_mode: {mode}")

        control_dir_value = str(
            getattr(self.config, "rollout_weight_sync_control_dir", "") or ""
        )
        if not control_dir_value:
            raise ValueError("restart weight synchronization requires rollout_weight_sync_control_dir")
        control_dir = Path(control_dir_value).resolve()
        control_dir.mkdir(parents=True, exist_ok=True)

        instance_ids = self._current_rollout_instance_ids()
        reload_strategy = str(
            getattr(self.config, "rollout_weight_reload_strategy", "parallel")
        ).lower()
        if reload_strategy not in {"parallel", "serial"}:
            raise ValueError(
                "rollout_weight_reload_strategy must be 'parallel' or 'serial', "
                f"got {reload_strategy!r}"
            )
        reload_method = str(
            getattr(self.config, "rollout_weight_reload_method", "restart")
        ).lower()
        if reload_method not in {"restart", "inplace"}:
            raise ValueError(
                "rollout_weight_reload_method must be 'restart' or 'inplace', "
                f"got {reload_method!r}"
            )

        for instance_id in instance_ids:
            (control_dir / f"ack_{instance_id}_{target_version}.json").unlink(
                missing_ok=True
            )
            (control_dir / f"error_{instance_id}_{target_version}.json").unlink(
                missing_ok=True
            )
        request = {
            "version": int(target_version),
            "checkpoint_path": checkpoint_path,
            "reload_method": reload_method,
            "reload_strategy": reload_strategy,
            "instance_ids": sorted(instance_ids),
            "created_at": time.time(),
        }
        request_tmp = control_dir / "reload_request.json.tmp"
        request_path = control_dir / "reload_request.json"
        request_tmp.write_text(json.dumps(request), encoding="utf-8")
        request_tmp.replace(request_path)

        timeout = float(
            getattr(self.config, "rollout_weight_sync_timeout_s", 1200.0)
        )
        deadline = time.monotonic() + timeout
        pending = set(instance_ids)
        while pending and time.monotonic() < deadline:
            failures = {}
            for instance_id in pending:
                error_path = control_dir / f"error_{instance_id}_{target_version}.json"
                if error_path.exists():
                    failures[instance_id] = error_path.read_text(encoding="utf-8")
            if failures:
                raise RuntimeError(
                    f"vLLM failed to loadversion={target_version}failed: {failures}"
                )
            pending = {
                instance_id
                for instance_id in pending
                if not (
                    control_dir / f"ack_{instance_id}_{target_version}.json"
                ).exists()
            }
            if pending:
                time.sleep(
                    float(
                        getattr(
                            self.config,
                            "rollout_weight_sync_poll_interval_s",
                            0.05,
                        )
                    )
                )
        if pending:
            raise TimeoutError(
                f"Timed out waiting for vLLM to loadversion={target_version}timed out，unconfirmed instances: "
                f"{sorted(pending)}"
            )
        acknowledgements = []
        for instance_id in instance_ids:
            ack_path = control_dir / f"ack_{instance_id}_{target_version}.json"
            try:
                ack = json.loads(ack_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid rollout reload ACK: {ack_path}") from exc
            if (
                str(ack.get("instance_id")) != instance_id
                or int(ack.get("version", -1)) != int(target_version)
                or str(ack.get("checkpoint_path")) != checkpoint_path
                or str(ack.get("reload_method", "restart")) != reload_method
                or str(ack.get("reload_strategy", "parallel")) != reload_strategy
            ):
                raise RuntimeError(
                    f"Mismatched rollout reload ACK for {instance_id}: {ack}"
                )
            acknowledgements.append(ack)

        reload_seconds = [
            float(ack["total_seconds"])
            for ack in acknowledgements
            if "total_seconds" in ack
        ]
        timing = (
            f", slowest_reload={max(reload_seconds):.2f}s"
            if reload_seconds
            else ""
        )
        print(
            f"Rollout weight refresh complete: version={target_version}, "
            f"instances={len(instance_ids)}, method={reload_method}, "
            f"strategy={reload_strategy}{timing}"
        )

    def _current_rollout_instance_ids(self) -> list[str]:
        if self._use_heterogeneous:
            planner_cfg = getattr(self.config, "global_resource_planner", None)
            runtime_manages_workers = bool(
                planner_cfg is not None
                and getattr(planner_cfg, "runtime_manage_rollout_processes", False)
            )
            if (
                not runtime_manages_workers
                and self._external_rollout_reload_instance_ids
            ):
                return list(self._external_rollout_reload_instance_ids)
            return [
                str(cfg["instance_id"])
                for cfg in self.rollout_engine.instance_configs
            ]
        return [
            f"instance_{index}"
            for index in range(self.rollout_engine.num_instances)
        ]

    def _record_rollout_weight_version(
        self,
        *,
        target_version: int,
        checkpoint_path: str,
        instance_ids: list[str],
    ) -> None:
        control_dir_value = str(
            getattr(self.config, "rollout_weight_sync_control_dir", "") or ""
        )
        if not control_dir_value:
            return
        self._write_json_atomic(
            Path(control_dir_value).resolve() / "rollout_weight_version.json",
            {
                "version": int(target_version),
                "global_step": int(self.global_step),
                "checkpoint_path": checkpoint_path,
                "instances": sorted(instance_ids),
                "updated_at": time.time(),
                "mode": str(getattr(self.config, "rollout_weight_sync_mode", "none")),
            },
        )

    def _save_checkpoint(self):
        """Save checkpoint."""
        if not self.is_main_process:
            return

        checkpoint_path = os.path.join(self.config.log_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "stats": self.stats,
            "config": self.config,
            "version": self.train_engine.get_version(),
        }

        torch.save(checkpoint, os.path.join(checkpoint_path, f"step_{self.global_step}.pt"))
        print(f"Checkpoint saved: step_{self.global_step}.pt")

    def _cleanup_old_weights(self):
        """Cleanup old weights."""
        if self.config.keep_latest_checkpoints <= 0:
            return

        weight_dirs = glob.glob(os.path.join(self.config.sync_path, "v*"))

        if len(weight_dirs) <= self.config.keep_latest_checkpoints:
            return


        version_dir_map = {}
        for wd in weight_dirs:
            basename = os.path.basename(wd)
            version_str = basename.replace("v", "")
            try:
                version_num = int(version_str)
                version_dir_map[version_num] = wd
            except ValueError:
                continue

        sorted_versions = sorted(version_dir_map.keys())
        to_delete = sorted_versions[:-self.config.keep_latest_checkpoints]

        for v in to_delete:
            try:
                shutil.rmtree(version_dir_map[v])
                if self.is_main_process:
                    print(f"Deleted old weights: {version_dir_map[v]}")
            except Exception as e:
                print(f"WARNING: Failed to delete weights {version_dir_map[v]}: {e}")

    def _cleanup(self):
        """Cleanup."""
        if self._hccl_weight_transfer is not None:
            self._hccl_weight_transfer.close()
            self._hccl_weight_transfer = None
        self._runtime_follow_stop.set()
        if self._runtime_follow_thread is not None:
            self._runtime_follow_thread.join(timeout=2.0)
            self._runtime_follow_thread = None
        if self._grp_executor is not None:
            self._grp_executor.shutdown(wait=False, cancel_futures=True)
            self._grp_executor = None
            self._grp_future = None
            self._grp_future_step = None


        if self.history_collector is not None:
            self.history_collector.finalize()

        if self.dispatcher is not None:
            self.dispatcher.destroy()
        elif self.async_runner is not None:
            self.async_runner.destroy()

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _init_wandb(self):
        """Init wandb."""
        if not self.is_main_process or not self.config.wandb_project:
            return

        try:
            import wandb

            run_name = self.config.wandb_run_name if self.config.wandb_run_name else None

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "model_path": self.config.model_path,
                    "learning_rate": self.config.learning_rate,
                    "ppo_epochs": self.config.ppo_epochs,
                    "batch_size": self.config.batch_size,
                    "kl_coef": self.config.kl_coef,
                    "clip_epsilon": self.config.clip_epsilon,
                    "max_new_tokens": self.config.max_new_tokens,
                    "n_samples": self.config.n_samples,
                    "temperature": self.config.temperature,
                    "total_steps": self.config.total_steps,
                    "max_concurrent_rollouts": self.config.max_concurrent_rollouts,
                    "max_head_offpolicyness": self.config.max_head_offpolicyness,
                    "recompute_logprobs": self.config.recompute_logprobs,
                },
            )

            print(f"Weights & Biases initialized: {self.wandb_run.name}")
        except ImportError:
            print("WARNING: wandb is not installed; skipping Weights & Biases logging")
        except Exception as e:
            print(f"WARNING: Weights & Biases initialization failed: {e}")

    def _log_to_wandb(self, stats: dict, step: int):
        """Log to wandb."""
        if self.wandb_run is None:
            return

        try:
            log_dict = {
                "step": step,
                "loss": stats.get("loss", 0),
                "pg_loss": stats.get("pg_loss", 0),
                "kl": stats.get("kl", 0),
                "reward_mean": stats.get("reward_mean", 0),
                "n_trajectories": stats.get("n_trajectories", 0),
                "version": stats.get("version", 0),
                "rollout_time": stats.get("rollout_time", 0),
                "train_time": stats.get("train_time", 0),
                "step_time": stats.get("step_time", 0),
                "reward_std": stats.get("reward_std", 0),
                "reward_min": stats.get("reward_min", 0),
                "reward_max": stats.get("reward_max", 0),
                "grpo_num_groups": stats.get("grpo_num_groups", 0),
                "grpo_mean_group_size": stats.get("grpo_mean_group_size", 0),
                "grpo_singleton_groups": stats.get("grpo_singleton_groups", 0),
                "grpo_zero_variance_groups": stats.get("grpo_zero_variance_groups", 0),
            }

            self.wandb_run.log(log_dict, step=step)
        except Exception as e:
            print(f"WARNING: Weights & Biases logging failed: {e}")



    def _init_global_resource_planner(self):
        """Init global resource planner."""
        if not self.is_main_process:
            self._start_runtime_reconfiguration_follower()
            return
        planner_cfg = getattr(self.config, "global_resource_planner", None)
        if planner_cfg is None or not getattr(planner_cfg, "enabled", False):
            return
        if not self.train_engine or not self.train_engine.is_batch_source():
            return

        self.global_resource_planner = GlobalResourcePlanner.from_config(self.config)
        self.runtime_elastic_executor = RuntimeElasticExecutor(
            config=self.config,
            planner=self.global_resource_planner,
            train_engine=self.train_engine,
            rollout_engine=self.rollout_engine,
            dispatcher=self.dispatcher,
        )
        if getattr(planner_cfg, "runtime_async_planning", True):
            self._grp_executor = ThreadPoolExecutor(
                max_workers=max(1, int(getattr(planner_cfg, "runtime_max_pending_plans", 1) or 1)),
                thread_name_prefix="global-resource-planner",
            )
        print(
            "[GlobalResourcePlanner] enabled "
            f"interval={planner_cfg.plan_interval} "
            f"min_history={planner_cfg.min_history_size} "
            f"gain_threshold={planner_cfg.min_gain_ratio:.2%} "
            f"async={getattr(planner_cfg, 'runtime_async_planning', True)}"
        )

    def _start_runtime_reconfiguration_follower(self) -> None:
        planner_cfg = getattr(self.config, "global_resource_planner", None)
        if planner_cfg is None or not getattr(planner_cfg, "enabled", False):
            return
        if not getattr(planner_cfg, "apply_to_runtime", False):
            return
        if not getattr(
            planner_cfg,
            "runtime_dynamic_reconfiguration_enabled",
            False,
        ):
            return
        if not getattr(planner_cfg, "runtime_coordinate_reconfiguration_ranks", True):
            return
        if self.train_engine is not None and not self.train_engine.is_batch_source():
            return
        if self._runtime_follow_thread is not None:
            return

        def follow_loop() -> None:
            while not self._runtime_follow_stop.is_set():
                try:
                    self._follow_runtime_reconfiguration_if_requested()
                except Exception as exc:
                    print(f"[RuntimeElasticExecutor] follower error: {exc}")
                self._runtime_follow_stop.wait(0.5)

        self._runtime_follow_thread = threading.Thread(
            target=follow_loop,
            name=f"runtime-reconfiguration-follower-rank{self.rank}",
            daemon=True,
        )
        self._runtime_follow_thread.start()

    def _runtime_reconfiguration_coord_dir(self) -> Path:
        planner_cfg = getattr(self.config, "global_resource_planner", None)
        configured = (
            getattr(planner_cfg, "runtime_reconfiguration_coord_dir", "")
            if planner_cfg is not None
            else ""
        )
        if configured:
            return Path(configured)
        return Path(self.config.log_dir) / "runtime_reconfiguration"

    def _runtime_reconfiguration_pending_path(self) -> Path:
        return self._runtime_reconfiguration_coord_dir() / "planning_pending.json"

    def _write_runtime_reconfiguration_pending(self, step: int) -> None:
        planner_cfg = getattr(self.config, "global_resource_planner", None)
        if planner_cfg is None or not getattr(planner_cfg, "enabled", False):
            return
        coord_dir = self._runtime_reconfiguration_coord_dir()
        coord_dir.mkdir(parents=True, exist_ok=True)
        wait_s = float(getattr(planner_cfg, "runtime_peer_request_wait_s", 45.0))
        self._write_json_atomic(
            self._runtime_reconfiguration_pending_path(),
            {
                "step": int(step),
                "rank": int(self.rank),
                "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
                "created_at": time.time(),
                "expires_at": time.time() + max(wait_s, 1.0) + 300.0,
            },
        )

    def _clear_runtime_reconfiguration_pending(self) -> None:
        try:
            self._runtime_reconfiguration_pending_path().unlink(missing_ok=True)
        except OSError:
            pass

    def _follow_runtime_reconfiguration_if_requested(self) -> None:
        planner_cfg = getattr(self.config, "global_resource_planner", None)
        if planner_cfg is None or not getattr(planner_cfg, "enabled", False):
            return
        if not getattr(planner_cfg, "apply_to_runtime", False):
            return
        if not getattr(
            planner_cfg,
            "runtime_dynamic_reconfiguration_enabled",
            False,
        ):
            return
        if not getattr(planner_cfg, "runtime_coordinate_reconfiguration_ranks", True):
            return

        coord_dir = self._runtime_reconfiguration_coord_dir()
        request_path = coord_dir / "request.json"
        if not request_path.exists():
            pending_path = self._runtime_reconfiguration_pending_path()
            if not pending_path.exists():
                # Normal training steps have no reconfiguration request.  Do
                # not add runtime_peer_request_wait_s to every rank and every
                # step; an actual planner operation first publishes pending.
                self._follow_applied_runtime_reconfiguration_if_available(
                    coord_dir
                )
                return
            wait_s = float(getattr(planner_cfg, "runtime_peer_request_wait_s", 45.0))
            deadline = time.time() + max(wait_s, 0.0)
            while time.time() < deadline:
                if pending_path.exists():
                    try:
                        pending = json.loads(pending_path.read_text(encoding="utf-8"))
                        current_job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", ""))
                        pending_job_id = str(pending.get("job_id", ""))
                        if current_job_id and pending_job_id != current_job_id:
                            return
                        if time.time() > float(pending.get("expires_at", 0.0)):
                            return
                    except Exception:
                        return
                if request_path.exists():
                    break
                if (coord_dir / "applied.json").exists() or (coord_dir / "aborted.json").exists():
                    break
                time.sleep(0.2)
        if not request_path.exists():
            self._follow_applied_runtime_reconfiguration_if_available(coord_dir)
            return
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception:
            return
        current_job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", ""))
        request_job_id = str(request.get("job_id", ""))
        if current_job_id and request_job_id != current_job_id:
            return
        coord_id = str(request.get("coord_id", ""))
        if not coord_id or coord_id == self._runtime_reconfiguration_coord_id:
            return

        # Cluster-swap is coordinated and drained by rank 0.  Peer ranks must
        # not wait for their old rollout queues to become idle here: rank 0
        # may already be stopping those endpoints, so a peer-side drain can
        # wait forever and prevent the rank from publishing its ready marker.
        strategy = str(
            getattr(
                planner_cfg,
                "runtime_rollout_reconfigure_strategy",
                getattr(planner_cfg, "rollout_reconfigure_strategy", "restart_all"),
            )
        ).lower()
        cluster_swap = bool(
            getattr(planner_cfg, "runtime_cluster_swap_enabled", False)
            or strategy == "cluster_swap"
        )
        paused = False
        try:
            if self.dispatcher is not None and hasattr(self.dispatcher, "pause"):
                self.dispatcher.pause()
                paused = True
                if not cluster_swap and hasattr(self.dispatcher, "wait_until_idle"):
                    self.dispatcher.wait_until_idle(
                        timeout=float(
                            getattr(planner_cfg, "runtime_drain_timeout_s", 3600.0)
                        )
                    )
            if (
                self.rollout_engine is not None
                and hasattr(self.rollout_engine, "wait_until_idle")
                and not cluster_swap
            ):
                self.rollout_engine.wait_until_idle(
                    timeout=float(
                        getattr(planner_cfg, "runtime_drain_timeout_s", 3600.0)
                    )
                )

            ready_dir = coord_dir / "ready"
            ready_dir.mkdir(parents=True, exist_ok=True)
            ready_path = ready_dir / f"rank_{self.rank}.json"
            self._write_json_atomic(
                ready_path,
                {
                    "coord_id": coord_id,
                    "rank": int(self.rank),
                    "job_id": current_job_id,
                    "updated_at": time.time(),
                },
            )

            deadline = time.time() + float(
                getattr(planner_cfg, "runtime_drain_timeout_s", 3600.0)
            )
            applied_path = coord_dir / "applied.json"
            aborted_path = coord_dir / "aborted.json"
            state: dict[str, Any] | None = None
            while time.time() < deadline:
                if aborted_path.exists():
                    aborted = json.loads(aborted_path.read_text(encoding="utf-8"))
                    aborted_job_id = str(aborted.get("job_id", ""))
                    if not current_job_id or aborted_job_id == current_job_id:
                        raise RuntimeError(
                            "runtime reconfiguration aborted on rank0: "
                            f"{aborted.get('error', '')}"
                        )
                if applied_path.exists():
                    state = json.loads(applied_path.read_text(encoding="utf-8"))
                    state_job_id = str(state.get("job_id", ""))
                    if (
                        str(state.get("coord_id", "")) == coord_id
                        and (not current_job_id or state_job_id == current_job_id)
                    ):
                        break
                time.sleep(0.5)
            if state is None:
                raise TimeoutError(
                    f"timed out waiting for runtime reconfiguration {coord_id} to apply"
                )
            self._apply_runtime_reconfiguration_state(state)
            self._reset_rollout_pipeline_after_reconfigure()
            self._runtime_reconfiguration_coord_id = coord_id
        finally:
            if paused and self.dispatcher is not None and hasattr(self.dispatcher, "resume"):
                self.dispatcher.resume()

    def _follow_applied_runtime_reconfiguration_if_available(self, coord_dir: Path) -> None:
        """Apply a completed non-draining runtime reconfiguration on peer ranks."""
        applied_path = coord_dir / "applied.json"
        state: dict[str, Any] | None = None
        if applied_path.exists():
            try:
                state = json.loads(applied_path.read_text(encoding="utf-8"))
            except Exception:
                return
        else:
            state = self._runtime_reconfiguration_state_from_manifest()
            if state is None:
                return
        current_job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", ""))
        state_job_id = str(state.get("job_id", ""))
        if current_job_id and state_job_id != current_job_id:
            return
        coord_id = str(state.get("coord_id", ""))
        if not coord_id:
            return
        desired_endpoints = self._runtime_reconfiguration_endpoint_keys(state)
        current_endpoints = self._rollout_engine_endpoint_keys()
        if (
            coord_id == self._runtime_reconfiguration_coord_id
            and desired_endpoints == current_endpoints
        ):
            return

        paused = False
        try:
            self._pause_rollout_pipeline()
            paused = self.dispatcher is not None
            self._reset_rollout_pipeline_after_reconfigure()
            self._apply_runtime_reconfiguration_state(state)
            self._reset_rollout_pipeline_after_reconfigure()
            self._runtime_reconfiguration_coord_id = coord_id
            print(
                "[RuntimeElasticExecutor] peer applied completed runtime "
                f"reconfiguration rank={self.rank} coord_id={coord_id} "
                f"endpoints={sorted(desired_endpoints)}"
            )
        finally:
            if paused:
                self._resume_rollout_pipeline()

    def _runtime_reconfiguration_endpoint_keys(
        self,
        state: dict[str, Any],
    ) -> set[str]:
        endpoints: set[str] = set()
        for idx, inst in enumerate(state.get("instances") or []):
            host = str(inst.get("host") or "")
            port = int(
                inst.get("port")
                or (self.config.heterogeneous_rollout.vllm_base_port + idx)
            )
            endpoints.add(f"{host}:{port}")
        return endpoints

    def _rollout_engine_endpoint_keys(self) -> set[str]:
        engine = self.rollout_engine
        if engine is None:
            return set()
        endpoints = set()
        for url in getattr(engine, "instance_urls", []):
            endpoint = str(url).removeprefix("http://").removeprefix("https://")
            endpoints.add(endpoint.rstrip("/"))
        return endpoints

    def _refresh_rollout_engine_from_manifest_for_task(self) -> None:
        """Best-effort task-time rollout endpoint refresh for batch-source ranks."""
        if not self._use_heterogeneous:
            return
        state = self._runtime_reconfiguration_state_from_manifest()
        if state is None:
            return
        desired_endpoints = self._runtime_reconfiguration_endpoint_keys(state)
        if not desired_endpoints or desired_endpoints == self._rollout_engine_endpoint_keys():
            return
        with self._rollout_engine_rebind_lock:
            if desired_endpoints == self._rollout_engine_endpoint_keys():
                return
            self._apply_training_reconfiguration_state(state.get("training") or {})
            instances = state.get("instances") or []
            hetero = self.config.heterogeneous_rollout
            hetero.instances = [
                HeterogeneousInstanceConfig(
                    instance_id=str(inst.get("instance_id", f"runtime_{idx}")),
                    tp=int(inst.get("tp", 1)),
                    gpus=[int(gpu) for gpu in inst.get("gpus", [])],
                    host=str(inst.get("host") or hetero.vllm_host),
                    port=int(inst.get("port") or (hetero.vllm_base_port + idx)),
                    description="runtime_reconfiguration_task_refresh",
                )
                for idx, inst in enumerate(instances)
            ]
            old_engine = self.rollout_engine
            try:
                for engine in getattr(old_engine, "engines", []):
                    if hasattr(engine, "close_sync"):
                        engine.close_sync()
            except Exception as exc:
                print(
                    "[RuntimeElasticExecutor] warning: failed to close old "
                    f"rollout clients during task refresh: {exc}"
                )
            self.rollout_engine = HeterogeneousRolloutEngine.from_config(self.config)
            if hasattr(self.rollout_engine, "wait_for_ready"):
                self.rollout_engine.wait_for_ready(
                    timeout=float(
                        getattr(
                            self.config.global_resource_planner,
                            "vllm_ready_timeout_s",
                            300.0,
                        )
                    )
                )
            self._runtime_reconfiguration_coord_id = str(state.get("coord_id", ""))
            print(
                "[RuntimeElasticExecutor] task refreshed rollout engine "
                f"rank={self.rank} endpoints={sorted(desired_endpoints)}",
                flush=True,
            )

    def _runtime_reconfiguration_state_from_manifest(self) -> dict[str, Any] | None:
        manifest_path = Path(self.config.log_dir) / "global_resource_rollout_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if str(manifest.get("phase", "")) != "applied":
            return None
        instances = manifest.get("instances") or []
        if not instances:
            return None
        updated_at = float(manifest.get("updated_at", time.time()) or time.time())
        return {
            "coord_id": f"manifest_{updated_at:.6f}",
            "phase": "applied",
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
            "updated_at": updated_at,
            "instances": instances,
            "training": manifest.get("training") or {},
        }

    def _rebind_rollout_engine_from_manifest_if_available(self, *, reason: str) -> None:
        state = self._runtime_reconfiguration_state_from_manifest()
        if state is None:
            return
        self._apply_runtime_reconfiguration_state(state)
        self._reset_rollout_pipeline_after_reconfigure()
        self._runtime_reconfiguration_coord_id = str(state.get("coord_id", ""))
        if self.is_main_process:
            instances = state.get("instances") or []
            endpoints = [
                f"{inst.get('host')}:{inst.get('port')}"
                for inst in instances
            ]
            print(
                "[RuntimeElasticExecutor] rebound rollout engine from manifest "
                f"reason={reason} endpoints={endpoints}"
            )

    def _rebind_rollout_engine_from_config(self, *, reason: str) -> None:
        if (
            self.rollout_engine is None
            or not hasattr(self.rollout_engine, "reconfigure_from_plan")
        ):
            return
        if self._use_heterogeneous:
            old_engine = self.rollout_engine
            try:
                for engine in getattr(old_engine, "engines", []):
                    if hasattr(engine, "close_sync"):
                        engine.close_sync()
            except Exception as exc:
                print(
                    "[RuntimeElasticExecutor] warning: failed to close old "
                    f"rollout clients during rebind: {exc}"
                )
            self.rollout_engine = HeterogeneousRolloutEngine.from_config(self.config)
        else:
            self.rollout_engine.reconfigure_from_plan(None, self.config)
        if hasattr(self.rollout_engine, "wait_for_ready"):
            self.rollout_engine.wait_for_ready(
                timeout=float(
                    getattr(
                        self.config.global_resource_planner,
                        "vllm_ready_timeout_s",
                        300.0,
                    )
                )
            )
        self._reset_rollout_pipeline_after_reconfigure()
        if self.is_main_process:
            instances = getattr(self.config.heterogeneous_rollout, "instances", [])
            default_host = (
                getattr(self.config.heterogeneous_rollout, "vllm_host", "")
                or "127.0.0.1"
            )
            if default_host == "0.0.0.0":
                default_host = "127.0.0.1"
            endpoints = [
                f"{getattr(inst, 'host', '') or default_host}:"
                f"{getattr(inst, 'port', '')}"
                for inst in instances
            ]
            print(
                "[RuntimeElasticExecutor] rebound rollout engine from config "
                f"reason={reason} endpoints={endpoints}"
            )

    def _apply_runtime_reconfiguration_state(self, state: dict[str, Any]) -> None:
        self._apply_training_reconfiguration_state(state.get("training") or {})
        instances = state.get("instances") or []
        if not instances:
            return
        hetero = self.config.heterogeneous_rollout
        hetero.instances = [
            HeterogeneousInstanceConfig(
                instance_id=str(inst.get("instance_id", f"runtime_{idx}")),
                tp=int(inst.get("tp", 1)),
                gpus=[int(gpu) for gpu in inst.get("gpus", [])],
                host=str(inst.get("host") or hetero.vllm_host),
                port=int(inst.get("port") or (hetero.vllm_base_port + idx)),
                description="runtime_reconfiguration_peer_apply",
            )
            for idx, inst in enumerate(instances)
        ]
        if (
            self.rollout_engine is not None
            and hasattr(self.rollout_engine, "reconfigure_from_plan")
        ):
            self._rebind_rollout_engine_from_config(reason="runtime_state_apply")

    def _apply_training_reconfiguration_state(self, state: dict[str, Any]) -> None:
        if not state or not bool(state.get("enabled", False)):
            return
        if bool(state.get("plan_only", True)):
            return
        if self.train_engine is None:
            return

        core_ids = [str(item) for item in state.get("core_replica_ids") or []]
        if not core_ids:
            dp = max(1, int(getattr(self.config, "train_dp_size", 1) or 1))
            core_ids = [f"dp{i}" for i in range(dp)]

        decoupled = bool(
            getattr(
                self.config.global_resource_planner,
                "decouple_communication_domains",
                True,
            )
        )
        configure = getattr(self.train_engine, "configure_elastic_training", None)
        if callable(configure):
            try:
                domain = configure(
                    core_ids,
                    decouple_communication_domains=decoupled,
                )
            except TypeError:
                domain = configure(core_ids)
        else:
            from RL_Framework.infra.elastic.hybrid_pool import InterReplicaGradientDomain

            domain = InterReplicaGradientDomain(
                core_replica_ids=core_ids,
                decouple_communication_domains=decoupled,
            )
            setter = getattr(self.train_engine, "set_elastic_gradient_domain", None)
            if callable(setter):
                setter(domain)

        hybrid_targets = {
            str(worker_id): str(target_core)
            for worker_id, target_core in (state.get("hybrid_targets") or {}).items()
        }
        for worker_id, target_core in hybrid_targets.items():
            domain.request_join(worker_id, target_core)
            if bool(state.get("activate_hybrids", False)):
                domain.mark_active(worker_id)

    def _run_global_resource_planner(self, step: int, batch: list, stats: dict):
        """Run global resource planner."""
        if self.global_resource_planner is None:
            return

        self._consume_global_resource_planner_result(stats)

        planner_cfg = self.config.global_resource_planner
        if not getattr(planner_cfg, "runtime_async_planning", True):
            dispatcher_metrics = (
                self.dispatcher.get_runtime_metrics()
                if self.dispatcher is not None
                and hasattr(self.dispatcher, "get_runtime_metrics")
                else {}
            )
            runtime_metrics = self.global_resource_planner.observe_runtime(
                step=step,
                dispatcher_metrics=dispatcher_metrics,
                step_stats={
                    **stats,
                    "max_concurrent_rollouts": self.config.max_concurrent_rollouts,
                },
            )
            self._write_runtime_reconfiguration_pending(step)
            try:
                decision = self.global_resource_planner.plan_if_needed(
                    step=step,
                    config=self.config,
                    batch=batch,
                    runtime_metrics=runtime_metrics,
                )
                self._apply_global_resource_planner_decision(step, decision, stats)
            finally:
                self._clear_runtime_reconfiguration_pending()
            return

        observed = self.global_resource_planner.observe_batch(batch)
        dispatcher_metrics = (
            self.dispatcher.get_runtime_metrics()
            if self.dispatcher is not None
            and hasattr(self.dispatcher, "get_runtime_metrics")
            else {}
        )
        runtime_metrics = self.global_resource_planner.observe_runtime(
            step=step,
            dispatcher_metrics=dispatcher_metrics,
            step_stats={
                **stats,
                "max_concurrent_rollouts": self.config.max_concurrent_rollouts,
            },
        )
        stats["global_resource_planner"] = {
            "step": step,
            "status": "observed",
            "observed_requests": observed,
            "runtime_metrics": runtime_metrics.to_dict(),
        }

        if step < self.global_resource_planner.warmup_steps:
            return
        trigger = self.global_resource_planner._planning_trigger(
            step,
            runtime_metrics,
        )
        if trigger in {"interval_skip", "cooldown"}:
            return
        if self.global_resource_planner.history_size < self.global_resource_planner.min_history_size:
            print(
                "[GlobalResourcePlanner] "
                f"step={step} async skip reason=insufficient_history "
                f"history={self.global_resource_planner.history_size}"
            )
            return
        if self._grp_future is not None and not self._grp_future.done():
            print(
                "[GlobalResourcePlanner] "
                f"step={step} async skip reason=planner_busy "
                f"pending_step={self._grp_future_step}"
            )
            return

        if self._grp_executor is None:
            self._grp_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="global-resource-planner",
            )

        self._grp_future_step = step
        self._write_runtime_reconfiguration_pending(step)
        self._grp_future = self._grp_executor.submit(
            self.global_resource_planner.plan_if_needed,
            step=step,
            config=self.config,
            batch=None,
            runtime_metrics=runtime_metrics,
        )
        print(
            "[GlobalResourcePlanner] "
            f"step={step} submitted async evaluation "
            f"trigger={trigger} "
            f"history={self.global_resource_planner.history_size}"
        )

    def _consume_global_resource_planner_result(self, stats: dict):
        if self._grp_future is None or not self._grp_future.done():
            return

        step = self._grp_future_step if self._grp_future_step is not None else -1
        future = self._grp_future
        self._grp_future = None
        self._grp_future_step = None
        try:
            decision = future.result()
        except Exception as exc:
            self._clear_runtime_reconfiguration_pending()
            stats["global_resource_planner"] = {
                "step": step,
                "status": "error",
                "error": str(exc),
            }
            print(f"[GlobalResourcePlanner] step={step} async evaluation failed: {exc}")
            return

        try:
            self._apply_global_resource_planner_decision(step, decision, stats)
        finally:
            self._clear_runtime_reconfiguration_pending()

    def _apply_global_resource_planner_decision(self, step: int, decision, stats: dict):
        stats["global_resource_planner"] = decision.to_dict()

        if not decision.should_reconfigure or decision.candidate_plan is None:
            if decision.reason not in {"interval_skip", "warmup"}:
                print(
                    "[GlobalResourcePlanner] "
                    f"step={step} skip reason={decision.reason} "
                    f"history={decision.num_requests}"
                )
            return

        plan = decision.candidate_plan
        print(
            "[GlobalResourcePlanner] "
            f"step={step} apply "
            f"train={plan.train_config.tp}x{plan.train_config.pp}x{plan.train_config.dp} "
            f"rollout_tp={plan.rollout_tp_list} "
            f"T={plan.t_global:.3f}s "
            f"net_gain={plan.expected_gain_s:.3f}s"
        )

        if self.runtime_elastic_executor is not None:
            pre_reset = self._should_pre_reset_rollout_pipeline_for_reconfigure()
            if pre_reset:
                self._pause_rollout_pipeline()
                if self.dispatcher is not None and hasattr(
                    self.dispatcher,
                    "wait_until_idle",
                ):
                    self.dispatcher.wait_until_idle(
                        timeout=float(
                            getattr(
                                self.config.global_resource_planner,
                                "runtime_drain_timeout_s",
                                3600.0,
                            )
                        )
                    )
                self._reset_rollout_pipeline_after_reconfigure()
                print(
                    "[RuntimeElasticExecutor] rollout pipeline drained and reset "
                    "before runtime reconfiguration"
                )
            try:
                result = self.runtime_elastic_executor.execute(decision)
                stats["global_resource_planner_runtime"] = result.to_dict()
                self._record_runtime_reconfiguration_event(step, decision, result)
                if result.applied:
                    self._rebind_rollout_engine_from_config(
                        reason="after_runtime_reconfiguration"
                    )
                    if not pre_reset:
                        self._reset_rollout_pipeline_after_reconfigure()
                    print(
                        "[RuntimeElasticExecutor] applied actions="
                        f"{','.join(result.actions)} "
                        f"training_actions={','.join(result.training_actions) or 'none'}"
                    )
            finally:
                if pre_reset:
                    self._resume_rollout_pipeline()
        else:
            self.global_resource_planner.apply_plan_to_config(plan, self.config)

        if self._resource_config_snapshot is not None:
            self._resource_config_snapshot = HistoryDataCollector.snapshot_resource_config(
                self.config
            )

    def _reset_rollout_pipeline_after_reconfigure(self) -> None:
        if self.dispatcher is not None and hasattr(
            self.dispatcher,
            "reset_after_reconfigure",
        ):
            self.dispatcher.reset_after_reconfigure()
        self._pending_grpo_groups.clear()

    def _pause_rollout_pipeline(self) -> None:
        if self.dispatcher is not None and hasattr(self.dispatcher, "pause"):
            self.dispatcher.pause()

    def _resume_rollout_pipeline(self) -> None:
        if self.dispatcher is not None and hasattr(self.dispatcher, "resume"):
            self.dispatcher.resume()

    def _should_pre_reset_rollout_pipeline_for_reconfigure(self) -> bool:
        grp_cfg = getattr(self.config, "global_resource_planner", None)
        if grp_cfg is None:
            return False
        strategy = str(
            getattr(
                grp_cfg,
                "runtime_rollout_reconfigure_strategy",
                getattr(grp_cfg, "rollout_reconfigure_strategy", "restart_all"),
            )
        ).lower()
        cluster_swap = bool(
            getattr(grp_cfg, "runtime_cluster_swap_enabled", False)
            or strategy == "cluster_swap"
        )
        drain = bool(
            getattr(
                grp_cfg,
                "runtime_drain_before_reconfigure",
                getattr(grp_cfg, "drain_before_reconfigure", True),
            )
        )
        return cluster_swap and not drain

    def _record_runtime_reconfiguration_event(self, step: int, decision, result) -> None:
        try:
            coord_dir = self._runtime_reconfiguration_coord_dir()
            coord_dir.mkdir(parents=True, exist_ok=True)
            path = coord_dir / "runtime_reconfiguration_events.jsonl"
            payload = {
                "step": int(step),
                "rank": int(self.rank),
                "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
                "timestamp": time.time(),
                "decision": decision.to_dict() if hasattr(decision, "to_dict") else {},
                "runtime": result.to_dict() if hasattr(result, "to_dict") else {},
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"[RuntimeElasticExecutor] failed to write event log: {exc}")

    def _init_history_collector(self):
        """Init history collector."""
        if not self.is_main_process:
            return
        if not getattr(self.config, "enable_history_collection", False):
            return

        output_dir = getattr(self.config, "history_output_dir", "") or ""
        if not output_dir:
            output_dir = os.path.join(self.config.log_dir, "history")

        experiment_name = getattr(self.config, "history_experiment_name", "") or ""

        self.history_collector = HistoryDataCollector(
            output_dir=output_dir,
            save_raw_lengths=getattr(self.config, "history_save_raw_lengths", False),
            flush_interval=getattr(self.config, "history_flush_interval", 10),
            experiment_name=experiment_name,
        )
        self.history_collector.initialize()


        self._resource_config_snapshot = HistoryDataCollector.snapshot_resource_config(
            self.config
        )
        if self.train_engine is not None:
            parallel_state = self.train_engine.get_parallel_state()
            self._resource_config_snapshot.train_backend = parallel_state.get(
                "backend",
                self._resource_config_snapshot.train_backend,
            )
            self._resource_config_snapshot.train_tp = parallel_state.get(
                "train_tp",
                self._resource_config_snapshot.train_tp,
            )
            self._resource_config_snapshot.train_pp = parallel_state.get(
                "train_pp",
                self._resource_config_snapshot.train_pp,
            )
            self._resource_config_snapshot.train_dp = parallel_state.get(
                "train_dp",
                self._resource_config_snapshot.train_dp,
            )

    def _record_history_step(
        self,
        step: int,
        batch: list,
        stats: dict,
        rollout_time: float,
        train_time: float,
        weight_sync_time: float,
        advantage_time: float,
        recompute_time: float,
        step_total_time: float,
    ):
        """Record history step."""
        if self.history_collector is None:
            return


        sequence_pairs = []
        sequence_records = []
        for traj in batch:
            if isinstance(traj, dict):
                pl = traj.get("input_len", 0)
                gl = traj.get("output_len", 0)
                if pl > 0 or gl > 0:
                    sequence_pairs.append((int(pl), int(gl)))
                    record = {
                        "input_len": int(pl),
                        "output_len": int(gl),
                    }
                    for key in (
                        "prompt_id",
                        "total_output_tokens",
                        "tool_returns",
                        "cmlfq_request_id",
                    ):
                        if key in traj:
                            record[key] = traj[key]
                    sequence_records.append(record)


        sm_stats = self.staleness_manager.get_stats() if self.staleness_manager else None
        control_plane = {}
        for key in (
            "global_resource_planner",
            "global_resource_planner_runtime",
        ):
            if key in stats:
                control_plane[key] = stats[key]

        self.history_collector.record_step(
            step=step,
            sequence_pairs=sequence_pairs,
            sequence_records=sequence_records,
            rollout_time=rollout_time,
            train_time=train_time,
            weight_sync_time=weight_sync_time,
            advantage_time=advantage_time,
            recompute_logprob_time=recompute_time,
            step_total_time=step_total_time,
            training_stats=stats,
            pipeline_running=getattr(sm_stats, "running", 0) if sm_stats else 0,
            pipeline_accepted=getattr(sm_stats, "accepted", 0) if sm_stats else 0,
            pipeline_rejected=getattr(sm_stats, "rejected", 0) if sm_stats else 0,
            model_version=stats.get("version", 0),
            resource_config=self._resource_config_snapshot,
            control_plane=control_plane,
        )

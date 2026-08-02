"""Megatron-Core training backend for Qwen3-MoE reinforcement learning."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from RL_Framework.engine.megatron_core_checkpointing import (
    MegatronDistributedCheckpointManager,
)
from RL_Framework.infra.elastic import (
    GradientPayload,
    InterReplicaGradientDomain,
)


class MegatronCoreTrainEngine:
    """MCore TP/EP/DDP backend with streaming HF interoperability."""

    def __init__(
        self,
        *,
        model_path: str,
        learning_rate: float = 1e-6,
        kl_coef: float = 0.001,
        clip_epsilon: float = 0.2,
        micro_batch_size: int = 1,
        train_tp_size: int = 1,
        train_pp_size: int = 1,
        train_cp_size: int = 1,
        train_ep_size: int = 1,
        expert_tensor_parallel_size: int = 1,
        sequence_parallel: bool = True,
        use_distributed_optimizer: bool = True,
        grad_reduce_in_fp32: bool = False,
        use_precision_aware_optimizer: bool | None = None,
        main_grads_dtype: torch.dtype = torch.bfloat16,
        main_params_dtype: torch.dtype = torch.bfloat16,
        exp_avg_dtype: torch.dtype = torch.bfloat16,
        exp_avg_sq_dtype: torch.dtype = torch.bfloat16,
        optimizer_cpu_offload: bool = False,
        optimizer_offload_fraction: float = 0.0,
        optimizer_pin_cpu_grads: bool = True,
        optimizer_pin_cpu_params: bool = True,
        checkpoint_format: str = "torch_dist",
        fully_parallel_save: bool = True,
        async_save: bool = False,
        streaming_export: bool = True,
        use_transformer_engine: bool = True,
        use_cpu_initialization: bool = False,
        recompute_num_layers: int = 1,
        sync_path: str = "./logs/async_rl_weights",
    ):
        self.train_backend = "megatron_core"
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.micro_batch_size = micro_batch_size
        self.train_tp_size = train_tp_size
        self.train_pp_size = train_pp_size
        self.train_cp_size = train_cp_size
        self.train_ep_size = train_ep_size
        self.expert_tensor_parallel_size = expert_tensor_parallel_size
        self.sequence_parallel = sequence_parallel
        self.use_distributed_optimizer = use_distributed_optimizer
        self.grad_reduce_in_fp32 = grad_reduce_in_fp32
        if use_precision_aware_optimizer is None:
            use_precision_aware_optimizer = optimizer_cpu_offload
        self.use_precision_aware_optimizer = use_precision_aware_optimizer
        if not self.use_precision_aware_optimizer:
            main_grads_dtype = torch.float32
            main_params_dtype = torch.float32
            exp_avg_dtype = torch.float32
            exp_avg_sq_dtype = torch.float32
        self.main_grads_dtype = main_grads_dtype
        self.main_params_dtype = main_params_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.optimizer_cpu_offload = optimizer_cpu_offload
        self.optimizer_offload_fraction = optimizer_offload_fraction
        self.optimizer_pin_cpu_grads = optimizer_pin_cpu_grads
        self.optimizer_pin_cpu_params = optimizer_pin_cpu_params
        self.streaming_export = streaming_export
        self.use_transformer_engine = use_transformer_engine
        self.use_cpu_initialization = use_cpu_initialization
        self.recompute_num_layers = recompute_num_layers

        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_main_process = self.rank == 0
        self.current_version = 0

        self.bridge = None
        self.provider = None
        self.model: list[Any] = []
        self.optimizer = None
        self.tokenizer = None
        self.max_seq_length = 0
        self.elastic_gradient_domain: InterReplicaGradientDomain | None = None
        self._pending_hybrid_gradients: list[GradientPayload] = []
        self._hybrid_gradient_lock = threading.Lock()
        self._hybrid_gradient_condition = threading.Condition(
            self._hybrid_gradient_lock
        )
        self._elastic_active_gradient_timeout_s = 300.0
        self._elastic_snapshot_wait_timeout_s = 300.0
        self._elastic_training_step = -1
        self._elastic_step_hybrid_workers: tuple[str, ...] = ()
        self._elastic_step_membership_frozen = False
        self._elastic_step_callback = None
        self._elastic_gradient_update_callback = None
        self._elastic_gradient_metrics = {
            "accepted": 0,
            "stale": 0,
            "future": 0,
        }
        self.checkpoints = MegatronDistributedCheckpointManager(
            sync_path=sync_path,
            checkpoint_format=checkpoint_format,
            fully_parallel_save=fully_parallel_save,
            async_save=async_save,
        )
        elastic_state_dir = Path(
            os.getenv(
                "ELASTIC_TRAINING_STATE_DIR",
                "./logs/elastic_training_state",
            )
        )
        self.elastic_checkpoints = MegatronDistributedCheckpointManager(
            sync_path=str(elastic_state_dir / "checkpoints"),
            checkpoint_format=checkpoint_format,
            fully_parallel_save=fully_parallel_save,
            async_save=True,
        )
        self._latest_elastic_snapshot_version = -1

    def initialize(
        self,
        max_seq_length: int = 2048,
        initialize_optimizer: bool = True,
    ):
        def init_phase(name: str) -> None:
            print(
                f"MCORE_INIT_PHASE rank={self.rank} local_rank={self.local_rank} {name}",
                flush=True,
            )

        init_phase("start")
        if not 0.0 <= self.optimizer_offload_fraction <= 1.0:
            raise ValueError("optimizer_offload_fraction must be between 0 and 1")
        if (
            self.optimizer_offload_fraction > 0.0
            and not self.optimizer_cpu_offload
        ):
            raise ValueError(
                "optimizer_offload_fraction requires optimizer_cpu_offload=True"
            )
        if self.train_pp_size != 1:
            raise ValueError(
                "MegatronCoreTrainEngine currently supports PP=1; use TP/EP/DP "
                "for Qwen3-30B-A3B"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("MegatronCoreTrainEngine requires CUDA")
        torch.cuda.set_device(self.local_rank)
        self.max_seq_length = max_seq_length

        init_phase("imports_start")
        try:
            from megatron.bridge import AutoBridge
            from megatron.core.distributed import DistributedDataParallelConfig
            from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
            from RL_Framework.engine.megatron_core_attention import (
                apply_megatron_core_014_compatibility,
                torch_sdpa_layer_spec,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Megatron-Core dependencies are unavailable. Install the pinned "
                "packages from requirements-megatron.txt."
            ) from exc
        init_phase("imports_complete")
        apply_megatron_core_014_compatibility()

        init_phase("tokenizer_load_start")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        init_phase("tokenizer_load_complete")

        init_phase("bridge_load_start")
        self.bridge = AutoBridge.from_hf_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        init_phase("bridge_load_complete")
        init_phase("provider_build_start")
        self.provider = self.bridge.to_megatron_provider(load_weights=True)
        if not self.use_transformer_engine:
            self.provider.transformer_layer_spec = torch_sdpa_layer_spec
        self._configure_provider(max_seq_length)
        init_phase("provider_finalize_start")
        self.provider.finalize()
        init_phase("provider_finalize_complete")
        init_phase("parallel_init_start")
        self.provider.initialize_model_parallel(seed=42)
        init_phase("parallel_init_complete")

        init_phase(
            "model_build_start "
            f"use_cpu_initialization={self.use_cpu_initialization}"
        )
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=self.grad_reduce_in_fp32,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=self.use_distributed_optimizer,
            check_for_nan_in_grad=True,
        )
        self.model = self.provider.provide_distributed_model(
            ddp_config=ddp_config,
            wrap_with_ddp=initialize_optimizer,
            bf16=True,
            use_cpu_initialization=self.use_cpu_initialization,
            data_parallel_random_init=False,
        )
        init_phase("model_build_complete")
        self._validate_conversion_coverage()
        if self.is_main_process:
            print("MCORE_INIT_PHASE conversion_validation_complete", flush=True)

        if initialize_optimizer:
            if self.is_main_process:
                print("MCORE_INIT_PHASE optimizer_build_start", flush=True)
            optimizer_config = OptimizerConfig(
                optimizer="adam",
                lr=self.learning_rate,
                min_lr=0.0,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_precision_aware_optimizer=(
                    self.use_precision_aware_optimizer
                ),
                main_grads_dtype=self.main_grads_dtype,
                main_params_dtype=self.main_params_dtype,
                exp_avg_dtype=self.exp_avg_dtype,
                exp_avg_sq_dtype=self.exp_avg_sq_dtype,
                clip_grad=1.0,
                use_distributed_optimizer=self.use_distributed_optimizer,
                overlap_param_gather=False,
                optimizer_cpu_offload=self.optimizer_cpu_offload,
                optimizer_offload_fraction=self.optimizer_offload_fraction,
                overlap_cpu_optimizer_d2h_h2d=False,
                pin_cpu_grads=self.optimizer_pin_cpu_grads,
                pin_cpu_params=self.optimizer_pin_cpu_params,
            )
            self.optimizer = get_megatron_optimizer(
                optimizer_config,
                self.model,
                use_gloo_process_groups=False,
            )
            if self.is_main_process:
                print("MCORE_INIT_PHASE optimizer_build_complete", flush=True)
        if self.is_main_process:
            print(
                "MegatronCoreTrainEngine initialized: "
                f"tp={self.train_tp_size}, pp={self.train_pp_size}, "
                f"cp={self.train_cp_size}, ep={self.train_ep_size}, "
                f"dp={self.get_data_parallel_world_size()}, "
                f"distributed_optimizer="
                f"{self.use_distributed_optimizer and initialize_optimizer}, "
                f"optimizer_cpu_offload="
                f"{self.optimizer_cpu_offload and initialize_optimizer}, "
                f"optimizer_offload_fraction="
                f"{self.optimizer_offload_fraction if initialize_optimizer else 0.0}",
                flush=True,
            )

    def _configure_provider(self, max_seq_length: int) -> None:
        provider = self.provider
        provider.tensor_model_parallel_size = self.train_tp_size
        provider.pipeline_model_parallel_size = self.train_pp_size
        provider.context_parallel_size = self.train_cp_size
        provider.expert_model_parallel_size = self.train_ep_size
        provider.expert_tensor_parallel_size = self.expert_tensor_parallel_size
        provider.sequence_parallel = (
            self.sequence_parallel and self.train_tp_size > 1
        )
        provider.seq_length = max_seq_length
        provider.pipeline_dtype = torch.bfloat16
        provider.params_dtype = torch.bfloat16
        provider.bf16 = True
        provider.fp16 = False
        provider.parallel_output = True
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = max(1, self.recompute_num_layers)
        # The CUDA grouped-GEMM extension is optional on the current cluster.
        provider.moe_grouped_gemm = (
            os.getenv("MCORE_MOE_GROUPED_GEMM", "1").lower()
            not in {"0", "false", "no"}
        )
        provider.moe_permute_fusion = False
        provider.moe_router_dtype = None
        provider.moe_token_dispatcher_type = "alltoall"
        if self.use_transformer_engine:
            try:
                import transformer_engine  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Transformer Engine was requested but is not installed. "
                    "Set megatron_use_transformer_engine=false to use the "
                    "PyTorch Flash SDPA adapter."
                ) from exc

    def _validate_conversion_coverage(self) -> None:
        tasks = self.bridge.get_conversion_tasks(self.model)
        local_missing = [
            task.param_name
            for task in tasks
            if task.mapping is None
        ]
        if dist.is_initialized():
            missing_by_rank: list[list[str] | None] = [
                None
            ] * dist.get_world_size()
            dist.all_gather_object(missing_by_rank, local_missing)
        else:
            missing_by_rank = [local_missing]
        missing_count = sum(len(names or []) for names in missing_by_rank)
        if missing_count:
            examples = [
                f"rank={rank}:{name}"
                for rank, names in enumerate(missing_by_rank)
                for name in (names or [])[:3]
            ][:12]
            raise RuntimeError(
                "Megatron Bridge left "
                f"{missing_count} local parameters unmapped; "
                f"examples={examples}"
            )

    def get_version(self) -> int:
        return self.current_version

    def set_version(self, version: int):
        self.current_version = version

    def get_data_parallel_world_size(self) -> int:
        parallel_state = self._parallel_state()
        return int(
            parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )
        )

    def get_data_parallel_rank(self) -> int:
        parallel_state = self._parallel_state()
        return int(
            parallel_state.get_data_parallel_rank(
                with_context_parallel=True
            )
        )

    def get_local_batch_size(self, global_batch_size: int) -> int:
        dp_size = self.get_data_parallel_world_size()
        if global_batch_size % dp_size != 0:
            raise ValueError(
                "global_batch_size must be divisible by MCore DP size: "
                f"{global_batch_size} vs {dp_size}"
            )
        return global_batch_size // dp_size

    def is_batch_source(self) -> bool:
        parallel_state = self._parallel_state()
        return (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.is_pipeline_first_stage()
        )

    def distribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if self.train_tp_size == 1 or not dist.is_initialized():
            return trajectories or []
        parallel_state = self._parallel_state()
        source_rank = parallel_state.get_tensor_model_parallel_src_rank()
        payload = [trajectories if self.rank == source_rank else None]
        dist.broadcast_object_list(
            payload,
            src=source_rank,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        return payload[0] or []

    def align_distributed_trajectories(
        self,
        trajectories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not dist.is_initialized() or self.world_size <= 1:
            return trajectories
        lengths = torch.tensor(
            [int(item["input_ids"].shape[-1]) for item in trajectories],
            device=torch.device(f"cuda:{self.local_rank}"),
            dtype=torch.long,
        )
        dist.all_reduce(lengths, op=dist.ReduceOp.MAX)
        alignment = self._sequence_parallel_alignment()
        pad_token_id = int(self.tokenizer.pad_token_id)
        for trajectory, target_length in zip(trajectories, lengths.tolist()):
            current_length = int(trajectory["input_ids"].shape[-1])
            aligned_length = self._round_sequence_length(int(target_length), alignment)
            pad_width = aligned_length - current_length
            if pad_width <= 0:
                continue
            trajectory["input_ids"] = F.pad(
                trajectory["input_ids"],
                (0, pad_width),
                value=pad_token_id,
            )
            trajectory["attention_mask"] = F.pad(
                trajectory["attention_mask"],
                (0, pad_width),
                value=0,
            )
            trajectory["logprobs"] = F.pad(
                trajectory["logprobs"],
                (0, pad_width),
                value=0.0,
            )
            trajectory["loss_mask"] = F.pad(
                trajectory["loss_mask"],
                (0, pad_width),
                value=0,
            )
        return trajectories

    def _sequence_parallel_alignment(self) -> int:
        if self.sequence_parallel and self.train_tp_size > 1:
            return int(self.train_tp_size)
        return 1

    @staticmethod
    def _round_sequence_length(length: int, alignment: int) -> int:
        if alignment <= 1:
            return int(length)
        return ((int(length) + alignment - 1) // alignment) * alignment

    def _pad_micro_batch_for_sequence_parallel(
        self,
        micro_batch: dict[str, Any],
    ) -> dict[str, Any]:
        alignment = self._sequence_parallel_alignment()
        input_ids = micro_batch["input_ids"]
        target_length = self._round_sequence_length(input_ids.shape[1], alignment)
        pad_width = target_length - input_ids.shape[1]
        if pad_width <= 0:
            return micro_batch

        pad_token_id = int(self.tokenizer.pad_token_id)
        micro_batch["input_ids"] = F.pad(
            input_ids,
            (0, pad_width),
            value=pad_token_id,
        )
        if "attention_mask" in micro_batch:
            micro_batch["attention_mask"] = F.pad(
                micro_batch["attention_mask"],
                (0, pad_width),
                value=0,
            )
        if "old_logprobs" in micro_batch:
            micro_batch["old_logprobs"] = F.pad(
                micro_batch["old_logprobs"],
                (0, pad_width),
                value=0.0,
            )
        if "loss_mask" in micro_batch:
            micro_batch["loss_mask"] = F.pad(
                micro_batch["loss_mask"],
                (0, pad_width),
                value=0,
            )
        return micro_batch

    def recompute_logprobs(self, trajectories: list[dict[str, Any]]):
        self._set_train_mode(False)
        with torch.no_grad():
            for micro_batch in self._iter_micro_batches(trajectories):
                logits = self._forward(micro_batch)
                action_log_probs = self._action_log_probs(
                    logits,
                    micro_batch["input_ids"],
                ).cpu()
                for row, trajectory in zip(
                    action_log_probs,
                    micro_batch["trajectories"],
                ):
                    updated = torch.zeros_like(trajectory["logprobs"])
                    length = min(row.shape[0], updated.shape[1] - 1)
                    updated[:, 1 : 1 + length] = row[:length].unsqueeze(0)
                    trajectory["logprobs"] = updated
        self._set_train_mode(True)

    def grpo_update(
        self,
        trajectories: list[dict[str, Any]],
        ppo_epochs: int = 1,
    ) -> dict[str, float]:
        from megatron.core.distributed import finalize_model_grads

        epoch_stats: list[dict[str, float]] = []
        for _ in range(ppo_epochs):
            micro_batches = list(self._iter_micro_batches(trajectories))
            self._zero_grad()
            self._set_train_mode(True)
            local_stats: list[dict[str, float]] = []
            for micro_batch in micro_batches:
                logits = self._forward(micro_batch)
                loss, stats = self._loss(micro_batch, logits)
                scaled_loss = self.optimizer.scale_loss(
                    loss / max(len(micro_batches), 1)
                )
                scaled_loss.backward()
                local_stats.append(stats)

            # Appendix D requires external gradients to be accumulated into the
            # target core replica before the immutable core DP All-Reduce.
            matched_hybrid_payloads = self._apply_elastic_inter_replica_gradients()
            finalize_model_grads(self.model)
            self._publish_elastic_gradient_updates(matched_hybrid_payloads)
            update_successful, grad_norm, _ = self.optimizer.step()
            if not update_successful:
                raise FloatingPointError("Megatron optimizer rejected the update")
            merged = self._merge_stats(local_stats)
            if isinstance(grad_norm, torch.Tensor):
                merged["grad_norm"] = float(grad_norm.detach().item())
            else:
                merged["grad_norm"] = float(grad_norm or 0.0)
            epoch_stats.append(self._reduce_stats(merged))
            if self._elastic_step_callback is not None:
                self._elastic_step_callback(
                    self._elastic_training_step,
                    self._elastic_training_step + 1,
                )
        result = self._merge_stats(epoch_stats)
        result["version"] = self.current_version
        return result

    def compute_elastic_gradient_payload(
        self,
        trajectories: list[dict[str, Any]],
        *,
        worker_id: str,
        target_core_id: str,
        step: int,
        state_version: int,
        membership_epoch: int,
    ) -> GradientPayload:
        """Compute one external replica gradient without applying an update."""
        from megatron.core.distributed import finalize_model_grads

        micro_batches = list(self._iter_micro_batches(trajectories))
        self._zero_grad()
        self._set_train_mode(True)
        for micro_batch in micro_batches:
            logits = self._forward(micro_batch)
            loss, _stats = self._loss(micro_batch, logits)
            scaled_loss = self.optimizer.scale_loss(
                loss / max(len(micro_batches), 1)
            )
            scaled_loss.backward()
        finalize_model_grads(self.model)
        gradients = []
        for model_chunk in self.model:
            for param in model_chunk.parameters():
                grad = getattr(param, "main_grad", None)
                if grad is not None:
                    gradients.append(grad.detach().cpu())
        return GradientPayload(
            replica_id=worker_id,
            target_core_id=target_core_id,
            tensors=tuple(gradients),
            step=int(step),
            state_version=int(state_version),
            membership_epoch=int(membership_epoch),
        )

    def apply_elastic_gradient_update(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        state_version: int,
    ) -> None:
        """Apply the Core's post-AllReduce gradient on a Hybrid replica."""
        local_gradients = self._model_main_gradients()
        if len(tensors) != len(local_gradients):
            raise ValueError(
                "Hybrid update tensor count mismatch: "
                f"received={len(tensors)}, local={len(local_gradients)}"
            )
        for local, remote in zip(local_gradients, tensors):
            local.copy_(remote.to(device=local.device, dtype=local.dtype))
        update_successful, _grad_norm, _zeros = self.optimizer.step()
        if not update_successful:
            raise FloatingPointError("Hybrid Megatron optimizer rejected the update")
        self.current_version = int(state_version)

    def save_weights(self, path: str, version: int):
        if Path(path) != self.checkpoints.sync_path:
            self.checkpoints.sync_path = Path(path)
        self.checkpoints.save(
            model=self.model,
            optimizer=self.optimizer,
            bridge=self.bridge,
            version=version,
            topology=self.get_parallel_state(),
            export_for_rollout=self.streaming_export,
        )

    def load_weights(self, path: str, version: int):
        if Path(path) != self.checkpoints.sync_path:
            self.checkpoints.sync_path = Path(path)
        self.checkpoints.load(
            model=self.model,
            optimizer=self.optimizer,
            version=version,
        )
        self.current_version = version

    def get_parallel_state(self) -> dict[str, Any]:
        return {
            "backend": self.train_backend,
            "train_tp": self.train_tp_size,
            "train_pp": self.train_pp_size,
            "train_cp": self.train_cp_size,
            "train_ep": self.train_ep_size,
            "train_dp": self.get_data_parallel_world_size(),
            "distributed_optimizer": self.use_distributed_optimizer,
            "checkpoint_format": self.checkpoints.checkpoint_format,
            "streaming_export": self.streaming_export,
            "transformer_engine": self.use_transformer_engine,
            "elastic_gradient_domain": self.elastic_gradient_domain is not None,
        }

    def get_elastic_core_replica_ids(self) -> list[str]:
        return [
            f"dp{index}"
            for index in range(max(1, self.get_data_parallel_world_size()))
        ]

    def get_elastic_lane_state(self) -> dict[str, int]:
        parallel_state = self._parallel_state()
        return {
            "global_rank": self.rank,
            "data_parallel_rank": self.get_data_parallel_rank(),
            "tensor_parallel_rank": int(
                parallel_state.get_tensor_model_parallel_rank()
            ),
            "pipeline_parallel_rank": int(
                parallel_state.get_pipeline_model_parallel_rank()
            ),
            "context_parallel_rank": int(
                parallel_state.get_context_parallel_rank()
            ),
        }

    def configure_elastic_training(
        self,
        core_replica_ids: list[str] | None = None,
        decouple_communication_domains: bool = True,
    ) -> InterReplicaGradientDomain:
        core_ids = core_replica_ids or self.get_elastic_core_replica_ids()
        self.elastic_gradient_domain = InterReplicaGradientDomain(
            core_replica_ids=core_ids,
            process_group=(
                None
                if decouple_communication_domains
                else self.get_elastic_core_process_group()
            ),
            decouple_communication_domains=decouple_communication_domains,
        )
        return self.elastic_gradient_domain

    def get_elastic_core_process_group(self):
        return self._parallel_state().get_data_parallel_group(
            with_context_parallel=True
        )

    def set_elastic_gradient_domain(
        self,
        domain: InterReplicaGradientDomain | None,
    ) -> None:
        self.elastic_gradient_domain = domain
        if domain is not None and not domain.decoupled_communication_domains:
            domain.process_group = self.get_elastic_core_process_group()

    def enqueue_hybrid_gradient_payload(self, payload: GradientPayload) -> None:
        with self._hybrid_gradient_condition:
            self._pending_hybrid_gradients.append(payload)
            self._hybrid_gradient_condition.notify_all()

    def set_elastic_training_step(
        self,
        step: int,
        active_hybrid_workers: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._elastic_training_step = int(step)
        if active_hybrid_workers is not None:
            self._elastic_step_hybrid_workers = tuple(active_hybrid_workers)
            self._elastic_step_membership_frozen = True

    def set_elastic_step_callback(self, callback) -> None:
        self._elastic_step_callback = callback

    def set_elastic_gradient_update_callback(self, callback) -> None:
        self._elastic_gradient_update_callback = callback

    def set_elastic_active_gradient_timeout(self, timeout_s: float) -> None:
        self._elastic_active_gradient_timeout_s = max(0.0, float(timeout_s))

    def set_elastic_snapshot_wait_timeout(self, timeout_s: float) -> None:
        self._elastic_snapshot_wait_timeout_s = max(0.0, float(timeout_s))

    def get_elastic_gradient_metrics(self) -> dict[str, int]:
        with self._hybrid_gradient_lock:
            return dict(self._elastic_gradient_metrics)

    def capture_elastic_state_snapshot(
        self,
        worker_id: str,
        target_core_id: str,
    ) -> int:
        """Return metadata for the latest completed asynchronous snapshot."""
        del worker_id, target_core_id
        deadline = time.monotonic() + self._elastic_snapshot_wait_timeout_s
        version = self._find_latest_elastic_snapshot_version()
        while version < 0 and time.monotonic() < deadline:
            time.sleep(0.1)
            version = self._find_latest_elastic_snapshot_version()
        if version < 0:
            raise TimeoutError(
                "no completed elastic snapshot became available before the "
                "join timeout; core ranks must publish a boundary snapshot"
            )
        snapshot_path = Path(
            self.get_elastic_state_snapshot_path(version)
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = (
            self.elastic_checkpoints.sync_path
            / f"v{version}"
            / "megatron_checkpoint_manifest.json"
        )
        torch.save(
            {
                "backend": self.train_backend,
                "version": version,
                "manifest_path": str(manifest_path),
                "checkpoint_root": str(self.elastic_checkpoints.sync_path),
                "parallel_state": self.get_parallel_state(),
            },
            snapshot_path,
        )
        return version

    def publish_elastic_state_snapshot(self, version: int) -> bool:
        """Collectively stage a non-blocking model+optimizer snapshot."""
        self.poll_elastic_state_snapshots()
        scheduled = self.elastic_checkpoints.save_async_snapshot(
            model=self.model,
            optimizer=self.optimizer,
            version=int(version),
            topology=self.get_parallel_state(),
        )
        if scheduled and not self.elastic_checkpoints.async_save:
            self._record_completed_elastic_snapshot(int(version))
        return scheduled

    def poll_elastic_state_snapshots(self, *, blocking: bool = False) -> list[int]:
        versions = self.elastic_checkpoints.poll_async_snapshots(blocking=blocking)
        for version in versions:
            self._record_completed_elastic_snapshot(int(version))
        return versions

    def close_elastic_state_snapshots(self) -> None:
        for version in self.elastic_checkpoints.close_async_snapshots():
            self._record_completed_elastic_snapshot(int(version))

    def prune_elastic_state_snapshots(
        self,
        *,
        keep_latest: int = 2,
        protected_versions: set[int] | None = None,
    ) -> list[int]:
        return self.elastic_checkpoints.prune_snapshots(
            keep_latest=keep_latest,
            protected_versions=protected_versions,
        )

    def _record_completed_elastic_snapshot(self, version: int) -> None:
        self._latest_elastic_snapshot_version = max(
            self._latest_elastic_snapshot_version,
            int(version),
        )
        snapshot_path = Path(self.get_elastic_state_snapshot_path(version))
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": self.train_backend,
                "version": int(version),
                "manifest_path": str(
                    self.elastic_checkpoints.sync_path
                    / f"v{version}"
                    / "megatron_checkpoint_manifest.json"
                ),
                "checkpoint_root": str(self.elastic_checkpoints.sync_path),
                "parallel_state": self.get_parallel_state(),
            },
            snapshot_path,
        )

    def _find_latest_elastic_snapshot_version(self) -> int:
        candidates = []
        for path in self.elastic_checkpoints.sync_path.glob(
            "v*/megatron_checkpoint_manifest.json"
        ):
            try:
                candidates.append(int(path.parent.name[1:]))
            except ValueError:
                continue
        if candidates:
            self._latest_elastic_snapshot_version = max(
                self._latest_elastic_snapshot_version,
                max(candidates),
            )
        return self._latest_elastic_snapshot_version

    def get_elastic_state_snapshot_path(self, version: int) -> str:
        state_dir = Path(
            os.getenv(
                "ELASTIC_TRAINING_STATE_DIR",
                "./logs/elastic_training_state",
            )
        )
        return str(state_dir / f"v{version}" / f"rank_{self.rank}.pt")

    def load_elastic_state_snapshot(self, snapshot_path: str) -> None:
        payload = torch.load(
            snapshot_path,
            map_location="cpu",
            weights_only=False,
        )
        version = int(payload["version"])
        checkpoint_root = str(
            payload.get("checkpoint_root", self.elastic_checkpoints.sync_path)
        )
        if Path(checkpoint_root) != self.elastic_checkpoints.sync_path:
            self.elastic_checkpoints.sync_path = Path(checkpoint_root)
        self.elastic_checkpoints.load(
            model=self.model,
            optimizer=self.optimizer,
            version=version,
        )
        self.current_version = version

    def _iter_micro_batches(self, trajectories: list[dict[str, Any]]):
        for start in range(0, len(trajectories), self.micro_batch_size):
            rows = trajectories[start : start + self.micro_batch_size]
            yield self._pad_micro_batch(rows)

    def _pad_micro_batch(
        self,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        device = torch.device(f"cuda:{self.local_rank}")
        max_length = max(int(item["input_ids"].shape[-1]) for item in trajectories)
        batch_size = len(trajectories)
        input_ids = torch.full(
            (batch_size, max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        old_logprobs = torch.zeros(
            (batch_size, max_length),
            dtype=torch.float32,
            device=device,
        )
        loss_mask = torch.zeros_like(old_logprobs)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
        advantages = torch.zeros_like(rewards)
        for index, trajectory in enumerate(trajectories):
            length = int(trajectory["input_ids"].shape[-1])
            input_ids[index, :length] = trajectory["input_ids"][0].to(device)
            old_logprobs[index, :length] = trajectory["logprobs"][0].to(device)
            loss_mask[index, :length] = trajectory["loss_mask"][0].to(device)
            rewards[index] = trajectory["rewards"].reshape(-1)[0].to(device)
            advantages[index] = trajectory.get(
                "advantages",
                trajectory["rewards"],
            ).reshape(-1)[0].to(device)
        return {
            "input_ids": input_ids,
            "old_logprobs": old_logprobs,
            "loss_mask": loss_mask,
            "rewards": rewards,
            "advantages": advantages,
            "trajectories": trajectories,
        }

    def _forward(self, micro_batch: dict[str, Any]) -> torch.Tensor:
        micro_batch = self._pad_micro_batch_for_sequence_parallel(micro_batch)
        input_ids = micro_batch["input_ids"]
        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand_as(input_ids)
        logits = self.model[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=None,
        )
        if logits.shape[0] == input_ids.shape[1] and logits.shape[1] == input_ids.shape[0]:
            logits = logits.transpose(0, 1).contiguous()
        return logits

    def _action_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        local_logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        if self.train_tp_size == 1:
            return torch.gather(
                F.log_softmax(local_logits.float(), dim=-1),
                dim=-1,
                index=targets.unsqueeze(-1),
            ).squeeze(-1)

        parallel_state = self._parallel_state()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        local_max = local_logits.max(dim=-1).values.detach()
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
        shifted = local_logits.float() - local_max.unsqueeze(-1)
        denominator = torch.exp(shifted).sum(dim=-1)
        denominator = dist_nn.all_reduce(
            denominator,
            op=dist.ReduceOp.SUM,
            group=tp_group,
        )
        local_vocab = local_logits.shape[-1]
        vocab_start = (
            parallel_state.get_tensor_model_parallel_rank() * local_vocab
        )
        vocab_end = vocab_start + local_vocab
        local_mask = (targets >= vocab_start) & (targets < vocab_end)
        local_targets = (targets - vocab_start).masked_fill(~local_mask, 0)
        selected = torch.gather(
            local_logits.float(),
            -1,
            local_targets.unsqueeze(-1),
        ).squeeze(-1)
        selected = selected.masked_fill(~local_mask, 0.0)
        selected = dist_nn.all_reduce(
            selected,
            op=dist.ReduceOp.SUM,
            group=tp_group,
        )
        return selected - local_max - torch.log(denominator)

    def _loss(
        self,
        micro_batch: dict[str, Any],
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_log_probs = self._action_log_probs(
            logits,
            micro_batch["input_ids"],
        )
        old_logprobs = micro_batch["old_logprobs"][:, 1:]
        loss_mask = micro_batch["loss_mask"][:, 1:]
        advantages = micro_batch["advantages"]
        log_ratio = action_log_probs - old_logprobs
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )
        pg_loss = torch.maximum(
            -advantages.unsqueeze(1) * ratio,
            -advantages.unsqueeze(1) * clipped_ratio,
        )
        count = loss_mask.sum().clamp(min=1.0)
        pg_loss = (pg_loss * loss_mask).sum() / count
        # Positive second-order approximation of KL(old_policy || new_policy).
        kl_per_token = (ratio - 1.0) - log_ratio
        kl = (kl_per_token * loss_mask).sum() / count
        loss = pg_loss + self.kl_coef * kl
        return loss, {
            "loss": float(loss.detach()),
            "pg_loss": float(pg_loss.detach()),
            "kl": float(kl.detach()),
            "reward_sum": float(micro_batch["rewards"].sum()),
            "n_samples": float(micro_batch["rewards"].numel()),
            "n_updates": 1.0,
        }

    def _reduce_stats(self, stats: dict[str, float]) -> dict[str, float]:
        updates = float(stats["n_updates"])
        values = torch.tensor(
            [
                stats["loss"] * updates,
                stats["pg_loss"] * updates,
                stats["kl"] * updates,
                stats["reward_sum"],
                stats["n_samples"],
                updates,
                stats.get("grad_norm", 0.0) * updates,
            ],
            dtype=torch.float64,
            device=torch.device(f"cuda:{self.local_rank}"),
        )
        group = self._parallel_state().get_data_parallel_group(
            with_context_parallel=True
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
        global_updates = values[5].clamp(min=1.0)
        return {
            "loss": float(values[0] / global_updates),
            "pg_loss": float(values[1] / global_updates),
            "kl": float(values[2] / global_updates),
            "reward_sum": float(values[3]),
            "n_samples": float(values[4]),
            "n_updates": float(values[5]),
            "reward_mean": float(values[3] / values[4].clamp(min=1.0)),
            "grad_norm": float(values[6] / global_updates),
        }

    @staticmethod
    def _merge_stats(stats: list[dict[str, float]]) -> dict[str, float]:
        if not stats:
            return {
                "loss": 0.0,
                "pg_loss": 0.0,
                "kl": 0.0,
                "reward_sum": 0.0,
                "reward_mean": 0.0,
                "n_samples": 0.0,
                "n_updates": 0.0,
            }
        updates = sum(float(item.get("n_updates", 1.0)) for item in stats)
        samples = sum(float(item.get("n_samples", 0.0)) for item in stats)
        reward_sum = sum(float(item.get("reward_sum", 0.0)) for item in stats)
        return {
            "loss": sum(
                float(item["loss"]) * float(item.get("n_updates", 1.0))
                for item in stats
            ) / max(updates, 1.0),
            "pg_loss": sum(
                float(item["pg_loss"]) * float(item.get("n_updates", 1.0))
                for item in stats
            ) / max(updates, 1.0),
            "kl": sum(
                float(item["kl"]) * float(item.get("n_updates", 1.0))
                for item in stats
            ) / max(updates, 1.0),
            "reward_sum": reward_sum,
            "reward_mean": reward_sum / max(samples, 1.0),
            "n_samples": samples,
            "n_updates": updates,
            "grad_norm": sum(
                float(item.get("grad_norm", 0.0))
                * float(item.get("n_updates", 1.0))
                for item in stats
            ) / max(updates, 1.0),
        }

    def _zero_grad(self) -> None:
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

    def _set_train_mode(self, enabled: bool) -> None:
        for model_chunk in self.model:
            model_chunk.train(enabled)

    def _model_main_gradients(self) -> list[torch.Tensor]:
        gradients = []
        for model_chunk in self.model:
            for param in model_chunk.parameters():
                grad = getattr(param, "main_grad", None)
                if grad is not None:
                    gradients.append(grad)
        return gradients

    def _apply_elastic_inter_replica_gradients(self) -> list[GradientPayload]:
        if self.elastic_gradient_domain is None:
            with self._hybrid_gradient_lock:
                self._pending_hybrid_gradients.clear()
            return []
        params_and_grads = [(None, grad) for grad in self._model_main_gradients()]
        if not params_and_grads:
            return []

        core_id = f"dp{self.get_data_parallel_rank()}"
        current_step = self._elastic_training_step
        expected_workers = set(
            self._elastic_step_hybrid_workers
            if self._elastic_step_membership_frozen
            else self.elastic_gradient_domain.active_hybrid_ids_for_core(core_id)
        )
        deadline = time.time() + self._elastic_active_gradient_timeout_s
        with self._hybrid_gradient_condition:
            while expected_workers:
                present = {
                    payload.replica_id
                    for payload in self._pending_hybrid_gradients
                    if payload.replica_id in expected_workers
                    and (payload.step < 0 or payload.step == current_step)
                    and payload.membership_epoch
                    == self.elastic_gradient_domain.membership_epoch(
                        payload.replica_id
                    )
                }
                missing = expected_workers - present
                if not missing:
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for active hybrid gradients before "
                        f"core All-Reduce: step={current_step}, missing={sorted(missing)}"
                    )
                self._hybrid_gradient_condition.wait(timeout=min(remaining, 0.1))

            matching: list[GradientPayload] = []
            future: list[GradientPayload] = []
            for payload in self._pending_hybrid_gradients:
                if (
                    payload.replica_id in expected_workers
                    and (payload.step < 0 or current_step < 0 or payload.step == current_step)
                ):
                    matching.append(payload)
                elif payload.step > current_step:
                    future.append(payload)
                    self._elastic_gradient_metrics["future"] += 1
                else:
                    self._elastic_gradient_metrics["stale"] += 1
            self._pending_hybrid_gradients = future
            self._elastic_gradient_metrics["accepted"] += len(matching)

        accumulated = self.elastic_gradient_domain.accumulate_local_gradients(
            core_replica_id=core_id,
            core_gradients=tuple(grad.detach() for _, grad in params_and_grads),
            hybrid_payloads=matching,
            step=current_step,
            state_version=current_step,
        )
        for (_, grad), accumulated_grad in zip(
            params_and_grads,
            accumulated,
        ):
            grad.copy_(
                accumulated_grad.to(device=grad.device, dtype=grad.dtype)
            )
        return matching

    def _publish_elastic_gradient_updates(
        self,
        payloads: list[GradientPayload],
    ) -> None:
        if not payloads or self._elastic_gradient_update_callback is None:
            return
        gradients = tuple(
            grad.detach().cpu() for grad in self._model_main_gradients()
        )
        self._elastic_gradient_update_callback(
            payloads,
            gradients,
            self._elastic_training_step + 1,
        )

    @staticmethod
    def _parallel_state():
        from megatron.core import parallel_state

        return parallel_state

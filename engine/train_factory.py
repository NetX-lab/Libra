"""Support code for Train factory."""

from __future__ import annotations

from RL_Framework.config import AsyncRLConfig
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.engine.megatron_train_engine import Megatron3DTrainEngine
from RL_Framework.engine.train_engine import FSDPTrainEngine, TrainEngine


def create_train_engine(config: AsyncRLConfig) -> TrainEngine:
    """Create train engine."""
    backend = getattr(config, "train_backend", "fsdp")

    if backend == "fsdp":
        return FSDPTrainEngine(
            model_path=config.model_path,
            learning_rate=config.learning_rate,
            kl_coef=config.kl_coef,
            clip_epsilon=config.clip_epsilon,
            micro_batch_size=config.micro_batch_size,
        )

    if backend == "megatron3d":
        return Megatron3DTrainEngine(
            model_path=config.model_path,
            learning_rate=config.learning_rate,
            kl_coef=config.kl_coef,
            clip_epsilon=config.clip_epsilon,
            micro_batch_size=config.micro_batch_size,
            train_tp_size=config.train_tp_size,
            train_pp_size=config.train_pp_size,
            train_dp_size=config.train_dp_size,
            pipeline_schedule=config.pipeline_schedule,
            virtual_pipeline_model_parallel_size=(
                config.virtual_pipeline_model_parallel_size
            ),
            sequence_parallel=config.sequence_parallel,
        )

    if backend == "megatron_core":
        return MegatronCoreTrainEngine(
            model_path=config.model_path,
            learning_rate=config.learning_rate,
            kl_coef=config.kl_coef,
            clip_epsilon=config.clip_epsilon,
            micro_batch_size=config.micro_batch_size,
            train_tp_size=config.train_tp_size,
            train_pp_size=config.train_pp_size,
            train_cp_size=config.train_cp_size,
            train_ep_size=config.train_ep_size,
            expert_tensor_parallel_size=config.expert_tensor_parallel_size,
            sequence_parallel=config.sequence_parallel,
            use_distributed_optimizer=config.use_distributed_optimizer,
            grad_reduce_in_fp32=config.megatron_grad_reduce_in_fp32,
            use_precision_aware_optimizer=(
                config.megatron_use_precision_aware_optimizer
            ),
            optimizer_cpu_offload=config.megatron_optimizer_cpu_offload,
            optimizer_offload_fraction=(
                config.megatron_optimizer_offload_fraction
            ),
            optimizer_pin_cpu_grads=config.megatron_optimizer_pin_cpu_grads,
            optimizer_pin_cpu_params=config.megatron_optimizer_pin_cpu_params,
            checkpoint_format=config.megatron_checkpoint_format,
            fully_parallel_save=config.megatron_fully_parallel_save,
            async_save=config.megatron_async_save,
            streaming_export=config.megatron_streaming_export,
            use_transformer_engine=config.megatron_use_transformer_engine,
            use_cpu_initialization=config.megatron_use_cpu_initialization,
            recompute_num_layers=config.megatron_recompute_num_layers,
            sync_path=config.sync_path,
            device_backend=config.device_backend,
        )

    raise ValueError(f"Unknown train_backend: {backend}")

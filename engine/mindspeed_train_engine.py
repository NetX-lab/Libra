"""DeepSeek V4 MindSpeed provider and PP execution for Libra.

Targets the pinned automodelwire v30-fixed runtime. The scheduling, GRPO loss,
policy versions and checkpoint manifest remain owned by Libra.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.infra.elastic import GradientPayload


def configure_mindspeed_runtime(overrides, *, validate=True):
    """Apply the pinned MindSpeed-LLM adaptor once, before importing MCore."""
    import argparse
    from mindspeed_llm import megatron_adaptor  # noqa: F401
    from megatron.training.arguments import add_megatron_arguments
    from mindspeed_llm.tasks.megatron_adaptor_v2 import repatch

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    flags = ["--" + key.replace("_", "-") for key, value in overrides.items() if value is True]
    args, _ = parser.parse_known_args(flags)
    values = {**vars(args), **overrides}
    repatch(values)
    if validate:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        # The caller is a Libra process; its CLI is not Megatron's CLI.
        import sys
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0]]
            args = parse_args(ignore_unknown_args=True)
        finally:
            sys.argv = old_argv
        for key, value in values.items():
            setattr(args, key, value)
        args = validate_args(args)
        set_global_variables(args)
    return values


class MBridgeAdapter:
    """Translate mbridge's public export API to Libra's checkpoint contract."""

    def __init__(self, bridge, tokenizer, model_path=None):
        self.bridge = bridge
        self.tokenizer = tokenizer
        self.model_path = Path(model_path) if model_path else None

    def _original_index(self):
        if self.model_path is None:
            return None
        return json.loads((self.model_path / "model.safetensors.index.json").read_text())

    @staticmethod
    def _missing_frozen_names(original, exported):
        missing = set(original["weight_map"]) - set(exported)
        unexpected = sorted(name for name in missing if not name.startswith("mtp."))
        if unexpected:
            raise RuntimeError(f"V4 export omitted non-MTP weights: {unexpected[:8]}")
        return sorted(missing)

    def export_hf_weights(self, model, *, cpu=False, show_progress=False):
        exported = set()
        device = None
        for name, tensor in self.bridge.export_weights(model):
            exported.add(name)
            device = tensor.device
            yield name, tensor.cpu() if cpu else tensor
        original = self._original_index()
        if original is not None:
            from safetensors import safe_open
            for name in self._missing_frozen_names(original, exported):
                with safe_open(str(self.model_path / original["weight_map"][name]), framework="pt", device="cpu") as shard:
                    tensor = shard.get_tensor(name)
                yield name, tensor if cpu else tensor.to(device)

    def save_hf_pretrained(self, model, path, *, show_progress=False):
        self.bridge.save_weights(model, str(path), memory_efficient=True)
        error = [None]
        if dist.get_rank() == 0:
            try:
                self.tokenizer.save_pretrained(path)
                original = self._original_index()
                if original is not None:
                    # MTP is frozen/not instantiated in the training model. Preserve
                    # its original shards rather than silently dropping DSpark.
                    import shutil
                    path = Path(path)
                    index_path = path / "model.safetensors.index.json"
                    exported = json.loads(index_path.read_text())
                    for name in self._missing_frozen_names(original, exported["weight_map"]):
                        source = self.model_path / original["weight_map"][name]
                        link = path / ("frozen-" + source.name)
                        if not link.exists():
                            link.symlink_to(source.resolve())
                        exported["weight_map"][name] = link.name
                    exported["metadata"] = original.get("metadata", {})
                    index_path.write_text(json.dumps(exported, indent=2))
                    for source in self.model_path.iterdir():
                        if source.suffix == ".py" or source.name in {"config.json", "generation_config.json"}:
                            shutil.copy2(source, path / source.name)
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"V4 HF export finalization failed: {error[0]}")


class MindSpeedTrainEngine(MegatronCoreTrainEngine):
    def __init__(self, *, mindspeed_args_path="", global_batch_size=32,
                 train_dp_size=0, **kwargs):
        super().__init__(**kwargs)
        self.mindspeed_args_path = mindspeed_args_path
        self.global_batch_size = global_batch_size
        self.train_dp_size = int(train_dp_size or 0)
        if self.train_dp_size <= 0:
            model_parallel = max(1, self.train_tp_size * self.train_pp_size)
            if self.world_size % model_parallel == 0:
                self.train_dp_size = self.world_size // model_parallel
            else:
                self.train_dp_size = 1

    def _prepare_accelerator(self):
        import torch_npu  # noqa: F401; register NPU and HCCL
        if not torch.npu.is_available():
            raise RuntimeError("MindSpeed V4 requires an available Ascend NPU")
        torch.npu.set_device(self.local_rank)

    def _device(self):
        return torch.device(f"npu:{self.local_rank}")

    def get_elastic_core_replica_ids(self) -> list[str]:
        # A DeepSeek-V4 replica includes all dense DP and expert-parallel
        # shards.  The generic Megatron hook models each dense-DP lane as a
        # separate replica, which would silently discard most EHP gradients.
        return ["model"]

    def get_elastic_local_core_id(self) -> str:
        return "model"

    def get_elastic_replica_size_gpus(self) -> int:
        # EP is distributed over the DP dimension; it must not be multiplied
        # into TP*PP*DP a second time. This is the complete V4 training mesh.
        return max(1, self.train_tp_size * self.train_pp_size * self.train_dp_size)

    def get_elastic_lane_state(self) -> dict[str, int]:
        state = super().get_elastic_lane_state()
        state["elastic_replica_rank"] = int(self.rank % self.get_elastic_replica_size_gpus())
        return state

    def configure_elastic_training(self, core_replica_ids=None,
                                   decouple_communication_domains=True,
                                   replica_world_size=None):
        width = self.get_elastic_replica_size_gpus()
        if replica_world_size is not None and int(replica_world_size) != width:
            raise ValueError(
                f"DeepSeek-V4 EHP requires a complete TP*PP*DP replica ({width} ranks); "
                f"got {replica_world_size}"
            )
        return super().configure_elastic_training(
            core_replica_ids=core_replica_ids or ["model"],
            decouple_communication_domains=decouple_communication_domains,
            replica_world_size=width,
        )

    def _apply_elastic_inter_replica_gradients(self):
        """Merge EHP gradients with the matching dense or expert DP lane.

        Megatron's finalizer has already reduced the core gradients. EHP's
        gradient must then be added and synchronized over the parameter's own
        DP group: expert parameters use expert-DP, while ordinary parameters
        use dense-DP. A single flat DP reduction corrupts EP-sharded weights.
        """
        pending = getattr(self, "_pending_hybrid_gradients", [])
        expected = set(getattr(self, "_elastic_step_hybrid_workers", ()))
        if not pending and not expected:
            return
        if self.elastic_gradient_domain is None:
            with self._hybrid_gradient_condition:
                self._pending_hybrid_gradients.clear()
            return
        deadline = __import__("time").monotonic() + self._elastic_active_gradient_timeout_s
        with self._hybrid_gradient_condition:
            while expected:
                present = {p.replica_id for p in self._pending_hybrid_gradients
                           if p.replica_id in expected and (p.step < 0 or p.step == self._elastic_training_step)}
                missing = expected - present
                if not missing:
                    break
                remaining = deadline - __import__("time").monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for active EHP gradients: {sorted(missing)}")
                self._hybrid_gradient_condition.wait(timeout=min(remaining, 0.1))
            pending = list(self._pending_hybrid_gradients)

        params_and_grads = []
        for chunk in self.model:
            for param in chunk.parameters():
                grad = getattr(param, "main_grad", None)
                if grad is not None:
                    params_and_grads.append((param, grad))
        if not params_and_grads:
            with self._hybrid_gradient_condition:
                self._pending_hybrid_gradients.clear()
            return
        core_id = self.get_elastic_local_core_id()
        reduced = self.elastic_gradient_domain.reduce_core_gradients(
            core_gradients={core_id: tuple(g.detach() for _, g in params_and_grads)},
            hybrid_payloads=pending,
            step=self._elastic_training_step,
            state_version=self.current_version,
        )[core_id]
        mpu = self._parallel_state()
        for (param, grad), value in zip(params_and_grads, reduced):
            grad.copy_(value.to(device=grad.device, dtype=grad.dtype))
            is_expert = bool(getattr(param, "is_expert_parallel", False)) or getattr(param, "allreduce", True) is False
            try:
                group = mpu.get_data_parallel_group(with_context_parallel=True, is_expert=is_expert)
            except TypeError as exc:
                if is_expert and self.train_ep_size > 1:
                    raise RuntimeError("MindSpeed EHP needs Megatron's expert-DP group accessor") from exc
                group = mpu.get_data_parallel_group(with_context_parallel=True)
            if dist.is_initialized() and dist.get_world_size(group=group) > 1:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
                grad.div_(dist.get_world_size(group=group))

        if self._elastic_gradient_update_callback is not None:
            from RL_Framework.infra.elastic import GradientUpdate
            for payload in pending:
                if payload.replica_id in expected and payload.replica_rank == self.rank % self.get_elastic_replica_size_gpus():
                    self._elastic_gradient_update_callback(GradientUpdate(
                        replica_id=payload.replica_id,
                        tensors=tuple(g.detach().cpu() for _, g in params_and_grads),
                        step=payload.step, state_version=self.current_version,
                        membership_epoch=payload.membership_epoch,
                    ))
        with self._hybrid_gradient_condition:
            self._pending_hybrid_gradients.clear()

    def build_mindspeed_args(self, max_seq_length):
        path = Path(self.mindspeed_args_path) if self.mindspeed_args_path else (
            Path(__file__).resolve().parents[1] / "configs/model_arch_config/deepseek_v4_mindspeed_args.json"
        )
        args = json.loads(path.read_text())
        args.update(
            tensor_model_parallel_size=self.train_tp_size,
            pipeline_model_parallel_size=self.train_pp_size,
            expert_model_parallel_size=self.train_ep_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            context_parallel_size=self.train_cp_size,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=self.sequence_parallel,
            use_distributed_optimizer=self.use_distributed_optimizer,
            tokenizer_name_or_path=self.model_path,
            seq_length=max_seq_length, micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size, lr=self.learning_rate,
            use_cpu_initialization=self.use_cpu_initialization,
            use_precision_aware_optimizer=False,
            recompute_num_layers=self.recompute_num_layers,
            overlap_grad_reduce=False, overlap_param_gather=False,
            seed=42,
        )
        return args

    def initialize(self, max_seq_length=2048, initialize_optimizer=True):
        if self.train_pp_size != 2 or self.train_cp_size != 1:
            raise ValueError("DeepSeek V4 MindSpeed requires PP=2 and CP=1")
        from RL_Framework.engine.mindspeed_v4_ops import require_mindspeed_v4_ops
        require_mindspeed_v4_ops()
        self._prepare_accelerator()
        print(f"LIBRA_MINDSPEED_PHASE rank={self.rank} runtime_init", flush=True)
        configure_mindspeed_runtime(self.build_mindspeed_args(max_seq_length))
        if not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group("hccl", timeout=timedelta(minutes=120))
        mpu = self._parallel_state()
        if mpu.model_parallel_is_initialized():
            raise RuntimeError("MindSpeed provider requires a fresh model-parallel process")
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.train_tp_size,
            pipeline_model_parallel_size=self.train_pp_size,
            context_parallel_size=self.train_cp_size,
            expert_model_parallel_size=self.train_ep_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            # Each NPU rank already participates in the HCCL groups. Megatron's
            # extra CPU/Gloo mirrors are unused by this engine and create a
            # second full mesh for every model-parallel group, which failed
            # during bootstrap on the 8-node MindSpeed job.
            create_gloo_process_groups=False,
            # Megatron's CPU optimizer offload needs only DP-with-CP Gloo
            # groups. The runtime patch keeps these without creating Gloo
            # mirrors for every tensor/expert/pipeline group.
            create_gloo_optimizer_process_groups=self.optimizer_cpu_offload,
        )
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(42)
        from megatron.core.enums import ModelType
        from megatron.training.training import get_model
        from mindspeed_llm.pretrain_deepseek4 import model_provider
        from mbridge import AutoBridge

        hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        if hf_config.model_type != "deepseek_v4":
            raise ValueError("MindSpeed provider currently targets deepseek_v4 only")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"LIBRA_MINDSPEED_PHASE rank={self.rank} bridge_init", flush=True)
        bridge = AutoBridge.from_config(hf_config, dtype=torch.bfloat16)
        print(f"LIBRA_MINDSPEED_PHASE rank={self.rank} model_build", flush=True)
        self.model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=initialize_optimizer)
        print(f"LIBRA_MINDSPEED_PHASE rank={self.rank} weights_load", flush=True)
        bridge.load_weights(self.model, self.model_path, memory_efficient=True)
        self.bridge = MBridgeAdapter(bridge, self.tokenizer, self.model_path)
        self.max_seq_length = max_seq_length
        from megatron.core.utils import get_model_config
        for chunk in self.model:
            cfg = get_model_config(chunk)
            # PP transfers use a fixed padded shape across every microbatch;
            # leaving the variable-length mode enabled made the upstream
            # schedule construct zero-length activations on PP=2.
            cfg.variable_seq_lengths = False
            cfg.calculate_per_token_loss = False
        if initialize_optimizer:
            print(f"LIBRA_MINDSPEED_PHASE rank={self.rank} optimizer_build", flush=True)
            from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
            from megatron.core.distributed import finalize_model_grads
            optim_config = OptimizerConfig(
                optimizer="adam", lr=self.learning_rate, min_lr=0.0,
                weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
                bf16=True, params_dtype=torch.bfloat16, clip_grad=1.0,
                use_distributed_optimizer=self.use_distributed_optimizer,
                use_precision_aware_optimizer=False,
                optimizer_cpu_offload=self.optimizer_cpu_offload,
                optimizer_offload_fraction=self.optimizer_offload_fraction,
            )
            self.optimizer = get_megatron_optimizer(optim_config, self.model)
            for chunk in self.model:
                cfg = get_model_config(chunk)
                cfg.grad_scale_func = self.optimizer.scale_loss
                cfg.finalize_model_grads_func = finalize_model_grads
        print(f"LIBRA_MINDSPEED_INITIALIZED rank={self.rank} pp={self.train_pp_size}", flush=True)

    def distribute_trajectories(self, trajectories):
        """First broadcast across TP on PP0, then along each PP lane."""
        mpu = self._parallel_state()
        payload = [trajectories if self.is_batch_source() else None]
        if mpu.is_pipeline_first_stage() and self.train_tp_size > 1:
            dist.broadcast_object_list(payload, src=mpu.get_tensor_model_parallel_src_rank(),
                                       group=mpu.get_tensor_model_parallel_group())
        dist.broadcast_object_list(payload, src=mpu.get_pipeline_model_parallel_first_rank(),
                                   group=mpu.get_pipeline_model_parallel_group())
        return payload[0] or []

    def align_distributed_trajectories(self, trajectories):
        """Use one sequence length for all 1F1B microbatches and PP lanes."""
        trajectories = super().align_distributed_trajectories(trajectories)
        if not trajectories:
            return trajectories
        # Sparse Flash MLA's index metadata requires a full 512-entry top-k
        # even when the generated response is short. The validated V4 path
        # uses a fixed sequence length, so keep each training microbatch at
        # the configured length rather than shrinking it to the longest row.
        length = max(self.max_seq_length, *(t["input_ids"].shape[-1] for t in trajectories))
        length = self._round_sequence_length(length, self._sequence_parallel_alignment())
        for trajectory in trajectories:
            width = length - trajectory["input_ids"].shape[-1]
            if width:
                for key in ("input_ids", "attention_mask", "logprobs", "loss_mask"):
                    value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                    trajectory[key] = torch.nn.functional.pad(trajectory[key], (0, width), value=value)
        return trajectories

    def _pipeline(self, trajectories, *, forward_only=False):
        from megatron.core.pipeline_parallel import get_forward_backward_func
        batches = list(self._iter_recompute_micro_batches(trajectories) if forward_only
                       else self._iter_micro_batches(trajectories))
        if not batches:
            raise ValueError("Pipeline batch must not be empty")
        # Uniform communication shape, including a short final microbatch.
        sizes = {tuple(b['input_ids'].shape) for b in batches}
        if len(sizes) != 1:
            raise ValueError("Align trajectories and use complete microbatches before PP execution")
        batches = [self._pad_micro_batch_for_sequence_parallel(b) for b in batches]
        rows = []

        def forward_step(iterator, model):
            batch = next(iterator)
            ids = batch["input_ids"]
            positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand_as(ids)
            output = model(input_ids=ids, position_ids=positions, attention_mask=None, labels=None)

            def loss_func(output_tensor):
                # DeepSeek4Model post_process returns [batch, sequence, vocabulary].
                if tuple(output_tensor.shape[:2]) != tuple(ids.shape):
                    raise RuntimeError("Unexpected DeepSeek4 output shape")
                if forward_only:
                    rows.append(self._action_log_probs(output_tensor, ids).detach())
                    return output_tensor.new_zeros(()), {}
                return self._loss(batch, output_tensor)

            return output, loss_func

        stats = get_forward_backward_func()(
            forward_step_func=forward_step, data_iterator=iter(batches), model=self.model,
            num_microbatches=len(batches), seq_length=batches[0]["input_ids"].shape[1],
            micro_batch_size=batches[0]["input_ids"].shape[0], forward_only=forward_only,
        )
        return stats, rows

    def recompute_logprobs(self, trajectories):
        self._set_train_mode(False)
        try:
            with torch.no_grad():
                _, rows = self._pipeline(trajectories, forward_only=True)
                mpu = self._parallel_state()
                if mpu.is_pipeline_last_stage():
                    logprobs = torch.cat(rows, dim=0).contiguous()
                else:
                    length = self._round_sequence_length(trajectories[0]["input_ids"].shape[-1],
                                                        self._sequence_parallel_alignment())
                    logprobs = torch.empty((len(trajectories), length - 1), device=self._device(), dtype=torch.float32)
                dist.broadcast(logprobs, src=mpu.get_pipeline_model_parallel_last_rank(),
                               group=mpu.get_pipeline_model_parallel_group())
                for row, trajectory in zip(logprobs.cpu(), trajectories):
                    updated = torch.zeros_like(trajectory["logprobs"])
                    length = min(row.numel(), updated.shape[-1] - 1)
                    updated[:, 1:1 + length] = row[:length]
                    trajectory["logprobs"] = updated
        finally:
            self._set_train_mode(True)

    def grpo_update(self, trajectories, ppo_epochs=1):
        epoch_stats = []
        for _ in range(ppo_epochs):
            self._zero_grad()
            self._set_train_mode(True)
            stats, _ = self._pipeline(trajectories)
            # The pipeline schedule already finalizes model gradients once.
            self._apply_elastic_inter_replica_gradients()
            ok, norm, _ = self.optimizer.step()
            if not ok:
                raise FloatingPointError("MindSpeed optimizer rejected the update")
            mpu = self._parallel_state()
            result = None
            if mpu.is_pipeline_last_stage():
                merged = self._merge_stats(stats)
                merged["grad_norm"] = float(norm or 0.0)
                result = self._reduce_stats(merged)
            payload = [result]
            dist.broadcast_object_list(payload, src=mpu.get_pipeline_model_parallel_last_rank(),
                                       group=mpu.get_pipeline_model_parallel_group())
            epoch_stats.append(payload[0])
        result = self._merge_stats(epoch_stats)
        result["version"] = self.current_version
        return result

    def compute_elastic_gradient_payload(self, trajectories, *, worker_id, target_core_id,
                                         step, state_version, membership_epoch):
        self._zero_grad()
        self._set_train_mode(True)
        self._pipeline(trajectories)
        return GradientPayload(
            replica_id=worker_id, target_core_id=target_core_id,
            tensors=tuple(g.detach().cpu() for g in self._model_main_gradients()),
            replica_rank=int(self.rank % self.get_elastic_replica_size_gpus()),
            replica_world_size=self.get_elastic_replica_size_gpus(),
            step=step, state_version=state_version, membership_epoch=membership_epoch,
        )

"""Support code for Megatron train engine."""

from __future__ import annotations

import contextlib
import gc
import json
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from RL_Framework.infra.elastic import GradientPayload, InterReplicaGradientDomain
from RL_Framework.engine.qwen3d_modules import (
    QwenPipelineStage,
    export_stage_state_dict_for_hf,
    vocab_parallel_token_logprobs,
)
from RL_Framework.engine.train_parallel import TrainParallelRuntime


def _maybe_barrier():
    if dist.is_initialized():
        dist.barrier()


class Megatron3DTrainEngine:
    """Megatron3 d train engine implementation."""

    def __init__(
        self,
        model_path: str,
        learning_rate: float = 1e-6,
        kl_coef: float = 0.001,
        clip_epsilon: float = 0.2,
        micro_batch_size: int = 4,
        train_tp_size: int = 1,
        train_pp_size: int = 1,
        train_dp_size: int = 1,
        pipeline_schedule: str = "1f1b",
        virtual_pipeline_model_parallel_size: int = 0,
        sequence_parallel: bool = True,
    ):
        self.train_backend = "megatron3d"
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.micro_batch_size = micro_batch_size
        self.train_tp_size = train_tp_size
        self.train_pp_size = train_pp_size
        self.train_dp_size = train_dp_size
        self.pipeline_schedule = pipeline_schedule
        self.virtual_pipeline_model_parallel_size = (
            virtual_pipeline_model_parallel_size
        )
        self.sequence_parallel = sequence_parallel

        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_main_process = self.rank == 0
        self.is_distributed = dist.is_initialized()

        self.runtime: TrainParallelRuntime | None = None
        self.tokenizer = None
        self.model = None
        self.model_config = None
        self.optimizer = None
        self.scheduler = None
        self.current_version = 0
        self.adapter_mode = "qwen_native_3d"
        self.elastic_gradient_domain: InterReplicaGradientDomain | None = None
        self._pending_hybrid_gradients: list[GradientPayload] = []

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def initialize(self, max_seq_length: int = 2048):
        self._validate_parallel_config()

        if not torch.cuda.is_available():
            raise RuntimeError("Megatron3DTrainEngine requires a CUDA environment")

        torch.cuda.set_device(self.local_rank)
        device = torch.device(f"cuda:{self.local_rank}")

        self.runtime = TrainParallelRuntime(
            rank=self.rank,
            world_size=self.world_size,
            tensor_parallel_size=self.train_tp_size,
            pipeline_parallel_size=self.train_pp_size,
            data_parallel_size=self.train_dp_size,
        )
        if self.is_distributed:
            self.runtime.initialize_process_groups()

        self.model_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._validate_model_architecture(self.model_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
        stage_model = QwenPipelineStage(hf_model, self.runtime).to(device)
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = stage_model
        if self.is_distributed and self.train_dp_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                process_group=self.runtime.dp_group,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        self.scheduler = None

        if self.is_main_process:
            spec = self._stage_model.stage_spec
            print(
                "Megatron3DTrainEngine initialized: "
                f"tp={self.train_tp_size}, pp={self.train_pp_size}, "
                f"dp={self.train_dp_size}, "
                f"stage={spec.stage_id}, layers=[{spec.layer_start}, {spec.layer_end})"
            )

    def _validate_parallel_config(self):
        expected_world = self.train_tp_size * self.train_pp_size * self.train_dp_size
        if expected_world != self.world_size:
            raise ValueError(
                "Megatron3DTrainEngine requires world_size to equal tp*pp*dp: "
                f"world_size={self.world_size}, tp={self.train_tp_size}, "
                f"pp={self.train_pp_size}, dp={self.train_dp_size}"
            )
        if self.pipeline_schedule != "1f1b":
            raise ValueError("Megatron3DTrainEngine currently supports only 1F1B")
        if self.virtual_pipeline_model_parallel_size != 0:
            raise ValueError("Megatron3DTrainEngine does not support virtual pipelines")

    @staticmethod
    def _validate_model_architecture(cfg):
        architectures = [a.lower() for a in getattr(cfg, "architectures", [])]
        model_type = str(getattr(cfg, "model_type", "")).lower()
        if "qwen" not in model_type and not any("qwen" in a for a in architectures):
            raise NotImplementedError(
                "Megatron3DTrainEngine v1 supports only Qwen-family causal language models"
            )

    @property
    def _stage_model(self) -> QwenPipelineStage:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def get_version(self) -> int:
        return self.current_version

    def set_version(self, version: int):
        self.current_version = version

    def get_data_parallel_world_size(self) -> int:
        return self.train_dp_size

    def get_data_parallel_rank(self) -> int:
        return self.runtime.data_parallel_rank if self.runtime else 0

    def get_local_batch_size(self, global_batch_size: int) -> int:
        if global_batch_size % self.train_dp_size != 0:
            raise ValueError(
                "global_batch_size must be divisible by train_dp_size: "
                f"{global_batch_size} vs {self.train_dp_size}"
            )
        return global_batch_size // self.train_dp_size

    def is_batch_source(self) -> bool:
        if self.runtime is None:
            return True
        return self.runtime.is_data_parallel_coordinator

    def distribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Distribute trajectories."""
        if self.runtime is None or not self.is_distributed:
            return trajectories or []

        ranks = self.runtime.get_model_replica_ranks()
        if len(ranks) == 1:
            return trajectories or []

        payload = [trajectories if self.is_batch_source() else None]
        dist.broadcast_object_list(
            payload,
            src=ranks[0],
            group=self.runtime.model_replica_group,
        )
        return payload[0] or []

    def get_parallel_state(self) -> dict[str, Any]:
        if self.runtime is None:
            return {
                "backend": self.train_backend,
                "train_tp": self.train_tp_size,
                "train_pp": self.train_pp_size,
                "train_dp": self.train_dp_size,
                "is_batch_source": True,
                "elastic_gradient_domain": self.elastic_gradient_domain is not None,
            }
        state = self.runtime.to_dict()
        state["backend"] = self.train_backend
        state["adapter_mode"] = self.adapter_mode
        state["elastic_gradient_domain"] = self.elastic_gradient_domain is not None
        return state

    def get_elastic_core_replica_ids(self) -> list[str]:
        return [f"dp{i}" for i in range(max(1, self.train_dp_size))]

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
        return self.runtime.dp_group if self.runtime is not None else None

    def set_elastic_gradient_domain(
        self,
        domain: InterReplicaGradientDomain | None,
    ):
        """Attach an elastic inter-replica gradient domain.

        The fixed TP/PP process groups remain unchanged. This hook only affects
        the replica/DP plane and is intentionally opt-in.
        """
        self.elastic_gradient_domain = domain
        if domain is not None and not domain.decoupled_communication_domains:
            domain.process_group = self.get_elastic_core_process_group()

    def enqueue_hybrid_gradient_payload(self, payload: GradientPayload):
        """Queue a hybrid worker gradient for the next optimizer step."""
        self._pending_hybrid_gradients.append(payload)

    def capture_elastic_state_snapshot(
        self,
        worker_id: str,
        target_core_id: str,
    ) -> int:
        """Persist local 3D shard state for a joining hybrid worker."""
        del worker_id, target_core_id
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Megatron3DTrainEngine is not initialized")
        snapshot_path = self.get_elastic_state_snapshot_path(self.current_version)
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        payload = {
            "model": self._stage_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "version": self.current_version,
            "parallel_state": self.get_parallel_state(),
        }
        torch.save(payload, snapshot_path)
        return self.current_version

    def get_elastic_state_snapshot_path(self, version: int) -> str:
        return os.path.join(
            os.getenv("ELASTIC_TRAINING_STATE_DIR", "./logs/elastic_training_state"),
            f"v{version}",
            f"rank_{self.rank}.pt",
        )

    def load_elastic_state_snapshot(self, snapshot_path: str):
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        self._stage_model.load_state_dict(payload["model"])
        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.current_version = int(payload.get("version", self.current_version))

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _pad_micro_batch(
        self,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        device = torch.device(f"cuda:{self.local_rank}")
        max_len = max(int(traj["input_ids"].shape[1]) for traj in trajectories)
        batch_size = len(trajectories)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )
        old_logprobs = torch.zeros(
            (batch_size, max_len),
            dtype=torch.float32,
            device=device,
        )
        loss_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.float32,
            device=device,
        )
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
        advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)

        for idx, traj in enumerate(trajectories):
            seq_len = int(traj["input_ids"].shape[1])
            input_ids[idx, :seq_len] = traj["input_ids"][0].to(device)
            attention_mask[idx, :seq_len] = traj["attention_mask"][0].to(device)
            old_logprobs[idx, :seq_len] = traj["logprobs"][0].to(device)
            loss_mask[idx, :seq_len] = traj["loss_mask"][0].to(device).float()
            rewards[idx] = traj["rewards"][0].to(device)
            advantages[idx] = traj.get("advantages", traj["rewards"])[0].to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_logprobs": old_logprobs,
            "loss_mask": loss_mask,
            "rewards": rewards,
            "advantages": advantages,
        }

    def _iter_micro_batches(self, trajectories: list[dict[str, Any]]):
        for start in range(0, len(trajectories), self.micro_batch_size):
            yield self._pad_micro_batch(trajectories[start:start + self.micro_batch_size])

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _recv_forward(self, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prev_rank = self.runtime.get_prev_pipeline_rank()
        if prev_rank is None:
            raise RuntimeError("The first stage must not receive forward hidden states")
        shape = (
            micro_batch["input_ids"].shape[0],
            micro_batch["input_ids"].shape[1],
            int(self.model_config.hidden_size),
        )
        hidden = torch.empty(
            shape,
            dtype=torch.bfloat16,
            device=torch.device(f"cuda:{self.local_rank}"),
            requires_grad=True,
        )
        dist.recv(hidden, src=prev_rank)
        hidden.requires_grad_(True)
        return hidden

    def _send_forward(self, hidden: torch.Tensor):
        next_rank = self.runtime.get_next_pipeline_rank()
        if next_rank is None:
            return
        dist.send(hidden.detach().contiguous(), dst=next_rank)

    def _recv_backward(self, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        next_rank = self.runtime.get_next_pipeline_rank()
        if next_rank is None:
            raise RuntimeError("The last stage must not receive backward gradients")
        shape = (
            micro_batch["input_ids"].shape[0],
            micro_batch["input_ids"].shape[1],
            int(self.model_config.hidden_size),
        )
        grad = torch.empty(
            shape,
            dtype=torch.bfloat16,
            device=torch.device(f"cuda:{self.local_rank}"),
        )
        dist.recv(grad, src=next_rank)
        return grad

    def _send_backward(self, grad: torch.Tensor | None):
        prev_rank = self.runtime.get_prev_pipeline_rank()
        if prev_rank is None or grad is None:
            return
        dist.send(grad.detach().contiguous(), dst=prev_rank)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _forward_stage(
        self,
        micro_batch: dict[str, torch.Tensor],
        hidden_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.runtime.is_pipeline_first_stage:
            return self._stage_model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
            )
        hidden_input.requires_grad_(True)
        return self._stage_model(
            hidden_states=hidden_input,
            attention_mask=micro_batch["attention_mask"],
        )

    def _compute_action_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_ids = input_ids[:, 1:]
        trimmed_logits = logits[:, :-1, :]
        if self.train_tp_size > 1:
            return vocab_parallel_token_logprobs(
                trimmed_logits,
                target_ids,
                self.runtime,
            )
        log_probs = F.log_softmax(trimmed_logits, dim=-1)
        return torch.gather(
            log_probs,
            2,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)

    def _compute_last_stage_loss(
        self,
        micro_batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        action_log_probs = self._compute_action_log_probs(
            logits,
            micro_batch["input_ids"],
        )
        old_logprobs_trimmed = micro_batch["old_logprobs"][:, 1:]
        loss_mask_trimmed = micro_batch["loss_mask"][:, 1:]
        advantages = micro_batch["advantages"]
        rewards = micro_batch["rewards"]

        log_ratio = action_log_probs - old_logprobs_trimmed
        ratio = torch.where(
            loss_mask_trimmed.bool(),
            torch.exp(log_ratio),
            torch.zeros_like(log_ratio),
        )
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon,
        )
        pg_loss1 = -advantages.unsqueeze(1) * ratio
        pg_loss2 = -advantages.unsqueeze(1) * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)
        loss_mask_count = loss_mask_trimmed.sum().clamp(min=1.0)
        pg_loss = (pg_loss * loss_mask_trimmed).sum() / loss_mask_count
        kl_div = (log_ratio * loss_mask_trimmed).sum() / loss_mask_count
        loss = pg_loss + self.kl_coef * kl_div

        stats = {
            "loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "kl": float(kl_div.detach().item()),
            "reward_sum": float(rewards.sum().detach().item()),
            "n_samples": int(rewards.numel()),
            "n_updates": 1,
        }
        return loss, stats, action_log_probs.detach()

    def _merge_stats(self, all_stats: list[dict[str, float]]) -> dict[str, float]:
        if not all_stats:
            return {
                "loss": 0.0,
                "pg_loss": 0.0,
                "kl": 0.0,
                "reward_mean": 0.0,
                "reward_sum": 0.0,
                "n_samples": 0,
                "n_updates": 0,
                "version": self.current_version,
            }
        n_updates = sum(s["n_updates"] for s in all_stats)
        n_samples = sum(s["n_samples"] for s in all_stats)
        reward_sum = sum(s["reward_sum"] for s in all_stats)
        return {
            "loss": sum(s["loss"] for s in all_stats) / max(n_updates, 1),
            "pg_loss": sum(s["pg_loss"] for s in all_stats) / max(n_updates, 1),
            "kl": sum(s["kl"] for s in all_stats) / max(n_updates, 1),
            "reward_mean": reward_sum / max(n_samples, 1),
            "reward_sum": reward_sum,
            "n_samples": n_samples,
            "n_updates": n_updates,
            "version": self.current_version,
        }

    def _run_pipeline_train_step(
        self,
        micro_batches: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        num_micro = len(micro_batches)
        pp_rank = self.runtime.pipeline_parallel_rank
        pp_world = self.runtime.pipeline_parallel_size
        warmup = min(pp_world - pp_rank - 1, num_micro)
        remaining = num_micro - warmup

        inputs_queue: list[torch.Tensor | None] = []
        outputs_queue: list[torch.Tensor] = []
        batch_queue: list[dict[str, torch.Tensor]] = []
        last_stage_stats: list[dict[str, float]] = []

        self.optimizer.zero_grad()
        self.model.train()
        micro_idx = 0

        for _ in range(warmup):
            mb = micro_batches[micro_idx]
            micro_idx += 1
            recv_hidden = None
            if not self.runtime.is_pipeline_first_stage:
                recv_hidden = self._recv_forward(mb)
            out = self._forward_stage(mb, recv_hidden)
            if not self.runtime.is_pipeline_last_stage:
                self._send_forward(out)
            inputs_queue.append(recv_hidden)
            outputs_queue.append(out)
            batch_queue.append(mb)

        for _ in range(remaining):
            mb = micro_batches[micro_idx]
            micro_idx += 1
            recv_hidden = None
            if not self.runtime.is_pipeline_first_stage:
                recv_hidden = self._recv_forward(mb)
            out = self._forward_stage(mb, recv_hidden)
            if not self.runtime.is_pipeline_last_stage:
                self._send_forward(out)
            inputs_queue.append(recv_hidden)
            outputs_queue.append(out)
            batch_queue.append(mb)

            old_in = inputs_queue.pop(0)
            old_out = outputs_queue.pop(0)
            old_mb = batch_queue.pop(0)

            if self.runtime.is_pipeline_last_stage:
                loss, stats, _ = self._compute_last_stage_loss(old_mb, old_out)
                loss.backward()
                last_stage_stats.append(stats)
                self._send_backward(old_in.grad if old_in is not None else None)
            else:
                grad_out = self._recv_backward(old_mb)
                torch.autograd.backward(old_out, grad_out)
                self._send_backward(old_in.grad if old_in is not None else None)

        while batch_queue:
            old_in = inputs_queue.pop(0)
            old_out = outputs_queue.pop(0)
            old_mb = batch_queue.pop(0)

            if self.runtime.is_pipeline_last_stage:
                loss, stats, _ = self._compute_last_stage_loss(old_mb, old_out)
                loss.backward()
                last_stage_stats.append(stats)
                self._send_backward(old_in.grad if old_in is not None else None)
            else:
                grad_out = self._recv_backward(old_mb)
                torch.autograd.backward(old_out, grad_out)
                self._send_backward(old_in.grad if old_in is not None else None)

        self._apply_elastic_inter_replica_gradients()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        stats = self._merge_stats(last_stage_stats)
        return self._broadcast_stats(stats)

    def _apply_elastic_inter_replica_gradients(self):
        if not getattr(self, "_pending_hybrid_gradients", []):
            return
        if self.elastic_gradient_domain is None or self.runtime is None:
            self._pending_hybrid_gradients.clear()
            return

        params_with_grad = [
            p for p in self.model.parameters()
            if p.grad is not None
        ]
        if not params_with_grad:
            self._pending_hybrid_gradients.clear()
            return

        core_id = f"dp{self.runtime.data_parallel_rank}"
        reduced = self.elastic_gradient_domain.reduce_core_gradients(
            core_gradients={
                core_id: tuple(p.grad.detach() for p in params_with_grad),
            },
            hybrid_payloads=self._pending_hybrid_gradients,
        )
        for param, grad in zip(params_with_grad, reduced[core_id]):
            param.grad.copy_(grad.to(device=param.grad.device, dtype=param.grad.dtype))
        self._pending_hybrid_gradients.clear()

    def _broadcast_stats(self, stats: dict[str, float]) -> dict[str, float]:
        reduced = dict(stats)
        if (
            dist.is_initialized()
            and self.runtime.is_pipeline_last_stage
            and self.runtime.is_tensor_parallel_source
        ):
            tensor = torch.tensor(
                [
                    reduced.get("loss", 0.0),
                    reduced.get("pg_loss", 0.0),
                    reduced.get("kl", 0.0),
                    reduced.get("reward_sum", 0.0),
                    float(reduced.get("n_samples", 0)),
                    float(reduced.get("n_updates", 0)),
                ],
                device=torch.device(f"cuda:{self.local_rank}"),
                dtype=torch.float32,
            )
            dist.all_reduce(tensor, group=self.runtime.dp_group)
            tensor /= max(self.train_dp_size, 1)
            reduced = {
                "loss": float(tensor[0].item()),
                "pg_loss": float(tensor[1].item()),
                "kl": float(tensor[2].item()),
                "reward_sum": float(tensor[3].item()),
                "n_samples": int(tensor[4].item()),
                "n_updates": int(tensor[5].item()),
                "reward_mean": float(tensor[3].item()) / max(int(tensor[4].item()), 1),
                "version": self.current_version,
            }

        src_rank = self.runtime.get_pipeline_rank(
            dp_rank=0,
            pp_rank=self.runtime.pipeline_parallel_size - 1,
            tp_rank=0,
        )
        payload = [reduced if self.rank == src_rank else None]
        if dist.is_initialized():
            dist.broadcast_object_list(payload, src=src_rank)
        return payload[0]

    def _run_pipeline_recompute(
        self,
        trajectories: list[dict[str, Any]],
    ):
        micro_batches = list(self._iter_micro_batches(trajectories))
        new_logprobs: list[torch.Tensor] = []
        self.model.eval()

        for mb in micro_batches:
            recv_hidden = None
            if not self.runtime.is_pipeline_first_stage:
                recv_hidden = self._recv_forward(mb)
            with torch.no_grad():
                out = self._forward_stage(mb, recv_hidden)
            if not self.runtime.is_pipeline_last_stage:
                self._send_forward(out)
            else:
                action_log_probs = self._compute_action_log_probs(
                    out,
                    mb["input_ids"],
                ).cpu()
                new_logprobs.append(action_log_probs)

        src_rank = self.runtime.get_pipeline_rank(
            dp_rank=self.runtime.data_parallel_rank,
            pp_rank=self.runtime.pipeline_parallel_size - 1,
            tp_rank=0,
        )
        payload = [new_logprobs if self.rank == src_rank else None]
        if dist.is_initialized():
            dist.broadcast_object_list(
                payload,
                src=src_rank,
                group=self.runtime.model_replica_group,
            )
        gathered = payload[0] or []

        traj_idx = 0
        for micro_lp in gathered:
            for row in micro_lp:
                traj = trajectories[traj_idx]
                updated = torch.zeros_like(traj["logprobs"])
                seq_len = min(row.shape[0], updated.shape[1] - 1)
                updated[:, 1:1 + seq_len] = row[:seq_len].unsqueeze(0)
                traj["logprobs"] = updated
                traj_idx += 1

        self.model.train()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def grpo_update(
        self,
        trajectories: list[dict[str, Any]],
        ppo_epochs: int = 1,
    ) -> dict[str, float]:
        all_stats = []
        for _ in range(ppo_epochs):
            micro_batches = list(self._iter_micro_batches(trajectories))
            stats = self._run_pipeline_train_step(micro_batches)
            all_stats.append(stats)
        merged = self._merge_stats(all_stats)
        merged["version"] = self.current_version
        return merged

    def recompute_logprobs(self, trajectories: list[dict[str, Any]]):
        self._run_pipeline_recompute(trajectories)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def save_weights(self, path: str, version: int):
        if self.runtime is None:
            raise RuntimeError("Megatron3DTrainEngine is not initialized")

        save_path = os.path.join(path, f"v{version}")
        shard_path = os.path.join(save_path, "train_shards")
        os.makedirs(shard_path, exist_ok=True)

        local_shard = {
            "model": self._stage_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "version": version,
            "parallel_state": self.get_parallel_state(),
        }
        torch.save(local_shard, os.path.join(shard_path, f"rank_{self.rank}.pt"))
        _maybe_barrier()

        fragment = None
        if self.runtime.data_parallel_rank == 0:
            fragment = export_stage_state_dict_for_hf(self._stage_model)
            if self.runtime.tensor_parallel_rank != 0:
                fragment = None

        gathered = [None for _ in range(self.world_size)] if self.is_main_process else None
        if dist.is_initialized():
            dist.gather_object(
                obj=fragment,
                object_gather_list=gathered,
                dst=0,
            )
        else:
            gathered = [fragment]

        if self.is_main_process:
            merged_state = {}
            for part in gathered:
                if not part:
                    continue
                merged_state.update(part)
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
            )
            missing, unexpected = hf_model.load_state_dict(merged_state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys while exporting Hugging Face weights: {unexpected}")
            hf_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            metadata = {
                "backend": self.train_backend,
                "version": version,
                "train_tp": self.train_tp_size,
                "train_pp": self.train_pp_size,
                "train_dp": self.train_dp_size,
                "pipeline_schedule": self.pipeline_schedule,
                "adapter_mode": self.adapter_mode,
                "missing_keys": missing,
            }
            with open(
                os.path.join(save_path, "training_parallel_metadata.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        _maybe_barrier()

    def load_weights(self, path: str, version: int):
        load_path = os.path.join(path, f"v{version}", "train_shards", f"rank_{self.rank}.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Training shard does not exist: {load_path}")
        payload = torch.load(load_path, map_location="cpu", weights_only=False)
        self._stage_model.load_state_dict(payload["model"])
        if self.optimizer is not None and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.current_version = payload.get("version", version)
        _maybe_barrier()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

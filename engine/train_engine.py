"""Support code for Train engine."""

import os
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from RL_Framework.infra.elastic import (
    GradientPayload,
    InterReplicaGradientDomain,
)


def _selected_token_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    selected_logits = torch.gather(
        logits,
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    return selected_logits - torch.logsumexp(logits, dim=-1)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    return value.lower() not in {"0", "false", "no", "off"}


def _checkpoint_only_with_grad(
    function: Any,
    *args: Any,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return torch_checkpoint(
        function,
        *args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
        **kwargs,
    )


def _apply_fsdp_activation_checkpointing(
    model: torch.nn.Module,
    transformer_layer_cls: type[torch.nn.Module],
) -> int:
    checkpointed_layers = sum(
        isinstance(module, transformer_layer_cls) for module in model.modules()
    )
    wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        checkpoint_fn=_checkpoint_only_with_grad,
        preserve_rng_state=True,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, transformer_layer_cls),
    )
    return checkpointed_layers


@runtime_checkable
class TrainEngine(Protocol):
    """Train engine implementation."""

    train_backend: str
    rank: int
    world_size: int
    is_main_process: bool

    def initialize(self, max_seq_length: int = 2048):
        """Initialize."""

    def get_version(self) -> int:
        """Get version."""

    def set_version(self, version: int):
        """Set version."""

    def grpo_update(
        self,
        trajectories: list[dict[str, Any]],
        ppo_epochs: int = 1,
    ) -> dict[str, float]:
        """Grpo update."""

    def recompute_logprobs(self, trajectories: list[dict[str, Any]]):
        """Recompute logprobs."""

    def save_weights(self, path: str, version: int):
        """Save weights."""

    def load_weights(self, path: str, version: int):
        """Load weights."""

    def get_data_parallel_world_size(self) -> int:
        """Get data parallel world size."""

    def get_data_parallel_rank(self) -> int:
        """Get data parallel rank."""

    def get_local_batch_size(self, global_batch_size: int) -> int:
        """Get local batch size."""

    def is_batch_source(self) -> bool:
        """Is batch source."""

    def distribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Distribute trajectories."""

    def align_distributed_trajectories(
        self,
        trajectories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Align micro-step sequence shapes across data-parallel ranks."""

    def get_parallel_state(self) -> dict[str, Any]:
        """Get parallel state."""

    def configure_elastic_training(
        self,
        core_replica_ids: list[str] | None = None,
    ) -> InterReplicaGradientDomain:
        """Attach an elastic gradient domain for hybrid workers"""

    def capture_elastic_state_snapshot(
        self,
        worker_id: str,
        target_core_id: str,
    ) -> int:
        """Export state used by a joining hybrid worker"""

    def load_elastic_state_snapshot(self, snapshot_path: str):
        """Load a previously exported elastic state snapshot"""

    def enqueue_hybrid_gradient_payload(self, payload: GradientPayload):
        """Queue a hybrid gradient payload for the next optimizer step"""

    def get_elastic_state_snapshot_path(self, version: int) -> str:
        """Return the local snapshot path for a model version"""


class FSDPTrainEngine:
    """F s d p train engine implementation."""

    def __init__(
        self,
        model_path: str,
        learning_rate: float = 1e-6,
        kl_coef: float = 0.001,
        clip_epsilon: float = 0.2,
        micro_batch_size: int = 4,
    ):
        self.train_backend = "fsdp"
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.micro_batch_size = micro_batch_size


        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_main_process = self.rank == 0


        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None


        self.current_version = 0


        self.fsdp_strategy = None
        self.elastic_gradient_domain: InterReplicaGradientDomain | None = None
        self._pending_hybrid_gradients: list[GradientPayload] = []
        self._fsdp_update_index = 0

    def get_version(self) -> int:
        """Get version."""
        return self.current_version

    def set_version(self, version: int):
        """Set version."""
        self.current_version = version

    def initialize(self, max_seq_length: int = 2048):
        """Initialize."""

        self.is_distributed = dist.is_initialized()


        torch.cuda.set_device(self.local_rank)
        device = torch.device(f"cuda:{self.local_rank}")


        if self.is_main_process:
            print(f"Loading tokenizer: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token


        if self.is_distributed:
            dist.barrier()
            if not self.is_main_process:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token


        if self.is_main_process:
            print(f"Loading model: {self.model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        gradient_checkpointing = _env_flag(
            "FSDP_GRADIENT_CHECKPOINTING", False
        )
        if gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        if self.is_main_process:
            print(
                "FSDP gradient checkpointing: "
                f"{'enabled' if gradient_checkpointing else 'disabled'}"
            )


        if self.is_distributed:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
            decoder_layers = getattr(
                getattr(self.model, "model", None),
                "layers",
                None,
            )
            if not decoder_layers:
                raise RuntimeError("Unable to locate Transformer decoder layers for FSDP wrapping")
            decoder_layer_cls = type(decoder_layers[0])
            activation_checkpointing = _env_flag(
                "FSDP_ACTIVATION_CHECKPOINTING", True
            )
            checkpointed_layers = 0
            fsdp_layer_cls = decoder_layer_cls
            if activation_checkpointing:
                checkpointed_layers = _apply_fsdp_activation_checkpointing(
                    self.model,
                    decoder_layer_cls,
                )
                fsdp_layer_cls = CheckpointWrapper
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={fsdp_layer_cls},
            )

            self.model = FSDP(
                self.model,
                mixed_precision=mp_policy,
                device_id=device,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                limit_all_gathers=True,
                backward_prefetch=None,
                forward_prefetch=False,
            )

            if self.is_main_process:
                print(f"FSDP model initialized (world_size={self.world_size})")
                print(
                    "FSDP activation checkpointing: "
                    f"{'enabled' if activation_checkpointing else 'disabled'} "
                    f"(layers={checkpointed_layers})"
                )
        else:

            self.model = self.model.to(device)
            print(f"Model loaded on GPU (single-GPU mode)")


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )



        self.scheduler = None

        if self.is_main_process:
            print(f"FSDP training engine initialized (rank={self.rank}, world_size={self.world_size})")

    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logprobs."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits


        log_probs = F.log_softmax(logits, dim=-1)


        log_probs = log_probs[:, :-1, :].contiguous()

        return log_probs

    def grpo_update(
        self,
        trajectories: list[dict[str, Any]],
        ppo_epochs: int = 1,
    ) -> dict[str, float]:
        """Grpo update."""
        self.model.train()

        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        n_updates = 0
        n_samples = 0
        update_index = self._fsdp_update_index

        for epoch in range(ppo_epochs):
            for micro_step, traj in enumerate(trajectories):

                input_ids = traj["input_ids"]  # [batch, seq_len]
                attention_mask = traj["attention_mask"]
                old_logprobs = traj["logprobs"]  # [batch, seq_len]
                loss_mask = traj.get("loss_mask", torch.ones_like(input_ids))  # [batch, seq_len]
                rewards = traj["rewards"]  # [batch]
                advantages = traj.get("advantages", rewards)  # [batch]


                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                old_logprobs = old_logprobs.to(device)
                loss_mask = loss_mask.to(device).float()
                advantages = advantages.to(device)

                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "before_forward",
                    int(input_ids.shape[-1]),
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits[:, :-1, :]
                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "after_forward",
                    int(input_ids.shape[-1]),
                )

                action_log_probs = _selected_token_logprobs(
                    logits,
                    input_ids[:, 1:],
                )


                old_logprobs_trimmed = old_logprobs[:, 1:]  # [batch, seq_len-1]
                loss_mask_trimmed = loss_mask[:, 1:]  # [batch, seq_len-1]




                log_ratio = action_log_probs - old_logprobs_trimmed
                ratio = torch.where(loss_mask_trimmed.bool(), torch.exp(log_ratio), torch.zeros_like(log_ratio))

                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )


                # AReaL: pg_loss = max(-adv * ratio, -adv * clipped_ratio)

                pg_loss1 = -advantages.unsqueeze(1) * ratio
                pg_loss2 = -advantages.unsqueeze(1) * clipped_ratio
                pg_loss = torch.max(pg_loss1, pg_loss2)


                loss_mask_count = loss_mask_trimmed.sum().clamp(min=1.0)
                pg_loss = (pg_loss * loss_mask_trimmed).sum() / loss_mask_count


                # approx_kl = E[log_ratio] = E[new_logp - old_logp]
                kl_div = (log_ratio * loss_mask_trimmed).sum() / loss_mask_count


                loss = pg_loss + self.kl_coef * kl_div


                loss.backward()
                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "after_backward",
                    int(input_ids.shape[-1]),
                )
                self._apply_elastic_inter_replica_gradients()
                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "after_optimizer",
                    int(input_ids.shape[-1]),
                )


                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_kl += kl_div.item()
                total_reward += rewards.sum().item()
                n_samples += rewards.numel()
                n_updates += 1

        self._fsdp_update_index += 1


        stats = {
            "loss": total_loss / max(n_updates, 1),
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "kl": total_kl / max(n_updates, 1),
            "reward_mean": total_reward / max(n_samples, 1),
            "n_updates": n_updates,
            "version": self.current_version,
        }

        return stats

    def align_distributed_trajectories(
        self,
        trajectories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pad matching micro-steps to a global length on every FSDP rank.

        FSDP all-gathers parameters between decoder layers. Large sequence
        length skew can leave one rank computing while peers enqueue later
        collectives, eventually tripping the NCCL watchdog. Sorting and global
        padding keeps each micro-step's kernel workload and collective order
        aligned across nodes.
        """
        if not self.is_distributed or self.world_size <= 1 or not trajectories:
            return trajectories
        if os.environ.get("FSDP_ALIGN_SEQUENCE_LENGTHS", "1").lower() in {
            "0", "false", "no"
        }:
            return trajectories

        device = next(self.model.parameters()).device
        local_count = torch.tensor(
            [len(trajectories)], dtype=torch.int64, device=device
        )
        min_count = local_count.clone()
        max_count = local_count.clone()
        dist.all_reduce(min_count, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
        if int(min_count.item()) != int(max_count.item()):
            raise RuntimeError(
                "FSDP ranks received different trajectory counts: "
                f"min={int(min_count.item())}, max={int(max_count.item())}"
            )

        aligned = sorted(
            trajectories,
            key=lambda traj: int(traj["input_ids"].shape[-1]),
            reverse=True,
        )
        original_lengths = [int(traj["input_ids"].shape[-1]) for traj in aligned]
        global_lengths = torch.tensor(
            original_lengths, dtype=torch.int64, device=device
        )
        dist.all_reduce(global_lengths, op=dist.ReduceOp.MAX)
        padded_lengths = [int(value) for value in global_lengths.cpu().tolist()]

        pad_token_id = int(self.tokenizer.pad_token_id or 0)
        for traj, original_length, target_length in zip(
            aligned, original_lengths, padded_lengths, strict=True
        ):
            pad_width = target_length - original_length
            traj["_original_seq_len"] = original_length
            traj["_padded_seq_len"] = target_length
            if pad_width <= 0:
                continue
            traj["input_ids"] = F.pad(
                traj["input_ids"], (0, pad_width), value=pad_token_id
            )
            # Mark right-padding as valid context so every rank executes the
            # same attention and MoE token workload. Causal masking prevents
            # original tokens from attending to these future pad positions;
            # loss_mask below keeps them out of the objective.
            traj["attention_mask"] = F.pad(
                traj["attention_mask"], (0, pad_width), value=1
            )
            traj["logprobs"] = F.pad(
                traj["logprobs"], (0, pad_width), value=0.0
            )
            traj["loss_mask"] = F.pad(
                traj.get("loss_mask", torch.ones_like(traj["input_ids"][..., :original_length])),
                (0, pad_width),
                value=0,
            )

        print(
            f"[FSDP shape sync] rank={self.rank} "
            f"original={original_lengths} global={padded_lengths}",
            flush=True,
        )
        return aligned

    def _fsdp_lockstep(
        self,
        update_index: int,
        micro_step: int,
        phase: str,
        sequence_length: int,
    ) -> None:
        if not self.is_distributed or self.world_size <= 1:
            return
        if os.environ.get("FSDP_LOCKSTEP_BARRIERS", "1").lower() in {
            "0", "false", "no"
        }:
            return

        # A process-group barrier alone can be enqueued while kernels from the
        # preceding phase are still running.  Synchronize the local stream so
        # every rank reaches the next FSDP all-gather from the same phase.
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        progress_root = Path(
            os.environ.get("FSDP_PROGRESS_DIR", "./logs/fsdp_progress")
        )
        progress_dir = (
            progress_root
            / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}"
            / f"update_{update_index}"
            / f"micro_{micro_step}"
        )
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path = progress_dir / f"rank_{self.rank}.json"
        tmp_path = progress_path.with_name(
            f".{progress_path.name}.{os.getpid()}.tmp"
        )
        tmp_path.write_text(
            json.dumps(
                {
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "update": update_index,
                    "micro_step": micro_step,
                    "phase": phase,
                    "sequence_length": sequence_length,
                    "timestamp": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_path, progress_path)
        print(
            f"[FSDP lockstep] rank={self.rank} update={update_index} "
            f"micro={micro_step} phase={phase} seq={sequence_length}",
            flush=True,
        )
        dist.barrier()

    def recompute_logprobs(self, trajectories: list[dict[str, Any]]):
        """Recompute logprobs."""
        self.model.eval()
        device = next(self.model.parameters()).device
        update_index = self._fsdp_update_index

        with torch.no_grad():
            for micro_step, traj in enumerate(trajectories):
                input_ids = traj["input_ids"].to(device)
                attention_mask = traj["attention_mask"].to(device)

                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "recompute_before_forward",
                    int(input_ids.shape[-1]),
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits[:, :-1, :]
                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "recompute_after_forward",
                    int(input_ids.shape[-1]),
                )

                action_log_probs = _selected_token_logprobs(
                    logits,
                    input_ids[:, 1:],
                )


                new_logprobs = torch.zeros_like(traj["logprobs"])
                seq_len = min(action_log_probs.shape[1], new_logprobs.shape[1] - 1)
                new_logprobs[:, 1:1+seq_len] = action_log_probs[:, :seq_len].cpu()


                traj["logprobs"] = new_logprobs
                self._fsdp_lockstep(
                    update_index,
                    micro_step,
                    "recompute_after_logprobs",
                    int(input_ids.shape[-1]),
                )

        self.model.train()

    def save_weights(self, path: str, version: int):
        """Save weights."""
        save_path = os.path.join(path, f"v{version}")
        os.makedirs(save_path, exist_ok=True)
        weight_file = os.path.join(save_path, "pytorch_model.bin")

        if self.is_distributed:
            self._record_weight_sync_phase(version, "before_state_dict", weight_file)
            dist.barrier()
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_cfg):
                state_dict = self.model.state_dict()
            self._record_weight_sync_phase(version, "after_state_dict", weight_file)

            if self.is_main_process:
                self._record_weight_sync_phase(version, "before_disk_save", weight_file)
                torch.save(state_dict, weight_file)
                self._record_weight_sync_phase(version, "after_disk_save", weight_file)
            else:
                self._record_weight_sync_phase(version, "waiting_for_disk_save", weight_file)
            dist.barrier()
            self._record_weight_sync_phase(version, "after_disk_save_barrier", weight_file)
        else:
            torch.save(self.model.state_dict(), weight_file)

        if self.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            model_config = (
                self.model.module.config
                if isinstance(self.model, FSDP)
                else self.model.config
            )
            model_config.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Weights saved to: {save_path}")

    def _record_weight_sync_phase(
        self,
        version: int,
        phase: str,
        weight_file: str,
    ) -> None:
        progress_root = Path(
            os.environ.get("FSDP_PROGRESS_DIR", "./logs/fsdp_progress")
        )
        progress_dir = (
            progress_root
            / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}"
            / f"update_{version}"
            / "weight_sync"
        )
        progress_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "update": version,
            "phase": phase,
            "weight_file": weight_file,
            "timestamp": time.time(),
        }
        progress_path = progress_dir / f"rank_{self.rank}.json"
        tmp_path = progress_path.with_name(
            f".{progress_path.name}.{os.getpid()}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, progress_path)
        print(
            f"[FSDP weight sync] rank={self.rank} update={version} phase={phase}",
            flush=True,
        )

    def load_weights(self, path: str, version: int):
        """Load weights."""
        load_path = os.path.join(path, f"v{version}")

        if not os.path.exists(load_path):
            print(f"WARNING: Weight path does not exist: {load_path}")
            return

        weight_file = os.path.join(load_path, "pytorch_model.bin")
        if not os.path.exists(weight_file):
            weight_file = os.path.join(load_path, "model.pt")
        state_dict = torch.load(
            weight_file,
            map_location="cpu",
            weights_only=True,
        )

        if self.is_distributed:
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_cfg):
                self.model.load_state_dict(state_dict)
            dist.barrier()
        else:
            self.model.load_state_dict(state_dict)

        print(f"Weights loaded from {load_path} loaded (rank={self.rank})")

    def get_device_stats(self) -> dict[str, float]:
        """Get device stats."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
            reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
            return {
                "gpu_memory_allocated_gb": allocated,
                "gpu_memory_reserved_gb": reserved,
            }
        return {}

    def step_scheduler(self):
        """Step scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def get_data_parallel_world_size(self) -> int:
        """Get data parallel world size."""
        return self.world_size

    def get_data_parallel_rank(self) -> int:
        """Get data parallel rank."""
        return self.rank

    def get_local_batch_size(self, global_batch_size: int) -> int:
        """Get local batch size."""
        if global_batch_size % self.world_size != 0:
            raise ValueError(
                "global_batch_size must be divisible by the FSDP world size: "
                f"{global_batch_size} vs {self.world_size}"
            )
        return global_batch_size // self.world_size

    def is_batch_source(self) -> bool:
        """Is batch source."""
        return True

    def distribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Distribute trajectories."""
        return trajectories or []

    def get_parallel_state(self) -> dict[str, Any]:
        """Get parallel state."""
        return {
            "backend": self.train_backend,
            "train_tp": 1,
            "train_pp": 1,
            "train_dp": self.world_size,
            "is_batch_source": True,
            "elastic_gradient_domain": self.elastic_gradient_domain is not None,
        }

    # ----------------------------------------------------------------
    # Elastic hybrid worker hooks
    # ----------------------------------------------------------------

    def get_elastic_core_replica_ids(self) -> list[str]:
        return [f"dp{i}" for i in range(max(1, self.world_size))]

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
        return dist.group.WORLD if dist.is_initialized() else None

    def set_elastic_gradient_domain(
        self,
        domain: InterReplicaGradientDomain | None,
    ):
        self.elastic_gradient_domain = domain
        if domain is not None and not domain.decoupled_communication_domains:
            domain.process_group = self.get_elastic_core_process_group()

    def enqueue_hybrid_gradient_payload(self, payload: GradientPayload):
        self._pending_hybrid_gradients.append(payload)

    def capture_elastic_state_snapshot(
        self,
        worker_id: str,
        target_core_id: str,
    ) -> int:
        """Persist a model/optimizer snapshot for a joining hybrid worker.

        The current control plane calls this on rank0. In a multi-rank launch,
        full-rank snapshot fan-out should be coordinated by the launcher; this
        method still provides the engine-side state materialization contract.
        """
        del worker_id, target_core_id
        if self.model is None:
            raise RuntimeError("FSDPTrainEngine is not initialized")
        snapshot_path = self.get_elastic_state_snapshot_path(self.current_version)
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        payload = {
            "model": self._elastic_model_state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
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
        self.model.load_state_dict(payload["model"])
        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.current_version = int(payload.get("version", self.current_version))

    def _elastic_model_state_dict(self) -> dict[str, Any]:
        if getattr(self, "is_distributed", False) and isinstance(self.model, FSDP):
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_cfg):
                return self.model.state_dict()
        return self.model.state_dict()

    def _apply_elastic_inter_replica_gradients(self):
        if self.elastic_gradient_domain is None:
            self._pending_hybrid_gradients.clear()
            return
        params_with_grad = [
            p for p in self.model.parameters()
            if p.grad is not None
        ]
        if not params_with_grad:
            self._pending_hybrid_gradients.clear()
            return

        core_id = f"dp{self.get_data_parallel_rank()}"
        reduced = self.elastic_gradient_domain.reduce_core_gradients(
            core_gradients={
                core_id: tuple(p.grad.detach() for p in params_with_grad),
            },
            hybrid_payloads=self._pending_hybrid_gradients,
        )
        for param, grad in zip(params_with_grad, reduced[core_id]):
            param.grad.copy_(grad.to(device=param.grad.device, dtype=param.grad.dtype))
        self._pending_hybrid_gradients.clear()

"""Megatron Bridge mappings for MCore's legacy GroupedMLP tensors."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def pack_grouped_weight1(
    expert_weights: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    etp_rank: int,
    etp_size: int,
) -> torch.Tensor:
    """Pack HF gate/up matrices into GroupedMLP's expert-major weight1."""
    blocks = []
    for gate, up in expert_weights:
        if gate.shape != up.shape or gate.ndim != 2:
            raise ValueError(
                "gate and up weights must be same-shaped rank-2 tensors"
            )
        if gate.shape[0] % etp_size:
            raise ValueError(
                f"expert FFN size {gate.shape[0]} is not divisible by "
                f"ETP size {etp_size}"
            )
        gate_shard = torch.chunk(gate, etp_size, dim=0)[etp_rank]
        up_shard = torch.chunk(up, etp_size, dim=0)[etp_rank]
        blocks.append(
            torch.cat((gate_shard, up_shard), dim=0).t().contiguous()
        )
    if not blocks:
        raise ValueError("at least one expert is required")

    # GroupedMLP reshapes the flat parameter to [experts, hidden, 2 * ffn].
    expert_major = torch.stack(blocks, dim=0)
    return expert_major.reshape(
        expert_major.shape[1],
        expert_major.shape[0] * expert_major.shape[2],
    )


def pack_grouped_weight2(
    expert_weights: list[torch.Tensor],
    *,
    etp_rank: int,
    etp_size: int,
) -> torch.Tensor:
    """Pack HF down matrices into GroupedMLP's expert-major weight2."""
    blocks = []
    for down in expert_weights:
        if down.ndim != 2:
            raise ValueError("down weights must be rank-2 tensors")
        if down.shape[1] % etp_size:
            raise ValueError(
                f"expert FFN size {down.shape[1]} is not divisible by "
                f"ETP size {etp_size}"
            )
        down_shard = torch.chunk(down, etp_size, dim=1)[etp_rank]
        blocks.append(down_shard.t().contiguous())
    if not blocks:
        raise ValueError("at least one expert is required")

    expert_major = torch.stack(blocks, dim=0)
    return expert_major.reshape(
        expert_major.shape[0] * expert_major.shape[1],
        expert_major.shape[2],
    )


_STACKED_MAPPING_CLASS = None


def stacked_expert_mapping_class():
    """Create the mapping class lazily so CPU-only installs can import tests."""
    global _STACKED_MAPPING_CLASS
    if _STACKED_MAPPING_CLASS is not None:
        return _STACKED_MAPPING_CLASS

    from megatron.bridge.models.conversion.param_mapping import (
        MegatronParamMapping,
    )

    class StackedExpertMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
        """Map all local HF experts to one MCore GroupedMLP parameter."""

        def __init__(
            self,
            megatron_param: str,
            *,
            kind: str,
            num_experts: int,
            hf_param: dict[str, str] | None = None,
            layer: str | None = None,
        ):
            if kind not in {"weight1", "weight2"}:
                raise ValueError(f"unsupported grouped expert kind: {kind}")
            self.kind = kind
            self.num_experts = num_experts
            self.layer = layer

            ep_size = self._group_size_before_base("ep")
            ep_rank = self._group_rank_before_base("ep")
            if num_experts % ep_size:
                raise ValueError(
                    f"{num_experts} experts are not divisible by EP={ep_size}"
                )
            self.num_local_experts = num_experts // ep_size
            self.local_expert_ids = list(
                range(
                    ep_rank * self.num_local_experts,
                    (ep_rank + 1) * self.num_local_experts,
                )
            )
            if hf_param is None:
                hf_param = self._hf_patterns()
            super().__init__(megatron_param, hf_param)

        @staticmethod
        def _group_size_before_base(group_type: str) -> int:
            from megatron.core import parallel_state

            if not parallel_state.model_parallel_is_initialized():
                return 1
            if group_type == "ep":
                group = parallel_state.get_expert_model_parallel_group()
            else:
                raise ValueError(f"unsupported group type: {group_type}")
            return dist.get_world_size(group=group)

        @staticmethod
        def _group_rank_before_base(group_type: str) -> int:
            from megatron.core import parallel_state

            if not parallel_state.model_parallel_is_initialized():
                return 0
            if group_type == "ep":
                group = parallel_state.get_expert_model_parallel_group()
            else:
                raise ValueError(f"unsupported group type: {group_type}")
            return dist.get_rank(group=group)

        @property
        def is_expert(self) -> bool:
            return True

        def _hf_patterns(self) -> dict[str, str]:
            patterns = {}
            for expert_id in self.local_expert_ids:
                prefix = f"model.layers.*.mlp.experts.{expert_id}"
                if self.kind == "weight1":
                    patterns[f"gate_{expert_id}"] = (
                        f"{prefix}.gate_proj.weight"
                    )
                    patterns[f"up_{expert_id}"] = f"{prefix}.up_proj.weight"
                else:
                    patterns[f"down_{expert_id}"] = (
                        f"{prefix}.down_proj.weight"
                    )
            return patterns

        def resolve(self, captures: tuple[str, ...]):
            resolved_megatron, resolved_hf = self._resolve_names(captures)
            if len(captures) != 1:
                raise ValueError(
                    "Grouped expert mapping expects exactly one layer wildcard"
                )
            return type(self)(
                resolved_megatron,
                kind=self.kind,
                num_experts=self.num_experts,
                hf_param=resolved_hf,
                layer=captures[0],
            )

        def hf_to_megatron(
            self,
            hf_weights: dict[str, torch.Tensor],
            megatron_module: Any,
        ) -> torch.Tensor:
            target = getattr(megatron_module, self.kind)
            if self.kind == "weight1":
                source = [
                    (
                        hf_weights[f"gate_{expert_id}"],
                        hf_weights[f"up_{expert_id}"],
                    )
                    for expert_id in self.local_expert_ids
                ]
                packed = pack_grouped_weight1(
                    source,
                    etp_rank=self.tp_rank,
                    etp_size=self.tp_size,
                )
            else:
                source = [
                    hf_weights[f"down_{expert_id}"]
                    for expert_id in self.local_expert_ids
                ]
                packed = pack_grouped_weight2(
                    source,
                    etp_rank=self.tp_rank,
                    etp_size=self.tp_size,
                )
            if packed.shape != target.shape:
                raise ValueError(
                    f"{self.megatron_param} shape mismatch: "
                    f"packed={tuple(packed.shape)} "
                    f"target={tuple(target.shape)}"
                )
            return packed.to(device=target.device, dtype=target.dtype)

        def megatron_to_hf(
            self,
            megatron_weights: torch.Tensor | None,
            megatron_module: Any,
        ) -> dict[str, torch.Tensor]:
            if megatron_weights is None:
                raise ValueError(
                    "PP=1 grouped expert export requires a local parameter"
                )
            if self.layer is None:
                raise ValueError("grouped expert mapping was not resolved")

            local_blocks = self._local_blocks(megatron_weights)
            output: dict[str, torch.Tensor] = {}
            for local_index, block in enumerate(local_blocks):
                projections = self._gather_etp_projections(block)
                for projection, local_weight in projections.items():
                    gathered_ep = self._all_gather(
                        local_weight,
                        self.ep_group,
                    )
                    for ep_rank, full_weight in enumerate(gathered_ep):
                        expert_id = (
                            ep_rank * self.num_local_experts + local_index
                        )
                        name = (
                            f"model.layers.{self.layer}.mlp.experts."
                            f"{expert_id}.{projection}_proj.weight"
                        )
                        output[name] = full_weight.cpu()
            return output

        def _local_blocks(self, weight: torch.Tensor) -> torch.Tensor:
            if self.kind == "weight1":
                hidden = weight.shape[0]
                return weight.view(self.num_local_experts, hidden, -1)
            hidden = weight.shape[1]
            return weight.view(self.num_local_experts, -1, hidden)

        def _gather_etp_projections(
            self,
            block: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            if self.kind == "weight1":
                gate_shard, up_shard = torch.chunk(block, 2, dim=1)
                gate = torch.cat(
                    self._all_gather(gate_shard.t().contiguous(), self.tp_group),
                    dim=0,
                )
                up = torch.cat(
                    self._all_gather(up_shard.t().contiguous(), self.tp_group),
                    dim=0,
                )
                return {"gate": gate, "up": up}
            down = torch.cat(
                self._all_gather(block, self.tp_group),
                dim=0,
            )
            return {"down": down.t().contiguous()}

        @staticmethod
        def _all_gather(
            tensor: torch.Tensor,
            group,
        ) -> list[torch.Tensor]:
            if dist.get_world_size(group=group) == 1:
                return [tensor]
            gathered = [
                torch.empty_like(tensor)
                for _ in range(dist.get_world_size(group=group))
            ]
            dist.all_gather(gathered, tensor, group=group)
            return gathered

    _STACKED_MAPPING_CLASS = StackedExpertMapping
    return StackedExpertMapping

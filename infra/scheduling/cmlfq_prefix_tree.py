"""Support code for Cmlfq prefix tree."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from RL_Framework.infra.scheduling.cmlfq_tool_state import ToolReturnState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class PrefixTreeNode:
    """Prefix tree node implementation."""
    key: str
    depth: int = 0
    mean_remaining_length: float = 0.0
    p90_remaining_length: float = 0.0
    visit_count: int = 0
    remaining_lengths: list[float] = field(default_factory=list)
    children: dict[str, "PrefixTreeNode"] = field(default_factory=dict)

    def update_statistics(self, remaining_length: float):
        """Update statistics."""
        self.visit_count += 1
        self.remaining_lengths.append(remaining_length)


        delta = remaining_length - self.mean_remaining_length
        self.mean_remaining_length += delta / self.visit_count


        if len(self.remaining_lengths) > 200:

            recent = self.remaining_lengths[-200:]
            self.remaining_lengths = recent

        self.p90_remaining_length = float(np.percentile(self.remaining_lengths, 90))

    def to_dict(self) -> dict:
        """Serialize the object to a dictionary."""
        return {
            "key": self.key,
            "depth": self.depth,
            "mean_remaining_length": round(self.mean_remaining_length, 2),
            "p90_remaining_length": round(self.p90_remaining_length, 2),
            "visit_count": self.visit_count,
            "remaining_lengths": self.remaining_lengths,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrefixTreeNode":
        """Build an instance from a dictionary."""
        node = cls(
            key=d["key"],
            depth=d.get("depth", 0),
            mean_remaining_length=d.get("mean_remaining_length", 0.0),
            p90_remaining_length=d.get("p90_remaining_length", 0.0),
            visit_count=d.get("visit_count", 0),
        )
        node.remaining_lengths = [
            float(value) for value in d.get("remaining_lengths", [])
        ]
        for k, child_d in d.get("children", {}).items():
            node.children[k] = cls.from_dict(child_d)
        return node

    def get_tree_stats(self) -> dict:
        """Get tree stats."""
        total_nodes = 1
        max_depth = self.depth
        leaf_count = 0 if self.children else 1
        for child in self.children.values():
            child_stats = child.get_tree_stats()
            total_nodes += child_stats["total_nodes"]
            max_depth = max(max_depth, child_stats["max_depth"])
            leaf_count += child_stats["leaf_count"]
        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "leaf_count": leaf_count,
        }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Trajectory implementation."""
    prompt_id: str
    return_states: list[ToolReturnState]
    total_remaining_lengths: list[float]
    total_length: float | None = None

    def __post_init__(self):
        assert len(self.return_states) == len(self.total_remaining_lengths), (
            f"return_states ({len(self.return_states)}) and "
            f"total_remaining_lengths ({len(self.total_remaining_lengths)}) have different lengths"
        )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class CausalPrefixTree:
    """Causal prefix tree implementation."""

    def __init__(self):
        # prompt_id -> root_node
        self._roots: dict[str, PrefixTreeNode] = {}
        # Aggregate causal evidence across prompts. Prompt ids are usually
        # unique online, while tool-return states recur across tasks.
        self._global_root: PrefixTreeNode | None = None
        self._lock = threading.RLock()
        self._total_insertions = 0

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def insert(self, trajectory: Trajectory):
        """Insert."""
        prompt_id = trajectory.prompt_id

        with self._lock:

            root = self._roots.get(prompt_id)
            if root is None:
                root = PrefixTreeNode(key=self._make_key(prompt_id, []), depth=0)
                self._roots[prompt_id] = root


            self._insert_into_root(root, trajectory, prompt_id)
            if self._global_root is None:
                self._global_root = PrefixTreeNode(key="__global__:root", depth=0)
            self._insert_into_root(self._global_root, trajectory, "__global__")

            self._total_insertions += 1

    def lookup(self, prompt_id: str, return_states: list[ToolReturnState]) -> Optional[PrefixTreeNode]:
        """Lookup."""
        with self._lock:
            root = self._roots.get(prompt_id)
            if root is None:
                return None

            current = root
            for state in return_states:
                state_key = state.to_key()
                child = current.children.get(state_key)
                if child is None:
                    return None
                current = child

            return current

    def lookup_with_fallback(
        self, prompt_id: str, return_states: list[ToolReturnState]
    ) -> Optional[PrefixTreeNode]:
        """Lookup with fallback."""
        with self._lock:
            root = self._roots.get(prompt_id)
            if root is None:
                return self._lookup_from_root(self._global_root, return_states)

            # Prefer the prompt's own deepest causal prefix when it exists.
            return self._lookup_from_root(root, return_states)

    def get_bucket_for_node(
        self, node: PrefixTreeNode, bucket_thresholds: dict[str, int]
    ) -> Optional[str]:
        """Get bucket for node."""
        if node is None or node.visit_count == 0:
            return None

        mean_bucket = self._length_to_bucket(node.mean_remaining_length, bucket_thresholds)
        p90_bucket = self._length_to_bucket(node.p90_remaining_length, bucket_thresholds)

        if mean_bucket == p90_bucket:
            return mean_bucket
        return None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def rebuild(self, trajectories: list[Trajectory]):
        """Rebuild."""
        with self._lock:
            self._roots.clear()
            self._global_root = None
            self._total_insertions = 0
            for traj in trajectories:
                self.insert(traj)
            logger.info(
                f"[CausalPrefixTree] Rebuild complete: "
                f"{len(self._roots)} prompts, "
                f"{self._total_insertions} trajectories"
            )

    def get_stats(self) -> dict:
        """Return runtime statistics."""
        with self._lock:
            total_nodes = 0
            max_depth = 0
            total_leaves = 0
            for root in self._roots.values():
                stats = root.get_tree_stats()
                total_nodes += stats["total_nodes"]
                max_depth = max(max_depth, stats["max_depth"])
                total_leaves += stats["leaf_count"]

            return {
                "n_prompts": len(self._roots),
                "total_nodes": total_nodes,
                "max_depth": max_depth,
                "total_leaves": total_leaves,
                "total_insertions": self._total_insertions,
                "global_nodes": (
                    self._global_root.get_tree_stats()["total_nodes"]
                    if self._global_root is not None
                    else 0
                ),
            }

    def merge(self, other: "CausalPrefixTree") -> None:
        """Merge another rank's observations into this tree."""
        if other is self:
            return
        with other._lock:
            other_roots = {
                prompt_id: PrefixTreeNode.from_dict(root.to_dict())
                for prompt_id, root in other._roots.items()
            }
            other_global_root = (
                PrefixTreeNode.from_dict(other._global_root.to_dict())
                if other._global_root is not None
                else None
            )
            other_insertions = other._total_insertions
        with self._lock:
            for prompt_id, source in other_roots.items():
                target = self._roots.get(prompt_id)
                if target is None:
                    self._roots[prompt_id] = source
                else:
                    self._merge_node(target, source)
            if other_global_root is not None:
                if self._global_root is None:
                    self._global_root = other_global_root
                else:
                    self._merge_node(self._global_root, other_global_root)
            self._total_insertions += other_insertions

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def save(self, path: str):
        """Save the current state."""
        with self._lock:
            data = {
                "version": 2,
                "total_insertions": self._total_insertions,
                "roots": {
                    prompt_id: root.to_dict()
                    for prompt_id, root in self._roots.items()
                },
                "global_root": (
                    self._global_root.to_dict()
                    if self._global_root is not None
                    else None
                ),
            }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.info(f"[CausalPrefixTree] Saved to {path}")

    def load(self, path: str):
        """Load saved state."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._roots.clear()
            self._global_root = None
            self._total_insertions = data.get("total_insertions", 0)
            for prompt_id, root_d in data.get("roots", {}).items():
                self._roots[prompt_id] = PrefixTreeNode.from_dict(root_d)
            global_root_d = data.get("global_root")
            if global_root_d is not None:
                self._global_root = PrefixTreeNode.from_dict(global_root_d)
            elif self._roots:
                # Version-1 checkpoints had only prompt-specific roots.
                self._global_root = PrefixTreeNode(key="__global__:root", depth=0)
                for root in self._roots.values():
                    self._merge_node(self._global_root, root)

        logger.info(
            f"[CausalPrefixTree] Loaded from {path}: "
            f"{len(self._roots)} prompts"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @staticmethod
    def _make_key(prompt_id: str, return_states: list[ToolReturnState]) -> str:
        """Make key."""
        if not return_states:
            return f"{prompt_id}:root"
        state_keys = ",".join(s.to_key() for s in return_states)
        return f"{prompt_id}:[{state_keys}]"

    @staticmethod
    def _lookup_from_root(
        root: PrefixTreeNode | None, return_states: list[ToolReturnState]
    ) -> Optional[PrefixTreeNode]:
        if root is None:
            return None
        current = root
        for state in return_states:
            child = current.children.get(state.to_key())
            if child is None:
                break
            current = child
        return current

    def _insert_into_root(
        self, root: PrefixTreeNode, trajectory: Trajectory, key_prefix: str
    ) -> None:
        remaining_lengths = trajectory.total_remaining_lengths
        if trajectory.total_length is not None:
            root.update_statistics(trajectory.total_length)
        elif remaining_lengths:
            root.update_statistics(remaining_lengths[0])

        current = root
        for i, state in enumerate(trajectory.return_states):
            state_key = state.to_key()
            child = current.children.get(state_key)
            if child is None:
                child = PrefixTreeNode(
                    key=self._make_key(key_prefix, trajectory.return_states[: i + 1]),
                    depth=current.depth + 1,
                )
                current.children[state_key] = child
            child.update_statistics(remaining_lengths[i])
            current = child

    @classmethod
    def _merge_node(cls, target: PrefixTreeNode, source: PrefixTreeNode) -> None:
        target_count = target.visit_count
        source_count = source.visit_count
        combined_count = target_count + source_count
        if combined_count:
            target.mean_remaining_length = (
                target.mean_remaining_length * target_count
                + source.mean_remaining_length * source_count
            ) / combined_count
        target.visit_count = combined_count
        target.remaining_lengths = (
            target.remaining_lengths + source.remaining_lengths
        )[-200:]
        if target.remaining_lengths:
            target.p90_remaining_length = float(
                np.percentile(target.remaining_lengths, 90)
            )
        for key, source_child in source.children.items():
            target_child = target.children.get(key)
            if target_child is None:
                target.children[key] = PrefixTreeNode.from_dict(
                    source_child.to_dict()
                )
            else:
                cls._merge_node(target_child, source_child)

    @staticmethod
    def _length_to_bucket(length: float, bucket_thresholds: dict[str, int]) -> str:
        """Length to bucket."""
        sorted_buckets = sorted(bucket_thresholds.items(), key=lambda x: x[1])
        for bucket_name, threshold in sorted_buckets:
            if length <= threshold:
                return bucket_name

        return sorted_buckets[-1][0] if sorted_buckets else "long"

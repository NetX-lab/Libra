"""Support code for Cmlfq offline profile."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from RL_Framework.infra.observability.history_collector import StepRecord
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import (
    CausalPrefixTree,
    Trajectory,
)
from RL_Framework.infra.scheduling.cmlfq_tool_state import (
    ToolReturnState,
    ToolStateExtractor,
    ToolStateExtractorRegistry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class ExtractedTrajectory:
    """Extracted trajectory implementation."""
    prompt_id: str
    return_states: list[ToolReturnState]
    total_remaining_lengths: list[float]
    total_output_tokens: int


class TrajectoryExtractor:
    """Trajectory extractor implementation."""

    def __init__(self, tool_extractor: ToolStateExtractor | None = None):
        self._tool_registry = ToolStateExtractorRegistry()
        if tool_extractor:
            self._tool_registry.register(tool_extractor)
        else:
            self._tool_registry = (
                ToolStateExtractorRegistry.create_default_registry()
            )

    def extract_from_step_records(
        self,
        records: list[StepRecord],
    ) -> list[Trajectory]:
        """Extract from step records."""
        trajectories = []
        for record in records:
            for seq in record.sequences:
                prompt_id = str(
                    seq.get(
                        "prompt_id",
                        f"step{record.step}_seq{seq.get('index', 0)}",
                    )
                )
                tool_returns = seq.get("tool_returns", [])
                if not tool_returns:
                    continue

                return_states = []
                remaining_lengths = []
                total_length = float(
                    seq.get("total_output_tokens", seq.get("output_len", 0))
                )
                for tool_return in tool_returns:
                    result = dict(tool_return)
                    result.setdefault("result", tool_return.get("output", ""))
                    return_states.append(self._tool_registry.extract(result))
                    if "remaining_length" in tool_return:
                        remaining = float(tool_return["remaining_length"])
                    else:
                        position = float(
                            tool_return.get("token_position", total_length)
                        )
                        remaining = max(0.0, total_length - position)
                    remaining_lengths.append(remaining)

                trajectories.append(
                    Trajectory(
                        prompt_id=prompt_id,
                        return_states=return_states,
                        total_remaining_lengths=remaining_lengths,
                        total_length=total_length,
                    )
                )

        return trajectories

    def extract_from_raw_data(
        self,
        raw_trajectories: list[dict],
    ) -> list[Trajectory]:
        """Extract from raw data."""
        trajectories = []
        for raw in raw_trajectories:
            prompt_id = raw.get("prompt_id", "")
            tool_returns = raw.get("tool_returns", [])
            if not tool_returns:
                continue

            return_states = []
            remaining_lengths = []
            for tr in tool_returns:
                tool_result = dict(tr)
                tool_result.setdefault("output", tr.get("result", ""))
                state = self._tool_registry.extract(tool_result)
                return_states.append(state)
                total_length = float(
                    raw.get("total_output_tokens", raw.get("output_len", 0))
                )
                remaining_lengths.append(
                    float(
                        tr.get(
                            "remaining_length",
                            max(
                                0.0,
                                total_length - float(
                                    tr.get("token_position", total_length)
                                ),
                            ),
                        )
                    )
                )

            trajectories.append(Trajectory(
                prompt_id=prompt_id,
                return_states=return_states,
                total_remaining_lengths=remaining_lengths,
                total_length=float(
                    raw.get("total_output_tokens", raw.get("output_len", 0))
                ),
            ))

        return trajectories


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class CMLFQOfflineProfiler:
    """C m l f q offline profiler implementation."""

    def __init__(
        self,
        prefix_tree: CausalPrefixTree | None = None,
        trajectory_extractor: TrajectoryExtractor | None = None,
    ):
        self.prefix_tree = prefix_tree or CausalPrefixTree()
        self.extractor = trajectory_extractor or TrajectoryExtractor()

    def profile_from_raw_trajectories(
        self,
        raw_trajectories: list[dict],
    ) -> dict:
        """Profile from raw trajectories."""
        start = time.time()
        trajectories = self.extractor.extract_from_raw_data(raw_trajectories)


        self.prefix_tree.rebuild(trajectories)

        elapsed = time.time() - start
        stats = self.prefix_tree.get_stats()

        result = {
            "n_trajectories": len(trajectories),
            "tree_stats": stats,
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(
            f"[CMLFQOfflineProfiler] Profiling complete: "
            f"{len(trajectories)} trajectories, "
            f"tree nodes={stats['total_nodes']}, "
            f"elapsed={elapsed:.2f}s"
        )
        return result

    def profile_from_step_records(
        self,
        records: list[StepRecord],
    ) -> dict:
        """Profile from step records."""
        start = time.time()
        trajectories = self.extractor.extract_from_step_records(records)
        self.prefix_tree.rebuild(trajectories)
        return {
            "n_trajectories": len(trajectories),
            "tree_stats": self.prefix_tree.get_stats(),
            "elapsed_seconds": round(time.time() - start, 2),
        }

    def save_tree(self, path: str):
        """Save tree."""
        self.prefix_tree.save(path)

    def load_tree(self, path: str):
        """Load tree."""
        self.prefix_tree.load(path)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class CMLFQTreeUpdater:
    """C m l f q tree updater implementation."""

    def __init__(
        self,
        prefix_tree: CausalPrefixTree,
        rebuild_interval: int = 50,
        max_recent_trajectories: int = 5000,
    ):
        self.prefix_tree = prefix_tree
        self.rebuild_interval = rebuild_interval
        self.max_recent_trajectories = max_recent_trajectories


        self._recent_trajectories: list[Trajectory] = []
        self._step_count = 0
        self._total_insertions = 0

    def update_from_step(
        self,
        step: int,
        trajectories: list[Trajectory],
    ):
        """Update from step."""
        self._step_count = step


        for traj in trajectories:
            self.prefix_tree.insert(traj)
            self._total_insertions += 1


        self._recent_trajectories.extend(trajectories)
        if len(self._recent_trajectories) > self.max_recent_trajectories:

            self._recent_trajectories = self._recent_trajectories[
                -self.max_recent_trajectories :
            ]


        if step > 0 and step % self.rebuild_interval == 0:
            self._periodic_rebuild()

    def _periodic_rebuild(self):
        """Periodic rebuild."""
        logger.info(
            f"[CMLFQTreeUpdater] Step {self._step_count}: "
            f"Triggered periodic rebuild with {len(self._recent_trajectories)} recent trajectories"
        )
        self.prefix_tree.rebuild(self._recent_trajectories)

    def force_rebuild(self, trajectories: list[Trajectory]):
        """Force rebuild."""
        self._recent_trajectories = list(trajectories)
        self.prefix_tree.rebuild(trajectories)

    def get_stats(self) -> dict:
        """Return runtime statistics."""
        return {
            "step_count": self._step_count,
            "total_insertions": self._total_insertions,
            "recent_trajectories": len(self._recent_trajectories),
            "rebuild_interval": self.rebuild_interval,
            "tree_stats": self.prefix_tree.get_stats(),
        }

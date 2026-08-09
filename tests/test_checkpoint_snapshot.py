import io
import os
from unittest.mock import patch

import torch

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def _max_container_depth(value):
    if isinstance(value, dict):
        return 1 + max((_max_container_depth(item) for item in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_max_container_depth(item) for item in value), default=0)
    return 0


def test_planner_runtime_stats_do_not_feed_previous_decisions_back_in():
    config = AsyncRLConfig(
        model_path="/tmp/fake-model",
        train_gpus=2,
        rollout_gpus=2,
        train_dp_size=2,
        batch_size=8,
        n_total_gpus=4,
    )
    config.global_resource_planner.enabled = True
    planner = GlobalResourcePlanner.from_config(config)
    stats = {}

    for step in range(200):
        planner_stats = AsyncRLTrainer._planner_step_stats(stats)
        metrics = planner.observe_runtime(
            step=step,
            dispatcher_metrics={"runner_max_queue_size": 16},
            step_stats={
                **planner_stats,
                "rollout_time": 20.0,
                "train_time": 5.0,
                "step_time": 30.0,
                "max_concurrent_rollouts": 8,
            },
        )
        stats["global_resource_planner"] = {
            "step": step,
            "runtime_metrics": metrics.to_dict(),
        }

    raw_step_stats = stats["global_resource_planner"]["runtime_metrics"]["raw"]["step_stats"]
    assert "global_resource_planner" not in raw_step_stats
    assert "global_resource_planner_runtime" not in raw_step_stats
    assert _max_container_depth(stats) < 10

    buffer = io.BytesIO()
    torch.save(
        {"stats": AsyncRLTrainer._checkpoint_safe_value(stats)},
        buffer,
    )
    assert buffer.tell() > 0


def test_checkpoint_snapshot_breaks_cycles_and_limits_depth():
    recursive = {"reward": 0.5}
    recursive["self"] = recursive
    nested = current = {}
    for index in range(100):
        current["next"] = {}
        current = current["next"]
        current["index"] = index

    snapshot = AsyncRLTrainer._checkpoint_safe_value(
        {"recursive": recursive, "nested": nested}
    )

    assert snapshot["recursive"]["self"] == "<recursive-reference>"
    assert _max_container_depth(snapshot) <= 34
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    assert buffer.tell() > 0


def test_resume_step_and_model_version_are_read_from_environment():
    config = AsyncRLConfig(model_path="/tmp/fake-model", total_steps=1000)
    with patch.dict(
        os.environ,
        {"RL_TRAIN_START_STEP": "151", "RL_INITIAL_MODEL_VERSION": "151"},
    ):
        trainer = AsyncRLTrainer(config)

    assert trainer.start_step == 151
    assert trainer.initial_model_version == 151

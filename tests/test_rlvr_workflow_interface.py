"""Regression coverage for the async RLVR workflow interface."""

import inspect

from RL_Framework.workflow.rlvr import RLVRWorkflow


def test_rlvr_run_episode_accepts_trainer_rollout_index():
    parameters = inspect.signature(RLVRWorkflow.run_episode).parameters
    assert "rollout_index" in parameters

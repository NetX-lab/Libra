from types import SimpleNamespace

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def _trainer(*, runtime_manages_workers: bool) -> AsyncRLTrainer:
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer._use_heterogeneous = True
    trainer._external_rollout_reload_instance_ids = [
        "short_tp1_0",
        "short_tp1_1",
        "medium_tp2",
        "long_tp4",
    ]
    trainer.rollout_engine = SimpleNamespace(
        instance_configs=[
            {"instance_id": "grp_tp4_0"},
            {"instance_id": "grp_tp2_1"},
            {"instance_id": "grp_tp2_2"},
        ]
    )
    trainer.config = SimpleNamespace(
        global_resource_planner=SimpleNamespace(
            runtime_manage_rollout_processes=runtime_manages_workers
        )
    )
    return trainer


def test_external_reload_workers_keep_launch_time_ids_after_grp_rebind():
    trainer = _trainer(runtime_manages_workers=False)

    assert trainer._current_rollout_instance_ids() == [
        "short_tp1_0",
        "short_tp1_1",
        "medium_tp2",
        "long_tp4",
    ]


def test_runtime_managed_reload_workers_follow_reconfigured_ids():
    trainer = _trainer(runtime_manages_workers=True)

    assert trainer._current_rollout_instance_ids() == [
        "grp_tp4_0",
        "grp_tp2_1",
        "grp_tp2_2",
    ]

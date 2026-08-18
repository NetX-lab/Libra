from itertools import islice
from types import SimpleNamespace

import torch

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class _FakeTrainEngine:
    def get_data_parallel_world_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_version(self):
        return 3


def _trainer(n_samples=4):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(n_samples=n_samples)
    trainer.train_engine = _FakeTrainEngine()
    trainer._task_counter = 0
    return trainer


def test_data_generator_repeats_each_prompt_as_a_contiguous_group():
    trainer = _trainer(n_samples=4)
    tasks = list(islice(trainer._create_data_generator([{"x": 1}, {"x": 2}], None), 8))

    assert len({task.group_id for task in tasks[:4]}) == 1
    assert len({task.group_id for task in tasks[4:]}) == 1
    assert tasks[0].group_id != tasks[4].group_id
    assert [task.rollout_index for task in tasks[:4]] == [0, 1, 2, 3]


def test_advantages_are_normalized_within_prompt_not_across_prompts():
    trainer = _trainer()
    trajectories = [
        {"grpo_group_id": "easy", "rewards": torch.tensor([0.8])},
        {"grpo_group_id": "easy", "rewards": torch.tensor([1.0])},
        {"grpo_group_id": "hard", "rewards": torch.tensor([0.0])},
        {"grpo_group_id": "hard", "rewards": torch.tensor([0.2])},
    ]

    trainer._compute_advantages(trajectories)

    easy = torch.cat([trajectories[0]["advantages"], trajectories[1]["advantages"]])
    hard = torch.cat([trajectories[2]["advantages"], trajectories[3]["advantages"]])
    assert torch.allclose(easy, hard, atol=1e-5)
    assert torch.allclose(easy.mean(), torch.tensor(0.0), atol=1e-6)
    assert trainer._advantage_stats["grpo_mean_group_size"] == 2.0


def test_constant_reward_group_has_zero_advantage():
    trainer = _trainer()
    trajectories = [
        {"grpo_group_id": "same", "rewards": torch.tensor([0.5])},
        {"grpo_group_id": "same", "rewards": torch.tensor([0.5])},
    ]

    trainer._compute_advantages(trajectories)

    assert all(float(traj["advantages"]) == 0.0 for traj in trajectories)
    assert trainer._advantage_stats["grpo_zero_variance_groups"] == 1


def test_batch_collector_keeps_only_complete_groups_after_rejection_gap():
    trainer = _trainer(n_samples=4)
    trainer._pending_grpo_groups = {}

    class Dispatcher:
        chunks = [
            [{"grpo_group_id": "a"}] * 3 + [{"grpo_group_id": "b"}],
            [{"grpo_group_id": "b"}] * 3 + [{"grpo_group_id": "c"}],
        ]

        def active_submit_and_wait(self, **kwargs):
            return self.chunks.pop(0)

    class Staleness:
        consumed = 0

        def on_batch_consumed(self, count):
            self.consumed += count

    trainer.dispatcher = Dispatcher()
    trainer.staleness_manager = Staleness()

    batch = trainer._collect_complete_grpo_batch(iter(()), batch_size=4)

    assert [trajectory["grpo_group_id"] for trajectory in batch] == ["b"] * 4
    assert trainer.staleness_manager.consumed == 8

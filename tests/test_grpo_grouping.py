from itertools import islice
from types import SimpleNamespace
import asyncio

import pytest
import torch

from RL_Framework.infra.execution.async_runner import AsyncTaskRunner, TaskResult
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
from RL_Framework.infra.sync.staleness import StalenessManager
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


@pytest.mark.parametrize('ready_results', [False, True])
def test_collector_consumes_only_accepted_packets_with_real_timeout_accounting(monkeypatch, ready_results):
    trainer = _trainer(n_samples=4)
    trainer.config.max_head_offpolicyness = 3
    trainer.config.enable_rollout_tracing = False
    trainer.is_main_process = True
    trainer._pending_grpo_groups = {}
    trainer._consumed_grpo_group_ids = {'consumed'}
    trainer.rollout_engine = object()
    trainer._refresh_rollout_engine_from_manifest_for_task = lambda: None
    manager = trainer.staleness_manager = StalenessManager(
        trainer.train_engine, max_concurrent_rollouts=16, consumer_batch_size=4, max_staleness=3)

    async def episode(engine, data, **kwargs):
        if data.get('timeout'):
            await asyncio.Future()
        return {}

    trainer.workflow = SimpleNamespace(run_episode=episode)
    monkeypatch.setenv('ROLLOUT_TASK_TIMEOUT_S', '.001')
    # One timeout never entered accepted. Successful duplicate members and a
    # late result for a consumed group did, and must still release their credit.
    tasks = [('incomplete', 0, False), ('incomplete', 1, False),
             ('incomplete', 3, True), ('consumed', 0, False),
             ('incomplete', 1, False), ('incomplete', 2, False)]
    tasks += [('complete', index, False) for index in range(4)]

    async def produce():
        results = []
        for task_id, (group_id, index, timeout) in enumerate(tasks):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()
            task = TaskInput(task_id, {'timeout': timeout}, version=3,
                group_id=group_id, rollout_index=index)
            results.append(await trainer._create_rollout_task(task)())
        return results

    packets = asyncio.run(produce())
    assert packets[2]['status'] == 'failed' and packets[2]['failure_reason'] == 'task_timeout'
    stats = manager.get_stats()
    assert (stats.accepted, stats.running, stats.rejected, stats.enqueued) == (9, 0, 1, 0)
    if ready_results:
        runner = AsyncTaskRunner(max_queue_size=32)
        dispatcher = BatchTaskDispatcher(runner, manager, trainer._create_rollout_task)
        for task_id, packet in enumerate(packets):
            dispatcher._pending_results[task_id] = TaskResult(packet, task_id, task_id, task_id)
            dispatcher._active_task_ids.add(task_id)
        dispatcher.active_submit_and_wait = lambda **kwargs: pytest.fail('Ready results must not submit work')
        trainer.dispatcher = dispatcher
    else:
        remaining = list(packets)

        def collect(**kwargs):
            count = kwargs['batch_size']
            chunk, remaining[:] = remaining[:count], remaining[count:]
            return chunk

        trainer.dispatcher = SimpleNamespace(active_submit_and_wait=collect)
    batch = trainer._collect_complete_grpo_batch(iter(()), batch_size=4)
    assert [member['grpo_group_id'] for member in batch] == ['complete'] * 4
    assert [member['grpo_rollout_index'] for member in batch] == list(range(4))
    assert set(trainer._pending_grpo_groups) == {'incomplete'}
    assert len(trainer._pending_grpo_groups['incomplete']) == 3
    assert packets[2]['status'] == 'failed'  # Failure notification is untouched.
    stats = manager.get_stats()
    assert (stats.accepted, stats.running, stats.rejected, stats.enqueued) == (0, 0, 1, 0)
    assert manager.get_capacity() == 16

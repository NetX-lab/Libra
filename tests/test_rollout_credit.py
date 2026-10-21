"""A timed-out task releases running credit, never accepted credit."""
import asyncio
from types import SimpleNamespace

from RL_Framework.infra.execution.batch_dispatcher import TaskInput
from RL_Framework.infra.sync.staleness import StalenessManager
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def test_timeout_notification_does_not_consume_successful_result_credit(monkeypatch):
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(n_samples=4, max_head_offpolicyness=3,
                                     enable_rollout_tracing=False)
    trainer.is_main_process = True
    trainer.train_engine = SimpleNamespace(get_version=lambda: 0)
    trainer.rollout_engine = object()
    trainer._refresh_rollout_engine_from_manifest_for_task = lambda: None
    trainer._pending_grpo_groups = {}
    manager = trainer.staleness_manager = StalenessManager(
        trainer.train_engine, max_concurrent_rollouts=8,
        consumer_batch_size=4, max_staleness=3)

    async def episode(engine, data, **kwargs):
        if data['timeout']:
            await asyncio.Future()
        return {}

    trainer.workflow = SimpleNamespace(run_episode=episode)
    monkeypatch.setenv('ROLLOUT_TASK_TIMEOUT_S', '.001')

    async def produce():
        packets = []
        for index in range(5):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()
            task = TaskInput(index, {'timeout': index == 0}, version=0,
                             group_id='failed' if index == 0 else 'complete',
                             rollout_index=max(0, index - 1))
            packets.append(await trainer._create_rollout_task(task)())
        return packets

    packets = asyncio.run(produce())
    assert (manager.get_stats().accepted, manager.get_stats().rejected) == (4, 1)
    chunks = [packets[:4], packets[4:]]
    trainer.dispatcher = SimpleNamespace(active_submit_and_wait=lambda **kw: chunks.pop(0))
    batch = trainer._collect_complete_grpo_batch(iter(()), batch_size=4)
    assert [row['grpo_rollout_index'] for row in batch] == [0, 1, 2, 3]
    assert (manager.get_stats().accepted, manager.get_stats().running,
            manager.get_stats().rejected) == (0, 0, 1)
    assert manager.get_capacity() == 8

from types import SimpleNamespace
import asyncio
import queue
import threading

import pytest
import torch

from RL_Framework.infra.execution.async_runner import AsyncTaskRunner
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
from RL_Framework.infra.sync.staleness import StalenessManager
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class _VersionProvider:
    def get_version(self):
        return 0


@pytest.mark.parametrize(
    "objective, dynamic, expected",
    [("legacy", False, False), ("decoupled", False, True), ("decoupled", True, False)],
)
def test_sync1_r2e_requires_exact_sampling_and_fixed_runtime_layout(objective, dynamic, expected):
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        grpo_objective=objective, train_backend="megatron_core", rollout_backend="vllm",
        sync_interval=1, ppo_epochs=1, weight_sync_mode="nccl",
        rollout_weight_sync_mode="nccl", require_rollout_weight_sync=True,
        global_resource_planner=SimpleNamespace(enabled=True, runtime_dynamic_reconfiguration_enabled=dynamic),
    )
    trainer.start_step = 0
    trainer.train_engine = SimpleNamespace(optimizer_revision=0, get_version=lambda: 0)
    trainer._configure_rollout_production()
    assert trainer._bounded_rollout_production is expected


def test_staleness_runtime_limit_reduces_and_restores_capacity():
    manager = StalenessManager(
        _VersionProvider(),
        max_concurrent_rollouts=32,
        consumer_batch_size=4,
        max_staleness=8,
    )

    manager.set_runtime_max_concurrent_rollouts(8)
    assert manager.get_capacity() == 8

    manager.set_runtime_max_concurrent_rollouts(None)
    assert manager.get_capacity() == 32


def test_dispatcher_cancels_pending_inputs_and_restores_accounting():
    runner = AsyncTaskRunner(max_queue_size=8)
    manager = StalenessManager(
        _VersionProvider(),
        max_concurrent_rollouts=8,
        consumer_batch_size=2,
        max_staleness=2,
    )
    dispatcher = BatchTaskDispatcher(
        runner,
        manager,
        task_factory=lambda _task: (lambda: None),
    )
    dispatcher.submit_task_input(TaskInput(task_id=1, data={}))
    dispatcher.submit_task_input(TaskInput(task_id=2, data={}))
    dispatcher.pause()

    cancelled = dispatcher.cancel_queued()

    assert cancelled == {
        "dispatcher_queue": 2,
        "runner_queue": 0,
        "task_ids": [1, 2],
    }
    assert dispatcher.get_runtime_metrics()["pending_inputs"] == 0
    assert dispatcher.get_runtime_metrics()["active_tasks"] == 0
    stats = manager.get_stats()
    assert stats.enqueued == 0
    assert stats.running == 0
    assert stats.cancelled == 2


def test_sync_drain_prefetch_limit_uses_two_then_one_batch():
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        sync_interval=5,
        rollout_sync_drain_lead_steps=2,
        max_concurrent_rollouts=32,
    )

    assert trainer._sync_aware_rollout_limit(step=2, batch_size=4) is None
    assert trainer._sync_aware_rollout_limit(step=3, batch_size=4) == 8
    assert trainer._sync_aware_rollout_limit(step=4, batch_size=4) == 4
    assert trainer._sync_aware_rollout_limit(step=5, batch_size=4) is None
    assert trainer._sync_aware_rollout_limit(step=6, batch_size=4) is None


def test_dispatcher_rechecks_pause_after_dequeue_before_submit():
    class _Runner:
        def __init__(self):
            self.paused = threading.Event()
            self.input_queue = queue.Queue()
            self.output_queue = queue.Queue()
            self.max_queue_size = 8
            self.submitted = []

        def submit(self, _fn, *, task_id):
            self.submitted.append(task_id)

        def cancel_queued(self):
            return []

    runner = _Runner()
    manager = StalenessManager(
        _VersionProvider(),
        max_concurrent_rollouts=8,
        consumer_batch_size=2,
        max_staleness=2,
    )
    dispatcher = BatchTaskDispatcher(
        runner,
        manager,
        task_factory=lambda _task: (lambda: None),
    )
    dispatcher.submit_task_input(TaskInput(task_id=7, data={}))
    original_get = dispatcher._get_next_task_for_submission

    def dequeue_at_pause_boundary():
        item = original_get()
        runner.paused.set()
        dispatcher._shutdown_event.set()
        return item

    dispatcher._get_next_task_for_submission = dequeue_at_pause_boundary

    dispatcher._commit_loop()

    assert runner.submitted == []
    assert manager.get_stats().enqueued == 1
    assert manager.get_stats().cancelled == 0
    assert dispatcher.cancel_queued()['task_ids'] == [7]
    assert manager.get_stats().enqueued == 0
    assert manager.get_stats().cancelled == 1
    assert dispatcher.cancel_queued()['task_ids'] == []
    assert manager.get_stats().cancelled == 1


@pytest.mark.parametrize("sync_interval", [5, 25])
def test_legacy_collector_keeps_training_without_sample_revision_between_publications(sync_interval):
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        n_samples=2, sync_interval=sync_interval, max_head_offpolicyness=1,
        global_resource_planner=SimpleNamespace(runtime_batch_collection_timeout_s=4),
    )
    trainer.is_main_process = True
    trainer._use_heterogeneous = False
    trainer._pending_grpo_groups = {}
    trainer.train_engine = _VersionProvider()
    trainer.rollout_engine = object()

    async def episode(engine, data, **kwargs):
        if data.get("timeout"):
            raise asyncio.TimeoutError
        return {"rewards": torch.tensor([0.0]), "versions": torch.tensor([kwargs["version"]])}

    trainer.workflow = SimpleNamespace(run_episode=episode)
    manager = trainer.staleness_manager = StalenessManager(trainer.train_engine, 8, 2, 1)
    runner = AsyncTaskRunner(max_queue_size=16)
    dispatcher = trainer.dispatcher = BatchTaskDispatcher(runner, manager, trainer._create_rollout_task)

    def inputs():
        for group in range(sync_interval + 4):
            for member in range(2):
                yield TaskInput(
                    2 * group + member, {"timeout": group == 0},
                    group_id=str(group), rollout_index=member,
                )

    work = inputs()
    runner.initialize()
    dispatcher.initialize()
    try:
        for step in range(sync_interval):
            rows = trainer._collect_complete_grpo_batch(work, 2)
            assert len(rows) == 2
            assert {row["grpo_group_id"] for row in rows} == {str(step + 1)}
            assert all("sample_revision" not in row for row in rows)
        assert manager.get_stats().rejected == 2
    finally:
        dispatcher.destroy()

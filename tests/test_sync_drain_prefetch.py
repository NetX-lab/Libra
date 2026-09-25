from types import SimpleNamespace
import queue
import threading

from RL_Framework.infra.execution.async_runner import AsyncTaskRunner
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
from RL_Framework.infra.sync.staleness import StalenessManager
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


class _VersionProvider:
    def get_version(self):
        return 0


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
    assert trainer._sync_aware_rollout_limit(step=5, batch_size=4) == 4
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
    assert manager.get_stats().enqueued == 0
    assert manager.get_stats().cancelled == 1

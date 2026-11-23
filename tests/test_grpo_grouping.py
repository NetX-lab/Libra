from itertools import count, islice
from types import SimpleNamespace
import asyncio
import threading
import time

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


def test_data_generator_uses_the_configured_sampler_seed():
    trainer = _trainer(n_samples=1)
    trainer.config.seed = 42
    dataset = [{"x": index} for index in range(8)]
    tasks = islice(trainer._create_data_generator(dataset, None), len(dataset))
    assert [task.data["x"] for task in tasks] == [6, 3, 0, 7, 2, 1, 4, 5]


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
    trainer.is_main_process = True
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

    inputs = (TaskInput(task_id, {'timeout': timeout}, version=3,
                        group_id=group_id, rollout_index=index)
              for task_id, (group_id, index, timeout) in enumerate(tasks))
    runner = AsyncTaskRunner(max_queue_size=32)
    dispatcher = trainer.dispatcher = BatchTaskDispatcher(runner, manager, trainer._create_rollout_task)
    runner.initialize()
    dispatcher.initialize()
    try:
        if ready_results:
            for task in inputs:
                dispatcher.submit_task_input(task)
            _until(lambda: len(dispatcher.peek_ready_data()) == len(tasks))
            packets = dispatcher.peek_ready_data()
            assert packets[2]['status'] == 'failed' and packets[2]['failure_reason'] == 'task_timeout'
            stats = manager.get_stats()
            assert (stats.accepted, stats.running, stats.rejected, stats.enqueued) == (9, 0, 1, 0)
            dispatcher.active_submit_and_wait = lambda **kwargs: pytest.fail('Ready results must not submit work')
        batch = trainer._collect_complete_grpo_batch(inputs, batch_size=4)
        assert [member['grpo_group_id'] for member in batch] == ['complete'] * 4
        assert [member['grpo_rollout_index'] for member in batch] == list(range(4))
        assert set(trainer._pending_grpo_groups) == {'incomplete'}
        assert len(trainer._pending_grpo_groups['incomplete']) == 3
        stats = manager.get_stats()
        assert (stats.accepted, stats.running, stats.rejected, stats.enqueued) == (0, 0, 1, 0)
        assert manager.get_capacity() == 16
    finally:
        dispatcher.destroy()


def _pipeline(factory, *, batch_size=16, group_size=4, concurrency=32):
    provider = SimpleNamespace(get_version=lambda: 0)
    manager = StalenessManager(provider, concurrency, batch_size, 1)
    runner = AsyncTaskRunner(max_queue_size=64)
    dispatcher = BatchTaskDispatcher(runner, manager, factory)
    runner.initialize()
    dispatcher.initialize()

    def inputs():
        for group in count():
            for index in range(group_size):
                yield TaskInput(group * group_size + index, {}, group_id=str(group), rollout_index=index)

    dispatcher.start_production(inputs(), group_size)
    return dispatcher, manager


def _packet(task):
    return {'task_id': task.task_id, 'grpo_group_id': task.group_id,
            'grpo_rollout_index': task.rollout_index, 'sample_revision': task.version, 'reward': 0.0}


def _until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail('condition did not become true')
        time.sleep(.005)


def test_production_fills_two_batches_and_consumption_holds_reservations_until_publication():
    started = []

    def factory(task):
        async def run():
            started.append(task.task_id)
            return _packet(task)
        return run

    dispatcher, manager = _pipeline(factory)
    try:
        _until(lambda: len(started) == 32)
        batch = dispatcher.collect_batch(16, optimizer_revision=0)
        assert len(batch) == 16 and all(member['reward'] == 0 for member in batch)
        assert len(manager.get_group_reservations()) == 8
        assert dispatcher.get_runtime_metrics()['pending_results'] == 16
        time.sleep(.03)
        assert len(started) == 32
        dispatcher.retire_batch(batch, optimizer_revision=1)
        time.sleep(.03)
        assert len(started) == 32  # Retirement alone does not open a new window.
        dispatcher.on_publication(1)
        _until(lambda: len(started) == 48)
        assert len(manager.get_group_reservations()) == 8
        assert {item['sample_revision'] for item in manager.get_group_reservations()} == {0, 1}
    finally:
        dispatcher.destroy()


def test_younger_ready_group_cannot_strand_two_old_groups_at_next_deadline():
    manager = StalenessManager(SimpleNamespace(get_version=lambda: 0), 4, 1, 2)
    manager.configure_groups(1)
    for index in range(3):
        manager.reserve_group(str(index), [index])
    manager.on_member_completed("0", 0, 0)
    assert manager.select_groups(1, 0) == ["0"]
    manager.retire_batch({"0"}, 1)
    manager.on_publication(1)
    manager.reserve_group("3", [3])
    manager.on_member_completed("3", 3, 1)

    assert manager.select_groups(1, 1) is None
    manager.on_member_completed("2", 2, 0)
    assert manager.select_groups(1, 1) == ["2"]
    manager.retire_batch({"2"}, 2)
    manager.on_publication(2)
    manager.on_member_completed("1", 1, 0)
    assert manager.select_groups(1, 2) == ["1"]


def test_retiring_a_batch_waits_for_a_group_whose_input_is_being_read():
    manager = StalenessManager(SimpleNamespace(get_version=lambda: 0), 2, 1, 1)
    reading_input = threading.Event()
    retirement_started = threading.Event()
    release_input = threading.Event()
    errors = []

    def factory(task):
        async def run():
            return _packet(task)
        return run

    def inputs():
        yield TaskInput(1, {}, group_id="training")
        reading_input.set()
        assert release_input.wait(3)
        yield TaskInput(2, {}, group_id="next")

    runner = AsyncTaskRunner(max_queue_size=4)
    dispatcher = BatchTaskDispatcher(runner, manager, factory)
    # Start with a batch already selected; release the producer into the next
    # dataset read so that retirement races with an open reservation.
    generator = inputs()
    dispatcher.start_production(generator, 1)
    with dispatcher._input_cv:
        dispatcher._admit_group()
    with dispatcher._reconfigure_lock:
        dispatcher._accept_group_result(TaskResult(_packet(TaskInput(1, {}, group_id="training")), 1, 0., 0.))
    batch = dispatcher.collect_batch(1, 0)
    with dispatcher._input_cv:
        dispatcher._pending_inputs.clear()

    def admit():
        try:
            with dispatcher._input_cv:
                dispatcher._admit_group()
        except Exception as error:
            errors.append(error)

    def retire():
        retirement_started.set()
        dispatcher.retire_batch(batch, 1)

    producer = threading.Thread(target=admit)
    updater = threading.Thread(target=retire)
    producer.start()
    try:
        assert reading_input.wait(2)
        updater.start()
        assert retirement_started.wait(2)
        updater.join(.1)
    finally:
        release_input.set()
        producer.join(2)
        if updater.ident is not None:
            updater.join(2)
    assert not errors
    assert not producer.is_alive() and not updater.is_alive()
    assert [group["group_id"] for group in manager.get_group_reservations()] == ["next"]


def test_timeout_terminates_whole_group_cancels_siblings_and_replenishes_once():
    cancelled = threading.Event()

    def factory(task):
        async def run():
            if task.group_id == '0':
                if task.rollout_index == 0:
                    return dict(_packet(task), status='failed', failure_reason='task_timeout')
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    return _packet(task)  # A real late completion after group termination.
                finally:
                    cancelled.set()
            return _packet(task)
        return run

    dispatcher, manager = _pipeline(factory, batch_size=4, group_size=4, concurrency=8)
    try:
        batch = dispatcher.collect_batch(4, optimizer_revision=0)
        assert {item['grpo_group_id'] for item in batch} == {'1'}
        _until(cancelled.is_set)
        _until(lambda: len(manager.get_group_reservations()) == 2)
        assert {item['group_id'] for item in manager.get_group_reservations()} == {'1', '2'}
        _until(lambda: manager.get_stats().accepted == 8)
        assert manager.get_stats().running == 0
        assert manager.get_stats().rejected == 1
        assert not any(row['grpo_group_id'] == '0' for row in dispatcher.peek_ready_data())
    finally:
        dispatcher.destroy()


def test_non_timeout_task_exception_stops_real_runner_dispatcher_collection():
    def factory(task):
        async def run():
            raise RuntimeError('actual tool failure')
        return run

    dispatcher, _ = _pipeline(factory, batch_size=1, group_size=1, concurrency=1)
    try:
        with pytest.raises(RuntimeError, match='actual tool failure'):
            dispatcher.collect_batch(1, optimizer_revision=0)
    finally:
        dispatcher.destroy()


def test_real_trainer_collects_from_one_continuous_producer_across_updates():
    trainer = _trainer(n_samples=4)
    trainer._bounded_rollout_production = True
    trainer.train_engine.optimizer_revision = 3
    trainer.is_main_process = True
    trainer.rollout_engine = object()

    async def episode(engine, data, **kwargs):
        return {'sample_revision': kwargs['version'], 'rewards': torch.tensor([0.0])}

    trainer.workflow = SimpleNamespace(run_episode=episode)
    manager = trainer.staleness_manager = StalenessManager(trainer.train_engine, 8, 4, 1)
    runner = AsyncTaskRunner(max_queue_size=16)
    dispatcher = trainer.dispatcher = BatchTaskDispatcher(runner, manager, trainer._create_rollout_task)
    generator = trainer._create_data_generator([{'x': 1}], trainer.workflow)
    runner.initialize()
    dispatcher.initialize()
    try:
        first = trainer._collect_complete_grpo_batch(generator, 4)
        assert len(first) == 4 and trainer._source_training_batch is first
        _until(lambda: manager.get_stats().accepted == 8)
        assert len(manager.get_group_reservations()) == 2
        trainer.train_engine.optimizer_revision = 4
        dispatcher.retire_batch(first, optimizer_revision=4)
        dispatcher.on_publication(4)
        second = trainer._collect_complete_grpo_batch(generator, 4)
        assert len(second) == 4
        assert {row['grpo_group_id'] for row in first}.isdisjoint(row['grpo_group_id'] for row in second)
        assert {row['sample_revision'] for row in second} == {3}
    finally:
        dispatcher.destroy()


def test_destroy_stops_production_and_releases_inflight_tasks_before_service_shutdown():
    started, released = [], []

    def factory(task):
        async def run():
            started.append(task.task_id)
            try:
                await asyncio.Future()
            finally:
                released.append(task.task_id)
        return run

    dispatcher, _ = _pipeline(factory, batch_size=4, group_size=1, concurrency=8)
    _until(lambda: len(started) == 8)
    dispatcher.destroy()
    assert not dispatcher._commit_thread.is_alive()
    assert not dispatcher._fetch_thread.is_alive()
    assert not dispatcher.runner.thread.is_alive()
    assert sorted(released) == sorted(started)
    assert len(started) == 8


@pytest.mark.parametrize(
    ("revision", "error"),
    [(2, "below the group's publication promise"), (None, "actual integer sample_revision"),
     (4, "ahead of the current optimizer revision")],
    ids=["below-promise", "missing-revision", "ahead-of-optimizer"],
)
def test_trainer_checks_actual_sampling_revision_against_publication_promise(revision, error):
    trainer = _trainer(n_samples=1)
    trainer.config.grpo_objective = "decoupled"
    trainer._bounded_rollout_production = True
    trainer.train_engine.optimizer_revision = 3
    trainer.is_main_process = True
    trainer.rollout_engine = object()

    async def episode(engine, data, *, version, rollout_index):
        row = {"versions": torch.tensor([version]), "rewards": torch.tensor([0.0])}
        if revision is not None:
            row["sample_revision"] = revision
        return row

    trainer.workflow = SimpleNamespace(run_episode=episode)
    manager = trainer.staleness_manager = StalenessManager(trainer.train_engine, 2, 1, 1)
    runner = AsyncTaskRunner(max_queue_size=4)
    dispatcher = trainer.dispatcher = BatchTaskDispatcher(runner, manager, trainer._create_rollout_task)
    generator = trainer._create_data_generator([{"x": 1}], trainer.workflow)
    runner.initialize()
    dispatcher.initialize()
    try:
        with pytest.raises(RuntimeError, match=error):
            trainer._collect_complete_grpo_batch(generator, 1)
    finally:
        dispatcher.destroy()

"""Already generated complete-group supply survives all-engine EHP loans."""
from types import SimpleNamespace

from RL_Framework.infra.elastic.rollout_supply import ready_complete_samples
from RL_Framework.infra.execution.async_runner import AsyncTaskRunner, TaskResult
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher
from RL_Framework.infra.sync.staleness import StalenessManager
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def bounded_trainer_fixture():
    engine = SimpleNamespace(get_version=lambda: 0, optimizer_revision=0)
    manager = StalenessManager(engine, max_concurrent_rollouts=4,
                               consumer_batch_size=2, max_staleness=1)
    manager.configure_groups(2)
    dispatcher = BatchTaskDispatcher(AsyncTaskRunner(max_queue_size=4), manager, lambda task: None)
    return SimpleNamespace(config=SimpleNamespace(n_samples=2), train_engine=engine, dispatcher=dispatcher)


def reserve(trainer, group, ids, *, ready):
    dispatcher = trainer.dispatcher
    manager = dispatcher.staleness_manager
    revision = manager.reserve_group(group, ids)
    assert revision is not None
    for index, task_id in enumerate(ids[:ready]):
        member = dict(grpo_group_id=group, grpo_rollout_index=index, task_id=task_id)
        manager.on_member_completed(group, task_id, revision)
        dispatcher._pending_results[task_id] = TaskResult(member, task_id, float(task_id), float(task_id))


def test_snapshot_is_non_consuming_and_collector_can_use_the_counted_group():
    trainer = bounded_trainer_fixture()
    reserve(trainer, "a", [1, 2], ready=2)
    assert ready_complete_samples(trainer) == 2
    assert ready_complete_samples(trainer) == 2
    assert [item["task_id"] for item in trainer.dispatcher.collect_batch(2, 0)] == [1, 2]


def test_ready_younger_group_cannot_hide_a_missing_due_member():
    trainer = bounded_trainer_fixture()
    reserve(trainer, "old", [1, 2], ready=2)
    reserve(trainer, "due", [3, 4], ready=1)
    dispatcher = trainer.dispatcher
    dispatcher.retire_batch(dispatcher.collect_batch(2, 0), 1)
    dispatcher.on_publication(1)
    trainer.train_engine.optimizer_revision = 1
    reserve(trainer, "young", [5, 6], ready=2)
    assert ready_complete_samples(trainer) == 0


def member(group, index, **kwargs):
    return dict(grpo_group_id=group, grpo_rollout_index=index, **kwargs)


def legacy_trainer_fixture(paused=False):
    runner = AsyncTaskRunner(max_queue_size=32)
    if paused:
        runner.pause()
    dispatcher = BatchTaskDispatcher(runner, SimpleNamespace(), lambda task: None)
    trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(n_samples=4, global_resource_planner=SimpleNamespace(
        runtime_batch_collection_timeout_s=.05, runtime_batch_collection_max_retries=0))
    trainer.is_main_process = True
    trainer._pending_grpo_groups = {}
    trainer.dispatcher = dispatcher
    trainer.staleness_manager = SimpleNamespace(on_batch_consumed=lambda count: None)
    return trainer


def add_result(dispatcher, task_id, data, created=None):
    dispatcher._pending_results[task_id] = TaskResult(data, task_id, created or float(task_id), float(task_id))
    dispatcher._active_task_ids.add(task_id)


def test_snapshot_is_non_consuming_and_matches_real_wait_order():
    trainer = legacy_trainer_fixture()
    dispatcher = trainer.dispatcher
    add_result(dispatcher, 10, member("b", 1), created=2)
    add_result(dispatcher, 20, member("a", 0), created=1)
    original = list(dispatcher._pending_results)
    data = dispatcher.peek_ready_data()
    assert [item["grpo_group_id"] for item in data] == ["a", "b"]
    data.clear()
    assert list(dispatcher._pending_results) == original
    assert [result.data["grpo_group_id"] for result in dispatcher.wait_results(2)] == ["a", "b"]


def test_ready_supply_combines_partial_groups_and_ignores_failed_duplicate_members():
    trainer = legacy_trainer_fixture(paused=True)
    trainer._pending_grpo_groups = {"a": [member("a", 0), member("a", 1)],
                                    "incomplete": [member("incomplete", 0)]}
    for task_id, value in enumerate([
        member("a", 1), member("a", 2), member("a", 3),
        member("b", 0), member("b", 1), member("b", 2), member("b", 3),
        member("incomplete", 1), member("incomplete", 2, status="failed"),
        member("incomplete", 0), None,
    ]):
        add_result(trainer.dispatcher, task_id, value)
    before = list(trainer.dispatcher._pending_results)
    pending_before = {group: list(values) for group, values in trainer._pending_grpo_groups.items()}
    assert ready_complete_samples(trainer) == 8
    assert list(trainer.dispatcher._pending_results) == before
    assert trainer._pending_grpo_groups == pending_before


def test_collector_returns_existing_complete_group_without_dispatcher_call():
    trainer = legacy_trainer_fixture(paused=True)
    group = [member("ready", index) for index in range(4)]
    trainer._pending_grpo_groups = {"ready": group + [dict(grpo_group_id="ready", task_id=99)]}
    trainer.dispatcher = SimpleNamespace()
    assert ready_complete_samples(trainer) == 4
    assert trainer._collect_complete_grpo_batch(iter(()), 4) == group
    assert trainer._pending_grpo_groups == {}


def test_paused_real_dispatcher_consumes_enough_partial_and_ready_data_without_submitting():
    trainer = legacy_trainer_fixture(paused=True)
    trainer._pending_grpo_groups = {"a": [member("a", 0), member("a", 1)]}
    add_result(trainer.dispatcher, 1, member("a", 2))
    add_result(trainer.dispatcher, 2, member("a", 3))
    add_result(trainer.dispatcher, 3, member("future", 0))
    def never_submit(**kwargs):
        raise AssertionError("paused ready data must not submit rollout work")
    trainer.dispatcher.active_submit_and_wait = never_submit
    batch = trainer._collect_complete_grpo_batch(iter(()), 4)
    assert [value["grpo_rollout_index"] for value in batch] == [0, 1, 2, 3]
    assert not trainer.dispatcher._pending_inputs
    # Reading extra already-ready members preserves them for the next group.
    assert trainer._pending_grpo_groups["future"] == [member("future", 0)]
    assert ready_complete_samples(trainer) == 0


def test_collector_deduplicates_ready_members_and_requests_only_deficit():
    trainer = legacy_trainer_fixture()
    trainer._pending_grpo_groups = {"a": [member("a", 0), member("a", 1), member("a", 2)]}
    requested = []
    def missing(**kwargs):
        requested.append(kwargs["batch_size"])
        return [member("a", 3)]
    trainer.dispatcher.active_submit_and_wait = missing
    batch = trainer._collect_complete_grpo_batch(iter(()), 4)
    assert requested == [1]
    assert len(batch) == 4
    # Retransmissions of a previously consumed group cannot create another batch.
    for index in range(4):
        add_result(trainer.dispatcher, index, member("a", index))
    assert ready_complete_samples(trainer) == 0
    next_group = [member("b", index) for index in range(4)]
    trainer.dispatcher.active_submit_and_wait = lambda **kwargs: next_group
    assert trainer._collect_complete_grpo_batch(iter(()), 4) == next_group


def test_duplicate_and_invalid_member_indices_do_not_complete_a_group():
    trainer = legacy_trainer_fixture()
    trainer._pending_grpo_groups = {"a": [member("a", 0), member("a", 1), member("a", 2),
                                          member("a", 2), member("a", 8)]}
    assert ready_complete_samples(trainer) == 0


def test_duplicate_task_identity_with_missing_member_index_is_not_counted_twice():
    trainer = legacy_trainer_fixture()
    trainer._pending_grpo_groups = {"a": [member("a", 0, task_id=10), member("a", 1, task_id=11),
                                          member("a", 2, task_id=12)]}
    add_result(trainer.dispatcher, 10, dict(grpo_group_id="a", task_id=10))
    assert ready_complete_samples(trainer) == 0


def test_paused_insufficient_supply_never_enqueues_new_rollouts():
    trainer = legacy_trainer_fixture(paused=True)
    trainer._pending_grpo_groups = {"a": [member("a", 0), member("a", 1)]}
    def cannot_submit(**kwargs):
        raise AssertionError("paused dispatcher submitted new work")
    trainer.dispatcher.active_submit_and_wait = cannot_submit
    assert trainer._collect_complete_grpo_batch(iter(()), 4) == []
    assert not trainer.dispatcher._pending_inputs
    assert len(trainer._pending_grpo_groups["a"]) == 2

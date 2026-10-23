"""Lent GPUs return before supply waits and stay closed until publication."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from RL_Framework.infra.elastic import rollout_lifecycle as lifecycle
from RL_Framework.infra.elastic import rollout_lending


class Collectives:
    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.packet = None
        self.values = [None, None]
        self.calls = []

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_world_size(group=None):
        return 2

    @staticmethod
    def get_rank(group=None):
        return group.rank if group is not None else 0

    def broadcast_object_list(self, packet, src, group):
        assert src == 0
        self.calls.append((group.rank, "broadcast"))
        if group.rank == 0:
            self.packet = copy.deepcopy(packet[0])
        self.barrier.wait(timeout=3)
        packet[0] = copy.deepcopy(self.packet)
        self.barrier.wait(timeout=3)

    def all_gather_object(self, records, value, group):
        self.calls.append((group.rank, "gather"))
        self.values[group.rank] = value
        self.barrier.wait(timeout=3)
        records[:] = self.values
        self.barrier.wait(timeout=3)


def trainer(rank, *, ready_groups=2, loan=True, publish_failure=False):
    instances = [SimpleNamespace(instance_id=name) for name in ("engine-a", "engine-b")]
    calls = []
    handles = [SimpleNamespace(instance_id=instance.instance_id, is_ready=not loan)
               for instance in instances]
    executor = SimpleNamespace(
        _lent_rollout={"engines": instances} if loan else None,
        _release_all_hybrid_workers=lambda **kwargs: calls.append("release-hybrids"),
        _rollout_lending_resume_pending=True,
    )
    state = SimpleNamespace(
        is_main_process=rank == 0, control=SimpleNamespace(rank=rank),
        runtime_elastic_executor=executor if rank == 0 else None,
        config=SimpleNamespace(
            n_samples=4, heterogeneous_rollout=SimpleNamespace(instances=instances),
            global_resource_planner=SimpleNamespace(
                enabled=True, runtime_reconfigure_training=True,
                hybrid_worker_launch_enabled=True, runtime_training_pool_plan_only=False,
            ),
        ),
        rollout_engine=SimpleNamespace(scheduler=SimpleNamespace(_instances=handles)),
        dispatcher=SimpleNamespace(peek_ready_data=lambda: []),
        _pending_grpo_groups={f"group-{i}": [
            {"grpo_group_id": f"group-{i}", "grpo_rollout_index": index}
            for index in range(4)] for i in range(ready_groups)},
        _consumed_grpo_group_ids=set(), _source_batch_sample_count=lambda: 8,
        _is_rollout_source=lambda: rank == 0,
        _pause_rollout_pipeline=lambda: calls.append("pause"),
        _resume_rollout_pipeline=lambda: calls.append("resume"),
        _refresh_nonblocking_elastic_membership=lambda: calls.append("refresh-membership"),
        _rebind_rollout_engine_from_config=lambda **kwargs: calls.append("rebind"),
        train_engine=SimpleNamespace(
            optimizer_revision=23, _elastic_step_hybrid_workers=("hybrid-old",),
            set_version=lambda version: calls.append(("set-version", version)),
        ),
        calls=calls, handles=handles,
    )

    def publish(*, target_version):
        assert target_version == 23
        assert all(not handle.is_ready for handle in handles)
        assert state._restored_rollouts_pending_publication == {"engine-a", "engine-b"}
        calls.append("publish")
        if publish_failure:
            raise RuntimeError("weight install failed")

    state._sync_weights = publish
    return state


def run_ranks(function, trainers):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(function, state) for state in trainers]
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=5))
            except Exception as exc:
                results.append(exc)
        return results


def install_transport(monkeypatch):
    collectives = Collectives()
    monkeypatch.setattr(lifecycle, "dist", collectives)

    def return_loan(executor):
        executor._lent_rollout = None
        return ["new-service-a", "new-service-b"]

    monkeypatch.setattr(rollout_lending, "return_rollout", return_loan)
    return collectives


def test_keep_whole_pool_lent_while_complete_batch_ready(monkeypatch):
    collectives = install_transport(monkeypatch)
    ranks = [trainer(rank) for rank in range(2)]
    assert run_ranks(lifecycle.ensure_batch_supply, ranks) == [None, None]
    assert all(state.calls == [] for state in ranks)
    assert ranks[0].runtime_elastic_executor._lent_rollout is not None
    assert len([call for call in collectives.calls if call[1] == "gather"]) == 2


def test_return_pool_before_supply_wait_then_publish_before_reopen(monkeypatch):
    install_transport(monkeypatch)
    ranks = [trainer(rank, ready_groups=1) for rank in range(2)]
    assert run_ranks(lifecycle.ensure_batch_supply, ranks) == [None, None]
    for state in ranks:
        assert state.calls.index("pause") < state.calls.index("publish")
        assert state.calls.index("refresh-membership") < state.calls.index("publish")
        assert state.calls.index("publish") < state.calls.index("rebind") < state.calls.index("resume")
        assert all(handle.is_ready for handle in state.handles)
        assert state._restored_rollouts_pending_publication == set()
        assert state.train_engine._elastic_step_hybrid_workers == ()
        assert ("set-version", 23) in state.calls
    assert ranks[0].calls.index("release-hybrids") < ranks[0].calls.index("publish")


def test_supply_restore_uses_publication_hook_on_every_rank(monkeypatch):
    install_transport(monkeypatch)
    ranks = [trainer(rank, ready_groups=1) for rank in range(2)]
    for state in ranks:
        def restore(state=state):
            state.calls.append("invalidate-old-publication")
            return lifecycle.restore_for_publication(state)
        state._restore_lent_rollouts_for_publication = restore
    assert run_ranks(lifecycle.ensure_batch_supply, ranks) == [None, None]
    for state in ranks:
        assert state.calls.index("pause") < state.calls.index("invalidate-old-publication")
        assert state.calls.index("invalidate-old-publication") < state.calls.index("publish")


def test_restore_failure_is_broadcast_to_every_rank_without_reopen(monkeypatch):
    install_transport(monkeypatch)

    def failed_return(executor):
        raise RuntimeError("service restart failed")

    monkeypatch.setattr(rollout_lending, "return_rollout", failed_return)
    ranks = [trainer(rank, ready_groups=1) for rank in range(2)]
    results = run_ranks(lifecycle.ensure_batch_supply, ranks)
    assert all(isinstance(result, RuntimeError) and "service restart failed" in str(result)
               for result in results)
    assert all("publish" not in state.calls and "resume" not in state.calls for state in ranks)
    assert all(not handle.is_ready for state in ranks for handle in state.handles)


def test_failed_publication_keeps_restored_engines_closed(monkeypatch):
    install_transport(monkeypatch)
    ranks = [trainer(rank, ready_groups=1, publish_failure=True) for rank in range(2)]
    results = run_ranks(lifecycle.ensure_batch_supply, ranks)
    assert all(isinstance(result, RuntimeError) and "weight install failed" in str(result)
               for result in results)
    assert all("resume" not in state.calls and "rebind" not in state.calls for state in ranks)
    assert all(not handle.is_ready for state in ranks for handle in state.handles)


def test_one_rank_publication_failure_prevents_every_rank_reopening(monkeypatch):
    install_transport(monkeypatch)
    ranks = [trainer(0, ready_groups=1, publish_failure=True),
             trainer(1, ready_groups=1, publish_failure=False)]
    results = run_ranks(lifecycle.ensure_batch_supply, ranks)
    assert all(isinstance(result, RuntimeError) and "weight install failed" in str(result)
               for result in results)
    assert all("resume" not in state.calls and "rebind" not in state.calls for state in ranks)
    assert all(not handle.is_ready for state in ranks for handle in state.handles)


def test_any_dp_source_shortage_returns_pool(monkeypatch):
    install_transport(monkeypatch)
    ranks = [trainer(0), trainer(1, ready_groups=1)]
    ranks[1]._is_rollout_source = lambda: True
    assert run_ranks(lifecycle.ensure_batch_supply, ranks) == [None, None]
    assert all("publish" in state.calls for state in ranks)


def test_partial_loan_keeps_available_rollout_supply_running(monkeypatch):
    collectives = install_transport(monkeypatch)
    ranks = [trainer(rank, ready_groups=0) for rank in range(2)]
    ranks[0].runtime_elastic_executor._lent_rollout["engines"] = [SimpleNamespace(instance_id="engine-a")]
    assert run_ranks(lifecycle.ensure_batch_supply, ranks) == [None, None]
    assert all(state.calls == [] for state in ranks)
    assert not any(call[1] == "gather" for call in collectives.calls)


def test_disabled_resource_management_avoids_collectives(monkeypatch):
    state = trainer(0, loan=False)
    state.config.global_resource_planner.enabled = False

    class ForbiddenCollectives:
        def __getattr__(self, name):
            raise AssertionError(f"disabled resource management entered dist.{name}")

    monkeypatch.setattr(lifecycle, "dist", ForbiddenCollectives())
    lifecycle.ensure_batch_supply(state)
    assert state.calls == []


def test_reopen_without_pending_install_is_noop():
    state = trainer(0)
    lifecycle.reopen_after_publication(state)
    assert all(not handle.is_ready for handle in state.handles)


def test_loan_publication_does_not_reenable_unrelated_unhealthy_engine():
    state = trainer(0)
    unhealthy = SimpleNamespace(instance_id="unhealthy-other", is_ready=False)
    state.handles.append(unhealthy)
    lifecycle.apply_availability(state, ("engine-a", "engine-b"))
    state._restored_rollouts_pending_publication = {"engine-a", "engine-b"}
    lifecycle.reopen_after_publication(state)
    assert all(handle.is_ready for handle in state.handles[:2])
    assert unhealthy.is_ready is False


def test_pending_hybrid_activation_gets_post_boundary_catchup_even_without_grp_gain():
    from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
    events = []
    current_plan = object()
    state = SimpleNamespace(
        config=object(),
        runtime_elastic_executor=SimpleNamespace(has_pending_boundary_work=lambda: True),
        global_resource_planner=SimpleNamespace(
            _history=[], _build_current_plan=lambda config, history: current_plan),
        _run_global_resource_planner=lambda *args: pytest.fail("pending join skipped catchup"),
        _runtime_reconfiguration_boundary=lambda *args: events.append(("noop", args)),
    )

    def apply(step, decision, stats):
        events.append((step, decision))
        state._runtime_boundary_published = True

    state._apply_global_resource_planner_decision = apply
    AsyncRLTrainer._run_global_resource_planner_at_boundary(state, 9, [], {})
    assert len(events) == 1
    step, decision = events[0]
    assert step == 9
    assert decision.reason == "hybrid_boundary_maintenance"
    assert decision.should_reconfigure is True
    assert decision.resource_should_reconfigure is False
    assert decision.current_plan is decision.candidate_plan is current_plan
    assert state._runtime_post_boundary is None


def test_no_pending_activation_keeps_normal_grp_post_boundary():
    from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
    events = []
    state = SimpleNamespace(
        runtime_elastic_executor=SimpleNamespace(has_pending_boundary_work=lambda: False),
        _run_global_resource_planner=lambda *args: events.append("planner"),
        _runtime_reconfiguration_boundary=lambda step, point: events.append((step, point)),
    )
    AsyncRLTrainer._run_global_resource_planner_at_boundary(state, 9, [], {})
    assert events == ["planner", (9, "post")]
    assert state._runtime_post_boundary is None


def test_repartition_publication_uses_new_topology_before_client_reopen():
    from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
    state = SimpleNamespace(
        _use_heterogeneous=True,
        rollout_engine=SimpleNamespace(instance_configs=[
            {"instance_id": "old-engine-a", "tp_degree": 2},
            {"instance_id": "old-engine-b", "tp_degree": 2},
        ]),
        config=SimpleNamespace(heterogeneous_rollout=SimpleNamespace(instances=[
            SimpleNamespace(instance_id="new-engine", tp=4),
        ])),
    )
    world_size, offsets = AsyncRLTrainer._nccl_rollout_topology(state)
    assert world_size == 5
    assert offsets == {"new-engine": 0}


def test_resource_repartition_publishes_current_config_targets(monkeypatch):
    from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor
    from RL_Framework.infra.elastic import core_reconfiguration
    commands = []
    monkeypatch.setattr(core_reconfiguration, "dispatch_core_command", lambda executor, operation, **kwargs:
                        commands.append((operation, kwargs)))
    state = SimpleNamespace(config=SimpleNamespace(heterogeneous_rollout=SimpleNamespace(instances=[
        SimpleNamespace(instance_id="new-engine", tp=4, gpus=[0, 1, 2, 3], host="rollout-node", port=8200),
    ])))
    RuntimeElasticExecutor._publish_current_rollout_weights(state)
    assert commands == [("publish", {"target": {"instances": [
        {"instance_id": "new-engine", "tp": 4, "gpus": [0, 1, 2, 3],
         "host": "rollout-node", "port": 8200},
    ]}})]


def test_subset_publication_leaves_routing_to_rollback_owner():
    state = trainer(0)
    lifecycle.apply_availability(state, ('engine-a','engine-b'))
    state._publication_target_ids = {'engine-b'}
    state._restored_rollouts_pending_publication = {'engine-b'}
    lifecycle.reopen_after_publication(state)
    assert state._rollout_unavailable_ids == {'engine-a','engine-b'}
    assert state._restored_rollouts_pending_publication == {'engine-b'}
    assert all(not handle.is_ready for handle in state.handles)
    assert state.runtime_elastic_executor._rollout_lending_resume_pending

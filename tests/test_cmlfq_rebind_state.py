"""Online CMLFQ evidence survives the engine lifecycle used by weight sync."""

from unittest.mock import patch

import pytest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import CausalPrefixTree, Trajectory
from RL_Framework.infra.scheduling.cmlfq_scheduler import CMLFQScheduler
from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor


EVENT = {"tool_type": "search", "output": "ok", "payload_tokens": 2}
STATE = DefaultToolStateExtractor().extract(EVENT)


def config_for(kind, tree_path=""):
    config = AsyncRLConfig(model_path="not-loaded")
    config.heterogeneous_rollout.instances = []
    sched = config.heterogeneous_rollout.scheduling
    sched.scheduler_type = kind
    sched.cmlfq_buckets = {
        "short": {"tp_degrees": [1], "max_tokens": 5000},
        "long": {"tp_degrees": [4], "max_tokens": 50000},
    }
    sched.cmlfq_tree_path = tree_path
    sched.cmlfq_tree_persist_interval = 3
    sched.cmlfq_rebuild_interval = 4
    return config


@pytest.mark.parametrize("kind", ["cmlfq", "cmlfq_cost"])
@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("lifecycle", ["from_config", "reconfigure"])
def test_online_window_and_rebuild_follow_real_engine_rebind(kind, persist, lifecycle, tmp_path):
    config = config_for(kind, str(tmp_path / "tree.json") if persist else "")
    engine = HeterogeneousRolloutEngine.from_config(config)
    # Offline evidence must survive rebind but should be retired at the first
    # periodic rebuild, which intentionally uses only the online window.
    engine.scheduler.prefix_tree.insert(Trajectory("offline", [STATE], [2000]))
    rebuilt = []
    original = CausalPrefixTree.rebuild

    def observe(tree, trajectories):
        rebuilt.append([trajectory.prompt_id for trajectory in trajectories])
        return original(tree, trajectories)

    with patch.object(CausalPrefixTree, "rebuild", observe):
        for step in range(9):
            engine.scheduler.register_instance(0, "short", 1)
            engine.notify_epoch_start(step)
            request = engine.begin_cmlfq_request(f"p-{step}", 100)
            engine.route_cmlfq_tool_return(request, EVENT, 100)
            engine.finish_cmlfq_request(request, 1000 + step)
            old = engine.scheduler
            if lifecycle == "from_config":
                engine = HeterogeneousRolloutEngine.from_config(
                    config, carry_scheduler_state_from=old,
                )
            else:
                engine.reconfigure_from_plan(None, config)
            current = engine.scheduler
            assert current._current_epoch == step
            assert current._tree_update_count == step + 1
            assert current._tool_return_count == step + 1
            assert len(current._recent_trajectories) == step + 1
            assert current.prefix_tree is not old.prefix_tree
            assert current._recent_trajectories is not old._recent_trajectories
            assert current.prefix_tree.lookup(f"p-{step}", [STATE]) is not None
            if step < 4:
                assert current.prefix_tree.lookup("offline", [STATE]) is not None
            engine.notify_epoch_end(step)

    assert rebuilt == [[f"p-{n}" for n in range(5)], [f"p-{n}" for n in range(9)]]
    assert engine.scheduler.prefix_tree.lookup("offline", [STATE]) is None
    assert engine.scheduler.prefix_tree.get_stats()["total_insertions"] == 9


def test_export_and_repeated_import_do_not_alias_or_round_evidence():
    scheduler = CMLFQScheduler(max_recent_trajectories=3)
    scheduler.prefix_tree.insert(Trajectory("precise", [STATE], [1 / 3]))
    scheduler._recent_trajectories = [Trajectory(str(i), [STATE], [i]) for i in range(3)]
    state = scheduler.export_learned_state()
    scheduler._recent_trajectories[0].total_remaining_lengths[0] = 999
    scheduler.prefix_tree.lookup("precise", [STATE]).remaining_lengths.append(999)

    restored = CMLFQScheduler(max_recent_trajectories=2)
    restored.import_learned_state(state)
    other = CMLFQScheduler()
    other.import_learned_state(state)
    assert [traj.prompt_id for traj in restored._recent_trajectories] == ["1", "2"]
    assert restored.prefix_tree.lookup("precise", [STATE]).mean_remaining_length == 1 / 3
    assert restored.prefix_tree.lookup("precise", [STATE]).remaining_lengths == [1 / 3]
    restored._recent_trajectories[0].total_remaining_lengths[0] = -1
    restored.prefix_tree.lookup("precise", [STATE]).remaining_lengths.append(-1)
    assert other._recent_trajectories[1].total_remaining_lengths == [1]
    assert other.prefix_tree.lookup("precise", [STATE]).remaining_lengths == [1 / 3]


def test_non_cmlfq_state_keeps_preloaded_tree():
    scheduler = CMLFQScheduler()
    scheduler.prefix_tree.insert(Trajectory("offline", [STATE], [100]))
    scheduler.import_learned_state({"output_ema": {"short": 42}})
    assert scheduler.prefix_tree.lookup("offline", [STATE]) is not None
    assert scheduler._output_ema["short"] == 42

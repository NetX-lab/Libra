"""Fixed-B tests through production trainer collection, dispatch and train paths."""
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL_Framework.infra.elastic.batch_partition import (
    complete_group_sample_counts, partition_complete_grpo_groups,
)
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def trajectories(prefix, count, n_samples=4):
    return [dict(grpo_group_id=f"{prefix}-g{i // n_samples}", rollout_index=i % n_samples,
                 rewards=torch.tensor([float(i % n_samples)]), input_ids=torch.ones(1, 2, dtype=torch.long))
            for i in range(count)]


@pytest.mark.parametrize("batch,replicas,expected", [(16, 3, [8, 4, 4]), (8, 4, [4, 4, 0, 0])])
def test_balances_whole_groups_including_zero_assignments(batch, replicas, expected):
    assert complete_group_sample_counts(batch, 4, replicas) == expected
    original = trajectories("global", batch)
    assignment = partition_complete_grpo_groups(original, 4, [str(i) for i in range(replicas)])
    assert [len(part) for part in assignment.values()] == expected
    assert [id(x) for part in assignment.values() for x in part] == [id(x) for x in original]
    assert all(len({x["grpo_group_id"] for x in part}) * 4 == len(part) for part in assignment.values())


def test_rejects_duplicate_member_and_incomplete_group():
    batch = trajectories("global", 8)
    batch[1]["rollout_index"] = 0
    with pytest.raises(ValueError, match="duplicate GRPO members"):
        partition_complete_grpo_groups(batch, 4, ["dp0"])
    with pytest.raises(ValueError, match="complete GRPO groups"):
        partition_complete_grpo_groups(batch[:-1], 4, ["dp0"])


class Engine:
    def __init__(self, dp_rank=0, dp_size=1, source=True):
        self.dp_rank, self.dp_size, self.source = dp_rank, dp_size, source
        self.assignments, self.updated = [], []
        self.optimizer_revision = 0

    def get_data_parallel_world_size(self):
        return self.dp_size

    def get_data_parallel_rank(self):
        return self.dp_rank

    def is_batch_source(self):
        return self.source

    def get_version(self):
        return 0

    def set_elastic_batch_partition(self, local_count, global_count, hybrid_counts):
        self.assignments.append((local_count, global_count, hybrid_counts))

    def grpo_update(self, batch, ppo_epochs):
        self.updated.append([x["grpo_group_id"] for x in batch])
        self.optimizer_revision += 1
        return {"loss": 0., "pg_loss": 0., "kl": 0.}


def make_trainer(tmp_path, rank=0, dp_size=1, dp_rank=0, source=True, hybrids=()):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.rank, trainer.local_rank = rank, rank
    trainer.world_size, trainer.is_main_process = dp_size, rank == 0
    trainer.config = SimpleNamespace(
        batch_size=16, n_samples=4, total_steps=2, micro_batch_size=1,
        ppo_epochs=1, recompute_logprobs=False, sync_interval=0, save_interval=0,
        log_dir=str(tmp_path), enable_rollout_tracing=False,
        heterogeneous_rollout=SimpleNamespace(instances=[]),
        global_resource_planner=SimpleNamespace(hybrid_worker_task_dir=str(tmp_path),
                                               hybrid_lockstep_gradient_sync=True),
    )
    trainer.train_engine = Engine(dp_rank, dp_size, source)
    trainer.start_step = 0
    trainer.runtime_elastic_executor = None
    trainer._use_heterogeneous = False
    trainer._bounded_rollout_production = False
    trainer.staleness_manager = None
    trainer._pending_hybrid_training_batches = {}
    trainer._elastic_membership_run_id = "fixed-b-test"
    trainer.setup = lambda workflow: None
    trainer._create_data_generator = lambda dataset, workflow: iter(())
    trainer._prepare_elastic_training_step = lambda step: list(hybrids)
    for name in ("_trace_train_phase", "_configure_sync_aware_rollout_prefetch",
                 "_runtime_reconfiguration_boundary", "_wait_for_all_ranks_after_rollout",
                 "_log_to_wandb", "_record_history_step", "_run_global_resource_planner_at_boundary",
                 "_run_periodic_evaluation", "_cleanup"):
        setattr(trainer, name, lambda *args, **kwargs: None)
    return trainer


@pytest.mark.parametrize("hybrid_count,expected", [(0, 16), (1, 8), (5, 4)])
def test_actual_train_collects_fixed_batch_and_dispatches_complete_groups(tmp_path, hybrid_count, expected):
    hybrids = [f"hybrid{i}" for i in range(hybrid_count)]
    trainer = make_trainer(tmp_path, hybrids=hybrids)
    (tmp_path / "membership").mkdir()
    for worker in hybrids:
        (tmp_path / "membership" / f"{worker}.json").write_text(json.dumps(
            dict(run_id="fixed-b-test", role="hybrid_training", membership_epoch=7)))
    collected, histories = [], []

    def collect(data_gen, batch_size):
        collected.append(batch_size)
        return trajectories(f"step{len(collected)}", batch_size)

    trainer._collect_complete_grpo_batch = collect
    trainer._record_history_step = lambda step, batch, stats, *args: histories.append((len(batch), stats.copy()))
    trainer.train(None, dataset=[1])
    assert collected == [16, 16]
    assert [len(batch) for batch in trainer.train_engine.updated] == [expected, expected]
    assert [row[0] for row in histories] == [16, 16]
    assert [row[1]["n_trajectories"] for row in histories] == [16, 16]
    for step in range(2):
        tasks = [torch.load(tmp_path / f"{worker}.step_{step}.pt", weights_only=False) for worker in hybrids]
        assert sum(len(task["trajectories"]) for task in tasks) + expected == 16
        assert all(task["global_sample_count"] == 16 for task in tasks)
        assert all(task["sample_count"] == len(task["trajectories"]) for task in tasks)
        assert all(task["state_version"] == step for task in tasks)
        all_groups = trainer.train_engine.updated[step] + [x["grpo_group_id"] for task in tasks for x in task["trajectories"]]
        assert len(all_groups) == 16
        assert len(set(all_groups)) == 4
        assert all(all_groups.count(group) == 4 for group in set(all_groups))


def _gloo_collection_worker(rank, rendezvous, directory):
    dist.init_process_group("gloo", rank=rank, world_size=4,
                            init_method=f"file://{rendezvous}", timeout=timedelta(seconds=30))
    # Gloo's async all-gather destructor can need the GIL to free tensors.
    # Keep them alive while the Python thread holds the GIL and joins its workers.
    from torch.distributed import distributed_c10d

    all_gather = distributed_c10d.all_gather
    collective_tensors = []

    def gather_with_owned_tensors(outputs, tensor, *args, **kwargs):
        collective_tensors.extend(outputs)
        collective_tensors.append(tensor)
        return all_gather(outputs, tensor, *args, **kwargs)

    distributed_c10d.all_gather = gather_with_owned_tensors
    try:
        path = Path(directory)
        results = []
        # Each case is a separately initialized fixed Core layout. These do
        # not exercise or claim support for runtime Core topology changes.
        for global_count, compute_dp_size in ((16, 2), (16, 4), (16, 1), (8, 4)):
            width = 4 // compute_dp_size
            trainer = make_trainer(path, rank, compute_dp_size, rank // width, rank % width == 0)
            trainer.world_size = 4
            trainer.config.batch_size = global_count
            trainer._initialize_rollout_source_layout()
            trainer._collect_complete_grpo_batch = lambda generator, count: trajectories(
                f"source{trainer.train_engine.dp_rank}", count)
            batch, participants = trainer._collect_fixed_global_batch(None, trainer._source_batch_sample_count(), [])
            own = trainer._assign_fixed_global_batch(0, batch, participants)
            results.append({"global_groups": [x["grpo_group_id"] for x in batch],
                            "own_groups": [x["grpo_group_id"] for x in own]})
        torch.save(results, path / f"rank{rank}.pt")
        trainer._collect_complete_grpo_batch = lambda generator, count: trajectories("partial", 0 if trainer.train_engine.dp_rank == 1 else count)
        with pytest.raises(RuntimeError, match="incomplete fixed batch"):
            trainer._collect_fixed_global_batch(None, trainer._source_batch_sample_count(), [])
    finally:
        distributed_c10d.all_gather = all_gather
        dist.destroy_process_group()


def test_real_gloo_dp_sources_survive_tp_followers(tmp_path):
    mp.spawn(_gloo_collection_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=4, join=True)
    results = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in range(4)]
    for topology_index, (global_count, dp_size) in enumerate(((16, 2), (16, 4), (16, 1), (8, 4))):
        topology = [result[topology_index] for result in results]
        assert all(len(set(item["global_groups"])) == global_count // 4 for item in topology)
        representatives = topology[::4 // dp_size]
        assigned = [group for item in representatives for group in item["own_groups"]]
        assert len(assigned) == global_count
        assert set(assigned) == set(topology[0]["global_groups"])
        assert all(assigned.count(group) == 4 for group in set(assigned))
    assert results[0][0] == results[1][0]
    assert results[2][0] == results[3][0]
    assert results[0][2] == results[3][2]
    assert results[2][3]["own_groups"] == results[3][3]["own_groups"] == []


def test_incomplete_source_batch_fails_before_optimizer_update(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._collect_complete_grpo_batch = lambda generator, count: trajectories("partial", 12)
    with pytest.raises(RuntimeError, match="incomplete fixed batch"):
        trainer.train(None, dataset=[1])
    assert not trainer.train_engine.updated


def test_logical_sampler_cursor_survives_rollout_client_rebind(tmp_path):
    trainer = make_trainer(tmp_path, dp_size=2)
    trainer._rollout_source_world_size = 2
    trainer._rollout_source_rank = 0
    trainer._task_counter = 0
    dataset = [{"id": index} for index in range(12)]
    stream = AsyncRLTrainer._create_data_generator(trainer, dataset, None)
    first = [next(stream) for _ in range(4)]
    rebound = []
    trainer.rollout_engine = SimpleNamespace(reconfigure_from_plan=lambda plan, config: rebound.append(config))
    trainer._rebind_rollout_engine_from_config(reason="runtime_topology_change")
    assert rebound == [trainer.config]
    second = [next(stream) for _ in range(4)]
    assert [task.task_id for task in first + second] == list(range(8))
    assert {task.group_id for task in first} == {"dp0:epoch0:sample0"}
    assert {task.group_id for task in second} == {"dp0:epoch0:sample1"}
    assert first[0].data != second[0].data
    assert trainer._is_rollout_source()
    assert trainer._source_batch_sample_count() == 8


def test_rebind_preserves_admitted_results_and_incomplete_group_members(tmp_path):
    trainer = make_trainer(tmp_path)
    pending = {"group": trajectories("part", 3)}
    trainer._pending_grpo_groups = pending
    trainer.dispatcher = SimpleNamespace(reset_after_reconfigure=lambda: pytest.fail("valid work discarded"))
    rebound = []
    trainer.rollout_engine = SimpleNamespace(reconfigure_from_plan=lambda plan, config: rebound.append(config))
    trainer._rebind_rollout_engine_from_config(reason="runtime_topology_change")
    assert rebound == [trainer.config]
    assert trainer._pending_grpo_groups is pending
    assert len(pending["group"]) == 3


def test_dispatch_rejects_membership_generation_change_before_update(tmp_path):
    trainer = make_trainer(tmp_path, hybrids=["hybrid0"])
    trainer.train_engine.elastic_gradient_domain = SimpleNamespace(
        membership_state=lambda: {"membership_epochs": {"hybrid0": 7}})
    (tmp_path / "membership").mkdir()
    (tmp_path / "membership" / "hybrid0.json").write_text(json.dumps(
        dict(run_id="fixed-b-test", role="hybrid_training", membership_epoch=8)))
    trainer._collect_complete_grpo_batch = lambda generator, count: trajectories("batch", count)
    with pytest.raises(RuntimeError, match="generation changed"):
        trainer.train(None, dataset=[1])
    assert not trainer.train_engine.updated
    assert not (tmp_path / "hybrid0.step_0.pt").exists()


def dispatch_trainer(tmp_path):
    trainer = make_trainer(tmp_path, hybrids=["hybrid0"])
    (tmp_path / "membership").mkdir()
    (tmp_path / "membership" / "hybrid0.json").write_text(json.dumps(
        dict(run_id="fixed-b-test", role="hybrid_training", membership_epoch=7)))
    trainer._pending_hybrid_training_batches = {"hybrid0": trajectories("batch", 4)}
    return trainer


@pytest.mark.parametrize("suffix", ["", ".done", ".failed", ".error", ".cancelled"])
def test_dispatch_preserves_existing_task_or_terminal_file(tmp_path, suffix):
    trainer = dispatch_trainer(tmp_path)
    existing = tmp_path / ("hybrid0.step_0.pt" + suffix)
    existing.write_bytes(b"previous task")
    with pytest.raises(RuntimeError, match="already exists"):
        trainer._dispatch_hybrid_training_batches(0)
    assert existing.read_bytes() == b"previous task"
    assert not (tmp_path / "hybrid0.step_0.pt.tmp").exists()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_dispatch_registers_before_publish_and_retires_unpublished_failure(tmp_path, monkeypatch, cleanup_fails):
    trainer = dispatch_trainer(tmp_path)
    pending = {}

    def record(worker, data, *task_path):
        pending[worker] = data

    def forget(worker, data):
        assert pending[worker] is data
        pending.pop(worker)

    trainer.runtime_elastic_executor = SimpleNamespace(
        hybrid_task_identity=lambda worker: {"launch_owner": "owner"},
        record_hybrid_task_assignment=record, forget_hybrid_task_assignment=forget)

    def fail_publish(source, target):
        assert source.exists() and "hybrid0" in pending
        assert not target.exists()
        raise OSError("publish denied")

    monkeypatch.setattr(Path, "replace", fail_publish)
    if cleanup_fails:
        def fail_cleanup(path, **kwargs):
            raise OSError("cleanup denied")
        monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError, match="publish denied"):
        trainer._dispatch_hybrid_training_batches(0)
    assert pending == {}
    assert (tmp_path / "hybrid0.step_0.pt.tmp").exists() is cleanup_fails
    assert not (tmp_path / "hybrid0.step_0.pt").exists()

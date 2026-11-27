"""Exercise the fixed R2E example at its dataset and publication boundaries."""

import asyncio
import importlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL_Framework.config import load_config
from RL_Framework.scripts.run_r2e_qwen3_14b_16gpu import bind_placement, train_command


def test_shipped_preset_binds_one_core_node_and_three_rollout_nodes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(str(Path(__file__).resolve().parents[1] / "configs/r2e_qwen3_14b_16gpu.yaml"))
    nodes = ["core", "rollout1", "rollout2", "rollout3"]
    bind_placement(config, nodes, tmp_path / "model", tmp_path)
    instances = config.heterogeneous_rollout.instances
    assert [service.tp for service in instances] == [4, 2, 2, 1, 1, 1, 1]
    assert {(s.host, gpu) for s in instances for gpu in s.gpus} == {
        (host, gpu) for host in nodes[1:] for gpu in range(4)
    }
    assert (config.train_tp_size, config.train_dp_size, config.grpo_objective) == (4, 1, "decoupled")
    assert (config.total_steps, config.batch_size, config.n_samples) == (100, 96, 4)
    argv = train_command(nodes[0], tmp_path / "planned.yaml", tmp_path / "tasks.jsonl")
    assert "--nodelist=core" in argv and "--nproc-per-node=4" in argv
    assert argv[argv.index("--module") + 1] == "RL_Framework.examples.r2e_gym_1938329"


def _example():
    name = "examples.r2e_gym_1938329"
    assert importlib.util.find_spec(name) is not None, "Missing formal R2E example entrypoint"
    return importlib.import_module(name)


def test_heldout_tasks_are_fixed_and_excluded_from_training():
    example = _example()
    rows = [{"repo_name": "repo", "commit_hash": str(index)} for index in range(1000)]
    rows.insert(1, rows[0])
    train, heldout = example.split_task_indices(rows)
    train_ids = {rows[index]["commit_hash"] for index in train}
    heldout_ids = [rows[index]["commit_hash"] for index in heldout]
    assert len(train) == len(train_ids) == 800
    assert len(heldout) == len(set(heldout_ids)) == 20
    assert train_ids.isdisjoint(heldout_ids)
    assert heldout_ids[:3] == ["930", "736", "739"]
    assert example.split_task_indices(rows) == (train, heldout)


def test_evaluation_rejects_a_generation_from_another_revision(monkeypatch):
    example = _example()

    async def generate(*args, **kwargs):
        return {"sample_revision": 21}

    monkeypatch.setattr(example.VLLMRolloutEngine, "generate", generate)
    engine = example._EvaluationEngine("localhost:8000", "model", revision=20)
    with pytest.raises(RuntimeError, match="21.*20"):
        asyncio.run(engine.generate("task"))


def _describe_rank_zero_failure(rank, store, result_dir):
    example = _example()
    dist.init_process_group(
        "gloo", init_method=store.as_uri(), rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        trainer = object.__new__(example.HeldoutTrainer)
        trainer.is_main_process = rank == 0
        trainer.control = None

        def evaluate():
            raise OSError("heldout output unavailable")

        with pytest.raises(RuntimeError, match="OSError: heldout output unavailable") as error:
            trainer._rank_zero(evaluate)
        (result_dir / str(rank)).write_text(str(error.value))
    finally:
        dist.destroy_process_group()


def test_rank_zero_exception_reaches_every_rank(tmp_path):
    mp.start_processes(
        _describe_rank_zero_failure, args=(tmp_path / "store", tmp_path),
        nprocs=2, start_method="fork",
    )
    assert [(tmp_path / str(rank)).read_text() for rank in range(2)] == [
        "OSError: heldout output unavailable", "OSError: heldout output unavailable",
    ]


def test_evaluation_reuses_the_published_survivor_and_writes_its_report(tmp_path, monkeypatch):
    example = _example()
    trainer = object.__new__(example.HeldoutTrainer)
    trainer.is_main_process, trainer.control = True, None
    trainer.config = SimpleNamespace(
        log_dir=tmp_path,
        model_path="model",
        heterogeneous_rollout=SimpleNamespace(
            enabled=True,
            vllm_host="localhost",
            instances=[
                SimpleNamespace(instance_id="kept", host="localhost", port=8000, tp=1),
                SimpleNamespace(instance_id="borrowed", host="localhost", port=8001, tp=1),
            ],
        ),
    )
    trainer.train_engine = SimpleNamespace(optimizer_revision=20)
    trainer._rollout_unavailable_ids = {"borrowed"}
    trainer.runtime_elastic_executor = SimpleNamespace(
        _lent_rollout={"engines": [SimpleNamespace(instance_id="borrowed")]},
    )
    target = dict(instance_id="kept", endpoint="localhost:8000", tp_size=1)
    state = dict(
        server_instance_id="active-server", server_generation=21, snapshot_id="snapshot-20",
        optimizer_revision=20, tp_size=1, server_generation_verified=True, closed=False,
    )
    trainer.last_publication = dict(
        optimizer_revision=20, targets=[target], target_ids=["kept"],
        acknowledgements=[dict(
            instance_id="kept", endpoint=target["endpoint"], expected_tp_size=1,
            **{key: state[key] for key in (
                "server_instance_id", "server_generation", "snapshot_id", "optimizer_revision",
            )},
        )],
    )
    trainer.heldout = [{"prompt_id": "repo:one"}, {"prompt_id": "repo:two"}]

    class Workflow:
        async def run_episode(self, engine, row, *, version):
            response = await engine.generate(row["prompt_id"])
            assert engine.port == 8000
            return dict(sample_revision=response["sample_revision"], rewards=torch.tensor([0.5]))

        def sampling_seed(self, row, *, version, rollout_index, turn):
            return 42

    async def generate(self, prompt):
        return {"sample_revision": 20}

    monkeypatch.setattr(
        "RL_Framework.infra.sync.publication_session.live_publication_state",
        {"localhost:8000": state}.__getitem__,
    )
    monkeypatch.setattr(example.VLLMRolloutEngine, "generate", generate)
    trainer._run_periodic_evaluation(Workflow(), None, 19)

    report = json.loads((tmp_path / "eval" / "heldout-step-020.json").read_text())
    assert report == {
        "version": 20, "evaluation_seed": 42, "task_count": 2, "reward_mean": 0.5,
        "rows": [{"task_id": "repo:one", "reward": 0.5}, {"task_id": "repo:two", "reward": 0.5}],
    }


def test_training_hook_evaluates_successful_versions_at_fixed_boundaries(monkeypatch):
    example = _example()
    trainer = object.__new__(example.HeldoutTrainer)
    trainer.config = SimpleNamespace(total_steps=100)
    trainer.is_main_process = True
    trainer.control = None
    trainer.train_engine = SimpleNamespace(optimizer_revision=0)
    published = []
    evaluated = []
    monkeypatch.setattr(example.AsyncRLTrainer, "setup", lambda self, workflow: None)
    monkeypatch.setattr(trainer, "_sync_weights_nccl", published.append)
    monkeypatch.setattr(trainer, "_needs_publication", lambda revision: False)
    monkeypatch.setattr(trainer, "_evaluate", lambda workflow, revision: evaluated.append(revision))

    trainer.setup(object())
    for step in range(100):
        trainer.train_engine.optimizer_revision = step + 1
        trainer._run_periodic_evaluation(object(), None, step)

    assert published == [0]
    assert evaluated == [0, 20, 40, 60, 80, 100]

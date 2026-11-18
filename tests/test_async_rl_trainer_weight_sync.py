import json
from types import SimpleNamespace

import pytest

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def test_runtime_topology_refreshes_tp_even_if_endpoint_is_unchanged():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(heterogeneous_rollout=SimpleNamespace(
        instances=[SimpleNamespace(instance_id="rollout", host="node", port=8000, tp=1, gpus=[0])],
    ))
    trainer.rollout_engine = SimpleNamespace(instance_urls=["http://node:8000"])
    rebound = []
    trainer._rebind_rollout_engine_from_config = lambda **kwargs: rebound.append(True)
    trainer._apply_runtime_reconfiguration_state({"instances": [
        {"instance_id": "rollout", "host": "node", "port": 8000, "tp": 2, "gpus": [0, 1]},
    ]})
    instance = trainer.config.heterogeneous_rollout.instances[0]
    assert (instance.tp, instance.gpus) == (2, [0, 1])
    assert rebound == [True]


def _last_update_trainer(leader):
    import torch
    from RL_Framework.config import AsyncRLConfig
    config = AsyncRLConfig(model_path="/unused", train_gpus=2, rollout_gpus=2,
                           n_total_gpus=4, train_tp_size=2, batch_size=4, n_samples=4, micro_batch_size=1,
                           total_steps=1, sync_interval=1, save_interval=0,
                           recompute_logprobs=False, rollout_weight_sync_mode="restart",
                           require_rollout_weight_sync=True)
    trainer = AsyncRLTrainer(config)
    trainer._bounded_rollout_production = True
    trainer.is_main_process = leader
    events = []
    revision = {"optimizer": 0, "published": 0}
    batch = [
        {"grpo_group_id": "a", "rewards": torch.tensor([float(i)]),
         "input_len": 3, "output_len": 2 + i, "input_ids": torch.zeros((1, 5 + i)),
         "total_output_tokens": str(7 + i)}
        for i in range(4)
    ]

    def update(*args, **kwargs):
        revision["optimizer"] = 1
        trainer.train_engine.optimizer_revision = 1
        events.append("update")
        return {}

    trainer.train_engine = SimpleNamespace(
        optimizer_revision=0, grpo_update=update,
        get_version=lambda: revision["published"],
        set_version=lambda version: revision.__setitem__("published", version),
    )
    trainer.dispatcher = SimpleNamespace(
        retire_batch=lambda rows, version: events.append("retire"),
        destroy=lambda: events.append("stop_production"),
        on_publication=lambda version: events.append("new_window"),
    )
    trainer._source_training_batch = batch
    trainer.setup = lambda workflow: None
    trainer._create_data_generator = lambda *args: iter(())
    trainer._source_batch_sample_count = lambda: 4
    trainer._prepare_elastic_training_step = lambda step: []
    trainer._collect_fixed_global_batch = lambda *args: (batch, ["dp0"])
    trainer._assign_fixed_global_batch = lambda step, rows, participants: rows
    trainer._runtime_reconfiguration_boundary = lambda step, point: events.append(point)
    trainer._run_global_resource_planner_at_boundary = lambda *args: events.append("post")
    trainer._sync_weights = lambda **kwargs: events.append("publication")
    trainer._wait_for_all_ranks_before_weight_sync = lambda step: None
    trainer._log_to_wandb = lambda *args: None
    trainer._run_periodic_evaluation = lambda *args: events.append("evaluation")
    trainer._cleanup = lambda **kwargs: None
    return trainer, events, revision


@pytest.mark.parametrize("leader", [True, False])
def test_last_update_runs_post_then_publication_and_stops_production(leader):
    trainer, events, revision = _last_update_trainer(leader)
    trainer.train(None, dataset=[{}])
    assert events == ["pre", "update", "retire", "stop_production", "post", "publication", "evaluation"]
    assert revision["published"] == 1


def test_bounded_history_exports_profile_before_post_and_records_publication_time(tmp_path, monkeypatch):
    from RL_Framework.infra.observability.history_collector import HistoryDataCollector, ResourceConfig
    from RL_Framework.trainer import async_rl_trainer

    trainer, _, _ = _last_update_trainer(True)
    profile = tmp_path / "lengths.jsonl"
    collector = trainer.history_collector = HistoryDataCollector(
        str(tmp_path), flush_interval=1, length_profile_path=str(profile),
    )
    trainer._resource_config_snapshot = ResourceConfig(rollout_gpus=2)
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(async_rl_trainer, "time", SimpleNamespace(time=lambda: clock.now))

    def post(step, batch, stats):
        rows = [json.loads(line) for line in profile.read_text().splitlines()]
        assert [row["output_len"] for row in rows] == [2, 3, 4, 5]
        assert [row["total_output_tokens"] for row in rows] == [7, 8, 9, 10]
        assert collector.get_records() == []
        trainer._resource_config_snapshot = ResourceConfig(rollout_gpus=1)
        clock.now += 1.0

    def publish(**kwargs):
        clock.now += 2.5

    trainer._run_global_resource_planner_at_boundary = post
    trainer._sync_weights = publish
    try:
        trainer.train(None, dataset=[{}])
        records = [json.loads(line) for line in next(tmp_path.glob("history_*.jsonl")).read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["timing"]["weight_sync_time"] == 2.5
        assert records[0]["timing"]["step_total_time"] == 3.5
        assert collector.get_records()[0].resource_config.rollout_gpus == 2
        assert collector.get_records()[0].pipeline_stats.model_version == 1
        assert len(profile.read_text().splitlines()) == 4
    finally:
        collector.finalize()


@pytest.mark.parametrize("install_outcome", ["ack", "error", "no_targets"])
def test_periodic_publication_preserves_work_and_opens_window_only_after_ack(install_outcome):
    trainer = object.__new__(AsyncRLTrainer)
    trainer._bounded_rollout_production = True
    trainer.staleness_manager = SimpleNamespace(group_reservations_enabled=True)
    trainer.config = SimpleNamespace(weight_sync_mode="nccl", rollout_weight_sync_mode="nccl",
        enable_rollout_tracing=False)
    events = []
    trainer.dispatcher = SimpleNamespace(
        on_publication=lambda revision: events.append(("window", revision)),
        pause=lambda: pytest.fail("periodic publication must preserve live group production"),
    )
    trainer.train_engine = SimpleNamespace(set_version=lambda revision: events.append(("version", revision)))
    trainer.is_main_process = True
    trainer._trace_train_phase = lambda *args, **kwargs: None
    trainer._wait_for_all_ranks_before_weight_sync = lambda step: events.append(("ranks", step))
    trainer._restored_rollouts_pending_publication = set()

    def install(*, target_version, reopen_restored):
        assert not reopen_restored
        events.append(("install", target_version))
        if install_outcome == "error":
            raise RuntimeError("install failed")
        if install_outcome == "no_targets":
            return None
        events.append(("all_ack", target_version))
        return {"optimizer_revision": target_version}

    trainer._sync_weights = install
    if install_outcome == "error":
        with pytest.raises(RuntimeError, match="install failed"):
            trainer._periodic_weight_sync(2)
        assert events == [("ranks", 2), ("install", 3)]
    elif install_outcome == "ack":
        trainer._periodic_weight_sync(2)
        assert events == [("ranks", 2), ("install", 3), ("all_ack", 3), ("version", 3), ("window", 3)]
    else:
        trainer._periodic_weight_sync(2)
        assert events == [("ranks", 2), ("install", 3)]


def test_rollout_reload_requests_batch_and_reports_slowest_instance(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    control_dir = tmp_path / "control"
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        rollout_weight_sync_mode="restart",
        require_rollout_weight_sync=True,
        rollout_weight_sync_control_dir=str(control_dir),
        rollout_weight_sync_timeout_s=5.0,
        rollout_weight_reload_method="inplace",
        rollout_weight_reload_strategy="parallel",
        vllm_endpoints="127.0.0.1:8000,127.0.0.1:8001",
        vllm_tp_size=1,
    )
    trainer._use_heterogeneous = False
    trainer.rollout_engine = SimpleNamespace(num_instances=2)
    responded = False

    def respond_to_batch(_seconds):
        nonlocal responded
        request_path = control_dir / "reload_request.json"
        if responded or not request_path.exists():
            return
        request = json.loads(request_path.read_text(encoding="utf-8"))
        for instance_id, total_seconds in zip(
            request["instance_ids"],
            (1.25, 2.5),
        ):
            (control_dir / f"ack_{instance_id}_7.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "version": 7,
                        "checkpoint_path": str(checkpoint_path),
                        "reload_method": "inplace",
                        "reload_strategy": "parallel",
                        "total_seconds": total_seconds,
                    }
                ),
                encoding="utf-8",
            )
        responded = True

    monkeypatch.setattr(
        "RL_Framework.trainer.async_rl_trainer.time.sleep",
        respond_to_batch,
    )

    trainer._reload_rollout_from_checkpoint(str(checkpoint_path), 7)

    request = json.loads(
        (control_dir / "reload_request.json").read_text(encoding="utf-8")
    )
    assert request["reload_method"] == "inplace"
    assert request["reload_strategy"] == "parallel"
    assert request["instance_ids"] == ["instance_0", "instance_1"]
    output = capsys.readouterr().out
    assert "method=inplace" in output
    assert "slowest_reload=2.50s" in output


@pytest.mark.parametrize('rollback', [True, False], ids=['rollback', 'periodic'])
def test_disk_publication_exports_only_its_targets_without_releasing_loan(tmp_path, monkeypatch, rollback):
    from RL_Framework.infra.elastic.core_reconfiguration import _run_operation

    trainer = object.__new__(AsyncRLTrainer)
    trainer.is_main_process = True
    trainer.global_step = 6
    trainer.config = SimpleNamespace(weight_sync_mode='disk', rollout_weight_sync_mode='restart',
        rollout_weight_sync_control_dir=str(tmp_path), rollout_weight_reload_method='inplace',
        rollout_weight_sync_timeout_s=1, sync_path=str(tmp_path), vllm_endpoints='a:8000,b:8000',
        vllm_tp_size=1)
    published = []
    def save(path, version):
        assert version == 6
        (tmp_path / 'v6').mkdir(exist_ok=True)
    trainer.train_engine = SimpleNamespace(train_backend='megatron_core', optimizer_revision=7,
        save_weights=save)
    trainer._cleanup_old_weights = lambda: None
    trainer._restore_lent_rollouts_for_publication = lambda: pytest.fail('rollback released old Hybrid')
    trainer._sync_weights_nccl = lambda *args, **kwargs: pytest.fail('checkpoint path entered NCCL')
    trainer._reopen_restored_rollouts = lambda: pytest.fail('publisher opened routing before caller')
    unavailable = {'instance_0', 'instance_1'} if rollback else {'instance_1'}
    trainer._rollout_unavailable_ids = unavailable
    target = 'instance_1' if rollback else 'instance_0'

    def finish_reload(_seconds):
        request = json.loads((tmp_path / 'reload_request.json').read_text())
        assert request['instance_ids'] == [target]
        assert request['version'] == 7 and request['checkpoint_path'] == str(tmp_path / 'v6')
        if published:
            trainer._write_json_atomic(tmp_path / f'error_{target}_7.json',
                dict(request, instance_id=target, snapshot_id=published[-1], error='previous attempt'))
        trainer._write_json_atomic(tmp_path / f'ack_{target}_7.json', dict(request, instance_id=target))
        published.append(request['snapshot_id'])
    monkeypatch.setattr('RL_Framework.trainer.async_rl_trainer.time.sleep', finish_reload)
    command = dict(operation='publish', target=dict(scope='returned_subset',
        instances=[dict(instance_id='instance_1')]))

    for _ in range(2):
        if rollback:
            result = _run_operation(trainer.train_engine, command, trainer)
            assert result == {'optimizer_revision': 7, 'acknowledged': True}
        else:
            trainer._sync_weights(7, reopen_restored=False)

    assert len(published) == len(set(published)) == 2
    assert trainer._rollout_unavailable_ids == unavailable


def test_serial_reload_strategy_is_preserved_in_batch_request(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    control_dir = tmp_path / "control"
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        rollout_weight_sync_mode="restart",
        require_rollout_weight_sync=True,
        rollout_weight_sync_control_dir=str(control_dir),
        rollout_weight_sync_timeout_s=5.0,
        rollout_weight_reload_method="restart",
        rollout_weight_reload_strategy="serial",
        vllm_endpoints="127.0.0.1:8000",
        vllm_tp_size=1,
    )
    trainer._use_heterogeneous = False
    trainer.rollout_engine = SimpleNamespace(num_instances=1)

    def respond_to_batch(_seconds):
        request = json.loads(
            (control_dir / "reload_request.json").read_text(encoding="utf-8")
        )
        (control_dir / "ack_instance_0_3.json").write_text(
            json.dumps(
                {
                    "instance_id": "instance_0",
                    "version": 3,
                    "checkpoint_path": str(checkpoint_path),
                    "reload_method": "restart",
                    "reload_strategy": request["reload_strategy"],
                    "total_seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "RL_Framework.trainer.async_rl_trainer.time.sleep",
        respond_to_batch,
    )

    trainer._reload_rollout_from_checkpoint(str(checkpoint_path), 3)

    request = json.loads(
        (control_dir / "reload_request.json").read_text(encoding="utf-8")
    )
    assert request["reload_method"] == "restart"
    assert request["reload_strategy"] == "serial"

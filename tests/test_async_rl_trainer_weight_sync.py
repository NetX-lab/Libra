import json
from types import SimpleNamespace

import pytest

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


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


def _last_update_trainer(leader):
    import torch
    from RL_Framework.config import AsyncRLConfig
    config = AsyncRLConfig(model_path="/unused", train_gpus=2, rollout_gpus=2,
                           n_total_gpus=4, train_tp_size=2, batch_size=4, n_samples=4, micro_batch_size=1,
                           total_steps=1, sync_interval=1, save_interval=0,
                           recompute_logprobs=False, rollout_weight_sync_mode="nccl",
                           weight_sync_mode="nccl", require_rollout_weight_sync=True)
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
    trainer._sync_weights = lambda **kwargs: events.append("publication") or True
    trainer._wait_for_all_ranks_before_weight_sync = lambda step: None
    trainer._log_to_wandb = lambda *args: None
    trainer._run_periodic_evaluation = lambda *args: events.append("evaluation")
    trainer._cleanup = lambda **kwargs: None
    return trainer, events, revision


@pytest.mark.parametrize("leader", [True, False])
def test_last_update_stops_production_before_publication_and_post(leader):
    trainer, events, revision = _last_update_trainer(leader)
    trainer.train(None, dataset=[{}])
    assert events == ["pre", "update", "retire", "stop_production", "publication", "post", "evaluation"]
    assert revision["published"] == 1


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

    def install(*, target_version):
        events.append(("install", target_version))
        if install_outcome == "error":
            raise RuntimeError("install failed")
        if install_outcome == "no_targets":
            return None
        events.append(("all_ack", target_version))
        return True

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

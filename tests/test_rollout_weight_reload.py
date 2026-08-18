import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def test_reload_requests_parallel_batch_and_validates_ack(tmp_path, monkeypatch, capsys):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    control_dir = tmp_path / "control"
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        rollout_weight_sync_mode="restart",
        require_rollout_weight_sync=True,
        rollout_weight_sync_control_dir=str(control_dir),
        rollout_weight_sync_timeout_s=5.0,
        rollout_weight_reload_strategy="parallel",
        rollout_weight_reload_method="inplace",
        rollout_weight_sync_poll_interval_s=0.01,
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
            request["instance_ids"], (1.25, 2.5)
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
        "RL_Framework.trainer.async_rl_trainer.time.sleep", respond_to_batch
    )

    trainer._reload_rollout_from_checkpoint(str(checkpoint_path), 7)

    request = json.loads(
        (control_dir / "reload_request.json").read_text(encoding="utf-8")
    )
    assert request["reload_strategy"] == "parallel"
    assert request["reload_method"] == "inplace"
    assert request["instance_ids"] == ["instance_0", "instance_1"]
    assert "slowest_reload=2.50s" in capsys.readouterr().out

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

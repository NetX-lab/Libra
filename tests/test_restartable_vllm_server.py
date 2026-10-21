import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "restartable_vllm_server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "restartable_vllm_server", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_atomic_ack_write_publishes_complete_payload(tmp_path):
    module = _load_module()
    ack_path = tmp_path / "ack_instance_1.json"

    module.write_json_atomic(
        ack_path,
        {"version": 1, "reload_strategy": "parallel"},
    )

    assert json.loads(ack_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "reload_strategy": "parallel",
    }
    assert list(tmp_path.iterdir()) == [ack_path]


def test_parallel_is_default_reload_strategy(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "restartable_vllm_server.py",
            "--instance-id",
            "instance_0",
            "--control-dir",
            "/tmp/control",
            "--health-url",
            "http://127.0.0.1:8000/health",
            "--initial-model",
            "/tmp/model",
            "--",
            "python",
            "server.py",
        ],
    )

    args = module.parse_args()
    assert args.reload_method == "restart"
    assert args.reload_strategy == "parallel"


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    module = _load_module()
    args = SimpleNamespace(instance_id='returned', control_dir=str(tmp_path),
        health_url='http://service/health', initial_model='/model', ready_timeout=1,
        reload_strategy='parallel', reload_method='inplace', poll_interval=.01,
        command=['server'])
    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(module, 'wait_until_ready', lambda *args: None)
    monkeypatch.setattr(module, 'stop_process', lambda *args: None)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    calls = []

    def reload(url, payload, timeout):
        calls.append(payload)
        return dict(service_instance_id='new-service', service_generation=len(calls))

    def stop(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, 'reload_nccl', reload)
    monkeypatch.setattr(module.time, 'sleep', stop)
    request = dict(mode='nccl', version=0, snapshot_id='same-snapshot', optimizer_revision=0,
        nccl_host='train', nccl_port=29620, nccl_world_size=2,
        nccl_rank_offsets={'returned': 0}, nccl_chunk_bytes=1024)
    return SimpleNamespace(module=module, args=args, calls=calls, request=request,
        request_path=tmp_path / 'reload_request.json', ack=tmp_path / 'ack_returned_0.json')


def test_startup_ignores_old_nccl_until_a_new_transfer_arrives(supervisor, monkeypatch):
    s = supervisor
    s.module.write_json_atomic(s.request_path, s.request)
    polls = 0

    def poll(seconds):
        nonlocal polls
        polls += 1
        if polls == 1:
            assert not s.calls and not s.ack.exists()
            s.request['snapshot_id'] = 'returned-transfer'
            s.module.write_json_atomic(s.request_path, s.request)
        else:
            raise KeyboardInterrupt()

    monkeypatch.setattr(s.module.time, 'sleep', poll)
    assert s.module.main() == 130
    assert len(s.calls) == 1
    ack = json.loads(s.ack.read_text())
    assert ack['snapshot_id'] == 'returned-transfer' and ack['optimizer_revision'] == 0
    assert ack['server_response'] == dict(service_instance_id='new-service', service_generation=1)


@pytest.mark.parametrize('old_request', [False, True])
def test_new_nccl_request_during_launch_is_not_missed(supervisor, monkeypatch, old_request):
    s = supervisor
    if old_request:
        s.module.write_json_atomic(s.request_path, s.request)
    monkeypatch.setattr(s.module, 'wait_until_ready',
        lambda *args: s.module.write_json_atomic(s.request_path, s.request))
    assert s.module.main() == 130
    assert len(s.calls) == 1
    assert json.loads(s.ack.read_text())['server_response']['service_generation'] == 1


def test_existing_disk_request_keeps_its_original_reload_behavior(supervisor, tmp_path, monkeypatch):
    s = supervisor
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    s.module.write_json_atomic(s.request_path,
        dict(mode='checkpoint', version=1, checkpoint_path=str(checkpoint)))
    disk_calls = []
    monkeypatch.setattr(s.module, 'reload_inplace', lambda *args: disk_calls.append(args) or {})
    assert s.module.main() == 130
    assert disk_calls == [(s.args.health_url, str(checkpoint), s.args.ready_timeout)]
    assert not s.calls
    assert json.loads((tmp_path / 'ack_returned_1.json').read_text())['version'] == 1

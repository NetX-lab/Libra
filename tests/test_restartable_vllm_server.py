import importlib.util
import json
from pathlib import Path


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

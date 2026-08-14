import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from RL_Framework.vllm_hot_reload import InplaceReloadWorkerExtension


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    return checkpoint


def _extension(monkeypatch, load_model):
    synchronize = Mock()
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)),
    )
    extension = InplaceReloadWorkerExtension()
    extension.rank = 3
    extension.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(model="/models/old"),
        load_model=load_model,
    )
    extension.model_config = SimpleNamespace(model="/models/old")
    extension.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="/models/old")
    )
    return extension, synchronize


def test_worker_extension_reloads_resident_model(monkeypatch, tmp_path):
    load_model = Mock()
    extension, synchronize = _extension(monkeypatch, load_model)
    checkpoint = _checkpoint(tmp_path)

    result = extension.reload_weights(str(checkpoint))

    load_model.assert_called_once_with()
    synchronize.assert_called_once_with()
    assert extension.model_runner.model_config.model == str(checkpoint)
    assert extension.model_config.model == str(checkpoint)
    assert extension.vllm_config.model_config.model == str(checkpoint)
    assert result["rank"] == 3
    assert result["previous_path"] == "/models/old"


def test_worker_extension_restores_paths_when_reload_fails(monkeypatch, tmp_path):
    extension, _ = _extension(monkeypatch, Mock(side_effect=RuntimeError("load failed")))
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(RuntimeError, match="load failed"):
        extension.reload_weights(str(checkpoint))

    assert extension.model_runner.model_config.model == "/models/old"
    assert extension.model_config.model == "/models/old"
    assert extension.vllm_config.model_config.model == "/models/old"


def test_reload_inplace_posts_checkpoint_to_reload_endpoint(monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "restartable_vllm_server.py"
    spec = importlib.util.spec_from_file_location("restartable_vllm_server", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"workers": [{"rank": 0}]}'

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    result = module.reload_inplace(
        "http://127.0.0.1:8000/health",
        "/checkpoints/v2",
        30.0,
    )

    assert captured == {
        "url": "http://127.0.0.1:8000/reload_weights",
        "payload": {
            "checkpoint_path": "/checkpoints/v2",
            "timeout_seconds": 30.0,
        },
        "timeout": 40.0,
    }
    assert result == {"workers": [{"rank": 0}]}

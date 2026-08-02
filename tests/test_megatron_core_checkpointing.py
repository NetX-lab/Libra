import json

from RL_Framework.engine.megatron_core_checkpointing import (
    MegatronDistributedCheckpointManager,
)


class FakeModel:
    def __init__(self):
        self.loaded = None

    def sharded_state_dict(self):
        return {"weight": "local-shard"}

    def load_state_dict(self, state):
        self.loaded = state


class FakeOptimizer:
    def __init__(self):
        self.loaded = None

    def sharded_state_dict(self, model_state, is_loading=False):
        assert "model" in model_state
        suffix = "load-template" if is_loading else "optimizer-shard"
        return {"state": suffix}

    def load_state_dict(self, state):
        self.loaded = state


class FakeDistCheckpointing:
    def __init__(self):
        self.saved = None

    def save(
        self,
        *,
        sharded_state_dict,
        checkpoint_dir,
        sharded_strategy,
        async_sharded_save,
    ):
        self.saved = {
            "state": sharded_state_dict,
            "checkpoint_dir": checkpoint_dir,
            "strategy": sharded_strategy,
            "async": async_sharded_save,
        }
        return None

    def load(self, *, sharded_state_dict, checkpoint_dir, sharded_strategy):
        assert "model" in sharded_state_dict
        assert checkpoint_dir.endswith("megatron_dist")
        assert sharded_strategy is None
        return {
            "model": {"weight": "restored-shard"},
            "optimizer": {"state": "restored-optimizer-shard"},
        }


class FakeBridge:
    def __init__(self):
        self.calls = []

    def save_hf_pretrained(self, model, path, show_progress):
        self.calls.append((model, path, show_progress))
        (path / "model-00001-of-00002.safetensors").write_bytes(b"shard")
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )


class FakeAsyncRequest:
    def __init__(self):
        self.finalize_fns = []

    def add_finalize_fn(self, fn):
        self.finalize_fns.append(fn)


class FakeAsyncQueue:
    def __init__(self):
        self.request = None

    def schedule_async_request(self, request):
        self.request = request
        return 0

    def maybe_finalize_async_calls(self, blocking=False):
        del blocking
        if self.request is None:
            return []
        request, self.request = self.request, None
        for fn in request.finalize_fns:
            fn()
        return [0]

    def close(self):
        pass


def test_distributed_checkpoint_and_streaming_export(tmp_path, monkeypatch):
    manager = MegatronDistributedCheckpointManager(
        sync_path=str(tmp_path),
        fully_parallel_save=False,
    )
    dist_checkpointing = FakeDistCheckpointing()
    monkeypatch.setattr(
        manager,
        "_dist_checkpointing",
        lambda: dist_checkpointing,
    )
    model = [FakeModel()]
    optimizer = FakeOptimizer()
    bridge = FakeBridge()

    version_dir = manager.save(
        model=model,
        optimizer=optimizer,
        bridge=bridge,
        version=5,
        topology={
            "train_tp": 1,
            "train_pp": 1,
            "train_cp": 1,
            "train_ep": 8,
            "train_dp": 8,
        },
    )

    assert dist_checkpointing.saved["state"]["model"] == {
        "weight": "local-shard"
    }
    assert bridge.calls == [(model, version_dir, True)]
    manifest = json.loads(
        (version_dir / "megatron_checkpoint_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["checkpoint_format"] == "torch_dist"
    assert manifest["expert_parallel_size"] == 8
    assert manifest["includes_optimizer"] is True
    progress = json.loads(
        (version_dir / "sync_progress" / "rank_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["phase"] == "complete"


def test_rollout_export_only_skips_distributed_checkpoint(tmp_path, monkeypatch):
    manager = MegatronDistributedCheckpointManager(
        sync_path=str(tmp_path),
        fully_parallel_save=False,
    )
    dist_checkpointing = FakeDistCheckpointing()
    monkeypatch.setattr(
        manager,
        "_dist_checkpointing",
        lambda: dist_checkpointing,
    )
    monkeypatch.setenv("MEGATRON_ROLLOUT_EXPORT_ONLY", "1")
    model = [FakeModel()]
    bridge = FakeBridge()

    version_dir = manager.save(
        model=model,
        optimizer=FakeOptimizer(),
        bridge=bridge,
        version=7,
        topology={
            "train_tp": 1,
            "train_pp": 1,
            "train_cp": 1,
            "train_ep": 1,
            "train_dp": 1,
        },
    )

    assert dist_checkpointing.saved is None
    assert bridge.calls == [(model, version_dir, True)]
    manifest = json.loads(
        (version_dir / "megatron_checkpoint_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["distributed_checkpoint"] == ""
    assert manifest["rollout_export"] == str(version_dir)
    assert manifest["includes_optimizer"] is False


def test_load_uses_sharded_templates(tmp_path, monkeypatch):
    manager = MegatronDistributedCheckpointManager(
        sync_path=str(tmp_path),
        fully_parallel_save=False,
    )
    dist_checkpointing = FakeDistCheckpointing()
    monkeypatch.setattr(
        manager,
        "_dist_checkpointing",
        lambda: dist_checkpointing,
    )
    version_dir = tmp_path / "v3"
    (version_dir / "megatron_dist").mkdir(parents=True)
    (version_dir / "megatron_checkpoint_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    model = [FakeModel()]
    optimizer = FakeOptimizer()

    manager.load(model=model, optimizer=optimizer, version=3)

    assert model[0].loaded == {"weight": "restored-shard"}
    assert optimizer.loaded == {"state": "restored-optimizer-shard"}


def test_async_snapshot_returns_before_manifest_finalization(tmp_path, monkeypatch):
    manager = MegatronDistributedCheckpointManager(
        sync_path=str(tmp_path),
        fully_parallel_save=False,
        async_save=True,
    )
    dist_checkpointing = FakeDistCheckpointing()
    request = FakeAsyncRequest()
    dist_checkpointing.save = lambda **kwargs: (
        setattr(dist_checkpointing, "saved", kwargs) or request
    )
    monkeypatch.setattr(manager, "_dist_checkpointing", lambda: dist_checkpointing)
    manager._async_calls = FakeAsyncQueue()

    scheduled = manager.save_async_snapshot(
        model=[FakeModel()],
        optimizer=FakeOptimizer(),
        version=8,
        topology={
            "train_tp": 2,
            "train_pp": 1,
            "train_cp": 1,
            "train_ep": 1,
            "train_dp": 2,
        },
    )

    manifest = tmp_path / "v8" / "megatron_checkpoint_manifest.json"
    assert scheduled
    assert dist_checkpointing.saved["async_sharded_save"] is True
    assert not manifest.exists()

    assert manager.poll_async_snapshots() == [8]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["includes_optimizer"] is True
    assert payload["data_parallel_size"] == 2


def test_prune_snapshots_retains_latest_completed_versions(tmp_path):
    manager = MegatronDistributedCheckpointManager(
        sync_path=str(tmp_path),
        fully_parallel_save=False,
    )
    for version in range(1, 5):
        version_dir = tmp_path / f"v{version}"
        version_dir.mkdir()
        (version_dir / "megatron_checkpoint_manifest.json").write_text(
            "{}",
            encoding="utf-8",
        )

    removed = manager.prune_snapshots(
        keep_latest=2,
        protected_versions={1},
    )

    assert removed == [2]
    assert (tmp_path / "v1").exists()
    assert not (tmp_path / "v2").exists()
    assert (tmp_path / "v3").exists()
    assert (tmp_path / "v4").exists()

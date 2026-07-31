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

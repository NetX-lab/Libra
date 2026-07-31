import os

import torch

from RL_Framework.engine.train_engine import FSDPTrainEngine
from RL_Framework.infra.elastic import GradientPayload


def test_fsdp_engine_merges_active_hybrid_gradients_before_step():
    engine = FSDPTrainEngine(model_path="unused")
    engine.model = torch.nn.Linear(2, 1)
    engine.optimizer = torch.optim.SGD(engine.model.parameters(), lr=0.1)
    engine.world_size = 1
    engine.rank = 0
    domain = engine.configure_elastic_training(["dp0"])
    domain.request_join("hybrid0", "dp0")
    domain.mark_active("hybrid0")

    params = list(engine.model.parameters())
    for param in params:
        param.grad = torch.ones_like(param)
    payload = GradientPayload(
        replica_id="hybrid0",
        target_core_id="dp0",
        tensors=tuple(torch.full_like(param.grad, 2.0) for param in params),
    )
    engine.enqueue_hybrid_gradient_payload(payload)

    engine._apply_elastic_inter_replica_gradients()

    for param in params:
        assert torch.allclose(param.grad, torch.full_like(param.grad, 3.0))


def test_fsdp_engine_captures_and_loads_elastic_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ELASTIC_TRAINING_STATE_DIR", str(tmp_path))
    engine = FSDPTrainEngine(model_path="unused")
    engine.model = torch.nn.Linear(2, 1)
    engine.optimizer = torch.optim.SGD(engine.model.parameters(), lr=0.1)
    engine.world_size = 1
    engine.rank = 0
    engine.current_version = 3

    version = engine.capture_elastic_state_snapshot("rollout0", "dp0")
    snapshot = tmp_path / "v3" / "rank_0.pt"

    assert version == 3
    assert snapshot.exists()

    with torch.no_grad():
        for param in engine.model.parameters():
            param.add_(10.0)

    engine.load_elastic_state_snapshot(str(snapshot))

    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    for name, tensor in engine.model.state_dict().items():
        assert torch.allclose(tensor, payload["model"][name])

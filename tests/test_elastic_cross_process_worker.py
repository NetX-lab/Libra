import subprocess
import sys
import time
from pathlib import Path

import torch

from RL_Framework.engine.train_engine import FSDPTrainEngine
from RL_Framework.infra.elastic.gradient_ipc import ElasticGradientServer


def test_cross_process_hybrid_worker_sends_gradient_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("ELASTIC_TRAINING_STATE_DIR", str(tmp_path / "state"))
    engine = FSDPTrainEngine(model_path="unused")
    engine.model = torch.nn.Linear(2, 1)
    engine.optimizer = torch.optim.SGD(engine.model.parameters(), lr=0.1)
    engine.world_size = 1
    engine.rank = 0
    engine.current_version = 5
    domain = engine.configure_elastic_training(["dp0"])
    domain.request_join("hybrid0", "dp0")
    domain.mark_active("hybrid0")

    snapshot_version = engine.capture_elastic_state_snapshot("hybrid0", "dp0")
    snapshot_path = engine.get_elastic_state_snapshot_path(snapshot_version)
    payload_file = tmp_path / "payload.pt"
    params = list(engine.model.parameters())
    torch.save(
        {
            "replica_id": "hybrid0",
            "target_core_id": "dp0",
            "tensors": tuple(torch.full_like(p, 2.0) for p in params),
        },
        payload_file,
    )

    server = ElasticGradientServer(
        host="127.0.0.1",
        port=0,
        authkey="secret",
        on_payload=engine.enqueue_hybrid_gradient_payload,
    )
    endpoint = server.start()
    try:
        script = Path(__file__).resolve().parents[1] / "scripts" / "elastic_hybrid_worker.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--worker-id",
                "hybrid0",
                "--target-core-id",
                "dp0",
                "--snapshot-path",
                snapshot_path,
                "--gradient-host",
                endpoint.host,
                "--gradient-port",
                str(endpoint.port),
                "--authkey",
                endpoint.authkey,
                "--payload-file",
                str(payload_file),
            ],
            check=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

        deadline = time.time() + 5
        while server.received_count < 1 and time.time() < deadline:
            time.sleep(0.05)

        for param in params:
            param.grad = torch.ones_like(param)
        engine._apply_elastic_inter_replica_gradients()

        assert server.received_count == 1
        for param in params:
            assert torch.allclose(param.grad, torch.full_like(param.grad, 3.0))
    finally:
        server.close()

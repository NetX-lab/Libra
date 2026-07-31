import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from RL_Framework.infra.elastic.native_rdma import (
    NativeRDMAConfig,
    NativeRDMAFileTransfer,
    NativeRDMAGradientClient,
    NativeRDMAGradientServer,
)
from RL_Framework.infra.elastic.gradient_ipc import GradientEndpoint
from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


def test_native_rdma_helper_builds():
    if not NativeRDMAFileTransfer.is_system_available():
        pytest.skip("native RDMA build prerequisites are not available")
    transfer = NativeRDMAFileTransfer(NativeRDMAConfig())
    helper = transfer.build_helper(force=True)
    assert helper.exists()
    assert os.access(helper, os.X_OK)


def test_native_rdma_loopback_file_transfer(tmp_path: Path):
    if not NativeRDMAFileTransfer.is_system_available():
        pytest.skip("native RDMA build prerequisites are not available")
    payload = bytes((i % 251 for i in range(256 * 1024)))
    src = tmp_path / "payload.bin"
    dst = tmp_path / "received.bin"
    src.write_bytes(payload)

    transfer = NativeRDMAFileTransfer(
        NativeRDMAConfig(
            device=os.environ.get("NATIVE_RDMA_DEVICE", "mlx5_0"),
            gid_index=int(os.environ.get("NATIVE_RDMA_GID_INDEX", "0")),
            ib_port=int(os.environ.get("NATIVE_RDMA_IB_PORT", "1")),
            max_bytes=1024 * 1024,
        )
    )
    transfer.transfer_file(input_path=src, output_path=dst)

    assert dst.read_bytes() == payload


def test_native_rdma_gradient_payload_transfer(tmp_path: Path):
    if not NativeRDMAFileTransfer.is_system_available():
        pytest.skip("native RDMA build prerequisites are not available")
    received = []
    config = NativeRDMAConfig(
        device=os.environ.get("NATIVE_RDMA_DEVICE", "mlx5_0"),
        gid_index=int(os.environ.get("NATIVE_RDMA_GID_INDEX", "0")),
        ib_port=int(os.environ.get("NATIVE_RDMA_IB_PORT", "1")),
        max_bytes=1024 * 1024,
    )
    server = NativeRDMAGradientServer(
        host="127.0.0.1",
        port=0,
        authkey="rdma-secret",
        on_payload=received.append,
        rdma_config=config,
        work_dir=tmp_path,
    )
    endpoint = server.start()
    try:
        client = NativeRDMAGradientClient(endpoint, rdma_config=config)
        client.send(
            GradientPayload(
                replica_id="hybrid0",
                target_core_id="dp0",
                tensors=(torch.tensor([1.0, 2.0]),),
            )
        )
        import time

        deadline = time.time() + 10
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert len(received) == 1
        assert received[0].replica_id == "hybrid0"
        assert received[0].target_core_id == "dp0"
    finally:
        server.close()


def test_elastic_hybrid_worker_cli_uses_native_rdma(tmp_path: Path):
    if not NativeRDMAFileTransfer.is_system_available():
        pytest.skip("native RDMA build prerequisites are not available")
    received = []
    config = NativeRDMAConfig(
        device=os.environ.get("NATIVE_RDMA_DEVICE", "mlx5_0"),
        gid_index=int(os.environ.get("NATIVE_RDMA_GID_INDEX", "0")),
        ib_port=int(os.environ.get("NATIVE_RDMA_IB_PORT", "1")),
        max_bytes=1024 * 1024,
    )
    server = NativeRDMAGradientServer(
        host="127.0.0.1",
        port=0,
        authkey="rdma-secret",
        on_payload=received.append,
        rdma_config=config,
        work_dir=tmp_path / "server",
    )
    endpoint = server.start()
    snapshot = tmp_path / "snapshot.pt"
    payload_file = tmp_path / "payload.pt"
    torch.save({"version": 1, "model": {}, "optimizer": None}, snapshot)
    torch.save(
        {
            "replica_id": "hybrid_cli",
            "target_core_id": "dp0",
            "tensors": (torch.tensor([3.0]),),
        },
        payload_file,
    )
    try:
        script = Path(__file__).resolve().parents[1] / "scripts" / "elastic_hybrid_worker.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--worker-id",
                "hybrid_cli",
                "--target-core-id",
                "dp0",
                "--snapshot-path",
                str(snapshot),
                "--gradient-host",
                endpoint.host,
                "--gradient-port",
                str(endpoint.port),
                "--authkey",
                endpoint.authkey,
                "--payload-file",
                str(payload_file),
                "--gradient-transport",
                "native_rdma",
                "--native-rdma-device",
                config.device,
                "--native-rdma-gid-index",
                str(config.gid_index),
                "--native-rdma-ib-port",
                str(config.ib_port),
                "--native-rdma-max-bytes",
                str(config.max_bytes),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
        )
        import time

        deadline = time.time() + 10
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert len(received) == 1
        assert received[0].replica_id == "hybrid_cli"
        assert torch.equal(received[0].tensors[0], torch.tensor([3.0]))
    finally:
        server.close()

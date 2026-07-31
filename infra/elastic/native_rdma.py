"""Native libibverbs backend wrapper for elastic data-plane tests."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from RL_Framework.infra.elastic.gradient_ipc import GradientEndpoint
from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


class NativeRDMAError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeRDMAConfig:
    device: str = "mlx5_0"
    gid_index: int = 0
    ib_port: int = 1
    max_bytes: int = 64 * 1024 * 1024
    helper_path: str = ""


class NativeRDMAFileTransfer:
    """One-way native RDMA SEND/RECV file transfer using libibverbs.

    The helper uses TCP only to exchange QP attributes; payload bytes are moved
    through an RC QP with ``ibv_post_send``/``ibv_post_recv``.
    """

    def __init__(self, config: NativeRDMAConfig | None = None):
        self.config = config or NativeRDMAConfig()

    @staticmethod
    def source_path() -> Path:
        return Path(__file__).with_name("native_rdma_verbs_xfer.c")

    def helper_path(self) -> Path:
        if self.config.helper_path:
            return Path(self.config.helper_path)
        return Path(tempfile.gettempdir()) / "rl_framework_native_rdma_verbs_xfer"

    @staticmethod
    def is_system_available() -> bool:
        return (
            shutil.which("gcc") is not None
            and Path("/usr/include/infiniband/verbs.h").exists()
            and bool(shutil.which("ibv_devinfo"))
        )

    def build_helper(self, *, force: bool = False) -> Path:
        helper = self.helper_path()
        src = self.source_path()
        if not src.exists():
            raise NativeRDMAError(f"native RDMA source missing: {src}")
        if helper.exists() and not force and helper.stat().st_mtime >= src.stat().st_mtime:
            return helper
        cmd = [
            "gcc",
            "-O2",
            "-Wall",
            "-Wextra",
            str(src),
            "-libverbs",
            "-o",
            str(helper),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise NativeRDMAError(
                "failed to build native RDMA helper\n"
                f"cmd={' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )
        return helper

    def transfer_file(
        self,
        *,
        input_path: str | os.PathLike,
        output_path: str | os.PathLike,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout_s: float = 30.0,
    ) -> None:
        helper = self.build_helper()
        if port == 0:
            port = self._free_tcp_port(host)
        server_cmd = [
            str(helper),
            "--server",
            "--bind",
            host,
            "--port",
            str(port),
            "--out",
            str(output_path),
            "--dev",
            self.config.device,
            "--ib-port",
            str(self.config.ib_port),
            "--gid-index",
            str(self.config.gid_index),
            "--max-bytes",
            str(self.config.max_bytes),
        ]
        client_cmd = [
            str(helper),
            "--client",
            "--host",
            host,
            "--port",
            str(port),
            "--in",
            str(input_path),
            "--dev",
            self.config.device,
            "--ib-port",
            str(self.config.ib_port),
            "--gid-index",
            str(self.config.gid_index),
            "--max-bytes",
            str(self.config.max_bytes),
        ]

        server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.25)
            client = subprocess.run(client_cmd, capture_output=True, text=True, timeout=timeout_s)
            try:
                server_out, server_err = server.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                server.kill()
                server_out, server_err = server.communicate(timeout=5)
                raise NativeRDMAError("native RDMA server timed out")
            if client.returncode != 0 or server.returncode != 0:
                raise NativeRDMAError(
                    "native RDMA transfer failed\n"
                    f"server_cmd={' '.join(server_cmd)}\n"
                    f"server_rc={server.returncode}\nserver_out={server_out}\nserver_err={server_err}\n"
                    f"client_cmd={' '.join(client_cmd)}\n"
                    f"client_rc={client.returncode}\nclient_out={client.stdout}\nclient_err={client.stderr}"
                )
        finally:
            if server.poll() is None:
                server.kill()

    def receive_file_once(
        self,
        *,
        output_path: str | os.PathLike,
        host: str,
        port: int,
        timeout_s: float = 60.0,
    ) -> None:
        helper = self.build_helper()
        cmd = [
            str(helper),
            "--server",
            "--bind",
            host,
            "--port",
            str(port),
            "--out",
            str(output_path),
            "--dev",
            self.config.device,
            "--ib-port",
            str(self.config.ib_port),
            "--gid-index",
            str(self.config.gid_index),
            "--max-bytes",
            str(self.config.max_bytes),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            raise NativeRDMAError(
                "native RDMA receive failed\n"
                f"cmd={' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

    def send_file_once(
        self,
        *,
        input_path: str | os.PathLike,
        host: str,
        port: int,
        timeout_s: float = 60.0,
    ) -> None:
        helper = self.build_helper()
        cmd = [
            str(helper),
            "--client",
            "--host",
            host,
            "--port",
            str(port),
            "--in",
            str(input_path),
            "--dev",
            self.config.device,
            "--ib-port",
            str(self.config.ib_port),
            "--gid-index",
            str(self.config.gid_index),
            "--max-bytes",
            str(self.config.max_bytes),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            raise NativeRDMAError(
                "native RDMA send failed\n"
                f"cmd={' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

    @staticmethod
    def _free_tcp_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])


class NativeRDMAGradientClient:
    """Send GradientPayload objects over native libibverbs data plane."""

    def __init__(
        self,
        endpoint: GradientEndpoint,
        *,
        rdma_config: NativeRDMAConfig | None = None,
        timeout_s: float = 60.0,
    ):
        self.endpoint = endpoint
        self.transfer = NativeRDMAFileTransfer(rdma_config)
        self.timeout_s = timeout_s

    def send(self, payload: GradientPayload):
        with tempfile.NamedTemporaryFile(prefix="rdma_gradient_", suffix=".pt", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            torch.save(
                {
                    "authkey": self.endpoint.authkey,
                    "replica_id": payload.replica_id,
                    "target_core_id": payload.target_core_id,
                    "tensors": tuple(t.detach().cpu() for t in payload.tensors),
                    "zero_placeholder": payload.zero_placeholder,
                },
                path,
            )
            self.transfer.send_file_once(
                input_path=path,
                host=self.endpoint.host,
                port=self.endpoint.port,
                timeout_s=self.timeout_s,
            )
        finally:
            path.unlink(missing_ok=True)


class NativeRDMAGradientServer:
    """Looping native RDMA receiver for GradientPayload objects."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        authkey: str = "",
        on_payload=None,
        rdma_config: NativeRDMAConfig | None = None,
        work_dir: str | os.PathLike | None = None,
    ):
        self.host = host
        self.port = int(port) if port else NativeRDMAFileTransfer._free_tcp_port(host)
        self.authkey = authkey
        self.on_payload = on_payload or (lambda payload: None)
        self.transfer = NativeRDMAFileTransfer(rdma_config)
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="native_rdma_gradients_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._received = 0
        self._last_error: Exception | None = None

    @property
    def endpoint(self) -> GradientEndpoint:
        return GradientEndpoint(self.host, self.port, self.authkey)

    @property
    def received_count(self) -> int:
        return self._received

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def start(self) -> GradientEndpoint:
        if self._thread is None:
            self.transfer.build_helper()
            self._thread = threading.Thread(
                target=self._serve,
                name="native-rdma-gradient-server",
                daemon=True,
            )
            self._thread.start()
            time.sleep(0.1)
        return self.endpoint

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _serve(self):
        idx = 0
        while not self._stop.is_set():
            out = self.work_dir / f"payload_{idx}.pt"
            idx += 1
            try:
                self.transfer.receive_file_once(
                    output_path=out,
                    host=self.host,
                    port=self.port,
                    timeout_s=5.0,
                )
            except subprocess.TimeoutExpired:
                continue
            except Exception as exc:
                self._last_error = exc
                continue
            try:
                data = torch.load(out, map_location="cpu", weights_only=False)
                if self.authkey and data.get("authkey") != self.authkey:
                    raise PermissionError("invalid native RDMA gradient authkey")
                payload = GradientPayload(
                    replica_id=str(data["replica_id"]),
                    target_core_id=str(data["target_core_id"]),
                    tensors=tuple(data["tensors"]),
                    zero_placeholder=bool(data.get("zero_placeholder", False)),
                )
                self.on_payload(payload)
                self._received += 1
            except Exception as exc:
                self._last_error = exc

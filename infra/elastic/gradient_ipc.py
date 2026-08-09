"""Cross-process gradient payload transport for elastic hybrid workers."""

from __future__ import annotations

import io
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Callable

import torch

from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


_HEADER = struct.Struct("!Q")


@dataclass(frozen=True)
class GradientEndpoint:
    host: str
    port: int
    authkey: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "authkey": self.authkey,
        }


class ElasticGradientClient:
    """Length-prefixed TCP client used by hybrid worker processes."""

    def __init__(self, endpoint: GradientEndpoint, timeout: float = 30.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def send(self, payload: GradientPayload):
        data = self._serialize(payload)
        with socket.create_connection(
            (self.endpoint.host, self.endpoint.port),
            timeout=self.timeout,
        ) as sock:
            sock.sendall(_HEADER.pack(len(data)))
            sock.sendall(data)

    def _serialize(self, payload: GradientPayload) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "authkey": self.endpoint.authkey,
                "replica_id": payload.replica_id,
                "target_core_id": payload.target_core_id,
                "tensors": tuple(t.detach().cpu() for t in payload.tensors),
                "zero_placeholder": payload.zero_placeholder,
                "replica_rank": payload.replica_rank,
                "replica_world_size": payload.replica_world_size,
            },
            buffer,
        )
        return buffer.getvalue()


class ElasticGradientServer:
    """Background TCP receiver that enqueues incoming hybrid gradients."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        authkey: str = "",
        on_payload: Callable[[GradientPayload], None],
    ):
        self.host = host
        self.port = int(port)
        self.authkey = authkey
        self.on_payload = on_payload
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._received = 0

    @property
    def endpoint(self) -> GradientEndpoint:
        if self._sock is None:
            raise RuntimeError("ElasticGradientServer is not started")
        host, port = self._sock.getsockname()
        public_host = self.host if self.host not in {"0.0.0.0", ""} else host
        return GradientEndpoint(public_host, int(port), self.authkey)

    @property
    def received_count(self) -> int:
        return self._received

    def start(self) -> GradientEndpoint:
        if self._sock is not None:
            return self.endpoint
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen()
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(
            target=self._serve,
            name="elastic-gradient-server",
            daemon=True,
        )
        self._thread.start()
        return self.endpoint

    def close(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sock = None
        self._thread = None

    def _serve(self):
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    payload = self._recv_payload(conn)
                    self.on_payload(payload)
                    self._received += 1
                except Exception:
                    continue

    def _recv_payload(self, conn: socket.socket) -> GradientPayload:
        header = self._recv_exact(conn, _HEADER.size)
        (length,) = _HEADER.unpack(header)
        raw = self._recv_exact(conn, length)
        data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if self.authkey and data.get("authkey") != self.authkey:
            raise PermissionError("invalid elastic gradient authkey")
        return GradientPayload(
            replica_id=str(data["replica_id"]),
            target_core_id=str(data["target_core_id"]),
            tensors=tuple(data["tensors"]),
            zero_placeholder=bool(data.get("zero_placeholder", False)),
            replica_rank=int(data.get("replica_rank", 0)),
            replica_world_size=int(data.get("replica_world_size", 1)),
        )

    @staticmethod
    def _recv_exact(conn: socket.socket, n_bytes: int) -> bytes:
        chunks = []
        remaining = n_bytes
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                raise ConnectionError("socket closed before payload completed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

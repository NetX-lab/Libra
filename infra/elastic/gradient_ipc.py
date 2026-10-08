"""Cross-process gradient payload transport for elastic hybrid workers."""

from __future__ import annotations

import io
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch

from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


_HEADER = struct.Struct("!Q")
_ACK = b"\x01"


@dataclass(frozen=True)
class GradientEndpoint:
    host: str
    port: int
    authkey: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"host": self.host, "port": self.port, "authkey": self.authkey}


@dataclass(frozen=True)
class GradientUpdate:
    """Post-AllReduce gradient returned to one Hybrid replica lane."""

    replica_id: str
    tensors: tuple[torch.Tensor, ...]
    step: int
    state_version: int
    membership_epoch: int


class ElasticGradientClient:
    """Length-prefixed TCP client with optional non-blocking update ACK."""

    def __init__(self, endpoint: GradientEndpoint, timeout: float = 300.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def send(self, payload: GradientPayload, *, expect_update: bool = False):
        with socket.create_connection(
            (self.endpoint.host, self.endpoint.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            metadata = self._serialize_metadata(payload, expect_update=expect_update)
            sock.sendall(_HEADER.pack(len(metadata)))
            sock.sendall(metadata)
            for tensor in payload.tensors:
                data = self._serialize_tensor(tensor)
                sock.sendall(_HEADER.pack(len(data)))
                sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            if self._recv_exact(sock, 1) != _ACK:
                raise ConnectionError("gradient receiver rejected the payload")
            if expect_update:
                return self._recv_update(sock)
        return None

    def _serialize_metadata(self, payload: GradientPayload, *, expect_update: bool) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "protocol_version": 3,
                "authkey": self.endpoint.authkey,
                "replica_id": payload.replica_id,
                "target_core_id": payload.target_core_id,
                "tensor_count": len(payload.tensors),
                "zero_placeholder": payload.zero_placeholder,
                "replica_rank": payload.replica_rank,
                "replica_world_size": payload.replica_world_size,
                "step": payload.step,
                "state_version": payload.state_version,
                "membership_epoch": payload.membership_epoch,
                "expect_update": bool(expect_update),
            },
            buffer,
        )
        return buffer.getvalue()

    @staticmethod
    def _serialize_tensor(tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor.detach().cpu(), buffer)
        return buffer.getvalue()

    def _recv_update(self, sock: socket.socket) -> GradientUpdate:
        (metadata_length,) = _HEADER.unpack(self._recv_exact(sock, _HEADER.size))
        metadata = torch.load(
            io.BytesIO(self._recv_exact(sock, metadata_length)),
            map_location="cpu",
            weights_only=False,
        )
        tensors = []
        for _ in range(int(metadata["tensor_count"])):
            (length,) = _HEADER.unpack(self._recv_exact(sock, _HEADER.size))
            tensors.append(
                torch.load(
                    io.BytesIO(self._recv_exact(sock, length)),
                    map_location="cpu",
                    weights_only=False,
                )
            )
        return GradientUpdate(
            replica_id=str(metadata["replica_id"]),
            tensors=tuple(tensors),
            step=int(metadata["step"]),
            state_version=int(metadata["state_version"]),
            membership_epoch=int(metadata["membership_epoch"]),
        )

    @staticmethod
    def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
        data = bytearray(n_bytes)
        view = memoryview(data)
        received = 0
        while received < n_bytes:
            count = sock.recv_into(view[received:])
            if count == 0:
                raise ConnectionError("socket closed before acknowledgement")
            received += count
        return bytes(data)


class ElasticGradientServer:
    """Background receiver with update publication for lockstep workers."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        authkey: str = "",
        on_payload: Callable[[GradientPayload], None],
        update_timeout: float = 300.0,
    ):
        self.host = host
        self.port = int(port)
        self.authkey = authkey
        self.on_payload = on_payload
        self.update_timeout = float(update_timeout)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._received = 0
        self._handlers: set[threading.Thread] = set()
        self._handler_lock = threading.Lock()
        self._update_condition = threading.Condition()
        self._updates: dict[tuple[str, int, int], GradientUpdate] = {}

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
        self._thread = threading.Thread(target=self._serve, name="elastic-gradient-server", daemon=True)
        self._thread.start()
        return self.endpoint

    def close(self):
        self._stop.set()
        with self._update_condition:
            self._update_condition.notify_all()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._handler_lock:
            handlers = list(self._handlers)
        for handler in handlers:
            handler.join(timeout=2.0)
        self._sock = None
        self._thread = None

    def publish_update(self, update: GradientUpdate) -> None:
        key = (update.replica_id, int(update.step), int(update.membership_epoch))
        with self._update_condition:
            self._updates[key] = update
            self._update_condition.notify_all()

    def _serve(self):
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            handler = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
            with self._handler_lock:
                self._handlers.add(handler)
            handler.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            with conn:
                try:
                    payload, expect_update = self._recv_payload(conn)
                    self.on_payload(payload)
                    self._received += 1
                    update = self._wait_for_update(payload) if expect_update else None
                    conn.sendall(_ACK)
                    if update is not None:
                        self._send_update(conn, update)
                except Exception:
                    try:
                        conn.sendall(b"\x00")
                    except OSError:
                        pass
        finally:
            with self._handler_lock:
                self._handlers.discard(threading.current_thread())

    def _wait_for_update(self, payload: GradientPayload) -> GradientUpdate:
        key = (payload.replica_id, int(payload.step), int(payload.membership_epoch))
        deadline = time.monotonic() + max(self.update_timeout, 0.0)
        with self._update_condition:
            while key not in self._updates and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for post-AllReduce update: {key}")
                self._update_condition.wait(timeout=min(remaining, 0.5))
            if key not in self._updates:
                raise RuntimeError("gradient server closed before update was published")
            return self._updates.pop(key)

    @staticmethod
    def _send_update(conn: socket.socket, update: GradientUpdate) -> None:
        buffer = io.BytesIO()
        torch.save(
            {
                "replica_id": update.replica_id,
                "tensor_count": len(update.tensors),
                "step": int(update.step),
                "state_version": int(update.state_version),
                "membership_epoch": int(update.membership_epoch),
            },
            buffer,
        )
        metadata = buffer.getvalue()
        conn.sendall(_HEADER.pack(len(metadata)))
        conn.sendall(metadata)
        for tensor in update.tensors:
            data = ElasticGradientClient._serialize_tensor(tensor)
            conn.sendall(_HEADER.pack(len(data)))
            conn.sendall(data)

    def _recv_payload(self, conn: socket.socket) -> tuple[GradientPayload, bool]:
        (length,) = _HEADER.unpack(self._recv_exact(conn, _HEADER.size))
        data = torch.load(io.BytesIO(self._recv_exact(conn, length)), map_location="cpu", weights_only=False)
        if self.authkey and data.get("authkey") != self.authkey:
            raise PermissionError("invalid elastic gradient authkey")
        if int(data.get("protocol_version", 0)) not in {2, 3}:
            raise ValueError("unsupported elastic gradient protocol")
        tensors = []
        for _ in range(int(data["tensor_count"])):
            (tensor_length,) = _HEADER.unpack(self._recv_exact(conn, _HEADER.size))
            tensors.append(torch.load(io.BytesIO(self._recv_exact(conn, tensor_length)), map_location="cpu", weights_only=False))
        return GradientPayload(
            replica_id=str(data["replica_id"]),
            target_core_id=str(data["target_core_id"]),
            tensors=tuple(tensors),
            zero_placeholder=bool(data.get("zero_placeholder", False)),
            replica_rank=int(data.get("replica_rank", 0)),
            replica_world_size=int(data.get("replica_world_size", 1)),
            step=int(data.get("step", -1)),
            state_version=int(data.get("state_version", -1)),
            membership_epoch=int(data.get("membership_epoch", 0)),
        ), bool(data.get("expect_update", False))

    @staticmethod
    def _recv_exact(conn: socket.socket, n_bytes: int) -> bytes:
        data = bytearray(n_bytes)
        view = memoryview(data)
        received = 0
        while received < n_bytes:
            count = conn.recv_into(view[received:])
            if count == 0:
                raise ConnectionError("socket closed before payload completed")
            received += count
        return bytes(data)

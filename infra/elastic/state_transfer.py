"""Transient, bounded-frame CPU tensor transport for live EHP state.

No model or optimizer checkpoint is written to disk. A capability URL names an
immutable in-memory snapshot; owners explicitly release it after installation.
"""
from __future__ import annotations

import io
import secrets
import socket
import struct
import threading
from urllib.parse import urlparse

import torch

_FRAME = struct.Struct('!Q')
_CHUNK = 4 * 1024 * 1024


def _read(sock, count):
    result = bytearray(count)
    view = memoryview(result)
    offset = 0
    while offset < count:
        received = sock.recv_into(view[offset:])
        if not received:
            raise ConnectionError('live state stream ended early')
        offset += received
    return result


def _pack_tree(value, tensors):
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value.detach().cpu().contiguous())
        return ('tensor', index)
    if isinstance(value, dict):
        return ('dict', [(_pack_tree(k, tensors), _pack_tree(v, tensors)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return ('tuple' if isinstance(value, tuple) else 'list', [_pack_tree(v, tensors) for v in value])
    return ('value', value)


def _unpack_tree(node, tensors):
    kind, value = node
    if kind == 'tensor':
        return tensors[value]
    if kind == 'dict':
        return {_unpack_tree(k, tensors): _unpack_tree(v, tensors) for k, v in value}
    if kind in ('tuple', 'list'):
        result = [_unpack_tree(v, tensors) for v in value]
        return tuple(result) if kind == 'tuple' else result
    return value


class LiveStateServer:
    def __init__(self, host='0.0.0.0', advertised_host=None):
        self._snapshots = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen()
        self._sock.settimeout(0.2)
        self.host = advertised_host or socket.gethostbyname(socket.gethostname())
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def publish(self, payload):
        token = secrets.token_hex(24)
        with self._lock:
            self._snapshots[token] = payload
        return f'ehp://{self.host}:{self.port}/{token}'

    def release(self, uri):
        with self._lock:
            self._snapshots.pop(urlparse(uri).path.lstrip('/'), None)

    def close(self):
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2)
        with self._lock:
            self._snapshots.clear()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._send, args=(conn,), daemon=True).start()

    def _send(self, conn):
        with conn:
            conn.settimeout(300)
            try:
                length, = _FRAME.unpack(_read(conn, _FRAME.size))
                if length > 256:
                    raise ValueError('invalid state capability')
                token = _read(conn, length).decode()
                with self._lock:
                    payload = self._snapshots.get(token)
                if payload is None:
                    conn.sendall(_FRAME.pack(0))
                    return
                tensors = []
                tree = _pack_tree(payload, tensors)
                metadata = {'tree': tree, 'tensors': [(str(t.dtype).removeprefix('torch.'), tuple(t.shape)) for t in tensors]}
                buffer = io.BytesIO()
                torch.save(metadata, buffer)
                blob = buffer.getvalue()
                conn.sendall(_FRAME.pack(len(blob)))
                conn.sendall(blob)
                for tensor in tensors:
                    flat = tensor.reshape(-1).view(torch.uint8)
                    for offset in range(0, flat.numel(), _CHUNK):
                        conn.sendall(flat[offset:offset + _CHUNK].numpy().tobytes())
            except (OSError, ValueError):
                return


def receive_live_state(uri, timeout=300):
    target = urlparse(uri)
    if target.scheme != 'ehp':
        raise ValueError('not a live EHP state endpoint')
    with socket.create_connection((target.hostname, target.port), timeout=timeout or None) as conn:
        token = target.path.lstrip('/').encode()
        conn.sendall(_FRAME.pack(len(token)))
        conn.sendall(token)
        length, = _FRAME.unpack(_read(conn, _FRAME.size))
        if not length:
            raise RuntimeError('live state snapshot expired or capability is invalid')
        metadata = torch.load(io.BytesIO(_read(conn, length)), weights_only=False)
        tensors = []
        for dtype_name, shape in metadata['tensors']:
            tensor = torch.empty(shape, dtype=getattr(torch, dtype_name))
            flat = tensor.reshape(-1).view(torch.uint8)
            for offset in range(0, flat.numel(), _CHUNK):
                count = min(_CHUNK, flat.numel() - offset)
                flat[offset:offset + count].copy_(torch.frombuffer(_read(conn, count), dtype=torch.uint8))
            tensors.append(tensor)
        return _unpack_tree(metadata['tree'], tensors)

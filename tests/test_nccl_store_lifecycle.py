"""The stateless rendezvous transfers its listening FD to the real TCPStore."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest

from RL_Framework.infra.sync import nccl_weight_sync
from RL_Framework.infra.sync.nccl_weight_sync import NcclReloadSpec


@pytest.mark.parametrize("rank", [0, 1])
def test_only_master_transfers_listening_socket_ownership(monkeypatch, rank):
    detached = []
    group = SimpleNamespace(socket=SimpleNamespace(detach=lambda: detached.append(True)))
    utils = ModuleType("vllm.distributed.utils")
    utils.StatelessProcessGroup = SimpleNamespace(create=lambda **_kwargs: group)
    monkeypatch.setitem(sys.modules, "vllm.distributed.utils", utils)

    assert nccl_weight_sync._group(NcclReloadSpec("127.0.0.1", 12345, 2, rank, "cpu")) is group
    assert detached == ([True] if rank == 0 else [])


def test_real_tcp_store_survives_python_socket_cleanup_and_cache_eviction():
    # A subprocess is essential: closing a socket that still owns TCPStore's FD
    # triggers SIGABRT in a C++ daemon, bypassing Python exception handling. Use
    # real CPU PyTorch but reproduce vLLM's small create() without importing GPU
    # dependencies. Multiple groups also exercise independent cache lifetimes.
    script = textwrap.dedent("""
        import gc
        import socket
        import sys
        from datetime import timedelta
        from types import ModuleType, SimpleNamespace
        import torch.distributed as dist
        from RL_Framework.infra.sync import nccl_weight_sync

        def create(host, port, rank, world_size, store_timeout):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.listen()
            port = sock.getsockname()[1]
            store = dist.TCPStore(host, port, world_size, True,
                timeout=timedelta(seconds=store_timeout), use_libuv=False,
                master_listen_fd=sock.fileno())
            return SimpleNamespace(socket=sock, store=store)

        utils = ModuleType('vllm.distributed.utils')
        utils.StatelessProcessGroup = SimpleNamespace(create=create)
        sys.modules['vllm.distributed.utils'] = utils
        for index in range(3):
            spec = nccl_weight_sync.NcclReloadSpec('127.0.0.1', 0, 1, 0, 'cpu', timeout_s=2)
            group = nccl_weight_sync._group(spec)
            assert group.socket.fileno() == -1
            group.socket.close()
            group.store.set('working', b'yes')
            assert group.store.get('working') == b'yes'
            nccl_weight_sync._COMMUNICATOR_CACHE[('test', index)] = SimpleNamespace(group=group)
            del group
        nccl_weight_sync.clear_communicator_cache()
        gc.collect()
        print('ORDERLY_TCPSTORE_EXIT', flush=True)
    """)
    root = Path(nccl_weight_sync.__file__).resolve().parents[2]
    # Preserve the package alias provided by the ordinary project test runner.
    environment = dict(os.environ)
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=root,
        env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ORDERLY_TCPSTORE_EXIT" in completed.stdout
    assert "TCPStoreMasterDaemon::run() failed" not in completed.stderr

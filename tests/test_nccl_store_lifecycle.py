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


def test_real_namespaced_store_reuses_listener_and_reconnects_new_member():
    # Keep TCPStore lifetime errors isolated in a real subprocess, as above.
    # Only vLLM's dataclass container is stubbed; all stores/connections are real.
    script = textwrap.dedent('''
        import gc
        import os
        import subprocess
        import sys
        import textwrap
        from types import ModuleType, SimpleNamespace
        from RL_Framework.infra.sync import nccl_weight_sync as transport

        utils = ModuleType('vllm.distributed.utils')
        utils.StatelessProcessGroup = lambda **kwargs: SimpleNamespace(**kwargs)
        sys.modules[utils.__name__] = utils
        sender = transport.NcclReloadSpec('127.0.0.1', 0, 2, 0, 'cpu', timeout_s=2, session_id='old')
        old_sender = transport._group(sender)
        port = transport._BASE_STORE_CACHE[('127.0.0.1', 0, True)].port
        receiver = transport.NcclReloadSpec('127.0.0.1', port, 2, 1, 'cpu', timeout_s=2, session_id='old')
        old_receiver = transport._group(receiver)
        old_sender.store.set('broadcast_from/0/0', b'old-nccl-id')
        assert old_receiver.store.get('broadcast_from/0/0') == b'old-nccl-id'

        from dataclasses import replace
        new_sender = transport._group(replace(sender, session_id='new'))
        new_receiver = transport._group(replace(receiver, session_id='new'))
        assert len(transport._BASE_STORE_CACHE) == 2
        assert transport._BASE_STORE_CACHE[('127.0.0.1', 0, True)].port == port
        assert not new_receiver.store.check(['broadcast_from/0/0'])
        new_sender.store.set('broadcast_from/0/0', b'new-nccl-id')
        assert new_receiver.store.get('broadcast_from/0/0') == b'new-nccl-id'
        assert old_receiver.store.get('broadcast_from/0/0') == b'old-nccl-id'
        assert all(group.socket is None for group in (old_sender, old_receiver, new_sender, new_receiver))

        client = textwrap.dedent("""
            import sys
            from types import ModuleType, SimpleNamespace
            from RL_Framework.infra.sync import nccl_weight_sync as transport
            utils = ModuleType('vllm.distributed.utils')
            utils.StatelessProcessGroup = lambda **kwargs: SimpleNamespace(**kwargs)
            sys.modules[utils.__name__] = utils
            group = transport._group(transport.NcclReloadSpec('127.0.0.1', int(sys.argv[1]),
                2, 1, 'cpu', timeout_s=2, session_id='new'))
            assert group.store.get('broadcast_from/0/0') == b'new-nccl-id'
            group.store.set('reconnected', b'yes')
            del group
            transport.clear_communicator_cache()
        """)
        result = subprocess.run([sys.executable, '-B', '-c', client, str(port)],
            env=dict(os.environ), capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert new_sender.store.get('reconnected') == b'yes'

        del old_sender, old_receiver, new_sender, new_receiver
        transport.clear_communicator_cache()
        gc.collect()
        assert not transport._BASE_STORE_CACHE
        # After whole cleanup, the actual port can be bound by a fresh master.
        fresh = transport._group(replace(sender, port=port, session_id='fresh'))
        fresh.store.set('healthy', b'yes')
        assert fresh.store.get('healthy') == b'yes'
        del fresh
        transport.clear_communicator_cache()
        gc.collect()
        print('NAMESPACED_STORE_RECONNECT_AND_CLEANUP_OK', flush=True)
    ''')
    root = Path(nccl_weight_sync.__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, '-B', '-c', script], cwd=root,
        env=dict(os.environ), capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'NAMESPACED_STORE_RECONNECT_AND_CLEANUP_OK' in completed.stdout

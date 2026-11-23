"""Stop only owned worker sessions and verify their live members exited."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


def _live_group_members(group):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_local_worker(proc, timeout=5.):
    """Use a dedicated process group when owned, retaining legacy mock support."""
    real_process = any(base.__module__ == "subprocess" and base.__name__ == "Popen"
                       for base in type(proc).__mro__)
    try:
        group = os.getpgid(proc.pid) if real_process else None
    except (ProcessLookupError, AttributeError):
        group = None
    # All actual runtime launches use start_new_session=True. Never signal a
    # shared shell/trainer process group for a legacy subprocess or test double.
    owned_group = group == proc.pid or (real_process and group is None and bool(_live_group_members(proc.pid)))
    if group is not None and not owned_group and proc.poll() is None:
        raise RuntimeError('Hybrid launcher lacks a dedicated owned process group; exit cannot be proved safely')
    if owned_group:
        group = proc.pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + timeout
            while _live_group_members(group) and time.monotonic() < deadline:
                time.sleep(.02)
            if not _live_group_members(group):
                break
        if _live_group_members(group):
            raise RuntimeError(f'Hybrid process group remains live: {group}')
        proc.wait(timeout=timeout)
    else:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout)
    if proc.poll() is None:
        raise RuntimeError('Hybrid launcher has not exited')

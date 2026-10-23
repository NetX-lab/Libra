#!/usr/bin/env python3
"""Jump-host controller for remote project-owned elastic hybrid workers.

Training ranks create ``.launch_<worker>.json`` requests in the shared task
directory.  This controller consumes them, launches the worker through the
existing password-authenticated internal SSH helper, and records the remote
PID.  A ``.stop_<worker>.json`` request terminates only the recorded worker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shlex
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--project-dir", default="/opt/libra/RL_Framework_NPU")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _remote_stop_code(pid):
    """Terminate the owned session and verify no live worker descendants remain."""
    return f"""import os, signal, time
pid = {int(pid)}
def live():
    result = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            fields = open('/proc/' + entry + '/stat').read().rsplit(')', 1)[1].split()
            if (int(fields[2]) == pid or int(entry) == pid) and fields[0] != 'Z':
                result.append(int(entry))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result
for sig in (signal.SIGTERM, signal.SIGKILL):
    for owned in live():
        try: os.kill(owned, sig)
        except ProcessLookupError: pass
    deadline = time.monotonic() + 10
    while live() and time.monotonic() < deadline:
        time.sleep(.1)
    if not live():
        break
if live():
    raise RuntimeError('owned worker session still alive')
"""


class Controller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.task_dir = args.task_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.stopped = False
        self.remote_pids: dict[str, tuple[str, int]] = {}

    def stop(self, *_args) -> None:
        self.stopped = True

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        while not self.stopped:
            handled = self._handle_stop_requests()
            handled |= self._handle_launch_requests()
            if not handled:
                time.sleep(self.args.poll_interval)
        self._stop_all_workers()

    def _handle_launch_requests(self) -> bool:
        handled = False
        for pending in sorted(self.task_dir.glob(".launch_*.json")):
            worker_id = pending.stem[len(".launch_") :]
            running = pending.with_suffix(".running")
            try:
                pending.replace(running)
            except OSError:
                continue
            handled = True
            started = self.task_dir / f".launch_{worker_id}.started"
            error = self.task_dir / f".launch_{worker_id}.error"
            try:
                request = _read_json(running)
                host = str(request["host"])
                command = str(request["command"])
                log_path = self.task_dir / f"{worker_id}.controller.log"
                remote_command = (
                    f"nohup setsid sh -c {shlex.quote(command)} >{shlex.quote(str(log_path))} 2>&1 & echo $!"
                )
                env = os.environ.copy()
                env.setdefault("INTERNAL_SSH_TIMEOUT", "30")
                helper = str(Path(self.args.project_dir) / "scripts" / "internal_ssh.sh")
                proc = subprocess.run(
                    [helper, host, "--", remote_command],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=False,
                )
                pids = re.findall(r"\b(\d+)\b", proc.stdout or "")
                if proc.returncode != 0 or not pids:
                    raise RuntimeError(
                        f"internal ssh launch failed rc={proc.returncode}: "
                        f"{(proc.stderr or proc.stdout)[-500:]}"
                    )
                pid = int(pids[-1])
                self.remote_pids[worker_id] = (host, pid)
                _write_json(started, {"worker_id": worker_id, "host": host, "pid": pid})
            except Exception as exc:
                _write_json(error, {"worker_id": worker_id, "error": str(exc)})
            finally:
                running.unlink(missing_ok=True)
        return handled

    def _handle_stop_requests(self) -> bool:
        handled = False
        for request_path in sorted(self.task_dir.glob(".stop_*.json")):
            worker_id = request_path.stem[len(".stop_") :]
            running = request_path.with_suffix(".running")
            try:
                request_path.replace(running)
            except FileNotFoundError:
                continue
            handled = True
            request = _read_json(running)
            request_id = request.get("request_id")
            try:
                # Withdraw launches that the controller has not claimed.
                (self.task_dir / f".launch_{worker_id}.json").unlink(missing_ok=True)
                record = self.remote_pids.get(worker_id)
                if record is None:
                    started = self.task_dir / f".launch_{worker_id}.started"
                    if started.exists():
                        raise RuntimeError("controller lost process ownership; refusing a stale PID")
                host, pid = record if record is not None else (str(request.get("host", "")), -1)
                if pid > 0:
                    helper = str(Path(self.args.project_dir) / "scripts" / "internal_ssh.sh")
                    code = _remote_stop_code(pid)
                    proc = subprocess.run(
                        [helper, host, "--", "python3 -c " + shlex.quote(code)],
                        env=os.environ.copy(), capture_output=True, text=True,
                        timeout=45, check=False,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(f"remote worker did not exit rc={proc.returncode}: {proc.stderr[-500:]}")
                self.remote_pids.pop(worker_id, None)
                _write_json(self.task_dir / f".stop_{worker_id}.done", {
                    "worker_id": worker_id, "host": host, "pid": pid,
                    "request_id": request_id, "exited": True, "finished_at": time.time(),
                })
            except Exception as exc:
                _write_json(self.task_dir / f".stop_{worker_id}.error", {
                    "worker_id": worker_id, "request_id": request_id, "error": str(exc),
                })
            finally:
                running.unlink(missing_ok=True)
        return handled

    def _stop_all_workers(self) -> None:
        for worker_id in list(self.remote_pids):
            path = self.task_dir / f".stop_{worker_id}.json"
            _write_json(path, {"worker_id": worker_id})
        self._handle_stop_requests()


def main() -> None:
    Controller(parse_args()).run()


if __name__ == "__main__":
    main()

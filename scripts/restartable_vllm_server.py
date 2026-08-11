#!/usr/bin/env python3
"""Run a vLLM server and reload full checkpoints via a file handshake."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--ready-timeout", type=float, default=1200.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("vLLM command is required after --")
    return args


def wait_until_ready(process: subprocess.Popen, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s: {url}")


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextlib.contextmanager
def node_reload_lock(control_dir: Path):
    node_name = os.uname().nodename.split(".")[0]
    lock_dir = control_dir / "node_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{node_name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main() -> int:
    args = parse_args()
    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    request_path = control_dir / "reload_request.json"
    current_version = 0
    process: subprocess.Popen | None = None

    def request_shutdown(signum, _frame):
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    def launch(model_path: str) -> subprocess.Popen:
        command = [model_path if item == "__MODEL_PATH__" else item for item in args.command]
        print(f"[vllm-reloader] launch version={current_version} model={model_path}", flush=True)
        child_env = os.environ.copy()
        child_env["MODEL_PATH"] = model_path
        child = subprocess.Popen(command, start_new_session=True, env=child_env)
        wait_until_ready(child, args.health_url, args.ready_timeout)
        return child

    try:
        process = launch(args.initial_model)
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited unexpectedly: {process.returncode}")
            if request_path.exists():
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    requested_version = int(request["version"])
                    checkpoint_path = str(request["checkpoint_path"])
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    time.sleep(1)
                    continue
                if requested_version > current_version:
                    if not Path(checkpoint_path).is_dir():
                        raise FileNotFoundError(checkpoint_path)
                    current_version = requested_version
                    try:
                        with node_reload_lock(control_dir):
                            stop_process(process)
                            process = launch(checkpoint_path)
                    except Exception as exc:
                        error_path = control_dir / f"error_{args.instance_id}_{current_version}.json"
                        error_path.write_text(
                            json.dumps(
                                {
                                    "instance_id": args.instance_id,
                                    "version": current_version,
                                    "checkpoint_path": checkpoint_path,
                                    "error": str(exc),
                                    "failed_at": time.time(),
                                }
                            ),
                            encoding="utf-8",
                        )
                        raise
                    ack_path = control_dir / f"ack_{args.instance_id}_{current_version}.json"
                    ack_path.write_text(
                        json.dumps(
                            {
                                "instance_id": args.instance_id,
                                "version": current_version,
                                "checkpoint_path": checkpoint_path,
                                "ready_at": time.time(),
                            }
                        ),
                        encoding="utf-8",
                    )
                    print(f"[vllm-reloader] ready version={current_version}", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        return 130
    finally:
        stop_process(process)


if __name__ == "__main__":
    sys.exit(main())

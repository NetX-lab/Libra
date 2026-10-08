#!/usr/bin/env python3
"""Run vLLM and refresh checkpoints through restart or in-place reload."""

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
    parser.add_argument(
        "--reload-strategy",
        choices=("parallel", "serial"),
        default="parallel",
    )
    parser.add_argument(
        "--reload-method",
        choices=("restart", "inplace"),
        default="restart",
    )
    parser.add_argument("--poll-interval", type=float, default=0.25)
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


def reload_inplace(health_url: str, checkpoint_path: str, timeout: float) -> dict:
    """Ask a running vLLM server to replace its resident model weights."""
    endpoint = health_url.rsplit("/", 1)[0] + "/reload_weights"
    payload = json.dumps(
        {"checkpoint_path": checkpoint_path, "timeout_seconds": timeout}
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout + 10) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM reload returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def reload_nccl(
    health_url: str,
    payload: dict,
    timeout: float,
) -> dict:
    """Ask a resident vLLM server to receive weights over NCCL."""
    endpoint = health_url.rsplit("/", 1)[0] + "/reload_weights_nccl"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout + 10) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM NCCL reload returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


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


def write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically publish a reload acknowledgement or error."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


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
                    reload_mode = str(request.get("mode", "checkpoint")).lower()
                    checkpoint_path = str(request.get("checkpoint_path", ""))
                    reload_method = str(
                        request.get("reload_method", args.reload_method)
                    ).lower()
                    reload_strategy = str(
                        request.get("reload_strategy", args.reload_strategy)
                    ).lower()
                    if reload_mode not in {"checkpoint", "nccl"}:
                        raise ValueError(f"unsupported reload mode: {reload_mode}")
                    if reload_mode == "checkpoint" and reload_method not in {"restart", "inplace"}:
                        raise ValueError(f"unsupported reload method: {reload_method}")
                    if reload_strategy not in {"parallel", "serial"}:
                        raise ValueError(
                            f"unsupported reload strategy: {reload_strategy}"
                        )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    time.sleep(max(0.05, args.poll_interval))
                    continue
                if requested_version > current_version:
                    if reload_mode == "checkpoint" and not Path(checkpoint_path).is_dir():
                        raise FileNotFoundError(checkpoint_path)
                    current_version = requested_version
                    reload_started_at = time.time()
                    response: dict = {}
                    try:
                        reload_guard = (
                            node_reload_lock(control_dir)
                            if reload_strategy == "serial" and reload_mode != "nccl"
                            else contextlib.nullcontext()
                        )
                        with reload_guard:
                            guard_acquired_at = time.time()
                            if reload_mode == "nccl":
                                stopped_at = guard_acquired_at
                                response = reload_nccl(
                                    args.health_url,
                                    {
                                        "host": str(request["nccl_host"]),
                                        "port": int(request["nccl_port"]),
                                        "world_size": int(request["nccl_world_size"]),
                                        "rank_offset": int(
                                            request["nccl_rank_offsets"][args.instance_id]
                                        ),
                                        "timeout_seconds": float(args.ready_timeout),
                                        "chunk_bytes": int(request["nccl_chunk_bytes"]),
                                    },
                                    args.ready_timeout,
                                )
                            elif reload_method == "restart":
                                stop_process(process)
                                stopped_at = time.time()
                                process = launch(checkpoint_path)
                            else:
                                stopped_at = guard_acquired_at
                                response = reload_inplace(
                                    args.health_url,
                                    checkpoint_path,
                                    args.ready_timeout,
                                )
                    except Exception as exc:
                        error_path = control_dir / f"error_{args.instance_id}_{current_version}.json"
                        write_json_atomic(
                            error_path,
                            {
                                "instance_id": args.instance_id,
                                "version": current_version,
                                "checkpoint_path": checkpoint_path,
                                "reload_strategy": reload_strategy,
                                "reload_method": reload_method,
                                "error": str(exc),
                                "failed_at": time.time(),
                            },
                        )
                        raise
                    ack_path = control_dir / f"ack_{args.instance_id}_{current_version}.json"
                    ready_at = time.time()
                    write_json_atomic(
                        ack_path,
                        {
                            "instance_id": args.instance_id,
                            "version": current_version,
                            "checkpoint_path": checkpoint_path,
                            "reload_strategy": reload_strategy,
                            "reload_method": reload_method,
                            "reload_started_at": reload_started_at,
                            "guard_acquired_at": guard_acquired_at,
                            "stopped_at": stopped_at,
                            "ready_at": ready_at,
                            "lock_wait_seconds": guard_acquired_at
                            - reload_started_at,
                            "stop_seconds": stopped_at - guard_acquired_at,
                            "load_seconds": ready_at - stopped_at,
                            "total_seconds": ready_at - reload_started_at,
                            "server_response": response,
                        },
                    )
                    print(
                        f"[vllm-reloader] ready version={current_version} "
                        f"method={reload_method} strategy={reload_strategy} "
                        f"reload_seconds={ready_at - reload_started_at:.2f}",
                        flush=True,
                    )
            time.sleep(max(0.05, args.poll_interval))
    except KeyboardInterrupt:
        return 130
    finally:
        stop_process(process)
    return 0


if __name__ == "__main__":
    sys.exit(main())

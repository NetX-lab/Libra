#!/usr/bin/env python3
"""GPU smoke test for parallel rollout weight reloads."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def _child(port: int, startup_delay: float) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Slurm allocation")
    value = (torch.ones(1024, device="cuda") * 2).sum().item()
    torch.cuda.synchronize()
    if value != 2048:
        raise RuntimeError(f"unexpected CUDA result: {value}")
    time.sleep(startup_delay)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/reload_weights":
                self.send_response(404)
                self.end_headers()
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            checkpoint_path = Path(payload["checkpoint_path"])
            if not checkpoint_path.is_dir():
                self.send_response(400)
                self.end_headers()
                return
            time.sleep(startup_delay)
            response = json.dumps(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "workers": [{"rank": 0}],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    return 0


def _wait_for_paths(paths: list[Path], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return
        time.sleep(0.1)
    missing = [str(path) for path in paths if not path.exists()]
    raise TimeoutError(f"timed out waiting for reload ACKs: {missing}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(ports: tuple[int, ...], timeout: float) -> None:
    pending = set(ports)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for port in list(pending):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health",
                    timeout=1,
                ) as response:
                    if response.status == 200:
                        pending.remove(port)
            except Exception:
                pass
        if pending:
            time.sleep(0.1)
    if pending:
        raise TimeoutError(
            f"initial services did not become healthy: {sorted(pending)}"
        )


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _run_reload(
    *,
    control_dir: Path,
    checkpoint: Path,
    instance_ids: tuple[str, ...],
    version: int,
    method: str,
    timeout: float,
) -> dict:
    request = {
        "version": version,
        "checkpoint_path": str(checkpoint),
        "reload_method": method,
        "reload_strategy": "parallel",
        "instance_ids": list(instance_ids),
        "created_at": time.time(),
    }
    request_tmp = control_dir / "reload_request.json.tmp"
    request_tmp.write_text(json.dumps(request), encoding="utf-8")
    started_at = time.monotonic()
    request_tmp.replace(control_dir / "reload_request.json")
    ack_paths = [
        control_dir / f"ack_{instance_id}_{version}.json"
        for instance_id in instance_ids
    ]
    _wait_for_paths(ack_paths, timeout)
    elapsed = time.monotonic() - started_at
    acks = [json.loads(path.read_text(encoding="utf-8")) for path in ack_paths]
    if any(ack["reload_method"] != method for ack in acks):
        raise AssertionError(f"unexpected reload method in ACK batch: {acks}")
    if any(ack["reload_strategy"] != "parallel" for ack in acks):
        raise AssertionError(f"unexpected reload strategy in ACK batch: {acks}")
    started = [float(ack["reload_started_at"]) for ack in acks]
    ready = [float(ack["ready_at"]) for ack in acks]
    start_skew = max(started) - min(started)
    reload_span = max(ready) - min(started)
    serial_work = sum(float(ack["total_seconds"]) for ack in acks)
    overlap_seconds = min(ready) - max(started)
    if overlap_seconds <= 0:
        raise AssertionError(
            f"reload intervals did not overlap: method={method}, "
            f"span={reload_span:.2f}s, serial_work={serial_work:.2f}s"
        )
    return {
        "method": method,
        "batch_elapsed_s": round(elapsed, 3),
        "reload_span_s": round(reload_span, 3),
        "serial_work_s": round(serial_work, 3),
        "reload_start_skew_s": round(start_skew, 3),
        "reload_overlap_s": round(overlap_seconds, 3),
        "slowest_instance_s": round(
            max(float(ack["total_seconds"]) for ack in acks),
            3,
        ),
    }


def _orchestrate(startup_delay: float, timeout: float) -> int:
    project_dir = Path(__file__).resolve().parents[1]
    reloader = project_dir / "scripts" / "restartable_vllm_server.py"
    with tempfile.TemporaryDirectory(prefix="rollout-reload-smoke-") as tmp:
        root = Path(tmp)
        control_dir = root / "control"
        initial_model = root / "initial_model"
        inplace_checkpoint = root / "checkpoint_v1"
        restart_checkpoint = root / "checkpoint_v2"
        control_dir.mkdir()
        initial_model.mkdir()
        inplace_checkpoint.mkdir()
        restart_checkpoint.mkdir()
        ports = (_free_port(), _free_port())
        instance_ids = ("smoke_0", "smoke_1")
        processes = []
        try:
            for instance_id, port in zip(instance_ids, ports):
                command = [
                    sys.executable,
                    str(reloader),
                    "--instance-id",
                    instance_id,
                    "--control-dir",
                    str(control_dir),
                    "--health-url",
                    f"http://127.0.0.1:{port}/health",
                    "--initial-model",
                    str(initial_model),
                    "--ready-timeout",
                    str(timeout),
                    "--poll-interval",
                    "0.05",
                    "--",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--port",
                    str(port),
                    "--startup-delay",
                    str(startup_delay),
                    "--model-path",
                    "__MODEL_PATH__",
                ]
                processes.append(subprocess.Popen(command, start_new_session=True))

            _wait_for_health(ports, timeout)
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a reload supervisor exited during initial startup")

            inplace_result = _run_reload(
                control_dir=control_dir,
                checkpoint=inplace_checkpoint,
                instance_ids=instance_ids,
                version=1,
                method="inplace",
                timeout=timeout,
            )
            restart_result = _run_reload(
                control_dir=control_dir,
                checkpoint=restart_checkpoint,
                instance_ids=instance_ids,
                version=2,
                method="restart",
                timeout=timeout,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "torch_cuda_smoke": True,
                        "instances": len(instance_ids),
                        "startup_delay_s": startup_delay,
                        "reloads": [inplace_result, restart_result],
                    },
                    sort_keys=True,
                )
            )
            return 0
        finally:
            for process in processes:
                _stop(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-delay", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()
    if args.child:
        del args.model_path
        return _child(args.port, args.startup_delay)
    return _orchestrate(args.startup_delay, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())

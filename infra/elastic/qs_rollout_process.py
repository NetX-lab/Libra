"""Dispatch rollout lifecycle requests to controllers in the existing QS Pods."""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from .rollout_process import _write, stop_owned_session


def _request(root, host, action, deadline_s, **fields):
    request_id = uuid4().hex
    directory = Path(root) / str(ipaddress.IPv4Address(host))
    path = directory / f"{request_id}.request"
    _write(path, dict(action=action, **fields))
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            reply = json.loads(path.with_suffix(".reply").read_text())
        except FileNotFoundError:
            time.sleep(.1)
            continue
        if reply.get("error"):
            raise RuntimeError(f"QS rollout {action} failed on {host}: {reply['error']}")
        return reply
    _write(path.with_suffix(".cancel"), {})
    raise TimeoutError(f"No QS rollout {action} receipt from {host}")


class QSProcess:
    def __init__(self, root, host, reply):
        self.root, self.host = root, host
        self.pid, self.token = reply["pid"], reply["token"]

    def poll(self):
        path = Path(self.root) / str(ipaddress.IPv4Address(self.host)) / f"{self.token}.exit"
        try:
            return json.loads(path.read_text())["code"]
        except FileNotFoundError:
            return None

    def stop(self, timeout):
        _request(self.root, self.host, "stop", timeout + 10, pid=self.pid,
                 token=self.token, timeout=timeout)


def launch(meta, env):
    root = env["LIBRA_QS_ROLLOUT_DIR"]
    fields = ("RL_FRAMEWORK_ROLE", "LIBRA_INSTANCE_ID", "LIBRA_SERVICE_LABEL",
              "LIBRA_EXPECTED_TP_SIZE", "LIBRA_CAPTURE_BEHAVIOR", "LIBRA_CHECK_PREFIX_CACHE",
              "VLLM_ENABLE_V1_MULTIPROCESSING", "LIBRA_KV_STORAGE_PATH")
    options = {name: env[name] for name in fields if name in env}
    reply = _request(root, meta.host, "launch", 60, command=meta.command,
                     instance_id=meta.instance_id, port=meta.port, gpus=meta.gpus,
                     options=options, log_path=meta.log_path)
    return QSProcess(root, meta.host, reply)


def stop_adopted(root, meta, timeout):
    return _request(root, meta.host, "stop", timeout + 10, pid=meta.pid,
                    instance_id=meta.instance_id, port=meta.port, timeout=timeout)


class Controller:
    def __init__(self, root, host, gpus):
        self.directory = Path(root) / str(ipaddress.IPv4Address(host))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gpus = set(gpus)
        self.processes = {}

    def register(self, proc, instance_id, port, gpus):
        token = uuid4().hex
        self.processes[token] = (proc, instance_id, port, set(gpus))
        return {"pid": proc.pid, "token": token}

    def _launch(self, request):
        gpus = request["gpus"]
        if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= self.gpus:
            raise ValueError("Rollout launch uses GPUs outside this QS controller")
        for _, _, port, used in self.processes.values():
            if port == request["port"] or used.intersection(gpus):
                raise RuntimeError("Rollout launch overlaps an owned port or GPU")
        devices = ",".join(map(str, gpus))
        env = dict(os.environ, **request["options"])
        env.update(CUDA_VISIBLE_DEVICES=devices, PLANNED_CUDA_VISIBLE_DEVICES=devices)
        with open(request["log_path"], "a", encoding="utf-8") as log:
            proc = subprocess.Popen(["bash", "-c", "exec " + request["command"]],
                                    env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        return self.register(proc, request["instance_id"], request["port"], gpus)

    def _stop(self, request):
        matches = [(token, row[0]) for token, row in self.processes.items()
                   if row[0].pid == request["pid"]
                   and (token == request["token"] if "token" in request else
                        row[1:3] == (request["instance_id"], request["port"]))]
        if len(matches) != 1:
            raise RuntimeError("QS controller cannot identify the owned rollout")
        token, proc = matches[0]
        stop_owned_session(proc, request["timeout"])
        _write(self.directory / f"{token}.exit", {"code": proc.returncode})
        del self.processes[token]
        return {"owned_processes_exited": True}

    def poll(self):
        for path in sorted(self.directory.glob("*.request")):
            running = path.with_suffix(".running")
            path.replace(running)
            request = json.loads(running.read_text())
            cancelled = path.with_suffix(".cancel")
            try:
                if cancelled.exists():
                    raise RuntimeError("QS rollout request was cancelled")
                handler = {"launch": self._launch, "stop": self._stop}[request["action"]]
                reply = handler(request)
                if cancelled.exists() and request["action"] == "launch":
                    self._stop(dict(reply, timeout=10))
                    raise RuntimeError("QS rollout launch was cancelled")
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                reply = {"error": str(exc)}
            _write(path.with_suffix(".reply"), reply)
            running.unlink()
        for token, (proc, _, _, _) in self.processes.items():
            code = proc.poll()
            if code is not None and not (self.directory / f"{token}.exit").exists():
                _write(self.directory / f"{token}.exit", {"code": code})

    def close(self):
        errors = []
        for token, (proc, _, _, _) in list(self.processes.items()):
            try:
                self._stop({"token": token, "pid": proc.pid, "timeout": 15})
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

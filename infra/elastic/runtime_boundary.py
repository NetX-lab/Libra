"""File-based decisions for trainer reconfiguration boundaries (no collectives)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def read_record(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RuntimeBoundaryProtocol:
    """One leader and one trainer per rank, sharing a job-specific directory.

    Fresh follower nonces must be acknowledged by the current leader. Reading
    an old leader announcement or old membership file cannot join an old run.
    Decisions are immutable and independent of planner Future completion.
    """

    def __init__(self, directory: Path, rank: int, world_size: int, timeout: float):
        self.directory = Path(directory)
        self.rank = rank
        self.world_size = world_size
        self.timeout = max(0.0, float(timeout))
        self.run_id = ""

    def join(self) -> None:
        session = self.directory / "session"
        nonce = uuid.uuid4().hex
        deadline = time.monotonic() + self.timeout
        if self.rank == 0:
            run_id = uuid.uuid4().hex
            write_record(session / "leader.json", {
                "run_id": run_id, "world_size": self.world_size,
            })
        else:
            run_id = ""
        announced = ""
        while True:
            if self.rank == 0:
                members = {"0": nonce}
                for rank in range(1, self.world_size):
                    peer = read_record(session / f"rank_{rank}.json")
                    if peer.get("run_id") == run_id and peer.get("nonce"):
                        members[str(rank)] = peer["nonce"]
                if len(members) == self.world_size:
                    write_record(session / run_id / "members.json", members)
                    self.run_id = run_id
                    return
            else:
                leader = read_record(session / "leader.json")
                candidate = leader.get("run_id", "")
                if candidate and leader.get("world_size") == self.world_size:
                    if candidate != announced:
                        write_record(session / f"rank_{self.rank}.json", {
                            "run_id": candidate, "nonce": nonce,
                        })
                        announced = candidate
                    members = read_record(session / candidate / "members.json")
                    if members.get(str(self.rank)) == nonce:
                        self.run_id = candidate
                        return
            if time.monotonic() >= deadline:
                raise TimeoutError("runtime reconfiguration session handshake timed out")
            time.sleep(0.05)

    def decision_path(self, step: int, point: str) -> Path:
        if not self.run_id or point not in {"pre", "post"}:
            raise ValueError("a joined session and a pre/post boundary are required")
        return self.directory / "decisions" / self.run_id / f"step_{step}_{point}.json"

    def publish(self, step: int, point: str, *, coord_id: str = "",
                drain_required: bool = False) -> None:
        if self.rank != 0:
            raise RuntimeError("only rank0 can publish a boundary decision")
        payload = {
            "run_id": self.run_id, "step": step, "point": point,
            "decision": "reconfig" if coord_id else "no_reconfig",
            "coord_id": coord_id, "drain_required": drain_required,
        }
        path = self.decision_path(step, point)
        if path.exists():
            raise RuntimeError(f"boundary already closed: {path}")
        write_record(path, payload)

    def wait(self, step: int, point: str) -> dict:
        path = self.decision_path(step, point)
        deadline = time.monotonic() + self.timeout
        while True:
            record = read_record(path)
            if (record.get("run_id") == self.run_id
                    and record.get("step") == step and record.get("point") == point):
                if record.get("decision") == "no_reconfig":
                    if record.get("coord_id") or record.get("drain_required"):
                        raise RuntimeError(f"invalid no-reconfiguration decision: {path}")
                    return record
                if (record.get("decision") == "reconfig" and record.get("coord_id")
                        and isinstance(record.get("drain_required"), bool)):
                    return record
                raise RuntimeError(f"invalid boundary decision: {path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"runtime reconfiguration boundary decision missing: {path}")
            time.sleep(0.05)

    def runtime_done_path(self) -> Path:
        if not self.run_id:
            raise ValueError("a joined session is required")
        return self.directory / "runtime_done" / f"{self.run_id}.json"

    def publish_runtime_done(self, *, status: str, error: str = "") -> None:
        """Publish the leader's final executor-close outcome exactly once."""
        if self.rank != 0:
            raise RuntimeError("only rank0 can publish runtime completion")
        if status not in {"success", "failed"}:
            raise ValueError(f"invalid runtime completion status: {status}")
        path = self.runtime_done_path()
        if path.exists():
            raise RuntimeError(f"runtime completion already published: {path}")
        write_record(path, {
            "run_id": self.run_id,
            "status": status,
            "error": str(error),
            "updated_at": time.time(),
        })

    def wait_runtime_done(self, *, timeout: float | None = None) -> dict:
        """Wait for current-run teardown permission, or surface leader failure."""
        path = self.runtime_done_path()
        deadline = time.monotonic() + (
            self.timeout if timeout is None else max(0.0, float(timeout))
        )
        while True:
            record = read_record(path)
            if record.get("run_id") == self.run_id:
                if record.get("status") == "success":
                    return record
                if record.get("status") == "failed":
                    raise RuntimeError(
                        "runtime executor shutdown failed on rank0: "
                        f"{record.get('error', '')}"
                    )
                if record:
                    raise RuntimeError(f"invalid runtime completion record: {path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"runtime completion missing: {path}")
            time.sleep(0.05)

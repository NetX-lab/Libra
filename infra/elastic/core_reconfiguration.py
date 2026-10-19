"""Collective core-state operations inside the existing training boundary.

Only small commands and receipts use the shared filesystem. Model/Adam tensors
stay in memory and move through the live-state or topology transport.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from RL_Framework.infra.elastic.runtime_boundary import read_record, write_record


def _run_operation(engine, command, trainer=None):
    operation = command["operation"]
    if operation == "capture":
        revision = int(engine.capture_elastic_state_snapshot("boundary", "all"))
        return {
            "optimizer_revision": revision,
            "snapshot_uri": engine.get_elastic_state_snapshot_path(revision),
            "lane": engine.get_elastic_lane_state(),
        }
    if operation == "release":
        for revision in command.get("revisions", []):
            engine.release_elastic_state_snapshot(int(revision))
        return {}
    if operation == "publish":
        if trainer is None:
            raise RuntimeError("collective publication requires the live trainer")
        target = dict(command["target"])
        ids = [instance["instance_id"] for instance in target["instances"]]
        target["unavailable_rollout_ids"] = ids
        trainer._restored_rollouts_pending_publication = set(ids)
        trainer._apply_runtime_reconfiguration_state(target)
        revision = int(engine.optimizer_revision)
        from .rollout_lifecycle import publish_with_consensus
        publish_with_consensus(trainer, revision)
        engine.set_version(revision)
        trainer._reopen_restored_rollouts()
        trainer._rebind_rollout_engine_from_config(reason="core_boundary_publication")
        return {"optimizer_revision": revision, "acknowledged": True}
    if operation != "rebuild":
        raise ValueError(f"unknown collective core operation: {operation}")
    if trainer is not None:
        engine._elastic_step_hybrid_workers = ()
        server = getattr(trainer, "_elastic_gradient_server", None)
        if server is not None:
            server.close()
            trainer._elastic_gradient_server = None
    receipt = engine.reconfigure_parallelism(
        command["target"], transaction_id=command["command_id"],
    )
    if not receipt.get("acknowledged"):
        raise RuntimeError("core topology rebuild did not acknowledge installed state")
    if trainer is not None:
        for field, value in command["target"].items():
            if hasattr(trainer.config, field):
                setattr(trainer.config, field, value)
        trainer.config.tp_size = trainer.config.train_tp_size
        trainer._initialize_elastic_communication_domains()
        trainer._init_nonblocking_elastic_core()
        hook = getattr(trainer, "_on_train_engine_reconfigured", None)
        if callable(hook):
            hook()
    return receipt


def execute_core_command(engine, command, directory: Path, trainer=None):
    rank = int(getattr(engine, "rank", 0))
    receipt = {
        "command_id": command["command_id"], "coord_id": command["coord_id"],
        "run_id": command["run_id"], "rank": rank,
    }
    try:
        receipt.update(_run_operation(engine, command, trainer))
        receipt["status"] = "success"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_record(directory / "core_receipts" / f"rank_{rank}.json", receipt)
    return receipt


def follow_core_command(trainer, directory: Path, coord_id: str):
    command = read_record(directory / "core_command.json")
    if (command.get("coord_id") != coord_id
            or not trainer._runtime_state_is_current(command)):
        return
    handled = getattr(trainer, "_handled_core_commands", set())
    command_id = command.get("command_id")
    if not command_id or command_id in handled:
        return
    receipt = execute_core_command(trainer.train_engine, command, directory, trainer)
    handled.add(command_id)
    trainer._handled_core_commands = handled
    if receipt["status"] != "success":
        raise RuntimeError(receipt["error"])


def dispatch_core_command(executor, operation: str, *, target=None, revisions=()):
    """Run on every core rank already parked in this transaction boundary."""
    engine = executor.train_engine
    directory = executor._coordination_dir()
    command = {
        "operation": operation, "target": target or {},
        "revisions": list(revisions), "command_id": uuid.uuid4().hex,
        "coord_id": executor._active_runtime_coord_id,
        "run_id": executor.runtime_run_id,
    }
    write_record(directory / "core_command.json", command)
    execute_core_command(engine, command, directory, getattr(executor, "trainer", None))
    timeout = float(getattr(executor.config.global_resource_planner, "runtime_drain_timeout_s", 3600))
    deadline = time.monotonic() + timeout if timeout > 0 else None
    size = int(getattr(engine, "world_size", 1))
    while True:
        receipts = [read_record(directory / "core_receipts" / f"rank_{r}.json") for r in range(size)]
        matches = [r for rank, r in enumerate(receipts)
                   if r.get("command_id") == command["command_id"]
                   and r.get("run_id") == command["run_id"]
                   and r.get("coord_id") == command["coord_id"]
                   and r.get("rank") == rank]
        errors = [r for r in matches if r.get("status") == "failed"]
        if errors:
            raise RuntimeError(f"collective core {operation} failed: {errors}")
        if len(matches) == size:
            return matches
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"core {operation} missing rank receipts: {len(matches)}/{size}")
        if executor._closing.wait(0.02):
            raise RuntimeError("runtime executor closed during core state operation")

"""Every parked training rank must execute fixed-topology live-state commands."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from RL_Framework.infra.elastic import core_reconfiguration as protocol
from RL_Framework.infra.elastic.runtime_boundary import read_record, write_record


class Engine:
    def __init__(self, rank, calls, barrier=None, failing_rank=None):
        self.rank, self.world_size = rank, 2
        self.calls, self.barrier, self.failing_rank = calls, barrier, failing_rank

    def capture_elastic_state_snapshot(self, worker, target):
        self.calls.append((self.rank, "capture", worker, target))
        if self.barrier:
            self.barrier.wait(timeout=3)
        if self.rank == self.failing_rank:
            raise RuntimeError("rank-local capture failed")
        return 17

    def get_elastic_state_snapshot_path(self, revision):
        return f"ehp://rank-{self.rank}/revision-{revision}"

    def get_elastic_lane_state(self):
        return {"global_rank": self.rank, "tensor_parallel_rank": self.rank,
                "data_parallel_rank": 0}

    def release_elastic_state_snapshot(self, revision):
        self.calls.append((self.rank, "release", revision))



class Trainer:
    def __init__(self, engine, calls):
        self.train_engine = engine
        self.calls = calls
        self.config = SimpleNamespace(train_tp_size=2, train_dp_size=1, tp_size=2)
        self._elastic_gradient_server = SimpleNamespace(
            close=lambda: calls.append((engine.rank, "close-gradient-server")))

    def _runtime_state_is_current(self, command):
        return command.get("run_id") == "run-current"

    def _initialize_elastic_communication_domains(self):
        self.calls.append((self.train_engine.rank, "new-comm-domain"))

    def _init_nonblocking_elastic_core(self):
        self.calls.append((self.train_engine.rank, "new-gradient-server"))

    def _on_train_engine_reconfigured(self):
        self.calls.append((self.train_engine.rank, "new-topology-hook"))


def executor(directory, engine, trainer=None, timeout=1):
    return SimpleNamespace(
        train_engine=engine, trainer=trainer,
        _coordination_dir=lambda: directory, _active_runtime_coord_id="coord-current",
        runtime_run_id="run-current", _closing=threading.Event(),
        config=SimpleNamespace(global_resource_planner=SimpleNamespace(runtime_drain_timeout_s=timeout)),
    )


def follow_one(trainer, directory):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if read_record(directory / "core_command.json").get("command_id"):
            protocol.follow_core_command(trainer, directory, "coord-current")
            return
        time.sleep(0.001)
    raise AssertionError("leader did not publish command")


@pytest.mark.parametrize("operation", ["capture", "release"])
def test_every_training_rank_executes_command(tmp_path, operation):
    calls = []
    barrier = threading.Barrier(2) if operation != "release" else None
    engines = [Engine(rank, calls, barrier) for rank in range(2)]
    trainers = [Trainer(engine, calls) for engine in engines]
    with ThreadPoolExecutor(max_workers=1) as pool:
        peer = pool.submit(follow_one, trainers[1], tmp_path)
        receipts = protocol.dispatch_core_command(
            executor(tmp_path, engines[0], trainers[0]), operation,
            target={"train_tp_size": 1, "train_dp_size": 2}, revisions=(16, 17),
        )
        peer.result(timeout=4)
    assert {item["rank"] for item in receipts} == {0, 1}
    assert all(item["status"] == "success" for item in receipts)
    assert len({item["command_id"] for item in receipts}) == 1
    if operation == "capture":
        assert {item["snapshot_uri"] for item in receipts} == {
            "ehp://rank-0/revision-17", "ehp://rank-1/revision-17"}
        assert all(item["optimizer_revision"] == 17 for item in receipts)
    elif operation == "release":
        assert {(item[0], item[2]) for item in calls if item[1] == "release"} == {
            (0, 16), (0, 17), (1, 16), (1, 17)}


def test_one_rank_failure_reaches_dispatch_and_follower(tmp_path):
    calls, barrier = [], threading.Barrier(2)
    engines = [Engine(rank, calls, barrier, failing_rank=1) for rank in range(2)]
    trainers = [Trainer(engine, calls) for engine in engines]
    with ThreadPoolExecutor(max_workers=1) as pool:
        peer = pool.submit(follow_one, trainers[1], tmp_path)
        with pytest.raises(RuntimeError, match="collective core capture failed"):
            protocol.dispatch_core_command(executor(tmp_path, engines[0], trainers[0]), "capture",
                                           target={"train_tp_size": 1, "train_dp_size": 2})
        with pytest.raises(RuntimeError, match="rank-local capture failed"):
            peer.result(timeout=4)
    failed = read_record(tmp_path / "core_receipts" / "rank_1.json")
    assert failed["status"] == "failed"
    assert trainers[1].config.train_tp_size == 2


def test_followers_ignore_stale_or_repeated_commands(tmp_path):
    calls = []
    trainer = Trainer(Engine(1, calls), calls)
    command = {"command_id": "release-once", "coord_id": "coord-current",
               "run_id": "run-stale", "operation": "release", "revisions": [17]}
    write_record(tmp_path / "core_command.json", command)
    protocol.follow_core_command(trainer, tmp_path, "coord-current")
    assert calls == []
    command["run_id"] = "run-current"
    write_record(tmp_path / "core_command.json", command)
    protocol.follow_core_command(trainer, tmp_path, "wrong-coord")
    assert calls == []
    protocol.follow_core_command(trainer, tmp_path, "coord-current")
    protocol.follow_core_command(trainer, tmp_path, "coord-current")
    assert calls == [(1, "release", 17)]


@pytest.mark.parametrize("wrong", [{"coord_id": "coord-stale"}, {"rank": 0}, {"run_id": "run-stale"}])
def test_rank_coverage_requires_current_coord_and_exact_rank(tmp_path, monkeypatch, wrong):
    original = protocol.execute_core_command

    def local_and_invalid_peer(engine, command, directory, trainer):
        receipt = original(engine, command, directory, trainer)
        stale = dict(receipt, rank=1)
        stale.update(wrong)
        write_record(directory / "core_receipts" / "rank_1.json", stale)
        return receipt

    monkeypatch.setattr(protocol, "execute_core_command", local_and_invalid_peer)
    with pytest.raises(TimeoutError, match="missing rank receipts"):
        protocol.dispatch_core_command(executor(tmp_path, Engine(0, []), timeout=0.02), "release")


def test_zero_timeout_waits_for_delayed_rank(tmp_path, monkeypatch):
    original = protocol.execute_core_command
    peer_thread = []

    def delayed_peer(engine, command, directory, trainer):
        receipt = original(engine, command, directory, trainer)

        def publish():
            time.sleep(0.05)
            write_record(directory / "core_receipts" / "rank_1.json", dict(receipt, rank=1))

        thread = threading.Thread(target=publish)
        thread.start()
        peer_thread.append(thread)
        return receipt

    monkeypatch.setattr(protocol, "execute_core_command", delayed_peer)
    receipts = protocol.dispatch_core_command(executor(tmp_path, Engine(0, []), timeout=0), "release")
    assert len(receipts) == 2
    peer_thread[0].join(timeout=1)


def test_closing_executor_does_not_report_partial_success(tmp_path):
    instance = executor(tmp_path, Engine(0, []))
    instance._closing.set()
    with pytest.raises(RuntimeError, match="executor closed"):
        protocol.dispatch_core_command(instance, "release")


def test_topology_rebuild_is_outside_fixed_core_protocol(tmp_path):
    receipt = protocol.execute_core_command(Engine(0, []), {
        "command_id": "unsupported", "coord_id": "coord-current",
        "run_id": "run-current", "operation": "rebuild",
    }, tmp_path)
    assert receipt["status"] == "failed"
    assert "unknown collective core operation: rebuild" in receipt["error"]

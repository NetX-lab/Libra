"""Physical GPU ownership during EHP loans and remote process exit ACKs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from RL_Framework.infra.elastic.rollout_lending import lend_rollout, return_rollout, wait_hybrid_stopped
from RL_Framework.infra.elastic.runtime_executor import ManagedRolloutProcess
from RL_Framework.scripts.elastic_hybrid_controller import Controller, _remote_stop_code


def executor_fixture(tmp_path, *, remote=False):
    metas = [ManagedRolloutProcess(f"engine{i}", "vllm", i + 10, [2 * i, 2 * i + 1], 8000 + i,
                                   host="rollout-host", tp=2) for i in range(2)]
    handles = [SimpleNamespace(instance_id=meta.instance_id, is_ready=True) for meta in metas]
    events = []
    dispatcher = SimpleNamespace(paused=False)
    dispatcher.is_paused = lambda: dispatcher.paused
    def pause():
        dispatcher.paused = True
        events.append("pause")
    dispatcher.pause = pause
    dispatcher.resume = lambda: setattr(dispatcher, "paused", False)
    dispatcher.wait_until_idle = lambda timeout: events.append("dispatcher-drained")
    executor = SimpleNamespace(
        config=SimpleNamespace(train_tp_size=4,
            global_resource_planner=SimpleNamespace(runtime_drain_timeout_s=0, hybrid_worker_stop_timeout_s=.05),
            heterogeneous_rollout=SimpleNamespace(instances=[SimpleNamespace(host=m.host, gpus=m.gpus) for m in metas])),
        _process_meta={meta.instance_id: meta for meta in metas}, _hybrid_worker_meta={},
        _hybrid_worker_remote_control_enabled=remote, dispatcher=dispatcher,
        rollout_engine=SimpleNamespace(scheduler=SimpleNamespace(_instances=handles),
                                      wait_until_idle=lambda timeout: events.append("engine-drained")),
    )
    executor._adopt_existing_rollout_processes = lambda: None
    executor._hybrid_worker_control_dir = lambda: tmp_path
    def stop(instance_id):
        assert all(not handle.is_ready for handle in handles)
        events.append(f"stop-{instance_id}")
        return [executor._process_meta.pop(instance_id)]
    executor._stop_rollout_process = stop
    def start(meta):
        assert not executor._hybrid_worker_meta
        assert all(not handle.is_ready for handle in handles)
        events.append(f"start-{meta.instance_id}")
        executor._process_meta[meta.instance_id] = meta
        return meta
    executor._start_one_rollout_process = start
    executor._wait_rollout_process_start_ready = lambda meta: events.append(f"ready-{meta.instance_id}")
    executor._command_for_instance_id = lambda instance_id, port: f"restart {instance_id}:{port}"
    return executor, events, metas, handles


def test_tp4_can_borrow_both_tp2_engines_and_restore_exact_allocation(tmp_path):
    executor, events, original, handles = executor_fixture(tmp_path)
    slots = lend_rollout(executor, 4)
    assert len(slots) == 4
    assert events == ["pause", "dispatcher-drained", "engine-drained", "stop-engine0", "stop-engine1"]
    assert not executor._process_meta
    assert lend_rollout(executor, 4) == slots
    assert executor.dispatcher.paused
    restored = return_rollout(executor)
    assert [(m.instance_id, m.host, m.port, m.gpus, m.tp) for m in restored] == [
        (m.instance_id, m.host, m.port, m.gpus, m.tp) for m in original]
    assert len({m.incarnation_id for m in restored}) == 2
    assert all(m.service_generation is None for m in restored)
    assert all(not handle.is_ready for handle in handles)
    assert return_rollout(executor) == []


def test_rejected_candidate_is_never_used_to_select_lending_gpu(tmp_path):
    executor, _, _, _ = executor_fixture(tmp_path)
    executor.rejected_plan = SimpleNamespace(rollout_tp_list=[4], gpus=[99])
    slots = lend_rollout(executor, 4)
    assert {gpu for slot in slots.values() for gpu in slot["gpus"]} == {0, 1, 2, 3}


def test_failed_engine_stop_exposes_no_gpu_slots_and_can_recover(tmp_path):
    executor, _, _, _ = executor_fixture(tmp_path)
    stop = executor._stop_rollout_process
    def fail_second(instance_id):
        if instance_id == "engine1":
            raise RuntimeError("still owns CUDA")
        return stop(instance_id)
    executor._stop_rollout_process = fail_second
    with pytest.raises(RuntimeError, match="CUDA"):
        lend_rollout(executor, 4)
    assert executor._lent_rollout["slots"] == {}
    with pytest.raises(RuntimeError, match="incomplete"):
        lend_rollout(executor, 4)
    assert [meta.instance_id for meta in return_rollout(executor)] == ["engine0"]


def test_return_waits_for_remote_hybrid_exit_and_never_starts_vllm_early(tmp_path):
    executor, events, _, _ = executor_fixture(tmp_path, remote=True)
    lend_rollout(executor, 4)
    executor._hybrid_worker_meta["rollout0"] = SimpleNamespace(host="rollout-host", gpus=[0])
    def request_stop(worker_id):
        (tmp_path / f".stop_{worker_id}.json").write_text(json.dumps({"request_id": "current"}))
        wait_hybrid_stopped(executor, worker_id, "current")
        executor._hybrid_worker_meta.pop(worker_id)
    executor._stop_hybrid_worker_process = request_stop
    with pytest.raises(TimeoutError, match="not acknowledged"):
        return_rollout(executor)
    assert not any(event.startswith("start-") for event in events)
    (tmp_path / ".stop_rollout0.done").write_text(json.dumps(
        dict(worker_id="rollout0", request_id="current", exited=True)))
    assert len(return_rollout(executor)) == 2


def test_stale_remote_stop_ack_cannot_release_gpu(tmp_path):
    executor, _, _, _ = executor_fixture(tmp_path)
    (tmp_path / ".stop_rollout0.done").write_text(json.dumps(
        dict(worker_id="rollout0", request_id="old", exited=True)))
    with pytest.raises(TimeoutError):
        wait_hybrid_stopped(executor, "rollout0", "new")


def test_controller_cancels_unclaimed_launch_and_emits_matching_exit_ack(tmp_path):
    controller = Controller(SimpleNamespace(task_dir=tmp_path, project_dir=str(tmp_path), poll_interval=.1))
    (tmp_path / ".launch_rollout0.json").write_text("{}")
    (tmp_path / ".stop_rollout0.json").write_text(json.dumps({"request_id": "cancel"}))
    assert controller._handle_stop_requests()
    assert not (tmp_path / ".launch_rollout0.json").exists()
    done = json.loads((tmp_path / ".stop_rollout0.done").read_text())
    assert done["exited"] and done["request_id"] == "cancel"


def test_controller_failed_stop_preserves_ownership_and_reports_error(tmp_path, monkeypatch):
    controller = Controller(SimpleNamespace(task_dir=tmp_path, project_dir=str(tmp_path), poll_interval=.1))
    controller.remote_pids["rollout0"] = ("host", 12345)
    (tmp_path / ".stop_rollout0.json").write_text(json.dumps({"request_id": "failed"}))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="still alive"))
    controller._handle_stop_requests()
    assert "rollout0" in controller.remote_pids
    assert not (tmp_path / ".stop_rollout0.done").exists()
    assert json.loads((tmp_path / ".stop_rollout0.error").read_text())["request_id"] == "failed"


def test_real_process_group_stop_does_not_kill_unrelated_process():
    child_code = "import subprocess,sys,time;child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);print(child.pid,flush=True);time.sleep(60)"
    owned = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True, stdout=subprocess.PIPE, text=True)
    child_pid = int(owned.stdout.readline())
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    try:
        subprocess.run([sys.executable, "-c", _remote_stop_code(owned.pid)], check=True, timeout=25)
        owned.wait(timeout=5)
        child_stat = Path(f"/proc/{child_pid}/stat")
        assert not child_stat.exists() or child_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        assert unrelated.poll() is None
    finally:
        for proc in (owned, unrelated):
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
            proc.wait()

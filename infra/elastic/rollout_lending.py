"""Lend complete live rollout engines to EHP without applying a GRP layout."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import time
import uuid


def _set_unavailable(executor, instance_ids):
    scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
    if scheduler is None:
        return
    with getattr(scheduler, "_lock", nullcontext()):
        for handle in getattr(scheduler, "_instances", []):
            if handle.instance_id in instance_ids:
                handle.is_ready = False


def _capacity(physical, width):
    by_host = {}
    for host, gpu in physical:
        by_host.setdefault(host, set()).add(gpu)
    return sum(len(gpus) // width * width for gpus in by_host.values())


def _choose_engines(executor, required):
    loan = getattr(executor, "_lent_rollout", None) or {}
    old_ids = {meta.instance_id for meta in loan.get("engines", [])}
    metas = sorted((meta for meta in executor._process_meta.values()
                    if meta.instance_id not in old_ids), key=lambda meta: meta.instance_id)
    width = executor._elastic_replica_size_gpus()
    reserve = max(0, int(getattr(executor.config.global_resource_planner,
                               "elastic_hybrid_min_rollout_gpus", 0) or 0))
    serving = {(meta.host, int(gpu)) for meta in metas for gpu in meta.gpus}
    existing = {(slot["host"], int(gpu)) for slot in loan.get("slots", {}).values()
                for gpu in slot["gpus"]}
    candidates = []
    for count in range(1, len(metas) + 1):
        for subset in combinations(metas, count):
            physical = {(meta.host, int(gpu)) for meta in subset for gpu in meta.gpus}
            if len(serving - physical) < reserve:
                continue
            if _capacity(existing | physical, width) >= required:
                candidates.append((len(physical), count, subset))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else ()


def _restart_engines(executor, saved):
    restored = saved["restored"]
    for meta in saved["engines"]:
        if meta.instance_id not in saved["stopped"]:
            continue
        started = next((item for item in restored if item.instance_id == meta.instance_id), None)
        if started is None:
            command = executor._command_for_instance_id(meta.instance_id, meta.port)
            target = replace(meta, command=command, pid=0, adopted=False)
            started = executor._start_one_rollout_process(target)
            # A readiness failure must not lose ownership of an already started
            # process or launch another process on its GPU when recovery retries.
            restored.append(started)
            started.incarnation_id = uuid.uuid4().hex
            started.service_generation = None
        executor._wait_rollout_process_start_ready(started)
    return list(restored)


def _rollback_expansion(executor):
    pending = getattr(executor, "_rollout_loan_rollback", None)
    if pending is None:
        return
    # A failed stop may have signalled a process without proving its exit.
    # Resolve that same ownership before launching replacement CUDA workers.
    for identity in pending["uncertain"][:]:
        stopped = executor._stop_rollout_process(identity)
        if not stopped:
            raise RuntimeError(f"rollout stop did not confirm ownership: {identity}")
        pending["stopped"].append(identity)
        pending["uncertain"].remove(identity)
    restored = _restart_engines(executor, pending)
    if restored:
        executor._publish_current_rollout_weights(
            instance_ids=[meta.instance_id for meta in restored])
    scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
    # The subset publisher only installs weights. Routing is restored here,
    # after its current-revision ACK; untouched engines keep prior readiness.
    for handle in getattr(scheduler, "_instances", []):
        if handle.instance_id in pending["readiness"]:
            handle.is_ready = pending["readiness"][handle.instance_id]
    executor._rollout_loan_rollback = None


def lend_rollout(executor, required_gpu_count):
    """Grow the loan to a total GPU target without replacing active Hybrids.

    Only newly selected whole engines are stopped. Their physical slots become
    eligible for training together, after every new stop is acknowledged.
    """
    required = int(required_gpu_count)
    previous = getattr(executor, "_lent_rollout", None)
    existing = previous["slots"] if previous else {}
    physical = {(slot["host"], int(gpu)) for slot in existing.values() for gpu in slot["gpus"]}
    if required < 1 or _capacity(physical, executor._elastic_replica_size_gpus()) >= required:
        return deepcopy(existing)
    if getattr(executor, "_rollout_loan_rollback", None):
        raise RuntimeError("an incomplete rollout loan expansion must be recovered before retry")
    executor._adopt_existing_rollout_processes()
    selected = _choose_engines(executor, required)
    if not selected:
        return deepcopy(existing)
    # Freeze the namespace against the complete physical rollout allocation at
    # the first loan. Removing an engine must not renumber later worker slots.
    namespace = previous["physical_slots"] if previous else {
        address: f"rollout{index}" for index, address in enumerate(sorted({
            (meta.host, int(gpu)) for meta in executor._process_meta.values() for gpu in meta.gpus}))}
    additional = {(meta.host, int(gpu)) for meta in selected for gpu in meta.gpus}
    slots = {namespace[address]: {"host": address[0], "gpus": [address[1]]}
             for address in sorted(additional)}
    scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
    selected_ids = {meta.instance_id for meta in selected}
    pending = {"engines": [deepcopy(meta) for meta in selected], "stopped": [],
               "uncertain": [], "restored": [],
               "readiness": {handle.instance_id: handle.is_ready
                             for handle in getattr(scheduler, "_instances", [])
                             if handle.instance_id in selected_ids}}
    executor._rollout_loan_rollback = pending
    dispatcher = getattr(executor, "dispatcher", None)
    paused_here = dispatcher is not None and not dispatcher.is_paused()
    if paused_here:
        dispatcher.pause()
    timeout = float(getattr(executor.config.global_resource_planner, "runtime_drain_timeout_s", 3600.))
    try:
        if dispatcher is not None:
            dispatcher.wait_until_idle(timeout=timeout)
        engine = getattr(executor, "rollout_engine", None)
        if engine is not None and hasattr(engine, "wait_until_idle"):
            engine.wait_until_idle(timeout=timeout)
        _set_unavailable(executor, {meta.instance_id for meta in selected})
        for meta in selected:
            pending["uncertain"].append(meta.instance_id)
            stopped = executor._stop_rollout_process(meta.instance_id)
            if not stopped:
                raise RuntimeError(f"rollout stop did not confirm ownership: {meta.instance_id}")
            pending["stopped"].append(meta.instance_id)
            pending["uncertain"].remove(meta.instance_id)
        saved = previous if previous is not None else {
            "engines": [], "slots": {}, "stopped": [], "restored": [],
            "generation": uuid.uuid4().hex, "physical_slots": namespace}
        saved["engines"].extend(pending["engines"])
        saved["stopped"].extend(pending["stopped"])
        saved["slots"].update(slots)
        executor._lent_rollout = saved
        executor._rollout_loan_rollback = None
        return deepcopy(saved["slots"])
    except Exception as error:
        try:
            _rollback_expansion(executor)
        except Exception as rollback_error:
            raise RuntimeError(f"rollout loan rollback incomplete: {rollback_error}") from error
        raise
    finally:
        if paused_here:
            scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
            if any(handle.is_ready for handle in getattr(scheduler, "_instances", [])):
                dispatcher.resume()
            else:
                saved = getattr(executor, "_lent_rollout", None)
                if saved is not None:
                    saved["resume_dispatcher_after_publication"] = True


def wait_hybrid_stopped(executor, worker_id, request_id=None):
    """Await the controller's proof that the recorded remote process exited."""
    directory = Path(executor._hybrid_worker_control_dir())
    timeout = float(getattr(executor.config.global_resource_planner, "hybrid_worker_stop_timeout_s", 60.))
    deadline = time.monotonic() + timeout if timeout > 0 else float("inf")
    while True:
        done = directory / f".stop_{worker_id}.done"
        error = directory / f".stop_{worker_id}.error"
        if error.exists():
            record = json.loads(error.read_text())
            if request_id is None or record.get("request_id") == request_id:
                raise RuntimeError(f"hybrid stop failed: {record.get('error')}")
        if done.exists():
            record = json.loads(done.read_text())
            if ((request_id is None or record.get("request_id") == request_id)
                    and record.get("worker_id") == worker_id and record.get("exited") is True):
                return record
        if time.monotonic() >= deadline:
            raise TimeoutError(f"hybrid exit not acknowledged: {worker_id}")
        time.sleep(.05)


def return_rollout(executor):
    """Restart borrowed allocations only after Hybrid stop acknowledgements.

    Returned engines remain unavailable until the caller publishes the current
    revision to the new service generation and explicitly reopens admission.
    """
    _rollback_expansion(executor)
    saved = getattr(executor, "_lent_rollout", None)
    if not saved:
        return []
    physical = {(meta.host, int(gpu)) for meta in saved["engines"] for gpu in meta.gpus}
    for worker_id, meta in list(getattr(executor, "_hybrid_worker_meta", {}).items()):
        if not any((meta.host, int(gpu)) in physical for gpu in meta.gpus):
            continue
        # The executor retains ownership until a matching remote exit ACK (or
        # a local process wait) completes. It also withdraws replica aliases.
        if worker_id in executor._hybrid_worker_meta:
            executor.wait_for_hybrid_training_tasks(worker_id)
            executor._stop_hybrid_worker_process(worker_id)
    # A preceding pool release may already have withdrawn the local metadata.
    directory = Path(executor._hybrid_worker_control_dir())
    for worker_id in saved["slots"]:
        for suffix in ("json", "running"):
            pending = directory / f".stop_{worker_id}.{suffix}"
            try:
                request = json.loads(pending.read_text())
            except FileNotFoundError:
                continue
            wait_hybrid_stopped(executor, worker_id, request.get("request_id"))
    _set_unavailable(executor, {meta.instance_id for meta in saved["engines"]})
    restored = _restart_engines(executor, saved)
    executor._rollout_lending_resume_pending = bool(saved.get("resume_dispatcher_after_publication", False))
    executor._lent_rollout = None
    return restored

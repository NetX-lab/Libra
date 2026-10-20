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


def _choose_engines(executor, required):
    metas = sorted(executor._process_meta.values(), key=lambda meta: meta.instance_id)
    width = max(1, int(getattr(executor.config, "train_tp_size", 1)))
    reserve = max(0, int(getattr(
        executor.config.global_resource_planner, "elastic_hybrid_min_rollout_gpus", 0,
    ) or 0))
    all_physical = {(meta.host, int(gpu)) for meta in metas for gpu in meta.gpus}
    # A current launcher starts each full Hybrid replica on one host. Include
    # only subsets whose hosts each contain whole replicas. Stopping a wider
    # engine lends all its GPUs, even when the Hybrid needs fewer of them.
    candidates = []
    for count in range(1, len(metas) + 1):
        for subset in combinations(metas, count):
            by_host = {}
            for meta in subset:
                by_host.setdefault(meta.host, set()).update(int(gpu) for gpu in meta.gpus)
            borrowed = {(host, gpu) for host, gpus in by_host.items() for gpu in gpus}
            if len(all_physical - borrowed) < reserve:
                continue
            capacity = sum(len(gpus) // width * width for gpus in by_host.values())
            if capacity >= required:
                candidates.append((len(borrowed), count, subset))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else ()


def lend_rollout(executor, required_gpu_count):
    """Drain/stop whole current engines, returning safe single-GPU worker slots.

    The current partition is the sole source of allocations. The rejected GRP
    candidate is never consulted. Existing loans are returned idempotently.
    """
    required = int(required_gpu_count)
    if required < 1:
        return {}
    previous = getattr(executor, "_lent_rollout", None)
    if previous:
        if not previous["slots"]:
            raise RuntimeError("an incomplete rollout loan must be returned before retry")
        return deepcopy(previous["slots"])
    executor._adopt_existing_rollout_processes()
    all_physical = sorted({(meta.host, int(gpu)) for meta in executor._process_meta.values() for gpu in meta.gpus})
    selected = _choose_engines(executor, required)
    if not selected:
        return {}
    ids = {meta.instance_id for meta in selected}
    # Keep this record even on a partial stop failure: no caller may allocate
    # slots until every stop has succeeded, and return_rollout can recover them.
    saved = {"engines": [deepcopy(meta) for meta in selected], "slots": {},
             "stopped": [], "restored": [], "generation": uuid.uuid4().hex}
    executor._lent_rollout = saved
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
        # Existing multi-turn trajectories may still need their engines at
        # tool boundaries. Close engine routing only after accepted work drains.
        _set_unavailable(executor, ids)
        for meta in selected:
            stopped = executor._stop_rollout_process(meta.instance_id)
            if not stopped:
                raise RuntimeError(f"rollout stop did not confirm ownership: {meta.instance_id}")
            saved["stopped"].append(meta.instance_id)
        physical = sorted({(meta.host, int(gpu)) for meta in selected for gpu in meta.gpus})
        # Preserve the existing worker namespace based on the current physical
        # allocation; never derive worker IDs from a speculative candidate.
        saved["slots"] = {f"rollout{all_physical.index(slot)}": {"host": slot[0], "gpus": [slot[1]]}
                          for slot in physical}
        return deepcopy(saved["slots"])
    finally:
        if paused_here:
            scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
            handles = getattr(scheduler, "_instances", [])
            if any(handle.is_ready for handle in handles):
                dispatcher.resume()
            else:
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
    restored = list(saved["restored"])
    for meta in saved["engines"]:
        if meta.instance_id not in saved["stopped"] or any(item.instance_id == meta.instance_id for item in restored):
            continue
        command = executor._command_for_instance_id(meta.instance_id, meta.port)
        target = replace(meta, command=command, pid=0, adopted=False)
        started = executor._start_one_rollout_process(target)
        executor._wait_rollout_process_start_ready(started)
        # An explicit new incarnation accompanies an unchanged engine identity.
        started.incarnation_id = uuid.uuid4().hex
        started.service_generation = None
        restored.append(started)
        saved["restored"] = list(restored)
    executor._rollout_lending_resume_pending = bool(saved.get("resume_dispatcher_after_publication", False))
    executor._lent_rollout = None
    return restored

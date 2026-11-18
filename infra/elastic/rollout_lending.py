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


def loan_replica_capacity(instances, selected, width, reserve):
    """Count host-local replicas after lending complete physical services."""
    all_physical = {(meta.host, int(gpu)) for meta in instances for gpu in meta.gpus}
    by_host = {}
    for meta in selected:
        by_host.setdefault(meta.host, set()).update(int(gpu) for gpu in meta.gpus)
    borrowed = {(host, gpu) for host, gpus in by_host.items() for gpu in gpus}
    if len(all_physical - borrowed) < reserve:
        return 0
    return sum(len(gpus) // width for gpus in by_host.values())


def can_expand_loan(instances, current_ids, target_ids, width):
    """New services must be able to supply the extra full replicas."""
    current, target = set(current_ids), set(target_ids)
    if not current < target:
        return False
    old = [meta for meta in instances if meta.instance_id in current]
    added = [meta for meta in instances if meta.instance_id in target - current]
    chosen = old + added
    return (loan_replica_capacity(old, old, width, 0)
            + loan_replica_capacity(added, added, width, 0)
            >= loan_replica_capacity(chosen, chosen, width, 0))


def supports_local_lending(executor):
    hetero = getattr(executor.config, "heterogeneous_rollout", None)
    scheduling = getattr(hetero, "scheduling", None)
    scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
    trainer = getattr(executor, "trainer", None)
    sources = getattr(trainer, "_rollout_source_world_size",
                      getattr(executor.config, "train_dp_size", 1))
    ranks = getattr(trainer, "_rollout_source_ranks", (0,))
    return (getattr(trainer, "_bounded_rollout_production", True)
            and sources == 1 and tuple(ranks) == (0,)
            and getattr(scheduling, "scheduler_type", None) == "cmlfq_cost"
            and getattr(scheduling, "cmlfq_kv_backend", None) == "recompute"
            and callable(getattr(scheduler, "withdraw_instances", None))
            and callable(getattr(scheduler, "wait_for_generations", None)))


def _choose_engines(executor, required, instance_ids=None):
    loan = getattr(executor, "_lent_rollout", None) or {}
    existing = loan.get("engines", [])
    old_ids = {meta.instance_id for meta in existing}
    live = {name: meta for name, meta in executor._process_meta.items() if name not in old_ids}
    instances = [*existing, *live.values()]
    width = executor._elastic_replica_size_gpus()
    reserve = max(0, int(getattr(executor.config.global_resource_planner,
                               "elastic_hybrid_min_rollout_gpus", 0) or 0))
    old_capacity = loan_replica_capacity(existing, existing, width, 0)

    def fits(added):
        total = loan_replica_capacity(instances, [*existing, *added], width, reserve)
        extra = loan_replica_capacity(added, added, width, 0)
        return min(total, old_capacity + extra) * width >= required

    if instance_ids is not None:
        added_ids = set(instance_ids) - old_ids
        if not added_ids.issubset(live):
            raise ValueError("rollout loan targets must have current physical owners")
        selected = tuple(live[name] for name in sorted(added_ids))
        if not fits(selected):
            raise ValueError("rollout loan targets cannot supply the requested complete replicas and reserve")
        return selected

    metas = tuple(live[name] for name in sorted(live))
    candidates = []
    for count in range(1, len(metas) + 1):
        for subset in combinations(metas, count):
            if fits(subset):
                physical = {(meta.host, int(gpu)) for meta in subset for gpu in meta.gpus}
                candidates.append((len(physical), count, subset))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else ()


def _restart_engines(executor, saved):
    restored = saved["restored"]
    for meta in saved["engines"]:
        if meta.instance_id not in saved["stopped"]:
            continue
        started = next((item for item in restored if item.instance_id == meta.instance_id), None)
        proc = getattr(executor, "_processes", {}).get(meta.instance_id)
        if started is not None and proc is not None and proc.poll() is not None:
            if not executor._stop_rollout_process(meta.instance_id):
                raise RuntimeError(f"cannot confirm exit of restored rollout {meta.instance_id}")
            restored.remove(started)
            started = None
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
    with getattr(scheduler, "_lock", nullcontext()):
        for handle in getattr(scheduler, "_instances", []):
            if handle.instance_id in pending["readiness"]:
                handle.is_ready = pending["readiness"][handle.instance_id]
    executor._rollout_loan_rollback = None
    if pending.get("paused_here") and any(handle.is_ready for handle in getattr(scheduler, "_instances", [])):
        executor.dispatcher.resume()


def lend_rollout(executor, required_gpu_count, *, instance_ids=None):
    """Grow the loan to a total GPU target without replacing active Hybrids.

    Only newly selected whole engines are stopped. Their physical slots become
    eligible for training together, after every new stop is acknowledged.
    """
    required = int(required_gpu_count)
    previous = getattr(executor, "_lent_rollout", None)
    existing = previous["engines"] if previous else []
    old_ids = {meta.instance_id for meta in existing}
    target_ids = set(instance_ids) if instance_ids is not None else None
    if target_ids is not None and not old_ids.issubset(target_ids):
        raise RuntimeError("a different physical rollout loan must be returned before switching")
    width = executor._elastic_replica_size_gpus()
    if required < 1 or (target_ids in (None, old_ids)
                        and loan_replica_capacity(existing, existing, width, 0) * width >= required):
        return deepcopy(previous["slots"]) if previous else {}
    if getattr(executor, "_rollout_loan_rollback", None):
        raise RuntimeError("an incomplete rollout loan expansion must be recovered before retry")
    executor._adopt_existing_rollout_processes()
    selected = _choose_engines(executor, required, target_ids)
    if not selected:
        return deepcopy(previous["slots"]) if previous else {}

    trainer = getattr(executor, "trainer", None)
    source_ranks = getattr(trainer, "_rollout_source_ranks", ())
    if any(rank != 0 for rank in source_ranks):
        coord_id = executor._active_runtime_coord_id
        if not coord_id or executor._peer_drain_completed_coord_id != coord_id:
            raise RuntimeError("rollout loan requires peer source drain before stopping shared services")
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
    timeout = float(getattr(executor.config.global_resource_planner, "runtime_drain_timeout_s", 3600.))
    local = supports_local_lending(executor) and len(selected) < len(executor._process_meta)
    paused_here = False
    try:
        if local:
            indices = [index for index, handle in enumerate(scheduler._instances)
                       if handle.instance_id in selected_ids]
            scheduler.withdraw_instances(indices)
            # Requests reserved before withdrawal may still need to enter HTTP.
            scheduler.wait_for_generations(indices, timeout=timeout if timeout > 0 else None)
        else:
            paused_here = dispatcher is not None and not dispatcher.is_paused()
            if paused_here:
                dispatcher.pause()
                pending["paused_here"] = True
            drain_timeout = timeout if timeout > 0 else float("inf")
            if dispatcher is not None:
                dispatcher.wait_until_idle(timeout=drain_timeout)
            engine = getattr(executor, "rollout_engine", None)
            if callable(getattr(engine, "wait_until_idle", None)):
                engine.wait_until_idle(timeout=drain_timeout)
            _set_unavailable(executor, selected_ids)
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
        if (paused_here and dispatcher.is_paused()
                and getattr(executor, "_rollout_loan_rollback", None) is None):
            scheduler = getattr(getattr(executor, "rollout_engine", None), "scheduler", None)
            if any(handle.is_ready for handle in getattr(scheduler, "_instances", [])):
                dispatcher.resume()
            else:
                if trainer is not None:
                    trainer._paused_for_rollout_loan = True


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
    executor._lent_rollout = None
    trainer = getattr(executor, "trainer", None)
    if trainer is not None:
        from .rollout_lifecycle import apply_availability
        pending = set(getattr(trainer, "_restored_rollouts_pending_publication", ()))
        pending.update(meta.instance_id for meta in restored)
        trainer._restored_rollouts_pending_publication = pending
        apply_availability(trainer, pending)
    return restored

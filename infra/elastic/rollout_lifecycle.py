"""Training-boundary coordination for rollout GPUs lent to EHP."""
from __future__ import annotations

from contextlib import nullcontext

import torch.distributed as dist


def _broadcast(trainer, value):
    packet = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(packet, src=0, group=getattr(trainer, "control", None))
    if packet[0].get("error"):
        raise RuntimeError(packet[0]["error"])
    return packet[0]


def unavailable_rollouts(executor):
    loan = getattr(executor, "_lent_rollout", None)
    pending = getattr(executor, "_rollout_loan_rollback", None)
    return list(dict.fromkeys(meta.instance_id for record in (loan, pending) if record
                              for meta in record["engines"]))


def apply_availability(trainer, unavailable):
    previous = set(getattr(trainer, "_rollout_unavailable_ids", ()))
    trainer._rollout_unavailable_ids = set(unavailable)
    scheduler = getattr(trainer.rollout_engine, "scheduler", None)
    update = getattr(scheduler, "set_instances_available", None)
    if update is not None:
        handles = getattr(scheduler, "_instances", [])
        with scheduler._lock:
            update([handle.index for handle in handles
                    if handle.instance_id in trainer._rollout_unavailable_ids], False)
            update([handle.index for handle in handles
                    if handle.instance_id in previous - trainer._rollout_unavailable_ids], True)
        return
    with getattr(scheduler, "_lock", nullcontext()):
        for handle in getattr(scheduler, "_instances", []):
            if handle.instance_id in trainer._rollout_unavailable_ids:
                handle.is_ready = False
            elif handle.instance_id in previous:
                handle.is_ready = True


def restore_for_publication(trainer):
    """Return physical GPUs on rank zero; every rank withdraws old membership.

    Restarted services remain unroutable until publication completes their
    installation; survivor requests and accepted groups keep their state.
    """
    outcome = None
    if trainer.is_main_process:
        executor = trainer.runtime_elastic_executor
        try:
            loan = getattr(executor, "_lent_rollout", None)
            ids = [meta.instance_id for meta in loan["engines"]] if loan else []
            if loan or getattr(executor, "_rollout_loan_rollback", None):
                from .rollout_lending import return_rollout
                executor._release_all_hybrid_workers(reason="restore_rollout_supply")
                return_rollout(executor)
            outcome = {"restored": ids}
        except Exception as exc:
            outcome = {"error": f"failed to restore rollout GPUs: {exc}"}
    outcome = _broadcast(trainer, outcome)
    if outcome["restored"]:
        pending = set(getattr(trainer, "_restored_rollouts_pending_publication", ()))
        pending.update(outcome["restored"])
        trainer._restored_rollouts_pending_publication = pending
        apply_availability(trainer, pending)
        trainer._refresh_nonblocking_elastic_membership()
        trainer.train_engine._elastic_step_hybrid_workers = ()
    return bool(outcome["restored"])


def reopen_after_publication(trainer):
    if getattr(trainer, "_publication_target_ids", None) is not None:
        return
    if not getattr(trainer, "_restored_rollouts_pending_publication", None):
        return
    pending = trainer._restored_rollouts_pending_publication
    apply_availability(trainer, set(getattr(trainer, "_rollout_unavailable_ids", ())) - pending)
    trainer._restored_rollouts_pending_publication = set()
    if getattr(trainer, "_paused_for_rollout_loan", False):
        trainer._paused_for_rollout_loan = False
        trainer._resume_rollout_pipeline()


def publish_with_consensus(trainer, revision, **sync_options):
    error = ""
    try:
        trainer._sync_weights(target_version=revision, **sync_options)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=getattr(trainer, "control", None))
    if any(errors):
        raise RuntimeError(f"Rollout publication failed across core ranks: {errors}")


def ensure_batch_supply(trainer):
    """Restore the generating pool before waiting for an impossible next batch."""
    cfg = getattr(trainer.config, "global_resource_planner", None)
    if not (getattr(cfg, "enabled", False)
            and getattr(cfg, "runtime_reconfigure_training", False)
            and getattr(cfg, "hybrid_worker_launch_enabled", False)
            and not getattr(cfg, "runtime_training_pool_plan_only", True)
            and int(getattr(cfg, "elastic_hybrid_min_rollout_gpus", 0) or 0) <= 0):
        return
    outcome = None
    if trainer.is_main_process:
        executor = trainer.runtime_elastic_executor
        loan = getattr(executor, "_lent_rollout", None)
        lent = {meta.instance_id for meta in loan["engines"]} if loan else set()
        configured = {instance.instance_id for instance in trainer.config.heterogeneous_rollout.instances}
        outcome = {"all_lent": bool(configured) and configured.issubset(lent)}
    if not _broadcast(trainer, outcome)["all_lent"]:
        return

    from .rollout_supply import ready_complete_samples
    sufficient = (not trainer._is_rollout_source() or
                  ready_complete_samples(trainer) >= trainer._source_batch_sample_count())
    records = [sufficient]
    if dist.is_available() and dist.is_initialized():
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, sufficient, group=getattr(trainer, "control", None))
    if all(records):
        return

    dispatcher = trainer.dispatcher
    paused_here = dispatcher is not None and not dispatcher.is_paused()
    if paused_here:
        trainer._pause_rollout_pipeline()
    restore = getattr(trainer, "_restore_lent_rollouts_for_publication", None)
    if (restore() if callable(restore) else restore_for_publication(trainer)):
        revision = int(trainer.train_engine.optimizer_revision)
        publish_with_consensus(trainer, revision, reopen_restored=False)
        trainer.train_engine.set_version(revision)
        if getattr(getattr(dispatcher, "staleness_manager", None), "group_reservations_enabled", False):
            dispatcher.on_publication(revision)
        if not getattr(trainer, "_bounded_rollout_production", False):
            trainer._rebind_rollout_engine_from_config(reason="ehp_supply_restore")
        reopen_after_publication(trainer)
    if paused_here:
        trainer._resume_rollout_pipeline()

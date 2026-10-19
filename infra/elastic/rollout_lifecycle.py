"""Training-boundary coordination for rollout GPUs lent to EHP."""
from __future__ import annotations

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
    return [meta.instance_id for meta in loan["engines"]] if loan else []


def apply_availability(trainer, unavailable):
    previous = set(getattr(trainer, "_rollout_unavailable_ids", ()))
    trainer._rollout_unavailable_ids = set(unavailable)
    scheduler = getattr(trainer.rollout_engine, "scheduler", None)
    for handle in getattr(scheduler, "_instances", []):
        if handle.instance_id in trainer._rollout_unavailable_ids:
            handle.is_ready = False
        elif handle.instance_id in previous:
            handle.is_ready = True


def restore_for_publication(trainer):
    """Return physical GPUs on rank zero; every rank withdraws old membership.

    The caller has closed admission on all logical sources. Restarted services
    remain unroutable until publication has validated their new worker receipts.
    """
    outcome = None
    if trainer.is_main_process:
        executor = trainer.runtime_elastic_executor
        try:
            ids = unavailable_rollouts(executor)
            if ids:
                from .rollout_lending import return_rollout
                executor._release_all_hybrid_workers(reason="restore_rollout_supply")
                return_rollout(executor)
            outcome = {"restored": ids}
        except Exception as exc:
            outcome = {"error": f"failed to restore rollout GPUs: {exc}"}
    outcome = _broadcast(trainer, outcome)
    if outcome["restored"]:
        apply_availability(trainer, outcome["restored"])
        trainer._restored_rollouts_pending_publication = set(outcome["restored"])
        trainer._refresh_nonblocking_elastic_membership()
        trainer.train_engine._elastic_step_hybrid_workers = ()
    return bool(outcome["restored"])


def reopen_after_publication(trainer):
    if not getattr(trainer, "_restored_rollouts_pending_publication", None):
        return
    apply_availability(trainer, ())
    trainer._restored_rollouts_pending_publication = set()
    executor = getattr(trainer, "runtime_elastic_executor", None)
    if executor is not None:
        executor._rollout_lending_resume_pending = False


def publish_with_consensus(trainer, revision):
    error = ""
    try:
        trainer._sync_weights(target_version=revision)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=getattr(trainer, "control", None))
    if any(errors):
        raise RuntimeError(f"Rollout publication failed across core ranks: {errors}")


def ensure_batch_supply(trainer):
    """Keep the loan while ready groups suffice; return it before a supply wait."""
    cfg = getattr(trainer.config, "global_resource_planner", None)
    if not (getattr(cfg, "enabled", False)
            and getattr(cfg, "runtime_reconfigure_training", False)
            and getattr(cfg, "hybrid_worker_launch_enabled", False)
            and not getattr(cfg, "runtime_training_pool_plan_only", True)):
        return
    outcome = None
    if trainer.is_main_process:
        executor = trainer.runtime_elastic_executor
        ids = unavailable_rollouts(executor)
        instances = getattr(trainer.config.heterogeneous_rollout, "instances", ())
        outcome = {"all_lent": bool(ids) and len(ids) == len(instances)}
    outcome = _broadcast(trainer, outcome)
    if not outcome["all_lent"]:
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
    trainer._pause_rollout_pipeline()
    # Publication implementations must invalidate old service receipts before
    # a loan is returned, even when the replacement keeps the same endpoints.
    restore = getattr(trainer, "_restore_lent_rollouts_for_publication", None)
    restored = restore() if callable(restore) else restore_for_publication(trainer)
    if restored:
        revision = int(trainer.train_engine.optimizer_revision)
        publish_with_consensus(trainer, revision)
        trainer.train_engine.set_version(revision)
        reopen_after_publication(trainer)
        trainer._rebind_rollout_engine_from_config(reason="ehp_supply_restore")
    trainer._resume_rollout_pipeline()

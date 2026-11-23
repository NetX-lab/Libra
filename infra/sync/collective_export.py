"""Keep every training rank in the same weight-export iterator protocol."""
from __future__ import annotations

import torch.distributed as dist


def _gather(value, group):
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


class _ExportIterator:
    """One control checkpoint per yielded tensor, before the next TP export.

    An iterator exception aborts this publication on every rank. No rank may
    advance to the next tensor after a peer failed. An exception *inside* a
    tensor's distributed collective remains a process-group failure and is
    subject to that group's timeout/abort; it cannot be repaired by draining.
    """

    def __init__(self, factory, group, prepare):
        self.group, self.prepare, self.finished = group, prepare, False
        self.iterator = None
        error = ""
        try:
            self.iterator = iter(factory())
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = _gather(error, group)
        if any(errors):
            self.finished = True
            raise RuntimeError(f"collective weight export initialization failed: {errors}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        item, status = None, {"done": False, "error": "", "name": None}
        try:
            item = next(self.iterator)
            if self.prepare is not None:
                item = self.prepare(item)
            status["name"] = str(item[0])
        except StopIteration:
            status["done"] = True
        except BaseException as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
        statuses = _gather(status, self.group)
        errors = [state["error"] for state in statuses if state["error"]]
        if errors:
            self.finished = True
            raise RuntimeError(f"collective weight export iterator aborted: {errors}")
        if any(state["done"] for state in statuses):
            self.finished = True
            if not all(state["done"] for state in statuses):
                raise RuntimeError("collective weight export ended at different tensor positions")
            raise StopIteration
        if len({state["name"] for state in statuses}) != 1:
            self.finished = True
            raise RuntimeError("collective weight export tensor identities differ across ranks")
        return item


def stream_export_with_consensus(factory, *, is_sender, send, group=None, prepare=None):
    """Drain training collectives after a transport failure, then fail together.

    ``send`` runs only on the sender and must consume the supplied iterator.
    A transport error leaves that iterator alive; the sender drains the same
    stream as its peers without sending any further weights. Initialization,
    preparation, iteration and final transport status all have common control
    positions, so an iterator error never silently becomes successful EOF.
    """
    weights = _ExportIterator(factory, group, prepare)
    error = ""
    try:
        if is_sender:
            send(weights)
        else:
            for _name, tensor in weights:
                del tensor
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    # Also handle a sender which returned without consuming the entire stream.
    # Draining is safe after transport errors, and a failed iterator marks
    # itself finished so it cannot enter another TP collective.
    try:
        for _name, tensor in weights:
            del tensor
    except BaseException as exc:
        error += f"; export drain failed: {type(exc).__name__}: {exc}"
    errors = _gather(error, group)
    if any(errors):
        raise RuntimeError(f"collective weight publication failed: {errors}")

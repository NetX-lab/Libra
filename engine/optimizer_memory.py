"""Bound temporary storage in the native PyTorch Adam fallback."""

import torch


def bound_default_torch_adam_memory(optimizer) -> int:
    """Disable automatic foreach selection in native Adam/AdamW leaves.

    CUDA's automatic foreach Adam path creates a full parameter-sized list of
    temporaries (for example, sqrt of every second moment at once). MCore's
    non-fused fallback can therefore exceed memory even when model, master
    parameters, gradients and Adam moments fit. The per-tensor implementation
    retains the same precision and bounds those temporaries to one parameter.

    Explicit foreach choices and fused implementations are preserved. This
    walks the MCore chained/single optimizer wrappers without accessing a
    ChainedOptimizer's single-child-only ``optimizer`` property. Return the
    number of parameter groups changed; repeated calls are idempotent.
    """
    pending = [optimizer]
    visited = set()
    changed = 0
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if type(current) in (torch.optim.Adam, torch.optim.AdamW):
            for group in current.param_groups:
                if group.get("foreach") is None and group.get("fused") is not True:
                    group["foreach"] = False
                    changed += 1
            if (current.defaults.get("foreach") is None
                    and current.defaults.get("fused") is not True):
                current.defaults["foreach"] = False
            continue
        children = getattr(current, "chained_optimizers", None)
        if children is not None:
            pending.extend(children)
        else:
            pending.append(getattr(current, "optimizer", None))
    return changed

"""Strict native MCore optimizer metadata restoration for fixed-topology EHP."""

from contextlib import contextmanager

import torch


@contextmanager
def optimizer_state_load_context(optimizer):
    """Preserve native metadata's Torch Adam step during MCore tensor loads.

    MCore 0.14's model-space loader deliberately excludes ``step`` from tensor
    state, but its low-level setter iterates every Torch Adam state key. Supply
    the step already loaded from common metadata to that setter only. Other
    missing master/moment tensors still fail through the native strict copy.
    This temporary instance adapter never edits installed MCore dependencies.
    """
    parts = getattr(optimizer, "chained_optimizers", (optimizer,))
    restores = []
    absent = object()
    completed = False
    try:
        for part in parts:
            inner = getattr(part, "optimizer", None)
            native = getattr(part, "_set_main_param_and_optimizer_states", None)
            if not isinstance(inner, (torch.optim.Adam, torch.optim.AdamW)) or not callable(native):
                continue
            name = "_set_main_param_and_optimizer_states"
            previous = vars(part).get(name, absent)

            def copy_with_metadata_step(model_param, tensors, *, part=part, inner=inner, native=native):
                group, order = part.model_param_group_index_map[model_param]
                main_param = inner.param_groups[group]["params"][order]
                step = inner.state[main_param].get("step")
                if "step" not in tensors and step is not None:
                    tensors = dict(tensors, step=step)
                return native(model_param, tensors)

            setattr(part, name, copy_with_metadata_step)
            restores.append((part, name, previous))
        yield
        completed = True
    finally:
        for part, name, previous in reversed(restores):
            if previous is absent:
                delattr(part, name)
            else:
                setattr(part, name, previous)
            # Native MCore's Torch fallback assigns one shared scalar tensor
            # to every Adam parameter during metadata loading. Torch Adam
            # increments each parameter's step in-place; shared references
            # would advance it once per parameter and change the next update.
            if completed:
                for state in part.optimizer.state.values():
                    step = state.get("step")
                    if isinstance(step, torch.Tensor):
                        state["step"] = step.detach().clone()


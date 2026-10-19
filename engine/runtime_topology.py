"""In-memory MCore topology transitions at a collective optimizer boundary.

The fully-sharded model-space representation is essential here: the ordinary
DistributedOptimizer state_dict omits the Adam/master tensors, and its DP-zero
format cannot change TP. Only metadata is gathered; tensor payloads are streamed
from their old owner, so no rank assembles a second full model and optimizer.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import random
from contextlib import contextmanager
from datetime import timedelta
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist


def _consensus_error(error, *, group=None):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError("core topology transaction failed: " + "; ".join(
            f"rank={rank}: {message}" for rank, message in enumerate(errors) if message
        ))


def _collective_phase(operation, *, group):
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _consensus_error(error, group=group)
    return result


def _tensor_shard(value):
    return all(hasattr(value, key) for key in (
        "key", "data", "local_shape", "global_shape", "global_offset"
    ))


def _factory(value):
    return callable(getattr(value, "build", None)) and callable(getattr(value, "merge_fn", None))


_SCALAR_IDENTITY = "__libra_tensor_state_scalar__"


def _state_child_path(container, path, key):
    if key == "step":
        anchors = tuple(sorted(str(item.key) for item in container.values()
                               if _tensor_shard(item) or _factory(item)))
        if anchors:
            # DistributedOptimizer's param_state integer keys enumerate only
            # the current rank's DP-owned fragments. They change after TP/DP
            # repartitioning; global tensor identities do not.
            return (_SCALAR_IDENTITY, anchors, str(key))
    return (*path, key)


def _is_scalar_identity(path):
    return bool(path) and path[0] == _SCALAR_IDENTITY


def _same_scalar(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and torch.equal(left, right))
    return left == right


@dataclass
class _Piece:
    key: str
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    prepend: int
    flat_start: int
    flat_stop: int
    dtype: torch.dtype
    digest: str
    owner: int = 0
    index: int = 0


class MemoryShardStore:
    """Rank-local CPU copies with a topology-independent, streamed loader.

    Accepts MCore ShardedTensor, ShardedTensorFactory, ShardedObject and
    LocalNonpersistentObject leaves. Factories preserve the MCore transforms
    for fused gated-MLP/QKV tensors and their corresponding Adam/master state.
    """

    def __init__(self, state):
        self.tensors: list[torch.Tensor] = []
        self.pieces: list[_Piece] = []
        self.objects = {}
        self.common = {}
        self._capture(state, ())

    def _capture(self, value, path):
        if _factory(value):
            self._capture(value.build(), path)
        elif _tensor_shard(value):
            tensor = value.data.detach().to("cpu", copy=True).contiguous()
            span = getattr(value, "flattened_range", None)
            start = 0 if span is None else int(span.start or 0)
            stop = tensor.numel() + start if span is None else int(span.stop)
            self.pieces.append(_Piece(
                key=value.key, shape=tuple(value.global_shape),
                offset=tuple(value.global_offset), local_shape=tuple(value.local_shape),
                prepend=int(getattr(value, "prepend_axis_num", 0)),
                flat_start=start, flat_stop=stop, dtype=tensor.dtype,
                digest=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                index=len(self.tensors),
            ))
            self.tensors.append(tensor.flatten())
        elif hasattr(value, "unique_key") and hasattr(value, "data"):
            self.objects[value.unique_key] = copy.deepcopy(value.data)
        elif callable(getattr(value, "unwrap", None)):
            # LocalNonpersistentObject is reconstructed from the new topology.
            return
        elif isinstance(value, dict):
            for key, item in value.items():
                self._capture(item, _state_child_path(value, path, key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._capture(item, (*path, index))
        else:
            if (_is_scalar_identity(path) and path in self.common
                    and not _same_scalar(self.common[path], value)):
                raise RuntimeError(f"replicated optimizer scalar diverged within rank: {path}")
            self.common[path] = copy.deepcopy(value)

    def restore(self, template, *, group=None, chunk_bytes=64 * 1024 * 1024):
        """Fill a new topology's template collectively, including DP fragments."""
        distributed = dist.is_initialized()
        world = dist.get_world_size(group) if distributed else 1
        rank = dist.get_rank(group) if distributed else 0
        catalogs = [None] * world
        local = (self.pieces, self.objects, self.common)
        if distributed:
            dist.all_gather_object(catalogs, local, group=group)
        else:
            catalogs[0] = local
        unique = {}
        for owner, (pieces, _, _) in enumerate(catalogs):
            for piece in pieces:
                piece = copy.copy(piece)
                piece.owner = owner
                identity = (piece.key, piece.shape, piece.offset, piece.local_shape,
                            piece.prepend, piece.flat_start, piece.flat_stop)
                if identity in unique:
                    if unique[identity].digest != piece.digest:
                        raise RuntimeError(f"replicated state diverged: {piece.key}")
                else:
                    unique[identity] = piece

        scalar_states = {}
        for _, _, common in catalogs:
            for identity, value in common.items():
                if not _is_scalar_identity(identity):
                    continue
                if identity in scalar_states:
                    previous = scalar_states[identity]
                    if not _same_scalar(value, previous):
                        raise RuntimeError(f"replicated optimizer scalar diverged: {identity}")
                else:
                    scalar_states[identity] = value

        destinations = {}

        def prepare(value, path):
            if _factory(value):
                built = prepare(value.build(), path)
                return ("factory", value, built)
            if _tensor_shard(value):
                tensor = torch.empty(tuple(value.local_shape), dtype=value.dtype, device="cpu")
                mask = torch.zeros(tuple(value.local_shape), dtype=torch.bool)
                destinations.setdefault(value.key, []).append((value, tensor, mask))
                return ("tensor", value, tensor, mask)
            if hasattr(value, "unique_key") and hasattr(value, "data"):
                objects = [objects[value.unique_key] for _, objects, _ in catalogs
                           if value.unique_key in objects]
                if not objects:
                    raise RuntimeError(f"missing sharded object {value.unique_key}")
                return copy.deepcopy(objects[0])
            if callable(getattr(value, "unwrap", None)):
                return value.unwrap()
            if isinstance(value, dict):
                return {key: prepare(item, _state_child_path(value, path, key))
                        for key, item in value.items()}
            if isinstance(value, list):
                return [prepare(item, (*path, index)) for index, item in enumerate(value)]
            if isinstance(value, tuple):
                return tuple(prepare(item, (*path, index)) for index, item in enumerate(value))
            common = scalar_states if _is_scalar_identity(path) else catalogs[0][2]
            if path not in common:
                raise RuntimeError(f"missing topology-independent state at {path}")
            return copy.deepcopy(common[path])

        prepared = _collective_phase(lambda: prepare(template, ()), group=group)
        copy_error = None
        for piece in sorted(unique.values(), key=lambda item: (
            item.key, item.offset, item.flat_start, item.owner, item.index
        )):
            elements = piece.flat_stop - piece.flat_start
            data = (self.tensors[piece.index] if rank == piece.owner
                    else torch.empty(elements, dtype=piece.dtype))
            width = max(1, chunk_bytes // data.element_size())
            if distributed:
                global_src = dist.get_global_rank(group, piece.owner) if group is not None else piece.owner
                for begin in range(0, elements, width):
                    dist.broadcast(data[begin:begin + width], src=global_src, group=group)
            for leaf, target, coverage in destinations.get(piece.key, []):
                try:
                    _copy_overlap(piece, data, leaf, target, coverage)
                except Exception as exc:
                    copy_error = f"{type(exc).__name__}: {exc}"
        _consensus_error(copy_error, group=group)

        def finish(value):
            if isinstance(value, tuple) and value and isinstance(value[0], str):
                if value[0] == "tensor":
                    _, leaf, data, coverage = value
                    span = getattr(leaf, "flattened_range", None)
                    if span is not None:
                        data, coverage = data.flatten()[span], coverage.flatten()[span]
                    if not bool(coverage.all()):
                        raise RuntimeError(f"incomplete state coverage: {leaf.key}")
                    # load_state_dict copies into the model/optimizer's newly
                    # allocated GPU buffers. Keeping payloads on CPU avoids a
                    # second full set of GPU model+Adam tensors during install.
                    return data
                if value[0] == "factory":
                    return value[1].merge_fn(finish(value[2]))
            if isinstance(value, dict):
                return {key: finish(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finish(item) for item in value]
            if isinstance(value, tuple):
                return tuple(finish(item) for item in value)
            return value

        return _collective_phase(lambda: finish(prepared), group=group)


def _copy_overlap(piece, source_data, leaf, target, coverage):
    if tuple(leaf.global_shape) != piece.shape:
        raise RuntimeError(f"global shape changed for {piece.key}")
    if leaf.dtype != piece.dtype:
        raise RuntimeError(f"state dtype changed for {piece.key}")
    prepend = piece.prepend
    if prepend != int(getattr(leaf, "prepend_axis_num", 0)):
        raise RuntimeError(f"prepend axes changed for {piece.key}")
    if piece.offset[:prepend] != tuple(leaf.global_offset)[:prepend]:
        return
    source_slices, target_slices = [], []
    for old, size, new, new_size in zip(
        piece.offset[prepend:], piece.local_shape,
        tuple(leaf.global_offset)[prepend:], tuple(leaf.local_shape)
    ):
        begin, end = max(old, new), min(old + size, new + new_size)
        if end <= begin:
            return
        source_slices.append(slice(begin - old, end - old))
        target_slices.append(slice(begin - new, end - new))
    source = torch.empty(piece.local_shape, dtype=piece.dtype)
    valid = torch.zeros(piece.local_shape, dtype=torch.bool)
    source.flatten()[piece.flat_start:piece.flat_stop].copy_(source_data)
    valid.flatten()[piece.flat_start:piece.flat_stop] = True
    incoming = source[tuple(source_slices)]
    incoming_valid = valid[tuple(source_slices)]
    dest, seen = target[tuple(target_slices)], coverage[tuple(target_slices)]
    overlap = incoming_valid & seen
    if bool(overlap.any()) and not torch.equal(dest[overlap], incoming[overlap]):
        raise RuntimeError(f"overlapping state diverged: {piece.key}")
    dest[incoming_valid] = incoming[incoming_valid]
    seen |= incoming_valid


def _checkpoint_modules(engine):
    # MCore DDP inherits MegatronModule.sharded_state_dict, which prefixes its
    # child with "module.". Its load_state_dict instead delegates directly to
    # the wrapped module and expects bare keys. Use the native unwrapping
    # contract on both capture and restore; parameter objects remain identical
    # for DistributedOptimizer's model-space master/Adam mapping.
    try:
        from megatron.core.utils import unwrap_model
    except ImportError:
        # Lightweight engine adapters used without an installed MCore package
        # are already unwrapped. Production MCore always takes the native path.
        modules = engine.model
    else:
        modules = unwrap_model(engine.model)
    return modules


def _state_template(engine, *, loading):
    state = {"model" if index == 0 else f"model{index}": chunk.sharded_state_dict()
             for index, chunk in enumerate(_checkpoint_modules(engine))}
    state["optimizer"] = engine.optimizer.sharded_state_dict(
        state, is_loading=loading,
        metadata={"distrib_optim_sharding_type": "fully_sharded_model_space"},
    )
    return state


def install_core_state(engine, state):
    """Strictly install canonical model keys and their matching optimizer state."""
    for index, chunk in enumerate(_checkpoint_modules(engine)):
        chunk.load_state_dict(state["model" if index == 0 else f"model{index}"], strict=True)
    with optimizer_state_load_context(engine.optimizer):
        engine.optimizer.load_state_dict(state["optimizer"])


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


def capture_core_optimizer_metadata(engine):
    """Copy DistributedOptimizer's small common metadata, never model tensors.

    In MCore 0.14 with Torch Adam, a fresh DistributedOptimizer cannot generate
    a loading template: state_dict first requires an existing Adam step. Its
    native load_state_dict can allocate the new DP-local slots directly from
    saved common metadata. The actual master/moment tensors still come solely
    from MemoryShardStore's fully sharded model-space installation.
    """
    try:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        return None
    optimizer = engine.optimizer
    parts = getattr(optimizer, "chained_optimizers", (optimizer,))
    if not parts or not all(isinstance(part, DistributedOptimizer) for part in parts):
        return None
    # DistributedOptimizer.state_dict deliberately excludes all DP-local
    # parameter, master and moment tensors. Other optimizer adapters must not
    # be copied here: their ordinary state_dict can contain an entire model.
    return copy.deepcopy(optimizer.state_dict())


def _rng_state():
    import numpy as np
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        state["cuda"] = torch.cuda.get_rng_state()
        state["tracker"] = copy.deepcopy(get_cuda_rng_tracker().get_states())
    return state


def _restore_rng(state):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        torch.cuda.set_rng_state(state["cuda"])
        get_cuda_rng_tracker().set_states(state["tracker"])


def invalidate_core_topology_state(engine):
    """Invalidate old lane endpoints and gradients after membership quiesces.

    The trainer recreates its gradient server and communication domain only
    after the new topology's all-rank installation receipt succeeds.
    """
    engine.close_elastic_communication_domain()
    engine.elastic_gradient_domain = None
    engine._elastic_batch_partition = None
    engine._elastic_gradient_update_callback = None
    getattr(engine, "_elastic_snapshot_urls", {}).clear()
    getattr(engine, "_elastic_seen_payloads", set()).clear()
    condition = getattr(engine, "_hybrid_gradient_condition", None)
    if condition is not None:
        with condition:
            getattr(engine, "_pending_hybrid_gradients", []).clear()
    else:
        getattr(engine, "_pending_hybrid_gradients", []).clear()


def _rebuild_mcore(engine, topology):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import get_megatron_optimizer

    parallel_state = engine._parallel_state()
    optimizer_config = copy.copy(engine.optimizer.config)
    # Destroy DDP buffers before constructing the replacement model. No old
    # parameter tensor remains referenced by MemoryShardStore (CPU copies only).
    engine.optimizer = None
    engine.model = []
    engine.provider = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MCore 0.14 destroy_model_parallel clears the NCCL references but only
    # destroys its Gloo groups. Release the old communicators explicitly.
    old_groups = list(dict.fromkeys(
        value for name, value in sorted(vars(parallel_state).items())
        if name.startswith("_") and isinstance(value, dist.ProcessGroup)
        and value is not dist.group.WORLD
    ))
    parallel_state.destroy_model_parallel()
    registered = dist.distributed_c10d._world.pg_map
    for process_group in old_groups:
        if process_group in registered:
            dist.destroy_process_group(process_group)
    engine.train_tp_size = topology["train_tp"]
    engine.provider = engine.bridge.to_megatron_provider(load_weights=False)
    if not engine.use_transformer_engine:
        from RL_Framework.engine.megatron_core_attention import torch_sdpa_layer_spec
        engine.provider.transformer_layer_spec = torch_sdpa_layer_spec
    engine._configure_provider(engine.max_seq_length)
    engine.provider.finalize()
    engine.provider.initialize_model_parallel(seed=None)
    config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=engine.grad_reduce_in_fp32,
        overlap_grad_reduce=False, overlap_param_gather=False,
        use_distributed_optimizer=engine.use_distributed_optimizer,
        check_for_nan_in_grad=True,
    )
    engine.model = engine.provider.provide_distributed_model(
        ddp_config=config, wrap_with_ddp=True, bf16=True,
        use_cpu_initialization=engine.use_cpu_initialization,
        data_parallel_random_init=False,
    )
    engine.optimizer = get_megatron_optimizer(
        optimizer_config, engine.model, use_gloo_process_groups=True,
    )
    engine._validate_conversion_coverage()


def reconfigure_core_engine(engine, target: Mapping[str, Any], transaction_id: str):
    """Collectively rebuild TP/DP on the existing physical training ranks.

    All ranks must enter after the same successful optimizer step, with EHP
    membership quiesced. Exceptions abort the transaction; a partially rebuilt
    topology must never be reported as applied by the caller.
    """
    if not dist.is_initialized():
        raise RuntimeError("core topology transition requires all initialized training ranks")
    world = dist.get_world_size()
    topology = {}
    for axis in ("tp", "pp", "cp", "ep"):
        old = int(getattr(engine, f"train_{axis}_size"))
        topology[f"train_{axis}"] = int(target.get(f"train_{axis}", target.get(f"train_{axis}_size", old)))
    tp, pp, cp, ep = (topology[f"train_{axis}"] for axis in ("tp", "pp", "cp", "ep"))
    dp = int(target.get("train_dp", target.get("train_dp_size", world // max(1, tp * pp * cp))))
    topology["train_dp"] = dp
    if min(tp, pp, cp, ep, dp) < 1 or tp * pp * cp * dp != world:
        raise ValueError("core TP/DP transition must retain the physical training rank set")
    if any(topology[f"train_{axis}"] != int(getattr(engine, f"train_{axis}_size"))
           for axis in ("pp", "cp", "ep")):
        raise ValueError("core transition changes TP/DP; PP/CP/EP must retain the active layout")
    if getattr(engine, "_elastic_step_hybrid_workers", ()):
        raise RuntimeError("quiesce Hybrid replicas before rebuilding core parallel groups")
    if not transaction_id:
        raise ValueError("topology transition requires a transaction identity")
    group = dist.new_group(backend="gloo", timeout=timedelta(
        seconds=float(getattr(engine, "_elastic_active_gradient_timeout_s", 300) or 300)
    ))
    revision, version = int(engine.optimizer_revision), int(engine.current_version)
    try:
        agreement = [None] * world
        contract = (transaction_id, topology, revision)
        dist.all_gather_object(agreement, contract, group=group)
        if any(item != contract for item in agreement):
            raise RuntimeError("training ranks disagree on topology transaction or optimizer revision")
        engine._synchronize_optimizer_parameters()
        rng = _rng_state()
        store = _collective_phase(lambda: MemoryShardStore(
            _state_template(engine, loading=False)), group=group)
        optimizer_metadata = _collective_phase(
            lambda: capture_core_optimizer_metadata(engine), group=group)
        dist.barrier(group=group)
        invalidate_core_topology_state(engine)
        _collective_phase(lambda: _rebuild_mcore(engine, topology), group=group)
        if optimizer_metadata is not None:
            _collective_phase(lambda: engine.optimizer.load_state_dict(optimizer_metadata), group=group)
        template = _collective_phase(lambda: _state_template(engine, loading=True), group=group)
        loaded = store.restore(template, group=group)

        def install():
            install_core_state(engine, loaded)
            engine.optimizer_revision, engine.current_version = revision, version
            engine._synchronize_optimizer_parameters()
            _restore_rng(rng)
            actual = dict(engine.get_parallel_state())
            parallel = engine._parallel_state()
            for axis, name in (("tp", "tensor_model"), ("pp", "pipeline_model"),
                               ("cp", "context"), ("ep", "expert_model")):
                actual[f"train_{axis}"] = int(getattr(parallel, f"get_{name}_parallel_world_size")())
            for key, expected in topology.items():
                if int(actual.get(key, -1)) != expected:
                    raise RuntimeError(f"rebuilt topology mismatch: {key} expected={expected} actual={actual.get(key)}")

        _collective_phase(install, group=group)
        receipt = {"acknowledged": True, "applied": True,
                   "transaction_id": transaction_id, "optimizer_revision": revision,
                   "topology": topology, "rank": dist.get_rank()}
        receipts = [None] * world
        dist.all_gather_object(receipts, receipt, group=group)
        if any(item["topology"] != topology or item["optimizer_revision"] != revision
               for item in receipts):
            raise RuntimeError("core topology receipts disagree")
        receipt["rank_receipts"] = receipts
        return receipt
    finally:
        dist.destroy_process_group(group)

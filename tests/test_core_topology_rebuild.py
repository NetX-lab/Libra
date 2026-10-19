"""Real CPU tensors/Adam across collective TP/DP state transitions."""

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
import threading

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL_Framework.engine import runtime_topology as topology


@dataclass
class Shard:
    key: str
    data: torch.Tensor
    global_shape: tuple
    global_offset: tuple
    local_shape: tuple
    flattened_range: slice | None = None
    prepend_axis_num: int = 0

    @property
    def dtype(self):
        return self.data.dtype


def shard(key, data, tp, rank, *, span=None):
    width = 8 // tp
    return Shard(key, data, (8,), ((rank % tp) * width,), (width,), span)


class Chunk:
    def __init__(self, parameter, tp, rank):
        self.parameter, self.tp, self.rank = parameter, tp, rank

    def sharded_state_dict(self):
        return {"weight": shard("weight", self.parameter, self.tp, self.rank)}

    def load_state_dict(self, state, strict=True):
        with torch.no_grad():
            self.parameter.copy_(state["weight"])


class Adam:
    def __init__(self, parameter, tp, rank):
        self.parameter, self.tp, self.rank = parameter, tp, rank
        self.inner = torch.optim.Adam([parameter], lr=0.01, betas=(0.9, 0.95))

    def sharded_state_dict(self, model, is_loading, metadata):
        assert metadata["distrib_optim_sharding_type"] == "fully_sharded_model_space"
        if is_loading:
            states = {"step": torch.tensor(0.), "exp_avg": torch.zeros_like(self.parameter),
                      "exp_avg_sq": torch.zeros_like(self.parameter)}
        else:
            states = self.inner.state[self.parameter]
        result = self.inner.state_dict()
        result["state"] = {0: {
            key: (shard(f"adam.{key}.weight", value, self.tp, self.rank)
                  if key != "step" else value)
            for key, value in states.items()
        }}
        return result

    def load_state_dict(self, state):
        self.inner.load_state_dict(state)


class Engine:
    def __init__(self, tp):
        self.train_tp_size, self.train_pp_size = tp, 1
        self.train_cp_size, self.train_ep_size = 1, 1
        self.optimizer_revision, self.current_version = 0, 0
        self.rank = dist.get_rank()
        self.rebuild(tp)

    def rebuild(self, tp):
        self.train_tp_size = tp
        width, rank = 8 // tp, self.rank % tp
        parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float64)[rank * width:(rank + 1) * width].clone())
        self.model = [Chunk(parameter, tp, self.rank)]
        self.optimizer = Adam(parameter, tp, self.rank)

    def _synchronize_optimizer_parameters(self):
        pass

    def close_elastic_communication_domain(self):
        pass

    def get_parallel_state(self):
        return {"train_tp": self.train_tp_size, "train_dp": 4 // self.train_tp_size,
                "train_pp": 1, "train_cp": 1, "train_ep": 1}

    def _parallel_state(self):
        return SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: self.train_tp_size,
            get_pipeline_model_parallel_world_size=lambda: 1,
            get_context_parallel_world_size=lambda: 1,
            get_expert_model_parallel_world_size=lambda: 1,
        )

    def step(self, gradient):
        width, rank = 8 // self.train_tp_size, self.rank % self.train_tp_size
        parameter = self.model[0].parameter
        parameter.grad = gradient[rank * width:(rank + 1) * width].clone()
        self.optimizer.inner.step()
        self.optimizer.inner.zero_grad()
        self.optimizer_revision += 1


def _distributed_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=4,
                            timeout=timedelta(seconds=45))
    try:
        engine = Engine(4)
        reference = torch.nn.Parameter(torch.arange(8, dtype=torch.float64))
        optimizer = torch.optim.Adam([reference], lr=0.01, betas=(0.9, 0.95))
        for iteration, tp in enumerate((4, 2, 1, 4)):
            gradient = torch.arange(1, 9, dtype=torch.float64) * (iteration + 1)
            engine.step(gradient)
            reference.grad = gradient
            optimizer.step()
            optimizer.zero_grad()
            before_rng = torch.get_rng_state().clone()
            with patch.object(topology, "_rebuild_mcore", lambda engine, target: engine.rebuild(target["train_tp"])):
                receipt = topology.reconfigure_core_engine(
                    engine, {"train_tp": tp, "train_dp": 4 // tp}, f"transition-{iteration}",
                )
            assert torch.equal(before_rng, torch.get_rng_state())
            assert receipt["optimizer_revision"] == iteration + 1
            assert len(receipt["rank_receipts"]) == 4
            width, lane = 8 // tp, rank % tp
            torch.testing.assert_close(engine.model[0].parameter, reference[lane * width:(lane + 1) * width], rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    engine.optimizer.inner.state[engine.model[0].parameter][key],
                    optimizer.state[reference][key][lane * width:(lane + 1) * width],
                    rtol=0, atol=0,
                )

        # DistributedOptimizer fragments within a TP shard do not have the
        # full local shape. Assemble these fragments before slicing new TP.
        full = torch.arange(8, dtype=torch.float64)
        start = (rank // 2) * 2
        local = full[(rank % 2) * 4 + start:(rank % 2) * 4 + start + 2]
        store = topology.MemoryShardStore({"adam": shard("adam", local, 2, rank, span=slice(start, start + 2))})
        restored = store.restore({"adam": shard("adam", torch.empty(8, dtype=torch.float64), 1, rank)}, chunk_bytes=8)
        torch.testing.assert_close(restored["adam"], full)

        # One rank's rejected install propagates to every core rank; none may
        # return an applied receipt after a peer fails.
        def failed_rebuild(engine, target):
            if rank == 1:
                raise ValueError("injected rebuild failure")
            engine.rebuild(target["train_tp"])

        with patch.object(topology, "_rebuild_mcore", failed_rebuild):
            with pytest.raises(RuntimeError, match="rank=1: ValueError: injected rebuild failure"):
                topology.reconfigure_core_engine(engine, {"train_tp": 2}, "failed-transition")

        with pytest.raises(RuntimeError, match="disagree on topology transaction"):
            topology.reconfigure_core_engine(engine, {"train_tp": 2}, f"disagreement-{rank}")
        with pytest.raises(ValueError, match="retain the physical training rank set"):
            topology.reconfigure_core_engine(engine, {"train_tp": 4, "train_dp": 2}, "wrong-ranks")
    finally:
        dist.destroy_process_group()


def test_collective_tp4_dp1_tp2_dp2_tp1_dp4_preserves_adam_updates(tmp_path):
    mp.start_processes(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=4,
                       join=True, start_method="spawn")


def test_missing_shard_is_rejected():
    source = shard("weight", torch.arange(4.), 2, 0)
    store = topology.MemoryShardStore({"weight": source})
    with pytest.raises(RuntimeError, match="incomplete state coverage"):
        store.restore({"weight": shard("weight", torch.empty(8), 1, 0)})


class Factory:
    def __init__(self, data):
        self.data = data

    def build(self):
        return {"gate": Shard("gate", self.data[:2], (2,), (0,), (2,)),
                "up": Shard("up", self.data[2:], (2,), (0,), (2,))}

    @staticmethod
    def merge_fn(state):
        return torch.cat((state["gate"], state["up"]))


def test_factory_transforms_and_master_dtype_are_preserved():
    original = torch.tensor([1.125, 2.25, 3.5, 4.75], dtype=torch.float64)
    store = topology.MemoryShardStore({"fused": Factory(original)})
    restored = store.restore({"fused": Factory(torch.empty(4, dtype=torch.float64))})
    torch.testing.assert_close(restored["fused"], original, rtol=0, atol=0)
    assert restored["fused"].dtype == torch.float64


def test_row_shards_and_dp_fragments_reshard_into_columns():
    original = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    pieces = [Shard("adam.matrix", original[row, start:start + 2], (2, 4),
                    (row, 0), (1, 4), slice(start, start + 2))
              for row in range(2) for start in (0, 2)]
    store = topology.MemoryShardStore({"pieces": pieces})
    target = {"columns": [Shard("adam.matrix", torch.empty((2, 2), dtype=torch.float64),
                                  (2, 4), (0, column), (2, 2))
                          for column in (0, 2)]}
    restored = store.restore(target)
    torch.testing.assert_close(torch.cat(restored["columns"], dim=1), original)


def test_mismatched_global_shape_does_not_silently_copy():
    store = topology.MemoryShardStore({"weight": shard("weight", torch.arange(8.), 1, 0)})
    target = Shard("weight", torch.empty(9), (9,), (0,), (9,))
    with pytest.raises(RuntimeError, match="global shape changed"):
        store.restore({"weight": target})


def test_rebuild_requires_collective_entry():
    with pytest.raises(RuntimeError, match="initialized training ranks"):
        topology.reconfigure_core_engine(SimpleNamespace(), {"train_tp": 2}, "change")


def test_topology_invalidation_drops_old_endpoints_and_lane_state():
    closed = []
    engine = SimpleNamespace(
        close_elastic_communication_domain=lambda: closed.append(True),
        elastic_gradient_domain=object(), _elastic_batch_partition={"dp0": [1]},
        _elastic_gradient_update_callback=object(),
        _elastic_snapshot_urls={17: "ehp://old-topology"},
        _elastic_seen_payloads={"old-payload"}, _pending_hybrid_gradients=[object()],
        _hybrid_gradient_condition=threading.Condition(),
    )
    topology.invalidate_core_topology_state(engine)
    assert closed == [True]
    assert engine.elastic_gradient_domain is None
    assert engine._elastic_batch_partition is None
    assert engine._elastic_gradient_update_callback is None
    assert engine._elastic_snapshot_urls == {}
    assert engine._elastic_seen_payloads == set()
    assert engine._pending_hybrid_gradients == []


def test_actual_mcore_metadata_when_package_available():
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    source = mapping.ShardedTensor.from_rank_offsets_flat(
        "adam.weight", torch.tensor([2., 3.]), (4,), (0, 0, 2),
        flattened_range=slice(1, 3), replica_id=(0, 0, 0),
    )
    target = mapping.ShardedTensor.from_rank_offsets_flat(
        "adam.weight", torch.empty(2), (4,), (0, 0, 2),
        flattened_range=slice(1, 3), replica_id=(0, 0, 0),
    )
    store = topology.MemoryShardStore({"adam": source})
    restored = store.restore({"adam": target})
    torch.testing.assert_close(restored["adam"], source.data)

    def build(key, data, replica, span):
        return {"gate": mapping.ShardedTensor.from_rank_offsets(
            f"{key}.gate", data[:2], replica_id=replica),
                "up": mapping.ShardedTensor.from_rank_offsets(
            f"{key}.up", data[2:], replica_id=replica)}

    factory = mapping.ShardedTensorFactory(
        key="fused", data=torch.arange(4.), build_fn=build,
        merge_fn=lambda state: torch.cat((state["gate"], state["up"])),
        replica_id=(0, 0, 0),
    )
    restored = topology.MemoryShardStore({"fused": factory}).restore({"fused": factory})
    torch.testing.assert_close(restored["fused"], factory.data)


def test_actual_mcore_ddp_float16_checkpoint_keys_load_strictly():
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.data_parallel_base import _BaseDataParallel
    from megatron.core.transformer.module import Float16Module, MegatronModule

    class Extra(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = {"marker": "trained-extra-state"}

        def get_extra_state(self):
            return self.value

        def set_extra_state(self, state):
            self.value = state

    class TinyModel(MegatronModule):
        def __init__(self, config):
            super().__init__(config)
            self.linear = torch.nn.Linear(4, 4, bias=False)
            self.extra = Extra()

        def sharded_state_dict(self, prefix="", *args, **kwargs):
            return {
                f"{prefix}linear.weight": mapping.ShardedTensor.from_rank_offsets(
                    f"{prefix}linear.weight", self.linear.weight),
                f"{prefix}extra._extra_state": mapping.ShardedObject(
                    f"{prefix}extra._extra_state", self.extra.get_extra_state(), (1,), (0,)),
            }

    config = SimpleNamespace(fp16=False, bf16=True)
    raw = TinyModel(config)
    half = Float16Module(config, raw)
    # Exercise actual DDP inherited state APIs on CPU without allocating its
    # CUDA gradient buckets. No methods involved in checkpointing are mocked.
    wrapped = DistributedDataParallel.__new__(DistributedDataParallel)
    _BaseDataParallel.__init__(wrapped, config, half)
    incorrect = topology.MemoryShardStore(wrapped.sharded_state_dict()).restore(
        wrapped.sharded_state_dict())
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        wrapped.load_state_dict(incorrect, strict=True)

    optimizer_loaded = []
    optimizer = SimpleNamespace(sharded_state_dict=lambda *args, **kwargs: {},
                                load_state_dict=lambda state: optimizer_loaded.append(state))
    engine = SimpleNamespace(model=[wrapped], optimizer=optimizer)
    source = topology._state_template(engine, loading=False)
    assert set(source["model"]) == {"linear.weight", "extra._extra_state"}
    expected = raw.linear.weight.detach().clone()
    store = topology.MemoryShardStore(source)
    with torch.no_grad():
        raw.linear.weight.zero_()
    raw.extra.value = {"marker": "mutated"}
    loaded = store.restore(topology._state_template(engine, loading=True))
    topology.install_core_state(engine, loaded)
    torch.testing.assert_close(raw.linear.weight, expected, rtol=0, atol=0)
    assert raw.extra.value == {"marker": "trained-extra-state"}
    assert optimizer_loaded == [{}]
    with pytest.raises(RuntimeError, match="Missing key"):
        topology.install_core_state(engine, {
            "model": {"extra._extra_state": raw.extra.get_extra_state()}, "optimizer": {},
        })


def test_actual_mcore_torch_adam_empty_loading_template_uses_saved_metadata(monkeypatch):
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    module = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")
    monkeypatch.setattr(module, "HAVE_APEX_OR_TE", False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    def engine(values):
        master = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        model = torch.nn.Parameter(master.detach().bfloat16().clone())
        inner = torch.optim.AdamW([{
            "params": [master], "wd_mult": 1., "lr_mult": 1.,
            "is_expert_parallel": False, "is_decoupled_lr": False,
        }], lr=0.01, betas=(0.9, 0.95))
        # Invoke real pinned MCore state APIs on CPU. Its CUDA bucket allocation
        # is replaced only by this minimal valid two-element range description.
        optimizer = module.DistributedOptimizer.__new__(module.DistributedOptimizer)
        optimizer.optimizer = inner
        optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        optimizer.config = SimpleNamespace(
            exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fp16=False,
        )
        optimizer.grad_scaler = None
        optimizer.distributed_optimizer_instance_id = 0
        optimizer.model_param_group_index_map = {model: (0, 0)}
        optimizer.gbuf_ranges = [{torch.float32: [{"param_map": {
            model: {"gbuf_world": module.Range(0, 2), "param": module.Range(0, 2)},
        }}]}]

        class Chunk:
            def sharded_state_dict(self):
                return {"weight": mapping.ShardedTensor.from_rank_offsets(
                    "weight", model, replica_id=(0, 0, 0))}

            def load_state_dict(self, state, strict=True):
                assert strict is True and set(state) == {"weight"}
                with torch.no_grad():
                    model.copy_(state["weight"])

        return SimpleNamespace(model=[Chunk()], optimizer=optimizer), master, model

    source, source_master, source_model = engine([1., 2.])
    for _ in range(3):
        source_master.grad = torch.tensor([0.1, -0.2])
        source.optimizer.optimizer.step()
    with torch.no_grad():
        source_model.copy_(source_master)
    metadata = topology.capture_core_optimizer_metadata(source)
    assert "state" not in metadata["optimizer"]
    assert metadata["optimizer"]["param_groups"][0]["step"] == 3
    store = topology.MemoryShardStore(topology._state_template(source, loading=False))
    target, target_master, _ = engine([-9., -9.])
    with pytest.raises(AssertionError):
        topology._state_template(target, loading=True)
    # Native loading allocates Adam slots for the new ranges. It neither runs
    # an optimizer update nor guesses Adam's step from optimizer_revision.
    target.optimizer.load_state_dict(metadata)
    torch.testing.assert_close(target_master, torch.tensor([-9., -9.]), rtol=0, atol=0)
    loaded = store.restore(topology._state_template(target, loading=True))
    setter = module.DistributedOptimizer._set_main_param_and_optimizer_states
    if not getattr(setter, "_rl_framework_step_compat", False):
        # Normal engine initialization may already install the general MCore
        # compatibility shim. Reproduce the native failure only without it;
        # exact restoration and the next Adam update are checked in both cases.
        with pytest.raises(KeyError, match="step"):
            target.optimizer.load_state_dict(loaded["optimizer"])
    topology.install_core_state(target, loaded)
    torch.testing.assert_close(target_master, source_master, rtol=0, atol=0)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(target.optimizer.optimizer.state[target_master][key],
                                   source.optimizer.optimizer.state[source_master][key], rtol=0, atol=0)
    for adapter, parameter in ((source, source_master), (target, target_master)):
        parameter.grad = torch.tensor([-0.3, 0.4])
        adapter.optimizer.optimizer.step()
    torch.testing.assert_close(target_master, source_master, rtol=0, atol=0)
    assert "_set_main_param_and_optimizer_states" not in vars(target.optimizer)


def test_common_metadata_does_not_copy_arbitrary_optimizer_tensors():
    class OtherOptimizer:
        def state_dict(self):
            pytest.fail("ordinary optimizer state may include full model/master tensors")

    assert topology.capture_core_optimizer_metadata(SimpleNamespace(optimizer=OtherOptimizer())) is None


def _native_partitioned_engine(rank, tp, *, references=None, reverse=False):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
    sizes = {"a": 2, "b": 4, "c": 6, "d": 8}
    dp, tp_rank, dp_rank = 2 // tp, rank % tp, rank // tp
    per_dp = sum(sizes.values()) // (tp * dp)
    begin, end = dp_rank * per_dp, (dp_rank + 1) * per_dp
    models, masters, spans, offsets = {}, {}, {}, {}
    cursor = 0
    for name, size in sizes.items():
        width = size // tp
        offsets[name] = tp_rank * width
        data = (references[name]["param"][offsets[name]:offsets[name] + width]
                if references is not None else torch.full((width,), -9.))
        models[name] = torch.nn.Parameter(data.bfloat16().clone())
        left, right = max(0, begin - cursor), min(width, end - cursor)
        if right > left:
            spans[name] = (left, right)
            masters[name] = torch.nn.Parameter(data[left:right].float().clone())
        cursor += width
    names = list(masters)
    if reverse:
        names.reverse()
    inner = torch.optim.AdamW([{
        "params": [masters[name] for name in names], "wd_mult": 1., "lr_mult": 1.,
        "is_expert_parallel": False, "is_decoupled_lr": False,
    }], lr=0.01, betas=(0.9, 0.95), weight_decay=0.01)
    if references is not None:
        for name, master in masters.items():
            left, right = spans[name]
            region = slice(offsets[name] + left, offsets[name] + right)
            inner.state[master] = {
                "step": references[name]["step"].clone(),
                "exp_avg": references[name]["exp_avg"][region].clone(),
                "exp_avg_sq": references[name]["exp_avg_sq"][region].clone(),
            }
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.optimizer = inner
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.config = SimpleNamespace(
        exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fp16=False,
    )
    optimizer.grad_scaler = None
    optimizer.distributed_optimizer_instance_id = 0
    optimizer.model_param_group_index_map = {models[name]: (0, index) for index, name in enumerate(names)}
    optimizer.gbuf_ranges = [{torch.float32: [{"param_map": {
        models[name]: {"gbuf_world": Range(0, spans[name][1] - spans[name][0]),
                       "param": Range(*spans[name])} for name in names
    }}]}]

    class Chunk:
        def sharded_state_dict(self):
            return {name: ShardedTensor.from_rank_offsets(
                name, model, (0, tp_rank, tp), replica_id=(0, 0, dp_rank))
                    for name, model in models.items()}

        def load_state_dict(self, state, strict=True):
            assert strict is True and set(state) == set(models)
            with torch.no_grad():
                for name, model in models.items():
                    model.copy_(state[name])

    return SimpleNamespace(model=[Chunk()], optimizer=optimizer), models, masters, spans, offsets


def _native_partition_worker(rank, rendezvous):
    from megatron.core.optimizer import distrib_optimizer as native
    torch.set_num_threads(1)
    native.HAVE_APEX_OR_TE = False
    torch.cuda.current_device = lambda: "cpu"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        parameters = {name: torch.nn.Parameter(torch.arange(size, dtype=torch.float32) + 1)
                      for name, size in {"a": 2, "b": 4, "c": 6, "d": 8}.items()}
        reference = torch.optim.AdamW(parameters.values(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.01)
        for _ in range(3):
            for parameter in parameters.values():
                parameter.grad = torch.full_like(parameter, 0.25)
            reference.step()
        states = {name: {"param": parameter.detach().clone(),
                         **{key: value.clone() for key, value in reference.state[parameter].items()}}
                  for name, parameter in parameters.items()}
        current, _, _, _, _ = _native_partitioned_engine(rank, 2, references=states)
        ownership_counts = []
        for tp in (1, 2):
            source = topology._state_template(current, loading=False)
            store = topology.MemoryShardStore(source)
            metadata = topology.capture_core_optimizer_metadata(current)
            target, models, masters, spans, offsets = _native_partitioned_engine(rank, tp, reverse=True)
            target.optimizer.load_state_dict(metadata)
            target_state = topology._state_template(target, loading=True)
            ownership_counts.append(len(target_state["optimizer"]["param_state"]))
            loaded = store.restore(target_state, chunk_bytes=16)
            topology.install_core_state(target, loaded)
            steps = [state["step"] for state in target.optimizer.optimizer.state.values()]
            assert len({id(step) for step in steps}) == len(steps)
            for name, master in masters.items():
                left, right = spans[name]
                region = slice(offsets[name] + left, offsets[name] + right)
                torch.testing.assert_close(master, states[name]["param"][region], rtol=0, atol=0)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    expected = states[name][key] if key == "step" else states[name][key][region]
                    torch.testing.assert_close(target.optimizer.optimizer.state[master][key], expected,
                                               rtol=0, atol=0)
            current = target
        assert ownership_counts == ([3, 4] if rank == 0 else [2, 4])
        for parameter in parameters.values():
            parameter.grad = torch.full_like(parameter, -0.15)
        reference.step()
        for master in masters.values():
            master.grad = torch.full_like(master, -0.15)
        current.optimizer.optimizer.step()
        for name, master in masters.items():
            left, right = spans[name]
            region = slice(offsets[name] + left, offsets[name] + right)
            torch.testing.assert_close(master, parameters[name][region], rtol=0, atol=0)

        # TP replicas refer to the same global parameter, so a contradictory
        # actual scalar step must fail rather than choosing rank zero's copy.
        invalid = topology._state_template(current, loading=False)
        for item in invalid["optimizer"]["param_state"].values():
            item["step"] = item["step"].clone() + rank
        bad_store = topology.MemoryShardStore(invalid)
        with pytest.raises(RuntimeError, match="replicated optimizer scalar diverged"):
            bad_store.restore(topology._state_template(current, loading=False))
    finally:
        dist.destroy_process_group()


def test_real_mcore_adam_steps_follow_global_parameter_identity_across_tp_dp(tmp_path):
    pytest.importorskip("megatron.core.optimizer.distrib_optimizer")
    mp.start_processes(_native_partition_worker, args=(str(tmp_path / "native-rendezvous"),),
                       nprocs=2, join=True, start_method="spawn")


def test_associated_adam_scalar_changes_container_index_without_defaulting():
    def item(step=None):
        result = {"exp_avg": shard("optimizer.state.exp_avg.weight", torch.arange(8.), 1, 0)}
        if step is not None:
            result["step"] = torch.tensor(float(step))
        return result

    target = {"optimizer": {"param_state": {8: item(0)}}}
    restored = topology.MemoryShardStore({"optimizer": {"param_state": {0: item(7)}}}).restore(target)
    assert restored["optimizer"]["param_state"][8]["step"].item() == 7
    with pytest.raises(RuntimeError, match="missing topology-independent state"):
        topology.MemoryShardStore({"optimizer": {"param_state": {0: item()}}}).restore(target)
    with pytest.raises(RuntimeError, match="diverged within rank"):
        topology.MemoryShardStore({"optimizer": {"param_state": {0: item(7), 1: item(8)}}})

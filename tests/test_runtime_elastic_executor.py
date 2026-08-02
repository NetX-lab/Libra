import json

import pytest

from RL_Framework.config import AsyncRLConfig, HeterogeneousInstanceConfig
from RL_Framework.infra.execution.batch_dispatcher import BatchTaskDispatcher, TaskInput
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner
from RL_Framework.infra.elastic.runtime_executor import (
    ManagedRolloutProcess,
    RuntimeElasticExecutor,
)
from RL_Framework.infra.sync.staleness import StalenessManager

try:
    from RL_Framework.infra.elastic.hybrid_pool import ElasticHybridPool
except ModuleNotFoundError:
    ElasticHybridPool = None


class FakeDispatcher:
    def __init__(self):
        self.events = []

    def pause(self):
        self.events.append("pause")

    def wait_until_idle(self, timeout=3600.0):
        self.events.append(f"drain:{timeout}")

    def resume(self):
        self.events.append("resume")


class FakeRolloutEngine:
    def __init__(self):
        self.calls = []

    def reconfigure_from_plan(self, plan, config):
        self.calls.append((plan.rollout_tp_list, config.rollout_gpus))


def _config() -> AsyncRLConfig:
    cfg = AsyncRLConfig(
        model_path="/tmp/fake-model",
        train_gpus=2,
        rollout_gpus=2,
        train_tp_size=1,
        train_pp_size=1,
        train_dp_size=2,
        micro_batch_size=1,
        batch_size=8,
        n_total_gpus=4,
        max_seq_length=4096,
    )
    cfg.global_resource_planner.enabled = True
    cfg.global_resource_planner.train_backend = "analytic"
    cfg.global_resource_planner.rollout_backend = "analytic"
    cfg.global_resource_planner.plan_interval = 1
    cfg.global_resource_planner.warmup_steps = 0
    cfg.global_resource_planner.min_history_size = 4
    cfg.global_resource_planner.min_gain_ratio = 0.0
    cfg.global_resource_planner.reconfiguration_cost_s = 0.0
    cfg.global_resource_planner.allowed_train_tp = [1, 2]
    cfg.global_resource_planner.allowed_train_pp = [1]
    cfg.global_resource_planner.allowed_rollout_tp = [1, 2, 4]
    cfg.global_resource_planner.micro_batch_sizes = [1]
    return cfg


def _decision(cfg: AsyncRLConfig):
    planner = GlobalResourcePlanner.from_config(cfg)
    decision = planner.plan_if_needed(
        step=0,
        config=cfg,
        batch=[
            {"input_len": 128, "output_len": 4096},
            {"input_len": 128, "output_len": 4096},
            {"input_len": 128, "output_len": 4096},
            {"input_len": 128, "output_len": 4096},
        ],
    )
    assert decision.candidate_plan is not None
    decision.should_reconfigure = True
    return planner, decision


def test_runtime_executor_applies_rollout_reconfiguration():
    cfg = _config()
    planner, decision = _decision(cfg)
    dispatcher = FakeDispatcher()
    engine = FakeRolloutEngine()
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        rollout_engine=engine,
        dispatcher=dispatcher,
    )

    result = executor.execute(decision)

    assert result.applied
    assert dispatcher.events == ["pause", "drain:3600.0", "resume"]
    assert "apply_config" in result.actions
    assert "reconfigure_rollout_engine" in result.actions
    assert engine.calls == [(decision.candidate_plan.rollout_tp_list, cfg.rollout_gpus)]
    assert cfg.heterogeneous_rollout.instances


def test_runtime_executor_requests_supervised_training_handoff(tmp_path):
    """A physical DP resize must be delegated to the Slurm supervisor."""

    cfg = _config()
    planner, decision = _decision(cfg)
    decision.candidate_plan.train_config.dp = 1
    decision.candidate_plan.rollout_config.tp_list = [1, 1, 1]
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 1
    cfg.global_resource_planner.runtime_training_handoff_enabled = True
    cfg.global_resource_planner.runtime_training_handoff_dir = str(tmp_path)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)

    result = executor.execute(decision)

    assert result.applied
    assert result.reason == "training_handoff_requested"
    request = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    assert request["current_train_gpus"] == 2
    assert request["target_train_gpus"] == 1
    assert request["plan"]["train"]["n_gpus"] == 1


def test_hybrid_nonblocking_growth_does_not_request_training_handoff(tmp_path):
    cfg = _config()
    planner, decision = _decision(cfg)
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    cfg.global_resource_planner.runtime_training_handoff_enabled = True
    cfg.global_resource_planner.runtime_training_handoff_dir = str(tmp_path)
    cfg.global_resource_planner.runtime_training_resize_mode = "hybrid_nonblocking"
    cfg.global_resource_planner.runtime_training_pool_only = False
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 4
    pool = ElasticHybridPool(
        core_train_workers=["core0", "core1"],
        core_rollout_workers=["rollout0", "rollout1"],
        zero_sync_steps=0,
    )
    executor = RuntimeElasticExecutor(config=cfg, planner=planner, elastic_pool=pool)

    result = executor.execute(decision)

    assert result.reason != "training_handoff_requested"
    assert not (tmp_path / "request.json").exists()
    assert cfg.train_gpus == 2
    assert cfg.train_dp_size == 2
    assert cfg.global_resource_planner.runtime_effective_train_gpus == 4
    pool.close()


def test_sibling_step_launch_env_drops_parent_slurm_step(monkeypatch):
    cfg = _config()
    planner, _decision_value = _decision(cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gn[003-005]")
    monkeypatch.setenv("SLURM_STEP_ID", "3")
    monkeypatch.setenv("SLURM_STEP_NODELIST", "gn003")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "gn003")

    env = executor._allocation_launch_env("hybrid_training")

    assert env["SLURM_JOB_ID"] == "123"
    assert env["SLURM_JOB_NODELIST"] == "gn[003-005]"
    assert env["RL_FRAMEWORK_ROLE"] == "hybrid_training"
    assert "LIBRA_HYBRID_CPU_INITIALIZATION" not in env
    assert "SLURM_STEP_ID" not in env
    assert "SLURM_STEP_NODELIST" not in env
    assert "RANK" not in env
    assert "MASTER_ADDR" not in env


def test_runtime_executor_requests_training_join_when_enabled():
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    cfg = _config()
    planner, decision = _decision(cfg)
    decision.candidate_plan.train_config.dp = 3
    decision.candidate_plan.rollout_config.tp_list = [1]
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0", "rollout1"],
        zero_sync_steps=0,
    )
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        elastic_pool=pool,
    )

    result = executor.execute(decision)

    assert any(action.startswith("join_training:") for action in result.training_actions)
    assert cfg.train_gpus == 2
    assert cfg.train_dp_size == 2
    pool.close()


def test_runtime_executor_uses_training_pool_target_without_core_resize():
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    cfg = _config()
    planner, decision = _decision(cfg)
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 4
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0", "rollout1", "rollout2"],
        zero_sync_steps=0,
    )
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        elastic_pool=pool,
    )

    result = executor.execute(decision)

    assert len(
        [
            action
            for action in result.training_actions
            if action.startswith("join_training:")
        ]
    ) == 2
    assert cfg.train_gpus == 2
    assert cfg.train_dp_size == 2
    pool.close()


def test_runtime_executor_auto_creates_pool_and_attaches_train_engine():
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    class FakeTrainEngine:
        current_version = 7

        def __init__(self):
            self.snapshots = []
            self.domain = None

        def get_elastic_core_replica_ids(self):
            return ["dp0"]

        def capture_elastic_state_snapshot(self, worker_id, target_core_id):
            self.snapshots.append((worker_id, target_core_id))
            return self.current_version

        def set_elastic_gradient_domain(self, domain):
            self.domain = domain

    cfg = _config()
    planner, decision = _decision(cfg)
    decision.candidate_plan.train_config.dp = 3
    decision.candidate_plan.rollout_config.tp_list = [1]
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    train_engine = FakeTrainEngine()
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        train_engine=train_engine,
    )

    result = executor.execute(decision)
    handles = [
        worker for worker in executor.elastic_pool.snapshot().values()
        if getattr(worker.role, "value", worker.role) == "hybrid_joining"
    ]

    assert result.applied
    assert train_engine.domain is executor.elastic_pool.gradient_domain
    assert any(action.startswith("join_training:") for action in result.training_actions)
    assert handles or train_engine.snapshots
    executor.close()


def test_runtime_executor_creates_decoupled_elastic_domain_by_default():
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    class FakeTrainEngine:
        current_version = 7

        def __init__(self):
            self.domain = None
            self.core_group_requested = False

        def get_elastic_core_replica_ids(self):
            return ["dp0"]

        def get_elastic_core_process_group(self):
            self.core_group_requested = True
            return "core-dp"

        def capture_elastic_state_snapshot(self, worker_id, target_core_id):
            del worker_id, target_core_id
            return self.current_version

        def set_elastic_gradient_domain(self, domain):
            self.domain = domain

    cfg = _config()
    planner, decision = _decision(cfg)
    decision.candidate_plan.train_config.dp = 3
    decision.candidate_plan.rollout_config.tp_list = [1]
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    train_engine = FakeTrainEngine()
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        train_engine=train_engine,
    )

    result = executor.execute(decision)

    assert result.applied
    assert train_engine.domain is executor.elastic_pool.gradient_domain
    assert train_engine.domain.decoupled_communication_domains
    assert train_engine.domain.process_group is None
    assert not train_engine.core_group_requested
    executor.close()


def test_runtime_executor_training_pool_plan_only_does_not_attach_domain():
    class FakeTrainEngine:
        current_version = 7

        def __init__(self):
            self.domain = None

        def get_elastic_core_replica_ids(self):
            return ["dp0", "dp1"]

        def set_elastic_gradient_domain(self, domain):
            self.domain = domain

    cfg = _config()
    planner, decision = _decision(cfg)
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = True
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 4
    train_engine = FakeTrainEngine()
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        train_engine=train_engine,
    )

    result = executor.execute(decision)

    assert result.applied
    assert "training_pool_plan_only" in result.training_actions
    assert any(
        action.startswith("plan_join_training:")
        for action in result.training_actions
    )
    assert executor.elastic_pool is None
    assert train_engine.domain is None


def test_runtime_executor_adopts_prewarmed_training_worker_before_pause(tmp_path):
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    class FakeTrainEngine:
        current_version = 3

        def __init__(self):
            self.snapshots = []
            self.domain = None

        def get_elastic_core_replica_ids(self):
            return ["dp0"]

        def capture_elastic_state_snapshot(self, worker_id, target_core_id):
            self.snapshots.append((worker_id, target_core_id))
            return self.current_version

        def get_elastic_state_snapshot_path(self, version):
            path = tmp_path / f"rank0_v{version}.pt"
            path.write_bytes(b"snapshot")
            return path

        def set_elastic_gradient_domain(self, domain):
            self.domain = domain

    class OrderedDispatcher(FakeDispatcher):
        def __init__(self, events):
            super().__init__()
            self.shared_events = events

        def pause(self):
            self.shared_events.append("pause")
            super().pause()

    class FakeRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, events, **kwargs):
            super().__init__(**kwargs)
            self.events = events

        def _ensure_gradient_server(self):
            class Endpoint:
                host = "127.0.0.1"
                port = 12345
                authkey = "auth"

            class Server:
                endpoint = Endpoint()

                def close(self):
                    pass

            self.gradient_server = Server()

        def _launch_prewarmed_hybrid_worker(
            self,
            *,
            worker_id,
            target_core_id,
            command,
            snapshot_path,
        ):
            self.events.append(f"prewarm:{worker_id}")
            meta = super()._launch_prewarmed_hybrid_worker(
                worker_id=worker_id,
                target_core_id=target_core_id,
                command=command,
                snapshot_path=snapshot_path,
            )
            return meta

        def _is_hybrid_worker_running(self, worker_id):
            return worker_id in self._hybrid_worker_meta

    cfg = _config()
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 4
    cfg.global_resource_planner.hybrid_worker_launch_enabled = True
    cfg.global_resource_planner.hybrid_training_prewarm_enabled = True
    cfg.global_resource_planner.hybrid_worker_command_template = "sleep 3600"
    planner, decision = _decision(cfg)
    events = []
    train_engine = FakeTrainEngine()
    executor = FakeRuntimeExecutor(
        events,
        config=cfg,
        planner=planner,
        train_engine=train_engine,
        dispatcher=OrderedDispatcher(events),
    )

    result = executor.execute(decision)

    assert result.applied
    assert events.index("prewarm:rollout0") < events.index("pause")
    assert any(
        action.startswith("prewarm_hybrid_worker:rollout0->")
        for action in result.training_actions
    )
    assert any(
        action.startswith("activate_prewarmed_hybrid_worker:rollout0->")
        for action in result.training_actions
    )
    assert not any(
        action.startswith("launch_hybrid_worker_after_join:")
        for action in result.training_actions
    )
    executor.close()


def test_runtime_executor_cluster_swap_gives_rollout_gpu_to_training():
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    class SwapRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, events, **kwargs):
            super().__init__(**kwargs)
            self.events = events

        def _stop_rollout_process(self, instance_id):
            self.events.append(f"stop_rollout:{instance_id}")
            meta = self._process_meta.pop(instance_id)
            return [meta]

        def _start_one_rollout_process(self, meta):
            self.events.append(f"start_rollout:{meta.instance_id}")
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.events),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            self._process_meta[started.instance_id] = started
            return started

        def _reconfigure_training_pool(self, current_train_gpus, plan, result):
            self.events.append("training_pool")
            return super()._reconfigure_training_pool(
                current_train_gpus,
                plan,
                result,
            )

    cfg = _config()
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    cfg.global_resource_planner.runtime_cluster_swap_enabled = True
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 3
    cfg.global_resource_planner.hybrid_worker_launch_enabled = False
    planner, decision = _decision(cfg)
    decision.candidate_plan.train_config.dp = 3
    decision.candidate_plan.rollout_config.tp_list = [1]
    pool = ElasticHybridPool(
        core_train_workers=["core0"],
        core_rollout_workers=["rollout0", "rollout1"],
        zero_sync_steps=0,
    )
    events = []
    executor = SwapRuntimeExecutor(
        events,
        config=cfg,
        planner=planner,
        elastic_pool=pool,
        rollout_engine=FakeRolloutEngine(),
    )
    executor._process_meta["old0"] = ManagedRolloutProcess(
        instance_id="old0",
        command="old0",
        pid=-1,
        gpus=[2],
        port=cfg.heterogeneous_rollout.vllm_base_port,
        host="127.0.0.1",
        tp=1,
        adopted=True,
    )
    executor._process_meta["old1"] = ManagedRolloutProcess(
        instance_id="old1",
        command="old1",
        pid=-1,
        gpus=[3],
        port=cfg.heterogeneous_rollout.vllm_base_port + 1,
        host="127.0.0.1",
        tp=1,
        adopted=True,
    )

    result = executor.execute(decision)

    assert result.applied
    assert "reconfigure_rollout_processes:cluster_swap" in result.actions
    assert events.index("stop_rollout:old0") < events.index("training_pool")
    assert any(
        action.startswith("join_training:rollout")
        for action in result.training_actions
    )
    pool.close()


def test_runtime_executor_cluster_swap_returns_training_gpu_to_rollout():
    if ElasticHybridPool is None:
        pytest.skip("torch is not installed in this local environment")

    class SwapRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, events, **kwargs):
            super().__init__(**kwargs)
            self.events = events

        def _start_one_rollout_process(self, meta):
            self.events.append(f"start_rollout:{meta.instance_id}")
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.events),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            self._process_meta[started.instance_id] = started
            return started

        def _reconfigure_training_pool(self, current_train_gpus, plan, result):
            self.events.append("training_pool")
            return super()._reconfigure_training_pool(
                current_train_gpus,
                plan,
                result,
            )

    cfg = _config()
    # The immutable core owns two GPUs; two external replicas account for the
    # current effective training allocation of four GPUs.
    cfg.train_gpus = 2
    cfg.rollout_gpus = 2
    cfg.train_dp_size = 2
    cfg.n_total_gpus = 4
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_reconfigure_training = True
    cfg.global_resource_planner.runtime_training_pool_plan_only = False
    cfg.global_resource_planner.runtime_cluster_swap_enabled = True
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 2
    planner, decision = _decision(cfg)
    decision.candidate_plan.train_config.dp = 2
    decision.candidate_plan.rollout_config.tp_list = [1, 1]
    pool = ElasticHybridPool(
        core_train_workers=["core0", "core1"],
        core_rollout_workers=["rollout0", "rollout1"],
        zero_sync_steps=0,
    )
    pool.join_training("rollout0", "core0").result(timeout=1)
    pool.join_training("rollout1", "core0").result(timeout=1)
    events = []
    executor = SwapRuntimeExecutor(
        events,
        config=cfg,
        planner=planner,
        elastic_pool=pool,
        rollout_engine=FakeRolloutEngine(),
    )

    result = executor.execute(decision)

    assert result.applied
    assert events.index("training_pool") < events.index("start_rollout:grp_tp1_0")
    assert any(
        action.startswith("release_to_rollout:rollout")
        for action in result.training_actions
    )
    assert "reconfigure_rollout_processes:cluster_swap" in result.actions
    pool.close()


def test_cluster_swap_adopts_same_physical_rollout_and_filters_training_slots():
    class SwapRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.started = []
            self.stopped = []

        def _stop_rollout_process(self, instance_id):
            self.stopped.append(instance_id)
            meta = self._process_meta.pop(instance_id)
            return [meta]

        def _start_one_rollout_process(self, meta):
            self.started.append(meta.instance_id)
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.started),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            self._process_meta[started.instance_id] = started
            return started

        def _wait_rollout_process_start_ready(self, meta):
            return None

        def _reconfigure_rollout_engine(self, plan, *, wait_ready):
            self._rollout_engine_reconfigured = True

    cfg = _config()
    cfg.train_gpus = 8
    cfg.rollout_gpus = 8
    cfg.global_resource_planner.runtime_training_pool_target_gpus = 10
    cfg.heterogeneous_rollout.vllm_base_port = 8000
    cfg.heterogeneous_rollout.instances = [
        HeterogeneousInstanceConfig(
            instance_id="grp_tp4_0",
            tp=4,
            gpus=[0, 1, 2, 3],
            host="node-a",
            port=8000,
        ),
        HeterogeneousInstanceConfig(
            instance_id="grp_tp2_1",
            tp=2,
            gpus=[0, 1],
            host="node-b",
            port=8001,
        ),
    ]
    planner, decision = _decision(cfg)
    executor = SwapRuntimeExecutor(config=cfg, planner=planner)
    executor._process_meta["initial_override_tp4_0"] = ManagedRolloutProcess(
        instance_id="initial_override_tp4_0",
        command="old-keep",
        pid=-1,
        gpus=[0, 1, 2, 3],
        port=8000,
        host="node-a",
        tp=4,
        adopted=True,
    )
    executor._process_meta["initial_override_tp4_1"] = ManagedRolloutProcess(
        instance_id="initial_override_tp4_1",
        command="old-split",
        pid=-1,
        gpus=[0, 1, 2, 3],
        port=8001,
        host="node-b",
        tp=4,
        adopted=True,
    )

    stopped, started = executor._cluster_swap_reconfigure_rollout_processes(
        decision.candidate_plan
    )
    assigned = executor._assign_cluster_swap_training_slots(
        stopped,
        current_train_gpus=8,
        plan=decision.candidate_plan,
    )

    assert "initial_override_tp4_0" not in executor.stopped
    assert "grp_tp4_0" in executor._process_meta
    assert executor._process_meta["grp_tp4_0"].pid == -1
    assert [meta.instance_id for meta in stopped] == ["initial_override_tp4_1"]
    assert executor.started == ["grp_tp2_1"]
    assert assigned == {
        "rollout0": {"host": "node-b", "gpus": [2]},
        "rollout1": {"host": "node-b", "gpus": [3]},
    }


def test_runtime_executor_diff_reconfigures_only_changed_rollout_instances():
    class FakeRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.started = []
            self.stopped = []

        def _start_one_rollout_process(self, meta):
            self.started.append(meta.instance_id)
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.started),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            old = self._process_meta.get(meta.instance_id)
            if old is not None and not self._same_rollout_process(old, meta):
                old_list = getattr(self, "_blue_green_old_processes", [])
                old_list.append((old, self._processes.get(meta.instance_id)))
                self._blue_green_old_processes = old_list
            self._process_meta[started.instance_id] = started
            return started

        def _stop_rollout_process(self, instance_id):
            self.stopped.append(instance_id)
            meta = self._process_meta.pop(instance_id)
            return [meta]

        def _stop_process_meta(self, meta, proc):
            del proc
            self.stopped.append(meta.instance_id)
            return [meta]

    cfg = _config()
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_rollout_reconfigure_strategy = "diff"
    planner, decision = _decision(cfg)
    planner.apply_plan_to_config(decision.candidate_plan, cfg)
    cfg.heterogeneous_rollout.instances[0].instance_id = "keep"
    cfg.heterogeneous_rollout.instances[0].gpus = [2]
    cfg.heterogeneous_rollout.instances[0].tp = 1
    if len(cfg.heterogeneous_rollout.instances) == 1:
        cfg.heterogeneous_rollout.instances.append(
            type(cfg.heterogeneous_rollout.instances[0])(
                instance_id="replace",
                tp=1,
                gpus=[3],
                host="127.0.0.1",
            )
        )
    cfg.heterogeneous_rollout.instances[1].instance_id = "replace"
    cfg.heterogeneous_rollout.instances[1].gpus = [3]
    cfg.heterogeneous_rollout.instances[1].tp = 1

    executor = FakeRuntimeExecutor(config=cfg, planner=planner)
    executor._process_meta["keep"] = ManagedRolloutProcess(
        instance_id="keep",
        command="old",
        pid=11,
        gpus=[2],
        port=cfg.heterogeneous_rollout.vllm_base_port,
        host="127.0.0.1",
        tp=1,
    )
    executor._process_meta["replace"] = ManagedRolloutProcess(
        instance_id="replace",
        command="old",
        pid=12,
        gpus=[9],
        port=cfg.heterogeneous_rollout.vllm_base_port + 1,
        host="127.0.0.1",
        tp=1,
    )

    stopped, started = executor._reconfigure_rollout_processes(
        decision.candidate_plan
    )

    assert [m.instance_id for m in stopped] == ["replace"]
    assert [m.instance_id for m in started] == ["replace"]
    assert executor.started == ["replace"]
    assert executor.stopped == ["replace"]
    assert "keep" in executor._process_meta


def test_runtime_executor_can_disable_dynamic_reconfiguration():
    cfg = _config()
    cfg.global_resource_planner.runtime_dynamic_reconfiguration_enabled = False
    planner, decision = _decision(cfg)
    dispatcher = FakeDispatcher()
    engine = FakeRolloutEngine()
    executor = RuntimeElasticExecutor(
        config=cfg,
        planner=planner,
        rollout_engine=engine,
        dispatcher=dispatcher,
    )

    result = executor.execute(decision)

    assert not result.applied
    assert result.reason == "runtime_dynamic_reconfiguration_disabled"
    assert dispatcher.events == []
    assert engine.calls == []


def test_runtime_executor_blue_green_uses_spare_port_before_cutover():
    class ReadyRolloutEngine(FakeRolloutEngine):
        def wait_for_ready(self, timeout=300.0):
            self.calls.append(("ready", timeout))

    class FakeRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.started = []
            self.stopped = []

        def _start_one_rollout_process(self, meta):
            self.started.append((meta.instance_id, meta.port, list(meta.gpus), meta.tp))
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.started),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            old = self._process_meta.get(meta.instance_id)
            if old is not None and not self._same_rollout_process(old, meta):
                old_list = getattr(self, "_blue_green_old_processes", [])
                old_list.append((old, self._processes.get(meta.instance_id)))
                self._blue_green_old_processes = old_list
            self._process_meta[started.instance_id] = started
            return started

        def _stop_process_meta(self, meta, proc):
            del proc
            self.stopped.append((meta.instance_id, meta.port))
            return [meta]

    cfg = _config()
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_rollout_reconfigure_strategy = "blue_green"
    cfg.global_resource_planner.vllm_ready_timeout_s = 12.0
    planner, decision = _decision(cfg)
    planner.apply_plan_to_config(decision.candidate_plan, cfg)
    cfg.heterogeneous_rollout.instances[0].instance_id = "replace"
    cfg.heterogeneous_rollout.instances[0].tp = 2
    cfg.heterogeneous_rollout.instances[0].gpus = [2, 3]
    cfg.heterogeneous_rollout.instances[0].port = cfg.heterogeneous_rollout.vllm_base_port
    cfg.heterogeneous_rollout.instances = [cfg.heterogeneous_rollout.instances[0]]

    engine = ReadyRolloutEngine()
    executor = FakeRuntimeExecutor(config=cfg, planner=planner, rollout_engine=engine)
    executor._process_meta["replace"] = ManagedRolloutProcess(
        instance_id="replace",
        command="old",
        pid=11,
        gpus=[2],
        port=cfg.heterogeneous_rollout.vllm_base_port,
        host="127.0.0.1",
        tp=1,
    )

    stopped, started = executor._reconfigure_rollout_processes(decision.candidate_plan)

    assert started[0].port == cfg.heterogeneous_rollout.vllm_base_port + 1
    assert cfg.heterogeneous_rollout.instances[0].port == started[0].port
    assert engine.calls[0][0] == decision.candidate_plan.rollout_tp_list
    assert ("ready", 12.0) in engine.calls
    assert stopped[0].instance_id == "replace"
    assert executor.stopped == [("replace", cfg.heterogeneous_rollout.vllm_base_port)]


def test_runtime_executor_prewarms_rollout_before_dispatcher_pause():
    class ReadyRolloutEngine(FakeRolloutEngine):
        def wait_for_ready(self, timeout=300.0):
            self.calls.append(("ready", timeout))

    class OrderedDispatcher(FakeDispatcher):
        def __init__(self, events):
            super().__init__()
            self.shared_events = events

        def pause(self):
            self.shared_events.append("pause")
            super().pause()

        def wait_until_idle(self, timeout=3600.0):
            self.shared_events.append("drain")
            super().wait_until_idle(timeout=timeout)

        def resume(self):
            self.shared_events.append("resume")
            super().resume()

    class FakeRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, events, **kwargs):
            super().__init__(**kwargs)
            self.events = events

        def _start_one_rollout_process(self, meta):
            self.events.append(f"start:{meta.instance_id}:{meta.port}")
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.events),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            old = self._process_meta.get(meta.instance_id)
            if old is not None and not self._same_rollout_process(old, meta):
                old_list = getattr(self, "_blue_green_old_processes", [])
                old_list.append((old, self._processes.get(meta.instance_id)))
                self._blue_green_old_processes = old_list
            self._process_meta[started.instance_id] = started
            return started

        def _wait_rollout_process_start_ready(self, meta):
            self.events.append(f"ready:{meta.instance_id}:{meta.port}")

        def _stop_rollout_process(self, instance_id):
            self.events.append(f"stop:{instance_id}")
            return super()._stop_rollout_process(instance_id)

        def _terminate_external_pid(self, meta):
            self.events.append(f"terminate:{meta.instance_id}")

        def _run_external_stop(self, meta):
            self.events.append(f"external_stop:{meta.instance_id}")

    cfg = _config()
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_rollout_reconfigure_strategy = "prewarm"
    cfg.global_resource_planner.runtime_wait_rollout_process_start_ready = True
    cfg.global_resource_planner.runtime_post_rollout_stop_grace_s = 0.0
    cfg.heterogeneous_rollout.available_gpus = [0, 1, 2, 3]
    planner, decision = _decision(cfg)
    events = []
    dispatcher = OrderedDispatcher(events)
    engine = ReadyRolloutEngine()
    executor = FakeRuntimeExecutor(
        events,
        config=cfg,
        planner=planner,
        rollout_engine=engine,
        dispatcher=dispatcher,
    )
    executor._process_meta["old"] = ManagedRolloutProcess(
        instance_id="old",
        command="old",
        pid=-1,
        gpus=[0],
        port=cfg.heterogeneous_rollout.vllm_base_port,
        host="127.0.0.1",
        tp=1,
        adopted=True,
    )

    result = executor.execute(decision)

    assert result.applied
    first_start = next(i for i, event in enumerate(events) if event.startswith("start:"))
    pause_index = events.index("pause")
    assert first_start < pause_index
    assert any(action == "prewarm_rollout_processes" for action in result.actions)
    assert any(
        action == "reconfigure_rollout_processes:prewarm"
        for action in result.actions
    )
    assert any(event == "stop:old" for event in events)
    assert engine.calls[0][0] == decision.candidate_plan.rollout_tp_list


def test_runtime_executor_prewarm_falls_back_when_no_spare_capacity():
    class OrderedDispatcher(FakeDispatcher):
        def __init__(self, events):
            super().__init__()
            self.shared_events = events

        def pause(self):
            self.shared_events.append("pause")
            super().pause()

    class FakeRuntimeExecutor(RuntimeElasticExecutor):
        def __init__(self, events, **kwargs):
            super().__init__(**kwargs)
            self.events = events

        def _start_one_rollout_process(self, meta):
            self.events.append(f"start:{meta.instance_id}:{meta.port}")
            started = ManagedRolloutProcess(
                instance_id=meta.instance_id,
                command=meta.command,
                pid=100 + len(self.events),
                gpus=list(meta.gpus),
                port=meta.port,
                host=meta.host,
                tp=meta.tp,
            )
            self._process_meta[started.instance_id] = started
            return started

        def _wait_rollout_process_start_ready(self, meta):
            self.events.append(f"ready:{meta.instance_id}:{meta.port}")

        def _terminate_external_pid(self, meta):
            self.events.append(f"terminate:{meta.instance_id}")

        def _run_external_stop(self, meta):
            self.events.append(f"external_stop:{meta.instance_id}")

    cfg = _config()
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_rollout_reconfigure_strategy = "prewarm"
    cfg.global_resource_planner.runtime_prewarm_no_spare_fallback_strategy = "restart_all"
    cfg.global_resource_planner.runtime_post_rollout_stop_grace_s = 0.0
    planner, decision = _decision(cfg)
    events = []
    executor = FakeRuntimeExecutor(
        events,
        config=cfg,
        planner=planner,
        rollout_engine=FakeRolloutEngine(),
        dispatcher=OrderedDispatcher(events),
    )
    executor._process_meta["old0"] = ManagedRolloutProcess(
        instance_id="old0",
        command="old0",
        pid=-1,
        gpus=[0],
        port=cfg.heterogeneous_rollout.vllm_base_port,
        host="127.0.0.1",
        tp=1,
        adopted=True,
    )
    executor._process_meta["old1"] = ManagedRolloutProcess(
        instance_id="old1",
        command="old1",
        pid=-1,
        gpus=[1],
        port=cfg.heterogeneous_rollout.vllm_base_port + 1,
        host="127.0.0.1",
        tp=1,
        adopted=True,
    )

    result = executor.execute(decision)

    assert result.applied
    assert any(
        action.startswith("prewarm_unavailable:no_spare_gpu_capacity:")
        for action in result.actions
    )
    assert "reconfigure_rollout_processes:restart_all" in result.actions
    assert "reconfigure_rollout_processes:prewarm" not in result.actions
    assert events.index("pause") < next(
        idx for idx, event in enumerate(events) if event.startswith("start:")
    )


def test_global_resource_planner_can_force_training_pool_resize(monkeypatch):
    cfg = _config()
    cfg.n_total_gpus = 6
    cfg.train_gpus = 2
    cfg.rollout_gpus = 2
    cfg.train_tp_size = 1
    cfg.train_pp_size = 1
    cfg.train_dp_size = 2
    cfg.global_resource_planner.reconfiguration_cost_s = 0.0
    monkeypatch.setenv("GRP_FORCE_TRAIN_GPUS", "4")
    planner = GlobalResourcePlanner.from_config(cfg)

    decision = planner.plan_if_needed(
        step=0,
        config=cfg,
        batch=[
            {"input_len": 128, "output_len": 512},
            {"input_len": 128, "output_len": 512},
            {"input_len": 128, "output_len": 512},
            {"input_len": 128, "output_len": 512},
        ],
    )

    assert decision.should_reconfigure
    assert decision.reason == "forced_reconfigure"
    assert decision.candidate_plan.train_gpus == 4
    assert decision.candidate_plan.train_config.dp == 4


def test_runtime_executor_writes_rollout_manifest(tmp_path):
    cfg = _config()
    cfg.global_resource_planner.runtime_rollout_manifest_path = str(
        tmp_path / "rollout_manifest.json"
    )
    planner, decision = _decision(cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)

    planner.apply_plan_to_config(decision.candidate_plan, cfg)
    executor._write_rollout_manifest(decision.candidate_plan, phase="planned")

    payload = json.loads((tmp_path / "rollout_manifest.json").read_text())
    assert payload["phase"] == "planned"
    assert payload["plan"]["rollout"]["tp_list"] == decision.candidate_plan.rollout_tp_list
    assert payload["instances"]


def test_runtime_executor_skips_already_applied_runtime_plan(tmp_path):
    cfg = _config()
    cfg.global_resource_planner.runtime_manage_rollout_processes = True
    cfg.global_resource_planner.runtime_cluster_swap_enabled = True
    cfg.global_resource_planner.runtime_rollout_reconfigure_strategy = "cluster_swap"
    cfg.global_resource_planner.runtime_rollout_manifest_path = str(
        tmp_path / "rollout_manifest.json"
    )
    planner, decision = _decision(cfg)
    planner.apply_plan_to_config(decision.candidate_plan, cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)
    for meta in executor._desired_rollout_processes(decision.candidate_plan):
        executor._process_meta[meta.instance_id] = ManagedRolloutProcess(
            instance_id=meta.instance_id,
            command=meta.command,
            pid=123,
            gpus=list(meta.gpus),
            port=meta.port,
            host=meta.host,
            tp=meta.tp,
            adopted=True,
            log_path=meta.log_path,
        )

    result = executor.execute(decision)

    assert result.applied is False
    assert result.reason == "runtime_plan_already_applied"
    assert result.actions == ["runtime_plan_already_applied"]


def test_runtime_executor_waits_for_megatron_batch_source_peers(monkeypatch):
    cfg = _config()
    cfg.train_backend = "megatron_core"
    cfg.train_gpus = 8
    cfg.train_tp_size = 4
    cfg.train_pp_size = 1
    cfg.train_cp_size = 1
    cfg.train_dp_size = 2
    planner, _ = _decision(cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)

    monkeypatch.setenv("WORLD_SIZE", "8")

    assert executor._expected_peer_ready_ranks() == [4]


def test_runtime_executor_can_wait_for_all_peers(monkeypatch):
    cfg = _config()
    cfg.global_resource_planner.runtime_coordinate_batch_source_only = False
    planner, _ = _decision(cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)

    monkeypatch.setenv("WORLD_SIZE", "4")

    assert executor._expected_peer_ready_ranks() == [1, 2, 3]


def test_runtime_executor_scrubs_training_env_for_rollout(monkeypatch):
    cfg = _config()
    planner, _ = _decision(cfg)
    executor = RuntimeElasticExecutor(config=cfg, planner=planner)
    meta = ManagedRolloutProcess(
        instance_id="rollout",
        command="true",
        pid=0,
        gpus=[0, 1],
        port=8000,
        host="127.0.0.1",
        tp=2,
    )

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "node-a")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "train")

    env = executor._rollout_launch_env(meta)

    for key in [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    ]:
        assert key not in env
    assert env["PLANNED_CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["RL_FRAMEWORK_ROLE"] == "runtime_rollout"


def test_dispatcher_reset_after_reconfigure_clears_old_rollout_state():
    class VersionProvider:
        def get_version(self):
            return 0

    class FakeRunner:
        def __init__(self):
            import queue
            import threading

            self.input_queue = queue.Queue()
            self.output_queue = queue.Queue()
            self.max_queue_size = 8
            self.paused = threading.Event()

        def pause(self):
            self.paused.set()

        def resume(self):
            self.paused.clear()

        def get_queue_sizes(self):
            return self.input_queue.qsize(), self.output_queue.qsize()

    staleness = StalenessManager(
        VersionProvider(),
        max_concurrent_rollouts=4,
        consumer_batch_size=2,
        max_staleness=2,
    )
    dispatcher = BatchTaskDispatcher(
        FakeRunner(),
        staleness,
        task_factory=lambda _: (lambda: None),
    )
    dispatcher.submit_task_input(TaskInput(task_id=1, data={}))
    staleness.on_rollout_submitted()
    staleness.on_rollout_accepted()
    with dispatcher._result_cv:
        dispatcher._pending_results[1] = object()

    dispatcher.reset_after_reconfigure()

    assert dispatcher.get_runtime_metrics()["pending_inputs"] == 0
    assert dispatcher.get_runtime_metrics()["pending_results"] == 0
    assert dispatcher.get_runtime_metrics()["active_tasks"] == 0
    stats = staleness.get_stats()
    assert stats.enqueued == 0
    assert stats.running == 0
    assert stats.accepted == 0

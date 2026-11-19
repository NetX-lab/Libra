"""Support code for Test hetero cmlfq integration."""

import unittest

from RL_Framework.config import (
    AsyncRLConfig,
    HeterogeneousInstanceConfig,
    HeterogeneousRolloutConfig,
    SchedulingConfig,
)
from RL_Framework.infra.scheduling.cmlfq_scheduler import (
    CMLFQMigrationDecision,
    CMLFQScheduler,
)
from RL_Framework.infra.scheduling.cmlfq_tool_state import (
    ExecutionStatus,
    PayloadSizeClass,
    ToolReturnState,
)
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import Trajectory
from RL_Framework.infra.scheduling.factory import SchedulerFactory
from RL_Framework.engine.heterogeneous_engine import (
    HeterogeneousRolloutEngine,
)
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlanner,
)


class TestSchedulerFactory(unittest.TestCase):
    """Test scheduler factory implementation."""

    def test_create_cmlfq_scheduler(self):
        hetero = HeterogeneousRolloutConfig(
            enabled=True,
            instances=[
                HeterogeneousInstanceConfig(tp=1, gpus=[0]),
                HeterogeneousInstanceConfig(tp=2, gpus=[1, 2]),
            ],
            scheduling=SchedulingConfig(
                scheduler_type="cmlfq",
                cmlfq_buckets={
                    "short": {"tp_degrees": [1], "max_tokens": 5000},
                    "long": {"tp_degrees": [2], "max_tokens": 50000},
                },
            ),
        )
        scheduler = SchedulerFactory.create("cmlfq", hetero)
        self.assertIsInstance(scheduler, CMLFQScheduler)
        self.assertEqual(scheduler.name, "C-MLFQ")

    def test_available_types_includes_cmlfq(self):
        types = SchedulerFactory.available_types()
        self.assertIn("cmlfq", types)


class TestGlobalResourcePlannerFlow(unittest.TestCase):
    """Test global resource planner flow implementation."""

    def test_planner_reconfigures_then_cmlfq_routes(self):
        config = AsyncRLConfig(
            model_path="/tmp/fake-model",
            train_gpus=2,
            rollout_gpus=2,
            train_tp_size=1,
            train_pp_size=1,
            train_dp_size=2,
            micro_batch_size=1,
            batch_size=8,
            n_total_gpus=4,
            max_concurrent_rollouts=8,
            heterogeneous_rollout=HeterogeneousRolloutConfig(
                enabled=True,
                total_gpus=2,
                available_gpus=[2, 3],
                instances=[
                    HeterogeneousInstanceConfig(tp=1, gpus=[2]),
                    HeterogeneousInstanceConfig(tp=1, gpus=[3]),
                ],
                scheduling=SchedulingConfig(
                    scheduler_type="cmlfq",
                    cmlfq_buckets={
                        "short": {"tp_degrees": [1], "max_tokens": 5000},
                        "long": {"tp_degrees": [2], "max_tokens": 50000},
                    },
                ),
            ),
        )
        config.global_resource_planner.enabled = True
        config.global_resource_planner.plan_interval = 1
        config.global_resource_planner.warmup_steps = 0
        config.global_resource_planner.min_history_size = 4
        config.global_resource_planner.min_gain_ratio = 0.0
        config.global_resource_planner.reconfiguration_cost_s = 0.0
        config.global_resource_planner.allowed_rollout_tp = [1, 2]
        config.global_resource_planner.allowed_train_tp = [1, 2]
        config.global_resource_planner.allowed_train_pp = [1]
        config.global_resource_planner.micro_batch_sizes = [1]

        planner = GlobalResourcePlanner.from_config(config)
        long_tail_batch = [
            {"input_len": 1024, "output_len": 18000},
            {"input_len": 1024, "output_len": 22000},
            {"input_len": 1024, "output_len": 26000},
            {"input_len": 1024, "output_len": 32000},
        ]
        decision = planner.plan_if_needed(
            step=1,
            config=config,
            batch=long_tail_batch,
        )

        self.assertIn(
            decision.reason,
            {"reconfigure", "same_plan", "gain_below_threshold"},
        )
        self.assertIsNotNone(decision.candidate_plan)
        planner.apply_plan_to_config(decision.candidate_plan, config)
        self.assertGreaterEqual(config.rollout_gpus, 1)
        self.assertEqual(
            sum(inst.tp for inst in config.heterogeneous_rollout.instances),
            config.rollout_gpus,
        )

        engine = HeterogeneousRolloutEngine.from_config(config)
        engine.reconfigure_from_plan(decision.candidate_plan, config)
        self.assertEqual(engine.tp_list, config.heterogeneous_rollout.tp_list)

        initial = engine.begin_cmlfq_request("grp_prompt", 1024)
        self.assertTrue(initial)
        state = ToolReturnState(
            tool_type="code",
            payload_size_class=PayloadSizeClass.SizeLarge,
            execution_status=ExecutionStatus.ToolFailure,
        )
        engine.scheduler.prefix_tree.insert(
            Trajectory(
                prompt_id="grp_prompt",
                return_states=[state],
                total_remaining_lengths=[30000],
            )
        )
        decision2 = engine.route_cmlfq_tool_return(
            initial,
            {
                "tool_type": "code",
                "status": "failure",
                "payload_tokens": 8000,
                "output": "x" * 24000,
            },
            generated_tokens=1200,
        )
        self.assertIsInstance(decision2, CMLFQMigrationDecision)
        engine.finish_cmlfq_request(initial, 32000)


class TestCMLFQEndToEnd(unittest.TestCase):
    """Test c m l f q end to end implementation."""

    def setUp(self):
        self.scheduler = CMLFQScheduler(
            buckets={
                "short": {"tp_degrees": [1], "max_tokens": 5000},
                "long": {"tp_degrees": [2], "max_tokens": 50000},
            },
            bucket_thresholds={
                "short": 5000,
                "long": 50000,
            },
        )
        self.scheduler.register_instance(0, "inst_0", 1)
        self.scheduler.register_instance(1, "inst_1", 2)

    def test_end_to_end_short_sequence(self):
        """Test end to end short sequence."""

        result = self.scheduler.schedule(
            input_tokens=1000,
            prompt_id="prompt_short",
        )
        self.assertEqual(result.category, "short")
        self.assertEqual(result.tp_degree, 1)
        request_id = list(self.scheduler._request_states.keys())[0]


        state = ToolReturnState(
            tool_type="search",
            payload_size_class=PayloadSizeClass.SizeSmall,
            execution_status=ExecutionStatus.ToolSuccess,
        )
        traj = Trajectory(
            prompt_id="prompt_short",
            return_states=[state],
            total_remaining_lengths=[2000],
        )
        self.scheduler.prefix_tree.insert(traj)


        decision = self.scheduler.on_tool_return(
            request_id=request_id,
            tool_result={"tool_type": "search", "output": "small result"},
            generated_tokens=500,
        )

        self.assertFalse(decision.should_migrate)


        self.scheduler.on_request_done(
            instance_index=result.instance_index,
            prompt_id="prompt_short",
            final_bucket="short",
            output_tokens=2500,
        )


        stats = self.scheduler.prefix_tree.get_stats()
        self.assertGreater(stats["total_nodes"], 0)

    def test_end_to_end_long_sequence_with_migration(self):
        """Test end to end long sequence with migration."""

        result = self.scheduler.schedule(
            input_tokens=1000,
            prompt_id="prompt_long",
        )
        self.assertEqual(result.category, "short")
        request_id = list(self.scheduler._request_states.keys())[0]


        state = ToolReturnState(
            tool_type="code",
            payload_size_class=PayloadSizeClass.SizeLarge,
            execution_status=ExecutionStatus.ToolSuccess,
        )
        traj = Trajectory(
            prompt_id="prompt_long",
            return_states=[state],
            total_remaining_lengths=[30000],
        )
        self.scheduler.prefix_tree.insert(traj)


        decision = self.scheduler.on_tool_return(
            request_id=request_id,
            tool_result={"tool_type": "code", "output": "x" * 6000},
            generated_tokens=2000,
        )
        self.assertTrue(decision.should_migrate)
        self.assertEqual(decision.target_bucket, "long")


        new_result = self.scheduler.execute_migration(request_id, decision)
        self.assertEqual(new_result.category, "long")
        self.assertEqual(new_result.tp_degree, 2)


        self.scheduler.on_request_done(
            instance_index=new_result.instance_index,
            prompt_id="prompt_long",
            final_bucket="long",
            output_tokens=32000,
        )

    def test_end_to_end_multiple_tool_returns(self):
        """Test end to end multiple tool returns."""
        result = self.scheduler.schedule(
            input_tokens=1000,
            prompt_id="prompt_multi",
        )
        request_id = list(self.scheduler._request_states.keys())[0]


        states = [
            ToolReturnState("search", PayloadSizeClass.SizeSmall, ExecutionStatus.ToolSuccess),
            ToolReturnState("code", PayloadSizeClass.SizeLarge, ExecutionStatus.ToolSuccess),
        ]
        traj = Trajectory(
            prompt_id="prompt_multi",
            return_states=states,
            total_remaining_lengths=[15000, 5000],
        )
        self.scheduler.prefix_tree.insert(traj)


        decision1 = self.scheduler.on_tool_return(
            request_id, {"tool_type": "search", "output": "small"}, 500
        )

        self.assertIsInstance(decision1, CMLFQMigrationDecision)


        decision2 = self.scheduler.on_tool_return(
            request_id, {"tool_type": "code", "output": "x" * 6000}, 2000
        )
        self.assertIsInstance(decision2, CMLFQMigrationDecision)


        self.scheduler.on_request_done(
            instance_index=result.instance_index,
            prompt_id="prompt_multi",
            final_bucket=result.category,
            output_tokens=20000,
        )

    def test_finish_records_exact_remaining_lengths(self):
        result = self.scheduler.schedule(
            input_tokens=100,
            prompt_id="prompt_exact",
        )
        self.scheduler.on_tool_return(
            result.request_id,
            {"tool_type": "search", "output": "small"},
            generated_tokens=120,
        )
        self.scheduler.on_tool_return(
            result.request_id,
            {"tool_type": "code", "output": "x" * 6000},
            generated_tokens=700,
        )
        states = list(
            self.scheduler._request_states[
                result.request_id
            ].collected_return_states
        )

        self.scheduler.finish_request(result.request_id, 1000)

        first = self.scheduler.prefix_tree.lookup(
            "prompt_exact", states[:1]
        )
        second = self.scheduler.prefix_tree.lookup(
            "prompt_exact", states
        )
        self.assertEqual(first.remaining_lengths, [880])
        self.assertEqual(second.remaining_lengths, [300])
        self.assertNotIn(result.request_id, self.scheduler._request_states)

    def test_finish_uses_instance_for_duplicate_prompt_ids(self):
        self.scheduler.register_instance(2, "inst_2", 1)
        first = self.scheduler.schedule(
            input_tokens=100,
            prompt_id="duplicate_prompt",
        )
        second = self.scheduler.schedule(
            input_tokens=100,
            prompt_id="duplicate_prompt",
        )
        self.assertNotEqual(first.instance_index, second.instance_index)
        self.scheduler.on_tool_return(
            second.request_id,
            {"tool_type": "search", "output": "small"},
            generated_tokens=200,
        )

        self.scheduler.on_request_done(
            instance_index=second.instance_index,
            prompt_id="duplicate_prompt",
            final_bucket=second.category,
            output_tokens=500,
        )

        self.assertIn(first.request_id, self.scheduler._request_states)
        self.assertNotIn(second.request_id, self.scheduler._request_states)
        self.scheduler.cancel_request(first.request_id)

    def test_migration_result_keeps_request_id(self):
        result = self.scheduler.schedule(
            input_tokens=100,
            prompt_id="prompt_migration_id",
        )
        state = ToolReturnState(
            "code",
            PayloadSizeClass.SizeLarge,
            ExecutionStatus.ToolSuccess,
        )
        self.scheduler.prefix_tree.insert(
            Trajectory(
                prompt_id="prompt_migration_id",
                return_states=[state],
                total_remaining_lengths=[30000],
            )
        )

        decision = self.scheduler.on_tool_return(
            result.request_id,
            {"tool_type": "code", "output": "x" * 6000},
            generated_tokens=250,
        )
        migrated = self.scheduler.execute_migration(result.request_id, decision)

        self.assertEqual(migrated.request_id, result.request_id)
        self.scheduler.cancel_request(result.request_id)

    def test_epoch_lifecycle(self):
        """Test epoch lifecycle."""
        self.scheduler.on_epoch_start(1)
        self.scheduler.on_epoch_end(1)

    def test_stats(self):
        """Test stats."""
        self.scheduler.schedule(input_tokens=1000, prompt_id="p1")
        stats = self.scheduler.get_stats()
        self.assertEqual(stats.total_requests, 1)


class TestConfigIntegration(unittest.TestCase):
    """Test config integration implementation."""

    def test_load_config_with_cmlfq(self):
        """Test load config with cmlfq."""
        config = AsyncRLConfig(
            model_path="Qwen/Qwen3-14B",
            heterogeneous_rollout=HeterogeneousRolloutConfig(
                enabled=True,
                instances=[
                    HeterogeneousInstanceConfig(tp=1, gpus=[0]),
                    HeterogeneousInstanceConfig(tp=2, gpus=[1, 2]),
                ],
                scheduling=SchedulingConfig(
                    scheduler_type="cmlfq",
                    cmlfq_buckets={
                        "short": {"tp_degrees": [1], "max_tokens": 5000},
                        "long": {"tp_degrees": [2], "max_tokens": 50000},
                    },
                    cmlfq_rebuild_interval=50,
                ),
            ),
        )
        self.assertEqual(config.heterogeneous_rollout.scheduling.scheduler_type, "cmlfq")
        self.assertEqual(config.heterogeneous_rollout.scheduling.cmlfq_rebuild_interval, 50)


class _FakeRolloutEngine:
    def __init__(self, name):
        self.name = name
        self.base_url = f"fake://{name}"
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return {
            "text": self.name,
            "tokens": [self.name],
            "logprobs": [0.0],
        }


class TestHeterogeneousEngineCMLFQ(unittest.IsolatedAsyncioTestCase):
    async def test_tool_return_changes_next_rollout_bucket(self):
        scheduler = CMLFQScheduler(
            buckets={
                "short": {"tp_degrees": [1], "max_tokens": 5000},
                "long": {"tp_degrees": [2], "max_tokens": 50000},
            }
        )
        engine = HeterogeneousRolloutEngine(
            model_path="Qwen/Qwen3-14B",
            scheduler=scheduler,
        )
        engine.add_instance("short", "127.0.0.1", 8000, 1)
        engine.add_instance("long", "127.0.0.1", 8001, 2)
        short_engine = _FakeRolloutEngine("short")
        long_engine = _FakeRolloutEngine("long")
        engine.engines = [short_engine, long_engine]

        historical_state = ToolReturnState(
            "search",
            PayloadSizeClass.SizeLarge,
            ExecutionStatus.ToolSuccess,
        )
        scheduler.prefix_tree.insert(
            Trajectory(
                prompt_id="p",
                return_states=[historical_state],
                total_remaining_lengths=[30000],
                total_length=32000,
            )
        )

        request_id = engine.begin_cmlfq_request("p", 100)
        first = await engine.generate(
            "prompt",
            prompt_id="p",
            request_id=request_id,
        )
        decision = engine.route_cmlfq_tool_return(
            request_id,
            {"tool_type": "search", "output": "x" * 6000},
            generated_tokens=100,
        )
        second = await engine.generate(
            "prompt + tool",
            prompt_id="p",
            request_id=request_id,
        )
        engine.finish_cmlfq_request(request_id, 1000)

        self.assertEqual(first["text"], "short")
        self.assertTrue(decision.should_migrate)
        self.assertEqual(second["text"], "long")
        self.assertEqual(short_engine.calls, 1)
        self.assertEqual(long_engine.calls, 1)



class TestSingleDynamicIntegration(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def config(kind="cmlfq_cost_kv_dynamic", enabled=False):
        config = AsyncRLConfig(model_path="not-loaded")
        config.heterogeneous_rollout.instances = []
        sched = config.heterogeneous_rollout.scheduling
        sched.scheduler_type = kind
        sched.enable_metrics_feedback = enabled
        sched.cmlfq_buckets = {
            "short": {"tp_degrees": [1], "max_tokens": 4000},
            "long": {"tp_degrees": [2], "max_tokens": 12000},
        }
        return config

    def test_factory_and_poller_rebinding_preserve_official_lifecycle(self):
        from unittest.mock import patch
        from RL_Framework.infra.scheduling.cmlfq_kv_cost_scheduler import CMLFQDynamicKVCostScheduler
        config = self.config()
        with patch("RL_Framework.infra.scheduling.metrics_feed.VLLMMetricsPoller") as poller:
            engine = HeterogeneousRolloutEngine.from_config(config)
            first_feed = engine._metrics_poller
            self.assertIsInstance(engine.scheduler, CMLFQDynamicKVCostScheduler)
            self.assertIs(engine.scheduler._metrics_feed, first_feed)
            self.assertFalse(engine.scheduler._feedback.enabled)
            poller.assert_called_once_with(interval_s=1.0, timeout_s=1.0, ttl_s=5.0)
            first_feed.start.assert_called_once()
            state = ToolReturnState("search", PayloadSizeClass.SizeSmall, ExecutionStatus.ToolSuccess)
            engine.scheduler.prefix_tree.insert(Trajectory("carry", [state], [1000]))
            old_scheduler = engine.scheduler
            old_scheduler._dynamic_boundary = 6000
            engine.reconfigure_from_plan(None, config)
            first_feed.stop.assert_called_once()
            self.assertIsNot(engine.scheduler, old_scheduler)
            self.assertIsNotNone(engine.scheduler.prefix_tree.lookup("carry", [state]))
            self.assertEqual(engine.scheduler._dynamic_boundary, 4000)
            self.assertIsNone(engine.scheduler._last_boundary_epoch)
            self.assertIs(engine.scheduler._metrics_feed, engine._metrics_poller)
            self.assertEqual(poller.call_count, 2)
            rebuilt = HeterogeneousRolloutEngine.from_config(
                config, carry_scheduler_state_from=engine.scheduler,
            )
            self.assertIsNotNone(rebuilt.scheduler.prefix_tree.lookup("carry", [state]))
            rebuilt.close_sync()
            engine.close_sync()
            self.assertIsNone(rebuilt.scheduler._metrics_feed)
            self.assertIsNone(engine.scheduler._metrics_feed)

    def test_other_alias_retains_configured_metrics_behavior(self):
        from unittest.mock import patch
        config = self.config("cmlfq_cost")
        with patch("RL_Framework.infra.scheduling.metrics_feed.VLLMMetricsPoller") as poller:
            engine = HeterogeneousRolloutEngine.from_config(config)
            poller.assert_not_called()
            self.assertIsNone(engine._metrics_poller)
            config.heterogeneous_rollout.scheduling.enable_metrics_feedback = True
            engine._setup_metrics_polling(config.heterogeneous_rollout)
            poller.assert_called_once_with(interval_s=3.0, timeout_s=1.0, ttl_s=10.0)
            engine.close_sync()

    async def test_single_generation_metadata_commit_and_backend_failure_rollback(self):
        from types import SimpleNamespace
        from RL_Framework.infra.scheduling.cmlfq_kv_cost_scheduler import CMLFQDynamicKVCostScheduler
        from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor
        event = {"tool_type": "search", "output": "ok", "payload_tokens": 2}
        state = DefaultToolStateExtractor().extract(event)
        scheduler = CMLFQDynamicKVCostScheduler(
            buckets=self.config().heterogeneous_rollout.scheduling.cmlfq_buckets,
        )
        engine = HeterogeneousRolloutEngine(scheduler=scheduler)
        engine.add_instance("short", "127.0.0.1", 8000, 1)
        engine.add_instance("long", "127.0.0.1", 8001, 2)
        class Costs:
            def decode_seconds(self, tp, *args):
                return 10.0 if tp == 1 else 1.0
            def migration_seconds(self, *args):
                return "recompute", 0.0
        class Feed:
            def snapshots(self):
                import time
                return {name: SimpleNamespace(
                    updated_at=time.time(), gpu_cache_usage=0.2, running=0,
                    waiting=0, kv_capacity_tokens=100,
                ) for name in ("short", "long")}
        class FailedRollout(_FakeRolloutEngine):
            async def generate(self, **kwargs):
                raise RuntimeError("backend failed")
        scheduler.routing_costs = Costs()
        scheduler.attach_metrics_feed(Feed())
        scheduler.prefix_tree.insert(Trajectory("p", [state], [5000]))
        engine.engines = [_FakeRolloutEngine("short"), _FakeRolloutEngine("long")]
        request_id = engine.begin_cmlfq_request("p", 100)
        first = await engine.generate("prompt", request_id=request_id, prompt_id="p")
        self.assertEqual(first["_schedule_info"]["kv_routing"]["policy"], "cmlfq_cost_kv_dynamic")
        self.assertEqual(first["_schedule_info"]["instance_index"], 0)
        engine.route_cmlfq_tool_return(request_id, event, generated_tokens=1)
        engine.engines[1] = FailedRollout("long")
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            await engine.generate("prompt+tool", request_id=request_id, prompt_id="p")
        self.assertEqual(sum(h.active_requests for h in scheduler._instances), 1)
        self.assertFalse(scheduler._reservations)
        self.assertEqual(scheduler.get_request_route(request_id).instance_index, 0)
        engine.engines[1] = _FakeRolloutEngine("long")
        second = await engine.generate("prompt+tool", request_id=request_id, prompt_id="p")
        self.assertEqual(second["_schedule_info"]["instance_index"], 1)
        self.assertEqual(second["_schedule_info"]["kv_routing"]["chosen_instance_id"], "long")
        self.assertEqual(sum(h.active_requests for h in scheduler._instances), 1)
        engine.finish_cmlfq_request(request_id, 100)
        self.assertEqual(sum(h.active_requests for h in scheduler._instances), 0)
        self.assertFalse(scheduler._reservations)
        self.assertFalse(scheduler._pending_cost_routes)
        await engine._cmlfq_backend.close()

if __name__ == "__main__":
    unittest.main()

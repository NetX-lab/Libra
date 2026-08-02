"""Support code for Test cmlfq scheduler."""

import os
import sys
import tempfile
import unittest

from RL_Framework.infra.scheduling.cmlfq_tool_state import (
    DefaultToolStateExtractor,
    ExecutionStatus,
    PayloadSizeClass,
    ToolReturnState,
    ToolStateExtractorRegistry,
)
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import (
    CausalPrefixTree,
    PrefixTreeNode,
    Trajectory,
)
from RL_Framework.infra.scheduling.cmlfq_scheduler import (
    CMLFQMigrationDecision,
    CMLFQRequestState,
    CMLFQScheduler,
)
from RL_Framework.infra.scheduling.cmlfq_shared_state import SharedCMLFQLoadState
from RL_Framework.infra.scheduling.cmlfq_migration import (
    KVCacheManager,
    MigrationCostProfiler,
)
from RL_Framework.infra.scheduling.cmlfq_offline_profile import (
    CMLFQOfflineProfiler,
    CMLFQTreeUpdater,
    TrajectoryExtractor,
)
from RL_Framework.infra.observability.history_collector import StepRecord


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TestToolStateExtractor(unittest.TestCase):
    def test_extract_from_string(self):
        extractor = DefaultToolStateExtractor()
        state = extractor.extract("Hello world")
        self.assertEqual(state.tool_type, "text")
        self.assertEqual(state.execution_status, ExecutionStatus.ToolSuccess)
        self.assertEqual(state.raw_payload_size, 11)

    def test_extract_from_dict(self):
        extractor = DefaultToolStateExtractor()
        state = extractor.extract({
            "tool_type": "python",
            "output": "result: 42" * 100,
        })
        self.assertEqual(state.tool_type, "python")
        self.assertEqual(state.execution_status, ExecutionStatus.ToolSuccess)

    def test_extract_error(self):
        extractor = DefaultToolStateExtractor()
        state = extractor.extract({
            "tool_type": "python",
            "output": "",
            "error": "SyntaxError",
        })
        self.assertEqual(state.execution_status, ExecutionStatus.ToolFailure)

    def test_registry(self):
        registry = ToolStateExtractorRegistry.create_default_registry()
        state = registry.extract("test string")
        self.assertEqual(state.tool_type, "text")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TestCausalPrefixTree(unittest.TestCase):
    def setUp(self):
        self.tree = CausalPrefixTree()
        self.bucket_thresholds = {
            "short": 5000,
            "long": 50000,
        }

    def _make_state(self, tool_type: str, size: str, status: str) -> ToolReturnState:
        return ToolReturnState(
            tool_type=tool_type,
            payload_size_class=PayloadSizeClass(size),
            execution_status=ExecutionStatus(status),
        )

    def test_insert_and_lookup(self):
        """Test insert and lookup."""
        traj = Trajectory(
            prompt_id="p1",
            return_states=[
                self._make_state("search", "small", "success"),
                self._make_state("code", "large", "success"),
            ],
            total_remaining_lengths=[8000, 3000],
        )
        self.tree.insert(traj)


        root = self.tree.lookup("p1", [])
        self.assertIsNotNone(root)
        self.assertEqual(root.depth, 0)


        node1 = self.tree.lookup("p1", [traj.return_states[0]])
        self.assertIsNotNone(node1)
        self.assertEqual(node1.depth, 1)


        node2 = self.tree.lookup("p1", traj.return_states)
        self.assertIsNotNone(node2)
        self.assertEqual(node2.depth, 2)

    def test_lookup_not_found(self):
        """Test lookup not found."""
        node = self.tree.lookup("nonexistent", [])
        self.assertIsNone(node)

    def test_fallback_lookup(self):
        """Test fallback lookup."""
        traj = Trajectory(
            prompt_id="p1",
            return_states=[
                self._make_state("search", "small", "success"),
            ],
            total_remaining_lengths=[8000],
        )
        self.tree.insert(traj)


        node = self.tree.lookup_with_fallback(
            "p1", [self._make_state("unknown", "small", "success")]
        )
        self.assertIsNotNone(node)
        self.assertEqual(node.depth, 0)

    def test_bucket_mapping(self):
        """Test bucket mapping."""
        node = PrefixTreeNode(key="test", depth=0)
        node.mean_remaining_length = 3000
        node.p90_remaining_length = 4000
        node.visit_count = 10

        bucket = self.tree.get_bucket_for_node(node, self.bucket_thresholds)
        self.assertEqual(bucket, "short")


        node.mean_remaining_length = 3000
        node.p90_remaining_length = 8000
        bucket = self.tree.get_bucket_for_node(node, self.bucket_thresholds)
        self.assertIsNone(bucket)

    def test_rebuild(self):
        """Test rebuild."""
        trajectories = [
            Trajectory(
                prompt_id=f"p{i}",
                return_states=[
                    self._make_state("search", "small", "success"),
                ],
                total_remaining_lengths=[5000],
            )
            for i in range(10)
        ]
        self.tree.rebuild(trajectories)
        stats = self.tree.get_stats()
        self.assertEqual(stats["n_prompts"], 10)
        self.assertGreater(stats["total_nodes"], 0)

    def test_save_load(self):
        """Test save load."""
        traj = Trajectory(
            prompt_id="p1",
            return_states=[self._make_state("search", "small", "success")],
            total_remaining_lengths=[5000],
        )
        self.tree.insert(traj)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.tree.save(path)

            new_tree = CausalPrefixTree()
            new_tree.load(path)
            stats = new_tree.get_stats()
            self.assertEqual(stats["n_prompts"], 1)
            node = new_tree.lookup("p1", traj.return_states)
            self.assertEqual(node.remaining_lengths, [5000.0])
        finally:
            os.unlink(path)

    def test_merge_rank_trees(self):
        state = self._make_state("code", "small", "success")
        other = CausalPrefixTree()
        self.tree.insert(
            Trajectory("p1", [state], [1000], total_length=1500)
        )
        other.insert(
            Trajectory("p1", [state], [3000], total_length=3500)
        )

        self.tree.merge(other)

        root = self.tree.lookup("p1", [])
        self.assertEqual(root.visit_count, 2)
        self.assertEqual(root.mean_remaining_length, 2500)
        self.assertEqual(self.tree.get_stats()["total_insertions"], 2)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TestCMLFQScheduler(unittest.TestCase):
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

        self.scheduler.register_instance(0, "inst_short", 1)
        self.scheduler.register_instance(1, "inst_long", 2)

    def test_initial_placement(self):
        """Test initial placement."""
        result = self.scheduler.schedule(
            input_tokens=1000,
            prompt_id="test_prompt",
        )
        self.assertEqual(result.category, "short")
        self.assertEqual(result.tp_degree, 1)
        self.assertTrue(result.request_id)

    def test_mixed_tp_preference_balances_across_ready_instances(self):
        scheduler = CMLFQScheduler(
            buckets={"long": {"tp_degrees": [4, 2], "max_tokens": 50000}},
            initial_bucket="long",
            load_balance_strategy="least_connections",
        )
        scheduler.register_instance(0, "tp4", 4)
        scheduler.register_instance(1, "tp2", 2)
        scheduler.register_instance(2, "tp2_b", 2)

        routes = [scheduler.schedule(input_tokens=100, prompt_id=f"p{i}")
                  for i in range(3)]
        self.assertEqual(routes[0].tp_degree, 4)
        self.assertCountEqual([route.tp_degree for route in routes], [4, 2, 2])

    def test_long_only_preference_reserves_tp4_for_long_requests(self):
        scheduler = CMLFQScheduler(
            buckets={"long": {"tp_degrees": [4], "max_tokens": 50000}},
            initial_bucket="long",
            load_balance_strategy="least_connections",
        )
        scheduler.register_instance(0, "tp4", 4)
        scheduler.register_instance(1, "tp2", 2)

        routes = [scheduler.schedule(input_tokens=100, prompt_id=f"p{i}")
                  for i in range(4)]

        self.assertTrue(all(route.tp_degree == 4 for route in routes))
        self.assertTrue(all(not route.is_fallback for route in routes))

    def test_on_tool_return_migration(self):
        """Test on tool return migration."""

        result = self.scheduler.schedule(input_tokens=1000, prompt_id="p1")
        request_id = list(self.scheduler._request_states.keys())[0]


        state = ToolReturnState(
            tool_type="search",
            payload_size_class=PayloadSizeClass.SizeLarge,
            execution_status=ExecutionStatus.ToolSuccess,
        )
        traj = Trajectory(
            prompt_id="p1",
            return_states=[state],
            total_remaining_lengths=[30000],
        )
        self.scheduler.prefix_tree.insert(traj)


        decision = self.scheduler.on_tool_return(
            request_id=request_id,
            tool_result={"tool_type": "search", "output": "x" * 6000},
            generated_tokens=2000,
        )

        self.assertIsInstance(decision, CMLFQMigrationDecision)
        self.assertTrue(decision.should_migrate)
        self.assertEqual(decision.target_bucket, "long")

    def test_on_tool_return_no_migration(self):
        """Test on tool return no migration."""
        result = self.scheduler.schedule(input_tokens=1000, prompt_id="p1")
        request_id = list(self.scheduler._request_states.keys())[0]


        state = ToolReturnState(
            tool_type="search",
            payload_size_class=PayloadSizeClass.SizeSmall,
            execution_status=ExecutionStatus.ToolSuccess,
        )
        traj = Trajectory(
            prompt_id="p1",
            return_states=[state],
            total_remaining_lengths=[3000],
        )
        self.scheduler.prefix_tree.insert(traj)

        decision = self.scheduler.on_tool_return(
            request_id=request_id,
            tool_result={"tool_type": "search", "output": "small"},
            generated_tokens=2000,
        )


        self.assertFalse(decision.should_migrate)

    def test_migration_floor_skips_short_continuations(self):
        scheduler = CMLFQScheduler(
            buckets={
                "short": {"tp_degrees": [1], "max_tokens": 50},
                "long": {"tp_degrees": [2], "max_tokens": 50000},
            },
            min_migration_remaining_tokens=256,
        )
        scheduler.register_instance(0, "short", 1)
        scheduler.register_instance(1, "long", 2)
        route = scheduler.schedule(input_tokens=100, prompt_id="p1")
        state = ToolReturnState(
            tool_type="search",
            payload_size_class=PayloadSizeClass.SizeSmall,
            execution_status=ExecutionStatus.ToolSuccess,
        )
        scheduler.prefix_tree.insert(
            Trajectory("p1", [state], [100])
        )

        decision = scheduler.on_tool_return(
            route.request_id,
            {"tool_type": "search", "output": "small"},
        )

        self.assertFalse(decision.should_migrate)
        self.assertEqual(
            decision.reason, "remaining_length_below_migration_floor"
        )

    def test_on_request_done_tree_update(self):
        """Test on request done tree update."""
        result = self.scheduler.schedule(input_tokens=1000, prompt_id="p1")
        request_id = list(self.scheduler._request_states.keys())[0]


        self.scheduler.on_tool_return(
            request_id=request_id,
            tool_result={"tool_type": "code", "output": "result"},
            generated_tokens=2000,
        )


        self.scheduler.on_request_done(
            instance_index=result.instance_index,
            prompt_id="p1",
            final_bucket="short",
            output_tokens=5000,
        )


        stats = self.scheduler.prefix_tree.get_stats()
        self.assertGreater(stats["total_nodes"], 0)

    def test_mean_p90_disagree(self):
        """Test mean p90 disagree."""
        result = self.scheduler.schedule(input_tokens=1000, prompt_id="p1")
        request_id = list(self.scheduler._request_states.keys())[0]


        node = PrefixTreeNode(key="test", depth=1)
        node.mean_remaining_length = 3000   # short
        node.p90_remaining_length = 30000   # long
        node.visit_count = 10
        self.scheduler.prefix_tree._roots["p1"] = node

        decision = self.scheduler.on_tool_return(
            request_id=request_id,
            tool_result={"tool_type": "search", "output": "x" * 6000},
        )

        self.assertFalse(decision.should_migrate)
        self.assertEqual(decision.reason, "mean_p90_disagree")

    def test_shared_pressure_triggers_tp_fallback(self):
        with tempfile.TemporaryDirectory() as shared_dir:
            remote = SharedCMLFQLoadState(
                shared_dir,
                writer_id="rank_99",
                heartbeat_interval_s=60,
                ttl_s=120,
            )
            remote.increment("inst_short")
            scheduler = CMLFQScheduler(
                buckets={
                    "short": {"tp_degrees": [1], "max_tokens": 5000},
                    "long": {"tp_degrees": [2], "max_tokens": 50000},
                },
                max_queue_length=1,
                shared_load_dir=shared_dir,
                shared_load_heartbeat_s=60,
                shared_load_ttl_s=120,
            )
            scheduler.register_instance(0, "inst_short", 1)
            scheduler.register_instance(1, "inst_long", 2)
            try:
                result = scheduler.schedule(100, prompt_id="shared-pressure")
                self.assertEqual(result.tp_degree, 2)
                self.assertTrue(result.is_fallback)
            finally:
                scheduler.cancel_request(result.request_id)
                scheduler._shared_load.close()
                remote.close()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TestKVCacheResharding(unittest.TestCase):
    def setUp(self):
        self.manager = KVCacheManager()
        try:
            import torch
        except ImportError:
            self.has_torch = False
        else:
            self.has_torch = hasattr(torch, "randn")

    def test_reshard_tensor_same_tp(self):
        """Test reshard tensor same tp."""
        if not self.has_torch:
            self.skipTest("torch not available")
        import torch

        tensor = torch.randn(4, 8, 16, 64)  # [blocks, block_size, heads, head_dim]
        shards = self.manager._reshard_tensor(tensor, source_tp=2, target_tp=2, dim=2)
        self.assertEqual(len(shards), 2)
        self.assertEqual(shards[0].shape[2], 8)  # 16 / 2 = 8 heads per shard

    def test_reshard_tensor_increase_tp(self):
        """Test reshard tensor increase tp."""
        if not self.has_torch:
            self.skipTest("torch not available")
        import torch

        tensor = torch.randn(4, 8, 16, 64)
        shards = self.manager._reshard_tensor(tensor, source_tp=2, target_tp=4, dim=2)
        self.assertEqual(len(shards), 4)
        self.assertEqual(shards[0].shape[2], 4)  # 16 / 4 = 4 heads per shard

    def test_reshard_tensor_decrease_tp(self):
        """Test reshard tensor decrease tp."""
        if not self.has_torch:
            self.skipTest("torch not available")
        import torch

        tensor = torch.randn(4, 8, 16, 64)
        shards = self.manager._reshard_tensor(tensor, source_tp=4, target_tp=2, dim=2)
        self.assertEqual(len(shards), 2)
        self.assertEqual(shards[0].shape[2], 8)  # 16 / 2 = 8 heads per shard


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TestMigrationCostProfiler(unittest.TestCase):
    def test_decide_with_profile(self):
        profiler = MigrationCostProfiler()
        profiler.record_measurement(
            source_tp=1, target_tp=2, seq_len=1000,
            offload_ms=100, reshard_ms=50, network_ms=200,
            reload_ms=100, recompute_prefill_ms=500,
        )

        decision, latency = profiler.decide_migration_path(1, 2, 1000)
        self.assertEqual(decision, "migrate")
        self.assertEqual(latency, 250)

    def test_decide_recompute_cheaper(self):
        profiler = MigrationCostProfiler()
        profiler.record_measurement(
            source_tp=1, target_tp=2, seq_len=1000,
            offload_ms=500, reshard_ms=200, network_ms=500,
            reload_ms=300, recompute_prefill_ms=200,
        )

        decision, latency = profiler.decide_migration_path(1, 2, 1000)
        self.assertEqual(decision, "recompute")
        self.assertEqual(latency, 200)

    def test_heuristic_short_seq(self):
        profiler = MigrationCostProfiler()
        decision, _ = profiler.decide_migration_path(1, 2, 1000, is_cross_node=False)
        self.assertEqual(decision, "recompute")

    def test_heuristic_long_seq(self):
        profiler = MigrationCostProfiler()
        decision, _ = profiler.decide_migration_path(1, 2, 10000, is_cross_node=False)
        self.assertEqual(decision, "migrate")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TestCMLFQOfflineProfiler(unittest.TestCase):
    def test_profile_from_raw_data(self):
        profiler = CMLFQOfflineProfiler()
        raw_data = [
            {
                "prompt_id": "p1",
                "tool_returns": [
                    {"result": "small output", "remaining_length": 5000},
                    {"result": "large output" * 100, "remaining_length": 2000},
                ],
            },
            {
                "prompt_id": "p2",
                "tool_returns": [
                    {"result": "medium output" * 10, "remaining_length": 8000},
                ],
            },
        ]
        result = profiler.profile_from_raw_trajectories(raw_data)
        self.assertIn("tree_stats", result)
        self.assertGreater(result["tree_stats"]["total_nodes"], 0)

    def test_extract_from_history_tool_return_records(self):
        record = StepRecord(
            step=1,
            sequences=[
                {
                    "prompt_id": "r2e_task_1",
                    "input_len": 100,
                    "output_len": 900,
                    "total_output_tokens": 1200,
                    "tool_returns": [
                        {
                            "tool_type": "code_executor",
                            "output": "3 failed, 1 passed",
                            "status": "failure",
                            "payload_tokens": 120,
                            "token_position": 300,
                            "remaining_length": 900,
                        },
                        {
                            "tool_type": "code_executor",
                            "output": "4 passed",
                            "status": "success",
                            "payload_tokens": 30,
                            "token_position": 1000,
                            "remaining_length": 200,
                        },
                    ],
                }
            ],
        )

        trajectories = TrajectoryExtractor().extract_from_step_records([record])

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].prompt_id, "r2e_task_1")
        self.assertEqual(trajectories[0].total_remaining_lengths, [900.0, 200.0])
        self.assertEqual(trajectories[0].return_states[0].tool_type, "code_executor")
        self.assertEqual(
            trajectories[0].return_states[0].execution_status,
            ExecutionStatus.ToolFailure,
        )


class TestCMLFQTreeUpdater(unittest.TestCase):
    def test_periodic_rebuild(self):
        tree = CausalPrefixTree()
        updater = CMLFQTreeUpdater(tree, rebuild_interval=3)

        state = ToolReturnState(
            tool_type="search",
            payload_size_class=PayloadSizeClass.SizeSmall,
            execution_status=ExecutionStatus.ToolSuccess,
        )

        for step in range(1, 10):
            traj = Trajectory(
                prompt_id=f"p{step}",
                return_states=[state],
                total_remaining_lengths=[5000],
            )
            updater.update_from_step(step, [traj])


        stats = updater.get_stats()
        self.assertEqual(stats["step_count"], 9)
        self.assertEqual(stats["total_insertions"], 9)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

"""Regression tests for distribution routing and real backend request wiring."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from RL_Framework.config import AsyncRLConfig, SchedulingConfig
from RL_Framework.engine.cmlfq_backend import CMLFQGenerationBackend
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.engine.rollout_engine import VLLMRolloutEngine
from RL_Framework.infra.scheduling.cmlfq_cost import CMLFQRoutingCosts
from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import CMLFQCostScheduler
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import CausalPrefixTree, Trajectory
from RL_Framework.infra.scheduling.cmlfq_scheduler import CMLFQMigrationDecision, CMLFQScheduler
from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor
from RL_Framework.infra.scheduling.factory import SchedulerFactory


BUCKETS = {
    "short": {"tp_degrees": [1], "max_tokens": 5000},
    "long": {"tp_degrees": [4], "max_tokens": 50000},
}
EVENT = {"tool_type": "search", "output": "ok", "payload_tokens": 2}
STATE = DefaultToolStateExtractor().extract(EVENT)
TOPOLOGY = [
    {"instance_id": "s", "host": "host-a", "tp_degree": 1},
    {"instance_id": "l", "host": "host-b", "tp_degree": 4},
]


def costs(recompute_ms=1000, transfer_ms=100):
    result = CMLFQRoutingCosts()
    result.decode_ms = {"1": {"short": 2000, "long": 20000},
                        "4": {"short": 4000, "long": 5000}}
    result.migrations = [{
        "source_tp": 1, "target_tp": 4, "source_host": "host-a",
        "target_host": "host-b", "seq_len": 1000,
        "transfer_ms": transfer_ms, "recompute_ms": recompute_ms,
    }]
    return result


def scheduler(lengths=(1000, 10000), **kwargs):
    s = CMLFQCostScheduler(buckets=BUCKETS, routing_costs=costs(), **kwargs)
    s.register_instance(0, "s", 1)
    s.register_instance(1, "l", 4)
    s.configure_runtime(TOPOLOGY)
    for length in lengths:
        s.prefix_tree.insert(Trajectory("p", [STATE], [length], length + 1000))
    return s


class TestDistributionCostRouting(unittest.TestCase):
    def test_distribution_changes_decision_when_legacy_mean_and_p90_disagree(self):
        s = scheduler(lengths=(1000, 1000, 1000, 15000))
        node = s.prefix_tree.lookup("p", [STATE])
        self.assertIsNone(s.prefix_tree.get_bucket_for_node(node, s._bucket_thresholds))
        request = s.schedule(100, "p").request_id
        decision = s.on_tool_return(request, EVENT, 900)
        # Short: .75*2 + .25*20 = 6.5; long: .75*4 + .25*5 + 1 = 5.25.
        self.assertTrue(decision.should_migrate)
        self.assertAlmostEqual(decision.decode_seconds, 4.25)
        self.assertEqual(decision.target_bucket, "long")
        self.assertEqual(s.get_request_route(request).category, "short")

    def test_migration_cost_and_tie_keep_current_instance(self):
        s = scheduler(lengths=(10000,))
        s.routing_costs = costs(recompute_ms=15000)
        request = s.schedule(100, "p").request_id
        self.assertFalse(s.on_tool_return(request, EVENT, 900).should_migrate)
        s.routing_costs = costs(recompute_ms=20000)
        self.assertFalse(s._choose(request).should_migrate)

    def test_profile_units_network_path_and_backend_capability(self):
        s = scheduler(lengths=(10000,))
        request = s.schedule(100, "p").request_id
        self.assertEqual(s.on_tool_return(request, EVENT, 900).execution_path, "recompute")
        s.configure_runtime(TOPOLOGY, lambda src, dst: True)
        decision = s._choose(request)
        self.assertEqual(decision.execution_path, "transfer")
        self.assertAlmostEqual(decision.migration_seconds, .1)
        s.routing_costs.migrations[0]["target_host"] = "different-path"
        self.assertFalse(s._choose(request).should_migrate)

    def test_unavailable_source_uses_recompute_even_when_transfer_is_cheaper(self):
        s = scheduler(lengths=(10000,))
        request = s.schedule(100, "p").request_id
        s.configure_runtime(TOPOLOGY, lambda src, dst: True)
        s.get_instance_handle(0).is_ready = False
        decision = s.on_tool_return(request, EVENT, 900)
        self.assertTrue(decision.should_migrate)
        self.assertEqual(decision.execution_path, "recompute")

    def test_reservation_rechecks_capacity_and_does_not_invent_bucket(self):
        s = scheduler(max_queue_length=1)
        request = s.schedule(100, "p").request_id
        self.assertTrue(s.on_tool_return(request, EVENT, 900).should_migrate)
        s.get_instance_handle(1).active_requests = 1
        decision = s.reserve_generation(request)
        self.assertFalse(decision.should_migrate)
        s.commit_generation(request, decision)
        self.assertEqual(s.get_request_route(request).category, "short")
        self.assertEqual(s._migration_count, 0)

    def test_commit_rollback_and_cancel_balance_load(self):
        s = scheduler()
        request = s.schedule(100, "p").request_id
        s.on_tool_return(request, EVENT, 900)
        s.reserve_generation(request)
        self.assertEqual([h.active_requests for h in s._instances], [1, 1])
        s.rollback_generation(request)
        self.assertEqual([h.active_requests for h in s._instances], [1, 0])
        decision = s.reserve_generation(request)
        s.commit_generation(request, decision)
        self.assertEqual([h.active_requests for h in s._instances], [0, 1])
        self.assertEqual(s.get_request_route(request).category, "long")
        s.cancel_request(request)
        self.assertEqual([h.active_requests for h in s._instances], [0, 0])

    def test_cancel_during_reservation_prevents_commit(self):
        s = scheduler()
        request = s.schedule(100, "p").request_id
        s.on_tool_return(request, EVENT, 900)
        decision = s.reserve_generation(request)
        s.cancel_request(request)
        with self.assertRaises(RuntimeError):
            s.commit_generation(request, decision)
        s.rollback_generation(request)
        self.assertEqual([h.active_requests for h in s._instances], [0, 0])

    def test_distribution_survives_save_load_rebuild_and_fallback(self):
        s = scheduler(lengths=(1000, 10000, 80000))
        expected = s.prefix_tree.residual_distribution("p", [STATE], s._bucket_thresholds)
        self.assertAlmostEqual(sum(p for _, p, _ in expected), 1)
        self.assertIn(("long", 2 / 3, 45000), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tree.json")
            s.prefix_tree.save(path)
            restored = CausalPrefixTree()
            restored.load(path)
            # Unseen deeper state uses the last matched parent's distribution.
            self.assertEqual(restored.residual_distribution(
                "p", [STATE, STATE], s._bucket_thresholds), expected)
            self.assertEqual(restored.residual_distribution(
                "new-prompt", [STATE], s._bucket_thresholds), expected)
            restored.rebuild([Trajectory("p", [STATE], [1000], 2000)])
            self.assertEqual(restored.residual_distribution(
                "p", [STATE], s._bucket_thresholds), [("short", 1, 1000)])

    def test_config_factory_retains_both_policies(self):
        config = AsyncRLConfig(model_path="test-model")
        config.heterogeneous_rollout.scheduling = SchedulingConfig(
            scheduler_type="cmlfq_cost", cmlfq_buckets=BUCKETS,
        )
        new = SchedulerFactory.create("cmlfq_cost", config.heterogeneous_rollout)
        old = SchedulerFactory.create("cmlfq", config.heterogeneous_rollout)
        self.assertIsInstance(new, CMLFQCostScheduler)
        self.assertNotIsInstance(old, CMLFQCostScheduler)
        engine = HeterogeneousRolloutEngine.from_config(config)
        self.assertIsNotNone(engine.scheduler.routing_costs.rollout_model)

    def test_legacy_fallback_reports_actual_bucket_and_no_false_migration(self):
        s = CMLFQScheduler(buckets=BUCKETS)
        s.register_instance(0, "s", 1)
        request = s.schedule(100, "p").request_id
        result = s.execute_migration(request, CMLFQMigrationDecision(True, "test", "short", "long"))
        self.assertEqual(result.category, "short")
        self.assertEqual(s.get_request_route(request).category, "short")
        self.assertEqual(s._migration_count, 0)
        s.finish_request(request, 100)
        self.assertEqual(s.get_instance_handle(0).active_requests, 0)

    def test_initial_fallback_records_actual_bucket(self):
        s = CMLFQScheduler(buckets=BUCKETS)
        s.register_instance(0, "l", 4)
        result = s.schedule(100, "p")
        self.assertEqual(s.get_request_route(result.request_id).category, "long")

    def test_cost_initial_placement_does_not_overbook_or_ignore_fallback_flag(self):
        s = scheduler(max_queue_length=1, enable_fallback=False)
        first = s.schedule(100, "p")
        second = s.schedule(100, "p")
        self.assertGreaterEqual(first.instance_index, 0)
        self.assertEqual(second.instance_index, -1)
        self.assertEqual([h.active_requests for h in s._instances], [1, 0])

    def test_invalid_profile_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.json"
            path.write_text(json.dumps({"decode_ms": {"1": {"short": -1}}}))
            with self.assertRaises(ValueError):
                CMLFQRoutingCosts(str(path))


class TestBackendIntegration(unittest.IsolatedAsyncioTestCase):
    def engine(self, handler, transfer=True):
        s = scheduler(lengths=(10000,))
        engine = HeterogeneousRolloutEngine(scheduler=s)
        engine.instance_configs = TOPOLOGY
        engine._cmlfq_backend = CMLFQGenerationBackend(
            "nixl" if transfer else "recompute", [(1, 4)],
        )
        engine.engines = [VLLMRolloutEngine(host="host-a"), VLLMRolloutEngine(host="host-b")]
        for backend in engine.engines:
            backend.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            backend._http_client_loop = asyncio.get_running_loop()
        engine._configure_cost_runtime()
        return engine

    @staticmethod
    def response(text="answer", kv=None):
        result = {"choices": [{"text": text, "logprobs": {"tokens": [text],
                  "token_logprobs": [-.1]}, "finish_reason": "stop"}]}
        if kv is not None:
            result["kv_transfer_params"] = kv
        return httpx.Response(200, json=result)

    @staticmethod
    def descriptor():
        return {"remote_engine_id": "s", "remote_request_id": "export-1",
                "remote_host": "host-a", "remote_port": 5600,
                "remote_block_ids": [[1, 2]], "tp_size": 1}

    async def test_transfer_http_wire_and_commit_after_target_success(self):
        seen = []
        def handler(request):
            payload = json.loads(request.content)
            seen.append((request.url.host, payload))
            self.assertEqual(engine.scheduler.get_request_route(rid).instance_index, 0)
            if request.url.host == "host-a":
                return self.response("discarded", self.descriptor())
            return self.response()
        engine = self.engine(handler)
        rid = engine.begin_cmlfq_request("p", 100)
        engine.route_cmlfq_tool_return(rid, EVENT, 900)
        try:
            result = await engine.generate("full prompt + tool", request_id=rid, max_new_tokens=17)
            self.assertEqual(result["text"], "answer")
            self.assertEqual(result["_schedule_info"]["execution_path"], "transfer")
            self.assertEqual(seen[0][1]["max_tokens"], 1)
            self.assertEqual(seen[1][1]["max_tokens"], 17)
            self.assertEqual(seen[1][1]["prompt"], "full prompt + tool")
            self.assertTrue(seen[1][1]["kv_transfer_params"]["do_remote_prefill"])
            self.assertEqual(engine.scheduler._migration_count, 1)
            self.assertEqual(engine.scheduler.get_request_route(rid).category, "long")
        finally:
            engine.cancel_cmlfq_request(rid)
            await engine.close()

    async def test_transfer_failure_recomputes_without_kv_params(self):
        seen = []
        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if request.url.host == "host-a":
                return self.response("discarded", self.descriptor())
            if "kv_transfer_params" in payload:
                return httpx.Response(500, text="KV load failed")
            return self.response()
        engine = self.engine(handler)
        rid = engine.begin_cmlfq_request("p", 100)
        engine.route_cmlfq_tool_return(rid, EVENT, 900)
        try:
            result = await engine.generate("full transcript", request_id=rid)
            info = result["_schedule_info"]
            self.assertEqual(info["execution_path"], "recompute")
            self.assertTrue(info["is_fallback"])
            self.assertNotIn("kv_transfer_params", seen[-1])
            self.assertEqual(len(seen), 3)
        finally:
            engine.cancel_cmlfq_request(rid)
            await engine.close()

    async def test_missing_descriptor_recomputes(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return self.response()
        engine = self.engine(handler)
        rid = engine.begin_cmlfq_request("p", 100)
        engine.route_cmlfq_tool_return(rid, EVENT, 900)
        try:
            result = await engine.generate("prompt", request_id=rid)
            self.assertEqual(result["_schedule_info"]["execution_path"], "recompute")
            self.assertNotIn("kv_transfer_params", seen[-1])
        finally:
            engine.cancel_cmlfq_request(rid)
            await engine.close()

    async def test_target_failure_rolls_back_and_can_retry(self):
        failed = True
        def handler(request):
            return httpx.Response(503, text="unavailable") if failed else self.response()
        engine = self.engine(handler, transfer=False)
        rid = engine.begin_cmlfq_request("p", 100)
        engine.route_cmlfq_tool_return(rid, EVENT, 900)
        try:
            with self.assertRaises(RuntimeError):
                await engine.generate("prompt", request_id=rid)
            self.assertEqual(engine.scheduler.get_request_route(rid).category, "short")
            self.assertEqual(engine.scheduler._migration_count, 0)
            self.assertEqual([h.active_requests for h in engine.scheduler._instances], [1, 0])
            failed = False
            result = await engine.generate("prompt", request_id=rid)
            self.assertEqual(result["_schedule_info"]["category"], "long")
        finally:
            engine.cancel_cmlfq_request(rid)
            await engine.close()

    async def test_cancellation_releases_target_reservation(self):
        async def handler(request):
            raise asyncio.CancelledError()
        engine = self.engine(handler, transfer=False)
        rid = engine.begin_cmlfq_request("p", 100)
        engine.route_cmlfq_tool_return(rid, EVENT, 900)
        try:
            with self.assertRaises(asyncio.CancelledError):
                await engine.generate("prompt", request_id=rid)
            self.assertEqual([h.active_requests for h in engine.scheduler._instances], [1, 0])
            self.assertEqual(engine.scheduler._migration_count, 0)
        finally:
            engine.cancel_cmlfq_request(rid)
            await engine.close()

    async def test_duplicate_generation_does_not_release_first_reservation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def handler(request):
            entered.set()
            await release.wait()
            return self.response()
        engine = self.engine(handler, transfer=False)
        rid = engine.begin_cmlfq_request("p", 100)
        engine.route_cmlfq_tool_return(rid, EVENT, 900)
        task = asyncio.create_task(engine.generate("prompt", request_id=rid))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            with self.assertRaisesRegex(RuntimeError, "concurrent generation"):
                await engine.generate("prompt", request_id=rid)
            self.assertEqual([h.active_requests for h in engine.scheduler._instances], [1, 1])
            release.set()
            result = await task
            self.assertEqual(result["_schedule_info"]["category"], "long")
            self.assertEqual([h.active_requests for h in engine.scheduler._instances], [0, 1])
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            engine.cancel_cmlfq_request(rid)
            await engine.close()


if __name__ == "__main__":
    unittest.main()

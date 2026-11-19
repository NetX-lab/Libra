"""Regression tests for distribution routing and real backend request wiring."""

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from RL_Framework.infra.cost_model.model import RolloutCostModel
from RL_Framework.infra.elastic.rollout_lifecycle import apply_availability


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


def rollout_availability(scheduler):
    owner = SimpleNamespace(rollout_engine=SimpleNamespace(scheduler=scheduler))
    return lambda unavailable: apply_availability(owner, unavailable)


class TestDistributionCostRouting(unittest.TestCase):
    def test_completion_rejects_a_request_that_exceeds_instance_kv_capacity(self):
        model = RolloutCostModel(
            num_params=1, d_model=1, n_layers=1, n_kv_heads=1, head_dim=1,
            dtype_bytes=1, flops_peak=1000, mem_bw=1000, mem_capacity=20,
            tp_comm_overhead=0, prefill_mfu=1, act_workspace_bytes=0, kv_frag_rate=0,
        )
        routing = CMLFQRoutingCosts(rollout_model=model)
        # TP1 has room for nine KV tokens; this request needs twenty.
        self.assertTrue(math.isinf(routing.completion_seconds(1, "short", 10, 10)))
        routing.decode_ms = {"1": {"short": 1}}
        self.assertTrue(math.isinf(routing.completion_seconds(1, "short", 10, 10)))
        self.assertTrue(math.isinf(routing.decode_seconds(1, "short", 10, 10)))

    def test_known_load_can_move_a_trace_to_a_slower_idle_service(self):
        s = scheduler(lengths=(10000,))
        s.routing_costs.decode_ms = {"1": {"long": 2000}, "4": {"long": 2500}}
        s.routing_costs.migrations[0]["recompute_ms"] = 100
        request = s.schedule(100, "p").request_id
        other = s.schedule(100, "other").request_id
        previous = s.reserve_generation(other)
        s.commit_generation(other, previous)
        # The other trajectory is waiting on its tool; the server is free.
        self.assertFalse(s.on_tool_return(request, EVENT, 900).should_migrate)
        s.reserve_generation(other)
        # Staying waits behind one equal representative workload: 4s.
        # The idle service needs 2.5s generation plus .1s movement.
        decision = s.reserve_generation(request)
        self.assertTrue(decision.should_migrate)
        self.assertEqual(decision.target_instance_index, 1)

    def test_recompute_prefill_is_charged_once_on_the_target(self):
        s = scheduler(lengths=(10000,))
        s.routing_costs.decode_ms = {"1": {"long": 2000}, "4": {"long": 1000}}
        s.routing_costs.migrations = []
        s.routing_costs.rollout_model = RolloutCostModel(
            num_params=1, d_model=1, n_layers=1, n_kv_heads=1, head_dim=1,
            dtype_bytes=1, flops_peak=1000, mem_bw=1000, mem_capacity=1e9,
            tp_comm_overhead=0, prefill_mfu=1, prefill_chunk_size=100, act_workspace_bytes=0,
        )
        request = s.schedule(100, "p").request_id
        decision = s.on_tool_return(request, EVENT, 0)
        # Both choices include prefill; migration pays no second target prefill.
        self.assertTrue(decision.should_migrate)
        self.assertAlmostEqual(decision.prefill_seconds, 0)
        self.assertAlmostEqual(decision.migration_seconds, 10.6)

        s.routing_costs.migrations = costs(recompute_ms=100).migrations
        measured = s.reserve_generation(request)
        self.assertEqual(measured.execution_path, "recompute")
        self.assertAlmostEqual(measured.prefill_seconds, 0)
        self.assertAlmostEqual(measured.migration_seconds, .1)

    def test_analytic_decode_prices_an_unprofiled_bin(self):
        model = RolloutCostModel(
            num_params=1000, d_model=8, n_layers=2, n_kv_heads=2, head_dim=4,
            dtype_bytes=2, flops_peak=1e9, mem_bw=1e9, mem_capacity=1e9,
            tp_comm_overhead=0, prefill_mfu=1, act_workspace_bytes=0,
        )
        routing = CMLFQRoutingCosts(rollout_model=model)
        seconds = routing.decode_work_seconds(1, "short", 8, 32, other_requests=1)
        self.assertTrue(math.isfinite(seconds))
        self.assertGreater(seconds, 0)

    def test_online_tree_excludes_feedback_from_remaining_generated_tokens(self):
        s = scheduler(lengths=())
        engine = HeterogeneousRolloutEngine(scheduler=s)
        request = engine.begin_cmlfq_request("p", 100)
        first = dict(EVENT, payload_tokens=1000, generated_token_position=20)
        second = dict(EVENT, payload_tokens=2000, generated_token_position=50)
        engine.route_cmlfq_tool_return(request, first, 1020)
        engine.route_cmlfq_tool_return(request, second, 3050)
        engine.finish_cmlfq_request(request, 3100, generated_tokens=100)
        tree = s.prefix_tree
        states = [DefaultToolStateExtractor().extract(e) for e in (first, second)]
        self.assertEqual(tree.lookup("p", states[:1]).remaining_lengths, [80])
        self.assertEqual(tree.lookup("p", states).remaining_lengths, [50])
        self.assertEqual(tree.lookup("p", []).remaining_lengths, [100])

    def test_legacy_tree_lengths_remain_readable_but_are_not_decode_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-tree.json"
            s = scheduler(lengths=(10000,))
            s.prefix_tree.save(str(path))
            raw = json.loads(path.read_text())
            raw["version"] = 2
            raw.pop("remaining_unit", None)
            path.write_text(json.dumps(raw))
            restored = CausalPrefixTree()
            restored.load(str(path))
            self.assertEqual(restored.lookup("p", [STATE]).mean_remaining_length, 10000)
            self.assertEqual(restored.residual_distribution("p", [STATE], s._bucket_thresholds), [])
            restored.insert(Trajectory("p", [STATE], [75], 100))
            self.assertEqual(restored.residual_distribution("p", [STATE], s._bucket_thresholds),
                             [("short", 1, 75)])

    def test_withdrawn_source_without_distribution_recomputes_on_survivor(self):
        s = scheduler(lengths=())
        request = s.schedule(100, "unseen").request_id
        rollout_availability(s)({"s"})
        decision = s.reserve_generation(request)
        self.assertEqual(decision.target_instance_index, 1)
        self.assertEqual(decision.execution_path, "recompute")
        s.commit_generation(request, decision)
        self.assertEqual(s.get_request_route(request).instance_index, 1)

    def test_missing_migration_price_falls_back_to_fitting_ready_service(self):
        for history in ((), (10000,)):
            with self.subTest(history=history):
                s = scheduler(lengths=history)
                s.routing_costs.migrations = []
                s.register_instance(2, "fit", 4)
                s.get_instance_handle(1).kv_capacity_tokens = 115
                s.get_instance_handle(2).kv_capacity_tokens = 120
                request = s.schedule(100, "p").request_id
                s.on_tool_return(request, EVENT, 0, preview=False)
                rollout_availability(s)({"s"})
                with self.assertRaisesRegex(RuntimeError, "fit 121 tokens"):
                    s.reserve_generation(request, 100, 21)
                decision = s.reserve_generation(request, 100, 20)
                self.assertEqual(decision.target_instance_index, 2)
                self.assertEqual(decision.execution_path, "recompute")
                self.assertEqual(decision.reason, "source_withdrawn")

    def test_fast_profile_cannot_route_to_an_instance_that_does_not_fit(self):
        s = scheduler(lengths=(10000,))
        s.get_instance_handle(1).kv_capacity_tokens = 119
        request = s.schedule(100, "p").request_id
        s.on_tool_return(request, EVENT, 0, preview=False)
        self.assertFalse(s.reserve_generation(request, 100, 20).should_migrate)
        s.rollback_generation(request)
        self.assertTrue(s.reserve_generation(request, 100, 19).should_migrate)

    def test_resume_without_history_checks_the_generation_budget(self):
        s = scheduler(lengths=())
        s.get_instance_handle(0).kv_capacity_tokens = 110
        request = s.schedule(100, "p").request_id
        decision = s.reserve_generation(request, 100, 11)
        self.assertEqual(decision.reason, "source_capacity")
        self.assertEqual(decision.target_instance_index, 1)

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
        s.attach_metrics_feed(SimpleNamespace(get=lambda instance_id:
            SimpleNamespace(running=0, waiting=1) if instance_id == "l" else None))
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
    async def test_workflow_cost_is_evaluated_when_the_next_generation_is_reserved(self):
        engine = self.engine(lambda request: self.response(), transfer=False)
        rid = engine.begin_cmlfq_request("p", 100)
        costs = engine.scheduler.routing_costs
        try:
            with patch.object(costs, "decode_work_seconds", wraps=costs.decode_work_seconds) as estimate:
                record = getattr(engine, "record_cmlfq_tool_return", engine.route_cmlfq_tool_return)
                record(rid, EVENT, 900)
                self.assertEqual(estimate.call_count, 0)
                rollout_availability(engine.scheduler)({"s"})
                result = await engine.generate("transcript", request_id=rid)
                self.assertEqual(result["_schedule_info"]["instance_index"], 1)
                self.assertGreater(estimate.call_count, 0)
        finally:
            engine.cancel_cmlfq_request(rid)
            await engine.close()

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

    async def test_continuation_waits_through_empty_loan_switch_until_survivor_reopens(self):
        seen = []
        def handler(request):
            seen.append(request.url.host)
            return self.response()
        engine = self.engine(handler, transfer=False)
        rid = engine.begin_cmlfq_request("p", 100)
        set_unavailable = rollout_availability(engine.scheduler)
        set_unavailable({"s", "l"})
        pending = None
        try:
            engine.route_cmlfq_tool_return(rid, EVENT, 900)
            pending = asyncio.create_task(engine.generate("full transcript", request_id=rid))
            await asyncio.sleep(.01)
            self.assertFalse(pending.done())
            self.assertEqual(seen, [])
            set_unavailable({"s"})
            result = await asyncio.wait_for(pending, 1)
            self.assertEqual(result["_schedule_info"]["instance_index"], 1)
            self.assertEqual(result["_schedule_info"]["execution_path"], "recompute")
            self.assertEqual(seen, ["host-b"])
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            engine.cancel_cmlfq_request(rid)
            await engine.close()

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

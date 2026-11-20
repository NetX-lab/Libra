"""Focused tests for KV-aware C-MLFQ routing policies."""
from __future__ import annotations
import json, math, sys, time, types, unittest
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
try:
    import RL_Framework  # noqa: F401
except ModuleNotFoundError:
    package = types.ModuleType("RL_Framework")
    package.__path__ = [str(ROOT)]
    sys.modules["RL_Framework"] = package
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import Trajectory
from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor
from RL_Framework.infra.scheduling.cmlfq_kv_cost_scheduler import CMLFQDynamicKVCostScheduler
import RL_Framework.infra.scheduling.cmlfq_kv_cost_scheduler as kv_module
BUCKETS = {"short": {"tp_degrees": [1], "max_tokens": 4000}, "long": {"tp_degrees": [2], "max_tokens": 12000}}
EVENT = {"tool_type": "search", "output": "ok", "payload_tokens": 2}
@contextmanager
def controlled_clocks():
    wall, mono = [1000.0], [0.0]
    with patch.object(kv_module.time, "time", side_effect=lambda: wall[0]), patch.object(
        kv_module.time, "monotonic", side_effect=lambda: mono[0]
    ):
        yield wall, mono
STATE = DefaultToolStateExtractor().extract(EVENT)
class FixedCosts:
    def __init__(self, decode_by_tp=None, migration=0.0):
        self.decode_by_tp = decode_by_tp or {1: 2.0, 2: 1.0}
        self.migration = migration
        self.rollout_model = None
    def decode_seconds(self, tp, bin_name, remaining, context):
        return self.decode_by_tp[tp]
    def prefill_seconds(self, tp, context):
        return 0.0
    def decode_work_seconds(self, tp, bin_name, remaining, context, other_requests):
        return self.decode_seconds(tp, bin_name, remaining, context)
    def migration_seconds(self, source, target, context, allow_transfer):
        return "recompute", self.migration
class Feed:
    def __init__(self):
        self.values = {}
        self.reads = 0
    def get(self, instance_id):
        return self.values.get(instance_id)
    def snapshots(self):
        self.reads += 1
        return dict(self.values)
    def set(self, **usage_by_id):
        now = time.time()
        self.values = {key: SimpleNamespace(instance_id=key, updated_at=now, gpu_cache_usage=usage, running=0, waiting=0, kv_capacity_tokens=-1) for key, usage in usage_by_id.items()}
class KVRouterTest(unittest.TestCase):
    def make_scheduler(self, lengths=(1000,), max_queue_length=128):
        s = CMLFQDynamicKVCostScheduler(
            buckets=BUCKETS, routing_costs=FixedCosts(),
            max_queue_length=max_queue_length,
        )
        for idx, ident, tp in ((0, "s0", 1), (1, "s1", 1), (2, "l0", 2)):
            s.register_instance(idx, ident, tp)
        s.configure_runtime([{"instance_id": "s0", "host": "host", "tp_degree": 1}, {"instance_id": "s1", "host": "host", "tp_degree": 1}, {"instance_id": "l0", "host": "host", "tp_degree": 2}])
        feed = Feed()
        feed.set(s0=0.10, s1=0.20, l0=0.30)
        s.attach_metrics_feed(feed)
        for length in lengths:
            s.prefix_tree.insert(Trajectory("p", [STATE], [length], length + 1000))
        return s, feed
    def test_finish_accepts_generated_tokens_separately_from_transcript(self):
        scheduler, _ = self.make_scheduler()
        request = scheduler.schedule(100, prompt_id="p").request_id
        scheduler.on_tool_return(request, {**EVENT, "generated_token_position": 7},
                                 generated_tokens=35, preview=False)
        scheduler.finish_request(request, 50, generated_tokens=20)
        self.assertFalse(scheduler.has_request(request))
        trajectory = scheduler._recent_trajectories[-1]
        self.assertEqual(trajectory.remaining_unit, "model_generated_tokens")
        self.assertEqual(trajectory.total_remaining_lengths, [13])

    def start_request(self, s):
        route = s.schedule(100, "p")
        self.assertGreaterEqual(route.instance_index, 0)
        return route.request_id
    def test_cost_beats_active_count_and_kv_rank_after_health_filter(self):
        s, feed = self.make_scheduler()
        request = self.start_request(s)
        feed.set(s0=0.10, s1=0.80, l0=0.90)
        s.get_instance_handle(1).active_requests = 127
        s.get_instance_handle(2).active_requests = 1
        s.routing_costs = FixedCosts({1: 8.0, 2: 0.1}, migration=0.1)
        decision = s.on_tool_return(request, EVENT, 20)
        self.assertEqual(decision.target_instance_index, 2)
        self.assertEqual(decision.kv_routing["chosenKV"], 0.90)
        self.assertEqual(decision.kv_routing["policy"], "cmlfq_cost_kv_dynamic")
    def test_cost_ties_use_kv_then_stable_index_and_nonpreferred_tp_still_competes(self):
        s, feed = self.make_scheduler()
        s.register_instance(3, "s2", 1)
        s.configure_runtime([
            {"instance_id": "s0", "host": "host", "tp_degree": 1},
            {"instance_id": "s1", "host": "host", "tp_degree": 1},
            {"instance_id": "l0", "host": "host", "tp_degree": 2},
            {"instance_id": "s2", "host": "host", "tp_degree": 1},
        ])
        feed.set(s0=0.10, s1=0.20, l0=0.30, s2=0.25)
        request = self.start_request(s)
        s.routing_costs = FixedCosts({1: 1.0, 2: 1.0}, migration=0.0)
        feed.set(s0=0.99, s1=0.30, s2=0.20, l0=0.99)
        decision = s.on_tool_return(request, EVENT, 20)
        self.assertEqual(decision.target_instance_index, 3)
        feed.set(s0=0.99, s1=0.20, s2=0.20, l0=0.99)
        self.assertEqual(s._choose(request).target_instance_index, 1)

        s, feed = self.make_scheduler(lengths=(4211,))
        request = self.start_request(s)
        s._dynamic_boundary = 4500
        s.routing_costs = FixedCosts({1: 8.0, 2: 0.1}, migration=0.0)
        feed.set(s0=0.99, s1=0.80, l0=0.70)
        decision = s.on_tool_return(request, EVENT, 20)
        self.assertEqual(decision.mean_remaining_length, 4211.0)
        self.assertEqual(decision.kv_routing["preferred_bucket"], "short")
        self.assertIn("l0", decision.kv_routing["candidate_instances"])
        self.assertEqual(decision.target_instance_index, 2)
        self.assertAlmostEqual(decision.kv_routing["candidate_costs"]["l0"]["penalty"], 0.005)
    def test_probe_aware_initial_placement_prefers_shortest_healthy_bucket(self):
        s, feed = self.make_scheduler()
        feed.set(s0=0.80, s1=0.20, l0=0.10)
        self.assertEqual(s.schedule(100, "p").instance_index, 1)
        feed.set(s0=0.97, s1=0.98, l0=0.10)
        route = s.schedule(100, "p")
        self.assertEqual(route.instance_index, 2)
        self.assertIn("healthy", route.reason)
    def test_missing_stale_invalid_and_busy_snapshots_fall_back_explicitly(self):
        s, feed = self.make_scheduler()
        feed.values = {}
        route = s.schedule(100, "p")
        request = route.request_id
        self.assertGreaterEqual(route.instance_index, 0)
        self.assertIn("no_fresh_healthy_instance", s.on_tool_return(request, EVENT, 20).kv_routing["fallback_reason"])
        feed.set(s0=0.2, s1=0.3, l0=0.4)
        for snap in feed.values.values(): snap.updated_at -= 6
        self.assertIn("stale", s._choose(request).kv_routing["fallback_reason"])
        feed.set(s0=-0.1, s1=float("nan"), l0=1.1)
        self.assertIn("no_fresh_healthy_instance", s._choose(request).kv_routing["fallback_reason"])
        feed.set(s0=0.99, s1=0.99, l0=0.99)
        for handle in s._instances: handle.active_requests = 128
        self.assertIn("capacity", s._choose(request).kv_routing["fallback_reason"])
    def test_first_generation_preserves_healthy_initial_route_and_resume_refreshes_metadata(self):
        s, feed = self.make_scheduler()
        s.routing_costs = FixedCosts({1: 9.0, 2: 0.1}, migration=0.1)
        request = self.start_request(s)
        initial = s.get_request_route(request).instance_index
        feed.set(s0=0.10, s1=0.20, l0=0.30)
        first = s.reserve_generation(request)
        self.assertEqual(first.target_instance_index, initial)
        initial_record = first.kv_routing["initial_route"]
        s.rollback_generation(request)
        second = s.reserve_generation(request)
        self.assertEqual(second.kv_routing["initial_route"], initial_record)
        s.commit_generation(request, second)
        feed.set(s0=0.10, s1=0.20, l0=0.90)
        resumed = s.reserve_generation(request)
        self.assertEqual(resumed.kv_routing["snapshots"]["l0"]["kv"], 0.90)
        s.rollback_generation(request)
    def test_generation_budget_applies_to_kv_candidates_and_baseline_fallback(self):
        for use_history in (True, False):
            with self.subTest(use_history=use_history):
                scheduler, _ = self.make_scheduler(lengths=(1000,) if use_history else ())
                scheduler.get_instance_handle(0).kv_capacity_tokens = 119
                scheduler.get_instance_handle(1).kv_capacity_tokens = 119
                scheduler.get_instance_handle(2).kv_capacity_tokens = 120
                request = self.start_request(scheduler)
                if use_history:
                    scheduler.on_tool_return(request, EVENT, 0, preview=False)
                with self.assertRaisesRegex(RuntimeError, "fit 121 tokens"):
                    scheduler.reserve_generation(request, 100, 21)
                decision = scheduler.reserve_generation(request, 100, 20)
                self.assertEqual(decision.target_instance_index, 2)
                expected = "kv_distribution_cost" if use_history else "kv_baseline_fallback:no_distribution"
                self.assertEqual(decision.reason, expected)
                scheduler.cancel_request(request)

    def test_first_generation_rechecks_pressured_source_before_reserving(self):
        scheduler, feed = self.make_scheduler()
        scheduler.routing_costs = FixedCosts({1: 1.0, 2: 5.0}, migration=1.0)
        request = self.start_request(scheduler)
        self.assertEqual(scheduler.get_request_route(request).instance_index, 0)
        feed.set(s0=0.97, s1=0.20, l0=0.30)
        decision = scheduler.reserve_generation(request)
        self.assertEqual(decision.target_instance_index, 1)
        self.assertLess(decision.kv_routing["chosenKV"], 0.95)
        scheduler.rollback_generation(request)
        scheduler.cancel_request(request)
        self.assertEqual(sum(h.active_requests for h in scheduler._instances), 0)
    def test_dynamic_boundary_requires_two_ordered_buckets(self):
        one_bucket = {"short": BUCKETS["short"]}
        with self.assertRaisesRegex(ValueError, "at least two ordered buckets"):
            CMLFQDynamicKVCostScheduler(buckets=one_bucket)

        three_buckets = dict(BUCKETS, middle={"tp_degrees": [3], "max_tokens": 8000})
        scheduler = CMLFQDynamicKVCostScheduler(
            buckets=three_buckets,
        )
        self.assertEqual(scheduler._sorted_bucket_names, ["short", "middle", "long"])
    def test_dynamic_boundary_primes_and_waits_for_new_epoch_and_ten_seconds(self):
        with controlled_clocks() as (wall, mono):
            s, feed = self.make_scheduler()
            feed.set(s0=0.20, s1=0.20, l0=0.20)
            request = self.start_request(s)
            self.assertIn("feedback", s._last_initial_routing)
            self.assertEqual(s._last_initial_routing["feedback"]["status"], "primed")
            self.assertEqual(s._last_initial_routing["feedback"]["service_source"], "cost_service_proxy")
            mono[0] = 9.0
            wall[0] += 1
            feed.set(s0=0.80, s1=0.80, l0=0.10)
            early = s._choose(request)
            self.assertEqual(early.kv_routing["boundary"], 4000)
            self.assertEqual(early.kv_routing["feedback"]["status"], "interval_wait")
            mono[0] = 10.0
            evaluated = s._choose(request)
            self.assertEqual(evaluated.kv_routing["boundary"], 3903)
            self.assertEqual(evaluated.kv_routing["feedback"]["status"], "evaluated")
            same_epoch = s._choose(request)
            self.assertEqual(same_epoch.kv_routing["boundary"], 3903)
            self.assertEqual(same_epoch.kv_routing["feedback"]["status"], "same_epoch")
            mono[0] = 20.0
            wall[0] = 500.0  # Wall-clock movement does not control the interval.
            feed.set(s0=0.0, s1=0.0, l0=0.99)
            jumped = s._choose(request)
            self.assertEqual(jumped.kv_routing["feedback"]["status"], "evaluated")
            self.assertIn("wall_time", s._boundary_ledger[-1])
            self.assertIn("monotonic_time", s._boundary_ledger[-1])
    def test_dynamic_boundary_moves_in_both_directions_below_kv_admission_limit(self):
        for values, want in (({"s0": 0.10, "s1": 0.10, "l0": 0.90}, 4118),
                             ({"s0": 0.90, "s1": 0.90, "l0": 0.10}, 3882)):
            with self.subTest(values=values), controlled_clocks() as (wall, mono):
                s, feed = self.make_scheduler()
                feed.set(s0=0.20, s1=0.20, l0=0.20)
                request = self.start_request(s)
                mono[0] = 10.0
                wall[0] += 1
                feed.set(**values)
                decision = s._choose(request)
                self.assertEqual(decision.kv_routing["boundary"], want)
                self.assertLess(max(values.values()), 0.95)
                self.assertEqual(decision.kv_routing["boundary_transition_count"], 1)
    def test_dynamic_boundary_uses_smoothed_deadband_capped_steps_and_bounds(self):
        with controlled_clocks() as (wall, mono):
            s, feed = self.make_scheduler()
            feed.set(s0=0.20, s1=0.20, l0=0.20)
            request = self.start_request(s)
            mono[0] = 10.0
            wall[0] += 1
            feed.set(s0=0.20, s1=0.20, l0=0.40)
            deadband = s._choose(request)
            self.assertEqual(deadband.kv_routing["boundary"], 4000)
            self.assertIn("feedback", deadband.kv_routing)
            self.assertEqual(deadband.kv_routing["feedback"]["status"], "deadband")
            for tick in range(2, 22):
                mono[0] = tick * 10.0
                wall[0] += 1
                feed.set(s0=0.10, s1=0.10, l0=0.90)
                decision = s._choose(request)
                if tick == 3:
                    self.assertEqual(decision.kv_routing["feedback"]["step"], 250)
            self.assertEqual(s._dynamic_boundary, 8000)
            for tick in range(22, 52):
                mono[0] = tick * 10.0
                wall[0] += 1
                feed.set(s0=0.90, s1=0.90, l0=0.10)
                s._choose(request)
            self.assertEqual(s._dynamic_boundary, 2000)
            event = s._boundary_ledger[-1]
            for key in ("from", "to", "reason", "origin_bucket", "epoch", "raw_difference",
                        "smoothed_difference", "service_source", "wall_time", "monotonic_time"):
                self.assertIn(key, event)
    def test_dynamic_pressure_includes_queue_and_capacity_weighted_bucket_kv(self):
        with controlled_clocks() as (wall, mono):
            s, feed = self.make_scheduler()
            feed.set(s0=0.0, s1=1.0, l0=0.3)
            for name, cap in (("s0", 10), ("s1", 90), ("l0", 100)):
                feed.values[name].kv_capacity_tokens = cap
                feed.values[name].running = 0
                feed.values[name].waiting = 0
            request = self.start_request(s)
            primed = s._choose(request)
            self.assertIn("feedback", primed.kv_routing)
            self.assertAlmostEqual(primed.kv_routing["feedback"]["bucket_pressure"]["short"]["kv"], 0.9)
            self.assertEqual(primed.kv_routing["feedback"]["bucket_pressure"]["short"]["kv_source"], "capacity_weighted")
            mono[0] = 10.0
            wall[0] += 1
            feed.set(s0=0.2, s1=0.2, l0=0.2)
            feed.values["l0"].running = 0
            feed.values["l0"].waiting = 20
            decision = s._choose(request)
            self.assertGreater(decision.kv_routing["feedback"]["bucket_pressure"]["long"]["queue"], 0.5)
            self.assertEqual(decision.kv_routing["feedback"]["bucket_pressure"]["long"]["kv_source"], "equal_replica")
            self.assertGreater(decision.kv_routing["feedback"]["raw_difference"], 0)
    def test_dynamic_service_proxy_falls_back_to_count_proxy_for_bad_estimates(self):
        with controlled_clocks():
            s, feed = self.make_scheduler()
            s.routing_costs = FixedCosts({1: 0.0, 2: float("inf")})
            request = self.start_request(s)
            self.assertIn("feedback", s._last_initial_routing)
            feedback = s._last_initial_routing["feedback"]
            self.assertEqual(feedback["service_source"], "tag_count_proxy")
            self.assertEqual(feedback["bucket_pressure"]["short"]["service_ratio"], 1.0)
            self.assertTrue(all(math.isfinite(value["pressure"]) for value in feedback["bucket_pressure"].values()))
    def test_invalid_queue_epoch_does_not_consume_feedback_gate(self):
        with controlled_clocks() as (wall, mono):
            s, feed = self.make_scheduler()
            feed.set(s0=0.20, s1=0.20, l0=0.20)
            request = self.start_request(s)
            primed = s._choose(request)
            anchor = s._last_boundary_epoch
            mono[0] = 10.0
            wall[0] += 1
            feed.set(s0=0.1, s1=0.1, l0=0.9)
            feed.values["l0"].waiting = -1
            frozen = s._choose(request)
            self.assertIn("feedback", frozen.kv_routing)
            self.assertEqual(frozen.kv_routing["feedback"]["reason"], "invalid_queue")
            self.assertEqual(s._last_boundary_epoch, anchor)
            feed.values["l0"].waiting = 0
            recovered = s._choose(request)
            self.assertEqual(recovered.kv_routing["feedback"]["status"], "evaluated")
            self.assertGreater(recovered.kv_routing["boundary"], primed.kv_routing["boundary"])
    def test_missing_bucket_and_stale_or_future_metrics_freeze_without_gate_consumption(self):
        with controlled_clocks() as (wall, mono):
            s, feed = self.make_scheduler()
            feed.set(s0=0.20, s1=0.20, l0=0.20)
            request = self.start_request(s)
            anchor = s._last_boundary_epoch
            s.get_instance_handle(2).is_ready = False
            missing = s._choose(request)
            self.assertIn("feedback", missing.kv_routing)
            self.assertEqual(missing.kv_routing["feedback"]["reason"], "missing_bucket")
            self.assertEqual(s._last_boundary_epoch, anchor)
            s.get_instance_handle(2).is_ready = True
            mono[0] = 10.0
            wall[0] += 1
            feed.set(s0=0.2, s1=0.2, l0=0.9)
            feed.values["l0"].updated_at = 994.0
            stale = s._choose(request)
            self.assertEqual(stale.kv_routing["feedback"]["reason"], "stale_snapshot")
            feed.values["l0"].updated_at = wall[0] + 1
            future = s._choose(request)
            self.assertEqual(future.kv_routing["feedback"]["reason"], "future_snapshot")
            self.assertEqual(s._last_boundary_epoch, anchor)
    def test_dynamic_boundary_crossing_changes_route_but_strong_cost_still_wins(self):
        with controlled_clocks() as (wall, mono):
            s, feed = self.make_scheduler(lengths=(4005,))
            feed.set(s0=0.15, s1=0.15, l0=0.15)
            request = self.start_request(s)
            s.routing_costs = FixedCosts({1: 1.02, 2: 1.0}, migration=0.0)
            before = s.on_tool_return(request, EVENT, 20)
            self.assertEqual(before.mean_remaining_length, 4005.0)
            self.assertEqual(before.kv_routing["boundary"], 4000)
            self.assertEqual(before.kv_routing["preferred_bucket"], "long")
            self.assertEqual(before.target_instance_index, 2)
            self.assertEqual(before.kv_routing["candidate_instances"], ["s0", "s1", "l0"])
            mono[0] = 10.0
            wall[0] += 1
            feed.set(s0=0.10, s1=0.10, l0=0.90)
            after = s._choose(request)
            self.assertGreater(after.kv_routing["boundary"], 4005)
            self.assertEqual(after.kv_routing["preferred_bucket"], "short")
            self.assertEqual(after.target_instance_index, 0)
            self.assertEqual(after.kv_routing["candidate_instances"], ["s0", "s1", "l0"])
            self.assertAlmostEqual(after.kv_routing["candidate_costs"]["l0"]["score"], 1.05)
            s.routing_costs = FixedCosts({1: 10.0, 2: 0.4}, migration=0.0)
            cheap = s._choose(request)
            self.assertEqual(cheap.target_instance_index, 2)
            self.assertAlmostEqual(cheap.decode_seconds, 0.4)
            self.assertAlmostEqual(cheap.kv_routing["candidate_costs"]["l0"]["raw_total"], 0.4)
    def test_dynamic_nonfinite_costs_are_trace_safe_and_diagnostic(self):
        with controlled_clocks():
            s, feed = self.make_scheduler()
            feed.set(s0=0.20, s1=0.20, l0=0.20)
            request = self.start_request(s)
            s.routing_costs = FixedCosts({1: float("nan"), 2: float("inf")})
            decision = s._choose(request)
            json.dumps(decision.kv_routing, allow_nan=False)
            self.assertEqual(decision.kv_routing["candidate_instances"], ["s0", "s1", "l0"])
            self.assertIn("candidate_costs", decision.kv_routing)
            for record in decision.kv_routing["candidate_costs"].values():
                self.assertIsNone(record["raw_total"])
                self.assertIsNone(record["score"])
                self.assertEqual(record["unusable_reason"], "non_finite_cost")
    def test_4211_middle_tp_can_win_and_penalty_uses_common_floor(self):
        buckets = {
            "short": {"tp_degrees": [1], "max_tokens": 4000},
            "middle": {"tp_degrees": [2], "max_tokens": 12000},
            "long": {"tp_degrees": [4], "max_tokens": 30000},
        }
        s = CMLFQDynamicKVCostScheduler(buckets=buckets, routing_costs=FixedCosts({1: 8, 2: 0.1, 4: 9}))
        topology = []
        for index, (name, tp) in enumerate((("s0", 1), ("s1", 1), ("m0", 2), ("l0", 4))):
            s.register_instance(index, name, tp)
            topology.append({"host": "host", "tp_degree": tp})
        s.configure_runtime(topology)
        feed = Feed()
        feed.set(s0=0.2, s1=0.2, m0=0.2, l0=0.2)
        s.attach_metrics_feed(feed)
        s.prefix_tree.insert(Trajectory("p", [STATE], [5000]))
        request = self.start_request(s)
        preview = s.on_tool_return(request, EVENT, 20)
        self.assertEqual(preview.kv_routing["preferred_bucket"], "long")
        self.assertEqual(preview.target_instance_index, 2)
        self.assertAlmostEqual(preview.kv_routing["candidate_costs"]["m0"]["penalty"], 0.005)
        s.routing_costs = FixedCosts({1: 2, 2: 3, 4: 0.0001})
        reserved = s.reserve_generation(request)
        self.assertEqual(reserved.target_instance_index, 3)
        self.assertEqual(s.get_request_route(request).instance_index, 0)
        self.assertAlmostEqual(reserved.kv_routing["candidate_costs"]["m0"]["penalty"], 0.00005)
        s.rollback_generation(request)
        s.cancel_request(request)
        self.assertEqual(sum(h.active_requests for h in s._instances), 0)

    def test_initial_lowest_kv_ties_rotate(self):
        s, feed = self.make_scheduler()
        feed.set(s0=0.2, s1=0.2, l0=0.1)
        routes = [s.schedule(100, "p") for _ in range(3)]
        self.assertEqual([r.instance_index for r in routes], [0, 1, 0])
        for route in routes:
            s.cancel_request(route.request_id)
    def test_generation_commit_rollback_cancel_remain_transactional(self):
        s, feed = self.make_scheduler()
        request = self.start_request(s)
        feed.set(s0=0.99, s1=0.20, l0=0.10)
        s.on_tool_return(request, EVENT, 20)
        decision = s.reserve_generation(request)
        self.assertEqual(sum(h.active_requests for h in s._instances), 2)
        s.rollback_generation(request)
        self.assertEqual(sum(h.active_requests for h in s._instances), 1)
        decision = s.reserve_generation(request)
        s.commit_generation(request, decision)
        self.assertEqual(s.get_request_route(request).instance_index, decision.target_instance_index)
        s.cancel_request(request)
        self.assertFalse(s.has_request(request))
        self.assertEqual(sum(h.active_requests for h in s._instances), 0)
        self.assertFalse(s._pending_cost_routes)
        self.assertFalse(s._reservations)
if __name__ == "__main__":
    unittest.main()

"""Behavioral regressions found by auditing the actual rollout call chain."""

import asyncio
import json
import multiprocessing
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.infra.scheduling.base import MetricsFeedbackConfig
from RL_Framework.infra.scheduling.cmlfq_scheduler import CMLFQScheduler
from RL_Framework.infra.scheduling.la_mlfq import LAMLFQScheduler, WaitingRequest
from RL_Framework.infra.scheduling.length_aware import LengthAwareScheduler
from RL_Framework.infra.scheduling.load_balance import LoadBalanceScheduler
from RL_Framework.infra.scheduling.metrics_feed import InstanceMetrics, VLLMMetricsPoller
from RL_Framework.infra.scheduling.shared_token_state import SharedTokenLoadState


@pytest.mark.parametrize("scheduler_cls", [LoadBalanceScheduler, LengthAwareScheduler, LAMLFQScheduler])
def test_short_completion_does_not_retire_long_request(scheduler_cls):
    async def run():
        scheduler = scheduler_cls(load_metric="tokens")
        engine = HeterogeneousRolloutEngine("m", scheduler)
        engine.add_instance("a", "localhost", 1, 1)
        entered, release = asyncio.Event(), asyncio.Event()

        class Endpoint:
            async def generate(self, prompt, **kwargs):
                if prompt == "long":
                    entered.set()
                    await release.wait()
                return {"text": "x", "tokens": [1], "logprobs": []}

        engine.engines = [Endpoint()]
        long = asyncio.create_task(engine.generate("long", input_tokens=9000, max_new_tokens=64))
        await entered.wait()
        initial = scheduler.get_instance_handle(0).active_tokens
        await engine.generate("short", input_tokens=20, max_new_tokens=8)
        assert scheduler.get_instance_handle(0).active_tokens == initial
        with pytest.raises(TimeoutError):
            engine.wait_until_idle(timeout=0)
        release.set()
        await long
        assert scheduler.get_instance_handle(0).active_tokens == 0
    asyncio.run(run())


def test_cmlfq_direct_generation_accepts_budget_and_retires():
    async def run():
        scheduler = CMLFQScheduler()
        engine = HeterogeneousRolloutEngine("m", scheduler)
        engine.add_instance("a", "localhost", 1, 1)
        class Endpoint:
            async def generate(self, **kwargs):
                return {"text": "x", "tokens": [1]}
        engine.engines = [Endpoint()]
        await engine.generate("p", input_tokens=10, max_new_tokens=5)
        assert scheduler.get_instance_handle(0).active_requests == 0
    asyncio.run(run())


@pytest.mark.parametrize("scheduler_cls", [LengthAwareScheduler, LAMLFQScheduler])
def test_preferred_bucket_cannot_resurrect_blocked_or_unknown_capacity(scheduler_cls):
    scheduler = scheduler_cls(load_metric="kv_tokens", kv_capacity_tokens_by_tp={2: 10000})
    scheduler.register_instance(0, "a", 1)
    scheduler.register_instance(1, "b", 2)
    assert scheduler.schedule(100).instance_index == 1
    scheduler._kv_capacity_by_tp[1] = 10000
    scheduler._feedback = MetricsFeedbackConfig(enabled=True)
    class Feed:
        def get(self, iid):
            return InstanceMetrics(iid, time.time(), gpu_cache_usage=0.99 if iid == "a" else 0.1)
    scheduler.attach_metrics_feed(Feed())
    assert scheduler.schedule(100).instance_index == 1


def test_capacity_is_per_instance_not_largest_tp_peer():
    scheduler = LoadBalanceScheduler(load_metric="kv_tokens", feedback=MetricsFeedbackConfig(enabled=True))
    for index in range(2):
        scheduler.register_instance(index, str(index), 1)
        scheduler.get_instance_handle(index).inc_active(100, 0)
    class Feed:
        def get(self, iid):
            return InstanceMetrics(iid, time.time(), kv_capacity_tokens=1000 if iid == "0" else 10000)
    scheduler.attach_metrics_feed(Feed())
    assert scheduler.load_of(scheduler.get_instance_handle(0)) == 0.1
    assert scheduler.load_of(scheduler.get_instance_handle(1)) == 0.01


def test_cancelled_scout_follower_is_not_scheduled():
    async def run():
        scheduler = LAMLFQScheduler(load_metric="tokens")
        scheduler.register_instance(0, "a", 1)
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        waiter = WaitingRequest("p", 10, 2, 0, 1, future=future, max_new_tokens=8)
        scheduler._process_released_waiting([waiter], "short")
        assert scheduler.get_instance_handle(0).active_requests == 0
        future2 = asyncio.get_running_loop().create_future()
        waiter.future = future2
        scheduler._process_released_waiting([waiter], "short")
        assert future2.result().reserved_tokens == 18
    asyncio.run(run())


def _rank_writer(directory, commands, replies):
    state = SharedTokenLoadState(directory, writer_id="rank_1", cache_ttl_s=0)
    try:
        for command in iter(commands.get, "stop"):
            state.add("a", *command)
            replies.put("published")
    finally:
        state.close()


def test_real_process_updates_are_visible_with_cache_disabled(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    commands, replies = ctx.Queue(), ctx.Queue()
    writer = ctx.Process(target=_rank_writer, args=(str(tmp_path), commands, replies))
    reader = SharedTokenLoadState(str(tmp_path), writer_id="rank_0", cache_ttl_s=0)
    writer.start()
    try:
        assert reader.totals() == {}  # prime empty cache
        commands.put((1, 9000))
        assert replies.get(timeout=20) == "published"
        assert reader.totals()["a"] == {"requests": 1, "tokens": 9000}
        commands.put((-1, -9000))
        replies.get(timeout=20)
        assert reader.totals()["a"]["tokens"] == 0
    finally:
        commands.put("stop")
        writer.join(timeout=20)
        if writer.is_alive():
            writer.terminate()
            writer.join()
        reader.close()
    assert writer.exitcode == 0


def test_closed_writer_cannot_resurrect_or_delete_replacement(tmp_path):
    old = SharedTokenLoadState(str(tmp_path), writer_id="rank_0")
    new = SharedTokenLoadState(str(tmp_path), writer_id="rank_0")
    try:
        old.close()
        old.add("a", 1, 200)
        old.reset()
        new.add("a", 1, 100)
        assert new.totals()["a"]["tokens"] == 100
        assert json.loads((tmp_path / "rank_0.json").read_text())["owner"] == new._owner
    finally:
        old.close()
        new.close()


@contextmanager
def metrics_server():
    state = {"status": 200, "counter": 4, "usage": 0.95}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(state["status"])
            self.end_headers()
            self.wfile.write((
                f'vllm:kv_cache_usage_perc{{model_name="name with spaces"}} {state["usage"]}\n'
                f'vllm:num_preemptions_total {state["counter"]}\n'
                'vllm:num_requests_running NaN\n'
                'vllm:num_requests_waiting 3\n'
            ).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}/metrics"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_feed_counter_reset_nan_and_stale_admission():
    with metrics_server() as (state, url):
        poller = VLLMMetricsPoller()
        poller._poll_one("a", url)
        assert poller.get("a").running == -1
        assert poller.get("a").gpu_cache_usage == 0.95
        state["counter"] += 1
        poller._poll_one("a", url)
        assert poller.get("a").last_preemption_at > 0
        state["counter"] = 0  # restarted server
        poller._poll_one("a", url)
        assert poller.get("a").last_preemption_at == 0
        scheduler = LoadBalanceScheduler(feedback=MetricsFeedbackConfig(enabled=True))
        scheduler.register_instance(0, "a", 1)
        scheduler.attach_metrics_feed(poller)
        assert not scheduler._admission_ok(scheduler.get_instance_handle(0))
        state["status"] = 500
        poller._poll_one("a", url)
        poller._snapshots["a"].updated_at -= 100
        assert scheduler._admission_ok(scheduler.get_instance_handle(0))
        assert scheduler._admission_blocked == {}

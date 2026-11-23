#!/usr/bin/env python3
"""Real vLLM/CUDA CPU-offload benchmark; tool latency is controlled synthetic I/O.

Run next to two Libra CPU-offload API servers with shared CPU storage. Records
raw timings and checks greedy continuation against a full-prefill baseline.
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx


async def engine_smoke(args):
    """Exercise the scheduler/backend/workflow hooks on the live endpoints.

    Synthetic costs force a move to test plumbing; they are not measurements.
    Performance measurements are collected separately by main().
    """
    from urllib.parse import urlsplit
    from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
    from RL_Framework.engine.cmlfq_backend import CMLFQGenerationBackend
    from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import CMLFQCostScheduler
    from RL_Framework.infra.scheduling.cmlfq_cost import CMLFQRoutingCosts
    from RL_Framework.infra.scheduling.cmlfq_prefix_tree import Trajectory
    from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor

    costs = CMLFQRoutingCosts()
    costs.decode_ms = {str(args.source_tp): {"long": 10000}, str(args.target_tp): {"long": 1000}}
    src, dst = urlsplit(args.source), urlsplit(args.target)
    costs.migrations = [{"source_tp": args.source_tp, "target_tp": args.target_tp,
                         "source_host": src.hostname, "target_host": dst.hostname,
                         "seq_len": 1000, "transfer_ms": 1, "recompute_ms": 100}]
    scheduler = CMLFQCostScheduler(buckets={
        "short": {"tp_degrees": [args.source_tp], "max_tokens": 100},
        "long": {"tp_degrees": [args.target_tp], "max_tokens": 10000},
    }, routing_costs=costs)
    engine = HeterogeneousRolloutEngine(model_path=args.model, scheduler=scheduler)
    engine._cmlfq_backend = CMLFQGenerationBackend("cpu_offload", [(args.source_tp, args.target_tp)])
    engine.add_instance("source", src.hostname, src.port, args.source_tp)
    engine.add_instance("target", dst.hostname, dst.port, args.target_tp)
    event = {"tool_type": "test", "output": "ok"}
    state = DefaultToolStateExtractor().extract(event)
    scheduler.prefix_tree.insert(Trajectory("live", [state], [1000], 2000))
    prompt = "The number is seven. " * 64
    request_id = engine.begin_cmlfq_request("live", 400)
    try:
        first = await engine.generate(prompt, request_id=request_id, temperature=0, max_new_tokens=8)
        await asyncio.sleep(.5)
        decision = engine.route_cmlfq_tool_return(request_id, event, 8)
        assert decision.should_migrate
        second = await engine.generate(prompt + first["text"] + "\nTool: ok. Continue:",
                                       request_id=request_id, temperature=0, max_new_tokens=8)
        assert second["_schedule_info"]["instance_index"] == 1
        assert second["_cpu_offload_info"]["imported_tokens"] > 0
        assert scheduler._migration_count == 1
        return {"schedule": second["_schedule_info"], "offload": second["_cpu_offload_info"]}
    finally:
        engine.finish_cmlfq_request(request_id, 16)
        await engine.close()


async def main(args):
    async with httpx.AsyncClient(timeout=300) as client:
        async def generate(url, prompt, params=None, max_tokens=16):
            payload = {"model": args.model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 123, "logprobs": 1}
            if params:
                payload["kv_transfer_params"] = params
            response = await client.post(url + "/v1/completions", json=payload)
            response.raise_for_status()
            return response.json()

        async def ready(key, tp):
            start = time.perf_counter()
            while True:
                response = await client.get(args.source + f"/libra/kv/{key}", params={"source_tp": tp})
                response.raise_for_status()
                status = response.json()
                if status.get("ready"):
                    return status, (time.perf_counter() - start) * 1000
                if time.perf_counter() - start > 60:
                    raise TimeoutError("CPU cache publication timed out")
                await asyncio.sleep(.002)

        for url in (args.source, args.target):
            for attempt in range(180):
                try:
                    response = await client.get(url + "/health")
                    if response.status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
            else:
                raise TimeoutError(f"Server not ready: {url}")

        rows = []
        for words in args.words:
            prompt = "Repeat the final number only.\n" + "The value is seven. " * words + "\nFinal number:"
            # Warm kernels/model before measured trials.
            await generate(args.source, "Hello", max_tokens=1)
            await generate(args.target, "Hello", max_tokens=1)
            for delay in args.tool_ms:
                for trial in range(args.trials):
                    # Alternate mode order to avoid always warming one mode first.
                    modes = ("serial", "overlap") if trial % 2 == 0 else ("overlap", "serial")
                    for mode in modes:
                        key, next_key = uuid.uuid4().hex, uuid.uuid4().hex
                        first = await generate(args.source, prompt, {"libra_offload_key": key})
                        descriptor = first["kv_transfer_params"]
                        returned_at = time.time()
                        after_generation = time.perf_counter()
                        if mode == "serial":
                            status, wait_ms = await ready(key, descriptor["source_tp"])
                        await asyncio.sleep(delay / 1000)  # synthetic asynchronous tool
                        if mode == "overlap":
                            status, wait_ms = await ready(key, descriptor["source_tp"])
                        continuation = prompt + first["choices"][0]["text"] + "\nTool result: correct. Continue:"
                        decode_start = time.perf_counter()
                        loaded = await generate(args.target, continuation, {
                            "libra_load_key": key, "libra_offload_key": next_key,
                        })
                        decode_ms = (time.perf_counter() - decode_start) * 1000
                        elapsed = (time.perf_counter() - after_generation) * 1000
                        # Drain the target's next-turn export before timing an
                        # independent recomputation baseline, otherwise its
                        # post-completion DMA would inflate baseline latency.
                        target_descriptor = loaded.get("kv_transfer_params", {})
                        if target_descriptor.get("libra_offload_key"):
                            await ready(next_key, target_descriptor["source_tp"])
                        baseline_start = time.perf_counter()
                        baseline = await generate(args.target, continuation)
                        baseline_ms = (time.perf_counter() - baseline_start) * 1000
                        match = loaded["choices"][0]["text"] == baseline["choices"][0]["text"]
                        loaded_tokens = loaded.get("kv_transfer_params", {}).get("libra_imported_tokens", 0)
                        row = {
                            "words": words, "tool_ms": delay, "trial": trial, "mode": mode,
                            "cached_tokens": descriptor["cached_tokens"], "loaded_tokens": loaded_tokens,
                            "wait_ms": wait_ms, "resume_ms": decode_ms, "recompute_ms": baseline_ms,
                            "after_generation_ms": elapsed, "output_matches_recompute": match,
                            "completion_returned_at": returned_at, "workers": status["workers"],
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        Path(args.output).write_text(json.dumps({"rows": rows}, indent=2))
                        for url, cache_key in ((args.source, key), (args.target, next_key)):
                            response = await client.delete(url + f"/libra/kv/{cache_key}")
                            response.raise_for_status()
                        if not match or not loaded_tokens:
                            raise AssertionError("KV continuation failed correctness/cache-hit check")
        summaries = []
        for words in args.words:
            for delay in args.tool_ms:
                group = [row for row in rows if row["words"] == words and row["tool_ms"] == delay]
                summary = {"words": words, "tool_ms": delay}
                for mode in ("serial", "overlap"):
                    mode_rows = [row for row in group if row["mode"] == mode]
                    for metric in ("wait_ms", "resume_ms", "recompute_ms", "after_generation_ms"):
                        summary[mode + "_" + metric] = statistics.median(row[metric] for row in mode_rows)
                summaries.append(summary)
        smoke = await engine_smoke(args) if args.source_tp != args.target_tp else None
        Path(args.output).write_text(json.dumps({"rows": rows, "summary": summaries,
                                                "engine_smoke": smoke}, indent=2))
        print(json.dumps({"summary": summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:18910")
    parser.add_argument("--target", default="http://127.0.0.1:18911")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--words", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--tool-ms", type=int, nargs="+", default=[0, 100, 500])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))

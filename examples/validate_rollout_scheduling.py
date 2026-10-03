"""Live HTTP validation; no fake lengths, injected loads or manual draining.

Deployment (model, instance IDs, TP sizes and endpoints) is supplied at runtime.
The output records both route decisions and real vLLM cache-counter deltas.
This is a correctness smoke, not a throughput benchmark or training run.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

from RL_Framework.config import AsyncRLConfig
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.infra.scheduling.metrics_feed import parse_prometheus_metrics


async def validate(args):
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw["model_path"] = args.model
    instances = []
    for endpoint in args.endpoint:
        iid, tp, address = endpoint.split(",", 2)
        url = urlparse(address)
        if url.scheme != "http" or not url.hostname or not url.port:
            raise ValueError("endpoint must be instance-id,tp,http://host:port")
        instances.append(dict(instance_id=iid, tp=int(tp), host=url.hostname, port=url.port))
    if len(instances) < 2:
        raise ValueError("At least two independent vLLM instances are required")
    raw["heterogeneous_rollout"]["instances"] = instances
    if args.shared_load_dir:
        raw["heterogeneous_rollout"]["scheduling"]["shared_load_dir"] = args.shared_load_dir
    config = AsyncRLConfig.from_dict(copy.deepcopy(raw))
    engine = HeterogeneousRolloutEngine.from_config(config)
    scheduler = engine.scheduler
    report = {"deployment": instances, "model": args.model, "phases": []}
    tasks = []
    async with httpx.AsyncClient(timeout=30) as client:
        async def counters():
            values = {}
            for cfg in engine.instance_configs:
                url = f"http://{cfg['host']}:{cfg['port']}/metrics"
                response = await client.get(url)
                response.raise_for_status()
                all_values = parse_prometheus_metrics(response.text)
                values[cfg["instance_id"]] = {
                    k: v for k, v in all_values.items()
                    if k in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
                }
            return values

        lengths = {}
        async def generate(prompt, pid, budget=32):
            if prompt not in lengths:
                response = await client.post(engine.engines[0].base_url + "/tokenize", json={
                    "model": args.model, "prompt": prompt, "add_special_tokens": False,
                })
                response.raise_for_status()
                payload = response.json()
                lengths[prompt] = payload.get("count", len(payload.get("tokens", [])))
                if lengths[prompt] <= 0:
                    raise AssertionError("/tokenize did not return a positive token count")
            result = await engine.generate(
                prompt, input_tokens=lengths[prompt], prompt_id=pid,
                max_new_tokens=budget, temperature=0.0,
            )
            if not result.get("tokens"):
                raise AssertionError("Generation returned no token logprobs")
            return {"input_tokens": lengths[prompt], "output_tokens": len(result["tokens"]),
                    "route": result["_schedule_info"]}

        try:
            await asyncio.to_thread(engine.wait_for_ready, 180)
            snapshots = engine.metrics_snapshots()
            report["snapshots"] = {key: vars(value) for key, value in snapshots.items()}
            for handle in scheduler._instances:
                snap = engine._metrics_poller.get(handle.instance_id)
                if snap is None or snap.kv_capacity_tokens <= 0:
                    raise AssertionError(f"No fresh measured capacity for {handle.instance_id}")
                if scheduler.kv_capacity_of(handle) != snap.kv_capacity_tokens:
                    raise AssertionError("Measured capacity is not used by scheduler")

            prompt = ("Systems share compute and memory. " * 96) + "Summarize this in one sentence."
            for enabled in (False, True):
                scheduler._prefix_affinity_enabled = enabled
                scheduler._affinity.clear()
                before = await counters()
                rows = [await generate(prompt, "repeated-prompt") for _ in range(4)]
                after = await counters()
                if enabled:
                    targets = {row["route"]["instance_id"] for row in rows}
                    if len(targets) != 1 or any(
                        row["route"]["reason"] != "prefix_affinity" for row in rows[1:]
                    ):
                        raise AssertionError("Repeated prompt did not follow affinity")
                report["phases"].append({"affinity": enabled, "requests": rows,
                                         "cache_before": before, "cache_after": after})

            tasks = [asyncio.create_task(generate(
                ("Read this. " * (30 + 100 * i)) + "Summarize briefly.", f"mixed-{i}", 64
            )) for i in range(4)]
            report["mixed_requests"] = await asyncio.gather(*tasks)
            for handle in scheduler._instances:
                if handle.active_requests != 0 or handle.active_tokens != 0:
                    raise AssertionError(f"Leaked accounting: {handle}")
            old_state = engine._shared_state
            engine.reconfigure_from_plan(None, config)
            if old_state is not None and engine._shared_state is not old_state:
                raise AssertionError("Shared owner was not reused after a drained reconfigure")
            report["after_reconfigure"] = await generate("Say hello.", "post-reconfigure", 8)
            report["passed"] = True
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await engine.close()
            if args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scheduling_adaptive.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--shared-load-dir", default="")
    parser.add_argument("--output")
    asyncio.run(validate(parser.parse_args()))

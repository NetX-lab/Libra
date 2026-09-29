# Distribution and cost based C-MLFQ

`scheduler_type: cmlfq_cost` is an opt-in alternative. `cmlfq` retains its
mean/P90 policy, and the existing tensor migration helpers remain available.
Both policies now record the bucket of the selected instance after fallback;
returning to the same instance no longer increments the migration count.

## Enable

Use `configs/search_r1_cmlfq_cost_qwen3_14b.yaml`, or change the scheduling section
of a configuration:

```yaml
heterogeneous_rollout:
  scheduling:
    scheduler_type: cmlfq_cost
    cmlfq_buckets:
      short: {tp_degrees: [1], max_tokens: 5000}
      long: {tp_degrees: [4], max_tokens: 50000}
    cmlfq_cost_profile_path: ""
    cmlfq_kv_backend: recompute
    cmlfq_kv_transfer_tp_pairs: []
```

New-mode buckets require one distinct TP degree each. All registered instances
must belong to a bucket; multiple replicas of one TP are supported. One managed
request represents one trajectory (`n = n_samples = 1`). Existing workflows
already use this convention and call the same begin/tool-return/finish hooks.

## Routing and statistics

The prefix tree exposes a histogram aligned to the configured bucket thresholds.
Each populated bin provides its probability and conditional mean residual length.
Overflow belongs to the last bin, retaining its observed length rather than
clamping it to the last threshold. Histograms use the persisted latest 200
observations per node; this bounded window is separate from the legacy all-time
mean. Insertion, persistence, merge, parent fallback, global fallback for unseen
prompts, and periodic rebuild use the existing tree infrastructure. Older
checkpoints without raw observations cannot supply a distribution: requests
stay in place until traces have populated it. Bootstrap a new tree from the
offline profiler for useful routing from the first training step.

For every ready instance with capacity, evaluate:

`sum(bin_probability * decode_seconds(TP, bin)) + movement_seconds(source, target)`

Staying on the current instance has zero movement cost. Exact ties prefer staying,
then lower load and instance index. Unavailable/full destinations are excluded,
and candidates are re-evaluated just before execution. New-mode initial placement
also respects queue limits and `enable_fallback`. This is not a distributed atomic
reservation across ranks: the existing shared-load snapshots remain advisory.

With no measured decode table, `HeterogeneousRolloutEngine.from_config` binds the
configured planner's **analytic** rollout model (hardware, model architecture and
profiling parameters). It integrates decode step time over the representative
residual length and estimates chunked prefill for recomputation. It does not
automatically reuse an external Vidur/Sailor backend or calibrate a new model.
Applications may inject `CMLFQRoutingCosts` or its rollout model explicitly.

With no measured transfer entry, only recomputation is priced; there is no
heuristic transfer estimate. Tool-return decisions initially use initial input
tokens plus observed trajectory tokens. Passing exact `input_tokens` to the next
`generate` replaces this estimate at reservation time.

## Measured profile format

`cmlfq_cost_profile_path` accepts a JSON file of this shape. These values are
illustrative, **not benchmark results**:

```json
{
  "decode_ms": {
    "1": {"short": 2000, "long": 20000},
    "4": {"short": 4000, "long": 5000}
  },
  "migrations": [
    {
      "source_tp": 1,
      "target_tp": 4,
      "source_host": "host-a",
      "target_host": "host-b",
      "seq_len": 10000,
      "transfer_ms": 400,
      "recompute_ms": 1500
    }
  ]
}
```

Decode entries measure the **total remaining decode time** conditional on each
bin, not time per token. Migration entries are directional and match the actual
configured source/target hosts and TP degrees. Among matching entries, select
the closest context length. Missing bins use the analytic model when available;
without a model an unknown cost cannot win selection. Milliseconds are converted
to seconds exactly once. Use consistent hardware/model/context/load conditions
when collecting profiles. Re-profile after changing models or connector layout.
The old `cmlfq_migration_profile_path` belongs to the legacy standalone helper and
is not the new combined profile format.

## Execution and failure behavior

Tool return queues routing intent; it does not move ownership. On the next
generation, reserve capacity on the selected target, execute the backend, then
commit the actual instance/bucket and migration count only after successful
non-streaming completion. On failure or cancellation, release the target
reservation and retain the source route. Finishing/cancelling a request also
cleans pending intent and reservations. Concurrent generation for one request
is rejected. Metadata in `_schedule_info` includes planned and actual execution
paths, estimated costs and any transfer fallback reason.

`recompute` works with the existing ordinary vLLM completions endpoint. It sends
the full updated workflow transcript to the target, which performs prefill before
decode (possibly benefiting from its local prefix cache).

`nixl` uses vLLM's existing disaggregated prefill protocol:

1. Prepare the exact next prompt on the source using `max_tokens=1` and
   `kv_transfer_params.do_remote_decode=true`. Discard this preparatory output.
2. Pass the returned opaque KV descriptor to the target with
   `do_remote_prefill=true`, along with the same full prompt and original
   generation parameters. vLLM's connector handles network movement and TP layout.
3. If preparation/transfer fails or the source returns no usable descriptor,
   retry the target once with the full prompt and no transfer parameters.

The measured `transfer_ms` **must include source preparation/export and target
import**, not only network copying. This is a backend-supported transport path;
it does not implement the paper's CPU offload hidden during tool execution.
No unsupported vLLM KV export HTTP endpoints or guessed CUDA device mappings
are used. Unconsumed export blocks are subject to the server connector's lease
expiry; this client does not implement a separate release RPC.

Enable transfer only on compatible vLLM deployments, for example with the
documented NIXL server options:

```text
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'
```

Set reachable side-channel hosts/unique ports and validate the model, attention
backend, block layout and **directional TP pairs** against the installed vLLM
version. Then opt in on the client:

```yaml
cmlfq_kv_backend: nixl
cmlfq_kv_transfer_tp_pairs: [[1, 4], [4, 1]]
cmlfq_cost_profile_path: /absolute/path/to/measured-costs.json
```

The pairs above are configuration examples, not a claim that every model/backend
supports those pairs. With an empty allowlist no transfer is attempted. Server
load-failure policy `fail` lets Libra observe failures and report its recompute
fallback; server-side silent recomputation cannot be distinguished from transfer
success using the ordinary completion response alone. Older vLLM servers without
this connector protocol must use `recompute`.

Protocol reference: [vLLM NIXL guide](https://docs.vllm.ai/en/v0.22.0/features/nixl_connector_usage/)
and [vLLM P/D proxy](https://github.com/vllm-project/vllm/blob/v0.22.0/examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py).

## Validation

### Tool-time CPU offload (vLLM V1 0.9.2)

`cmlfq_kv_backend: cpu_offload` adds a separate path using
`LibraCPUOffloadConnector`. NIXL and recomputation remain available. Start servers
with `scripts/vllm_hot_reload_api_server.py` and:

```text
--kv-transfer-config '{"kv_connector":"LibraCPUOffloadConnector","kv_role":"kv_both","kv_connector_module_path":"RL_Framework.infra.scheduling.vllm_cpu_offload","kv_connector_extra_config":{"storage_path":"/dev/shm/libra-kv"}}'
```

Set `LIBRA_KV_STORAGE_PATH` on the API server to the same directory. Source and
target workers must see the same storage; tmpfs is suitable on one host. The
current transport is shared CPU storage, not direct RDMA. Do not configure a
node-local `/dev/shm` directory for endpoints on different hosts.

Each managed completion carries an export key. When generation ends, the
connector returns a descriptor immediately while retaining the request's blocks.
On the following worker tick it gathers those blocks, enqueues nonblocking copies
on a CUDA stream into pinned CPU tensors, and releases the GPU blocks only after
the copy event completes. CPU shard publication runs in a background thread.
The client starts its normal asynchronous tool operation as soon as completion
returns, overlapping DMA and publication without a separate synchronous offload
call. On resumption it waits only for any unfinished publication, verifies the
exact token prefix, reshards the CPU head ranges for the target TP, reloads, and
prefills the tool suffix. Full-block alignment leaves at least one token to
compute logits. Same-instance resumption uses the CPU cache too.

`_cpu_offload_info` reports actual imported tokens, visible waiting time, and
source DMA/publication timestamps. An absent, incompatible or failed cache falls
back to full prefill. Finish/cancel and successful resumption release prior
exports; `close()` drains client release tasks. The server keeps tiny release
tombstones to prevent a late publisher from resurrecting deleted payloads.
Use a per-job storage directory and remove it after servers stop (the validation
script does this). A crashed client can leave exports behind; long-lived
deployments need external storage quotas/expiry management.

The connector currently supports the FlashAttention five-dimensional KV layout,
one KV cache group, and non-replicated KV heads divisible by both TP degrees. MLA,
hybrid attention and replicated-head TP configurations are not supported. It is
validated against vLLM 0.9.2; do not assume its experimental connector API is
compatible with other versions. Updating weights requires draining requests and
clearing old exports just as for existing prefix caches.

For this backend, measured movement costs should include the **remaining visible
wait + CPU transport/reshard/reload**, without charging the full already-hidden
offload time again. Whether transfer beats recomputation still depends on context
length, tool duration, CPU storage and TP pair; overlap is not a universal speedup.

Reproduce real-GPU validation with three disjoint GPUs:

```bash
MODEL_PATH=/path/to/Qwen3-4B sbatch scripts/validate_cpu_offload.slurm
# Reverse direction:
SOURCE_TP=2 TARGET_TP=1 MODEL_PATH=/path/to/Qwen3-4B sbatch scripts/validate_cpu_offload.slurm
```

The benchmark uses real model KV tensors and continuations, with **synthetic
asynchronous tool delays** for controlled comparisons. It compares serialized
offload against overlapping offload, checks greedy outputs against full prefill,
and exercises the scheduler/backend hooks on live endpoints. Timings include CPU
file publication and loading, not just raw PCIe bandwidth.

```bash
python -m pytest -q tests/test_cmlfq_cost_scheduler.py tests/test_cmlfq_scheduler.py \
  tests/test_hetero_cmlfq_integration.py tests/test_rollout_engine.py \
  tests/test_cpu_offload_backend.py tests/test_vllm_cpu_offload.py
```

Tests cover distribution decisions, movement cost and ties, network-path matching,
capacity revalidation, rollback/cancellation, checkpoint distributions, legacy
fallback, and source/target HTTP request bodies using real client code with mock
transports. These tests do not replace a multi-GPU NIXL deployment test or measure
throughput/transfer latency.

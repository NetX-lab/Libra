"""Transport failures must not abandon peers inside the next TP export."""

import io
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=12))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=12))
    try:
        from RL_Framework.infra.sync.collective_export import stream_export_with_consensus
        from RL_Framework.infra.sync import nccl_weight_sync, publication_session
        from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer
        for entrypoint in ("helper", "native"):
            for failure in ("before-first", "after-first", "iterator", "prepare", "request"):
                if failure == "prepare" and entrypoint != "helper":
                    continue
                if failure == "request" and entrypoint == "helper":
                    continue
                exported, sent = [], []

                def weights():
                    for index in range(3):
                        # Real inter-rank compute/export synchronization.
                        dist.barrier()
                        if failure == "iterator" and rank == 1 and index == 1:
                            raise RuntimeError("iterator failed on rank 1")
                        exported.append(index)
                        yield f"weight-{index}", torch.ones(2) * index

                def prepare(item):
                    if failure == "prepare" and rank == 0 and item[0] == "weight-1":
                        raise RuntimeError("tensor preparation failed")
                    return item

                def send(items, *args, **kwargs):
                    if failure == "before-first":
                        raise RuntimeError("transport bootstrap failed")
                    for item in items:
                        sent.append(item[0])
                        if failure == "after-first":
                            raise RuntimeError("transport failed after first tensor")

                folder = Path(directory) / entrypoint / failure
                config = SimpleNamespace(
                    rollout_weight_sync_control_dir=str(folder), rollout_nccl_host="127.0.0.1",
                    rollout_nccl_port=29620, rollout_nccl_chunk_mb=1,
                    rollout_nccl_rate_limit_gbps=0, rollout_weight_sync_timeout_s=2,
                    rollout_weight_sync_poll_interval_s=0.001,
                    heterogeneous_rollout=SimpleNamespace(instances=[
                        SimpleNamespace(instance_id="engine", host="127.0.0.1", port=8000, tp=2),
                    ]),
                )

                def command(value):
                    packet = [value]
                    dist.broadcast_object_list(packet, src=0, group=control)
                    if packet[0].get("error"):
                        raise RuntimeError(packet[0]["error"])
                    return packet[0]

                def write_request(path, value):
                    if failure == "request":
                        raise OSError("control request write failed")
                    path.write_text(json.dumps(value))

                trainer = SimpleNamespace(
                    config=config, is_main_process=rank == 0, control=control,
                    train_engine=SimpleNamespace(train_backend="megatron_core", optimizer_revision=23,
                                                 stream_rollout_weights=weights),
                    optimizer_revision=23, service_generations={},
                    events=SimpleNamespace(emit=lambda *args, **kwargs: None), command=command,
                    _nccl_rollout_topology=lambda: (3, {"engine": 0}),
                    _write_json_atomic=write_request,
                )
                caught = ""
                with patch.object(nccl_weight_sync, "send_weights", send), \
                     patch.object(publication_session.urllib.request, "urlopen",
                         lambda *args, **kwargs: io.StringIO('{"server_instance_id":"live-service"}')), \
                     patch.object(torch.cuda, "current_device", lambda: "cpu"), \
                     patch.object(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)):
                    try:
                        if entrypoint == "helper":
                            stream_export_with_consensus(weights, is_sender=rank == 0, send=send,
                                                         group=control, prepare=prepare)
                        elif entrypoint == "native":
                            AsyncRLTrainer._sync_weights_nccl(trainer, 23)
                    except RuntimeError as exc:
                        caught = str(exc)
                if failure == "request":
                    assert "control request write failed" in caught, caught
                    assert exported == []
                else:
                    assert "collective weight publication failed" in caught, caught
                errors = [None, None]
                dist.all_gather_object(errors, caught, group=control)
                assert all(errors)
                if failure in ("before-first", "after-first"):
                    assert exported == [0, 1, 2]
                    assert sent == (["weight-0"] if rank == 0 and failure == "after-first" else [])
                elif failure in ("iterator", "prepare"):
                    assert 2 not in exported  # nobody enters another export collective
                # A matching barrier proves the failed publication left no rank
                # waiting in an orphaned export iteration.
                dist.barrier()

        captured = []
        stream_export_with_consensus(lambda: iter([("ok", torch.ones(1))]),
                                     is_sender=rank == 0,
                                     send=lambda items: captured.extend(items), group=control)
        assert len(captured) == (1 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


def test_two_rank_stream_failures_drain_then_fail_all_entrypoints(tmp_path):
    mp.start_processes(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)),
                       nprocs=2, join=True, start_method="spawn")

"""Run in the supported vLLM environment; GPU tests use real CUDA tensors."""

import json
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from RL_Framework.infra.scheduling.vllm_cpu_offload import LibraCPUOffloadConnector, cache_directory


def bare_connector(root, tp=1, rank=0):
    connector = object.__new__(LibraCPUOffloadConnector)
    connector.root, connector.tp, connector.rank = root, tp, rank
    connector.block_size, connector.num_heads = 16, 8
    connector.matches, connector.imported = {}, {}
    return connector


def test_prefix_identity_and_complete_block_boundaries():
    with tempfile.TemporaryDirectory() as root:
        key = uuid.uuid4().hex
        directory = cache_directory(root, key)
        directory.mkdir()
        (directory / "rank_0.json").write_text(json.dumps({
            "tokens": list(range(64)), "source_tp": 1, "block_size": 16,
        }))
        connector = bare_connector(root)
        request = SimpleNamespace(kv_transfer_params={"libra_load_key": key},
                                  request_id="r", prompt_token_ids=list(range(65)))
        assert connector.get_num_new_matched_tokens(request, 16) == (48, False)
        request.prompt_token_ids[1] = 999
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        request.prompt_token_ids = list(range(64))
        # Must recompute at least one token for the next logits.
        assert connector.get_num_new_matched_tokens(request, 0) == (48, False)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two real CUDA GPUs")
@pytest.mark.parametrize("source_tp,target_tp", [(1, 2), (2, 1)])
def test_real_cuda_cpu_head_resharding(source_tp, target_tp):
    with tempfile.TemporaryDirectory() as root:
        key = uuid.uuid4().hex
        directory = cache_directory(root, key)
        directory.mkdir()
        full = torch.arange(2 * 2 * 16 * 8 * 4, dtype=torch.float32).reshape(2, 2, 16, 8, 4)
        for rank in range(source_tp):
            with torch.cuda.device(rank):
                gpu = full.chunk(source_tp, dim=-2)[rank].contiguous().cuda()
                pinned = torch.empty_like(gpu, device="cpu", pin_memory=True)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    pinned.copy_(gpu, non_blocking=True)
                stream.synchronize()
                torch.save({"layer": pinned}, directory / f"rank_{rank}.pt")
        for rank in range(target_tp):
            with torch.cuda.device(rank):
                connector = bare_connector(root, target_tp, rank)
                destination = torch.zeros(2, 3, 16, 8 // target_tp, 4, device="cuda")
                connector.caches = {"layer": destination}
                connector._reload({"key": key, "source_tp": source_tp,
                                   "num_tokens": 32, "block_ids": [2, 0]})
                observed = destination[:, [2, 0]].cpu()
                assert torch.equal(observed, full.chunk(target_tp, dim=-2)[rank])

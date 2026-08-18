"""Support code for Test megatron3d backend."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import tempfile
import types

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("RL_Framework")
_pkg.__path__ = [_project_root]
_pkg.__package__ = "RL_Framework"
sys.modules.setdefault("RL_Framework", _pkg)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL_Framework.engine.megatron_train_engine import Megatron3DTrainEngine


def _require_transformers():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    return Tokenizer, WordLevel, Whitespace, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


def _find_free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def build_tiny_qwen_checkpoint(base_dir: str) -> str:
    (
        Tokenizer,
        WordLevel,
        Whitespace,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    ) = _require_transformers()

    ckpt_dir = os.path.join(base_dir, "tiny_qwen_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    vocab = {"<pad>": 0, "<eos>": 1}
    for i in range(2, 128):
        vocab[f"tok_{i}"] = i

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<eos>",
        eos_token="<eos>",
        unk_token="<eos>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(ckpt_dir)

    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(ckpt_dir)
    return ckpt_dir


def make_dummy_trajectories() -> list[dict]:
    torch.manual_seed(7)
    trajectories = []
    seqs = [
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19],
        [20, 21, 22],
    ]
    for idx, seq in enumerate(seqs):
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        logprobs = torch.zeros_like(seq_tensor, dtype=torch.float32)
        loss_mask = torch.zeros_like(seq_tensor, dtype=torch.long)
        input_len = max(1, len(seq) // 2)
        loss_mask[:, input_len:] = 1
        trajectories.append(
            {
                "input_ids": seq_tensor,
                "attention_mask": torch.ones_like(seq_tensor),
                "logprobs": logprobs,
                "loss_mask": loss_mask,
                "rewards": torch.tensor([1.0 + idx * 0.1], dtype=torch.float32),
                "advantages": torch.tensor([0.5 + idx * 0.2], dtype=torch.float32),
                "versions": torch.tensor([0], dtype=torch.long),
                "input_len": input_len,
                "output_len": len(seq) - input_len,
            }
        )
    return trajectories


def run_single_case():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA available")
        return

    tmpdir = tempfile.mkdtemp(prefix="megatron3d_single_")
    try:
        model_path = build_tiny_qwen_checkpoint(tmpdir)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        engine = Megatron3DTrainEngine(
            model_path=model_path,
            train_tp_size=1,
            train_pp_size=1,
            train_dp_size=1,
            micro_batch_size=2,
        )
        engine.initialize()
        trajectories = make_dummy_trajectories()
        engine.recompute_logprobs(trajectories)
        stats = engine.grpo_update(trajectories, ppo_epochs=1)
        assert "loss" in stats and torch.isfinite(torch.tensor(stats["loss"]))
        save_dir = os.path.join(tmpdir, "weights")
        engine.save_weights(save_dir, version=0)
        engine.load_weights(save_dir, version=0)
        print("single-case smoke test passed")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _distributed_worker(rank: int, world_size: int, port: int, model_path: str, tp: int, pp: int, dp: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        engine = Megatron3DTrainEngine(
            model_path=model_path,
            train_tp_size=tp,
            train_pp_size=pp,
            train_dp_size=dp,
            micro_batch_size=2,
        )
        engine.initialize()

        trajectories = make_dummy_trajectories() if engine.is_batch_source() else None
        trajectories = engine.distribute_trajectories(trajectories)
        engine.recompute_logprobs(trajectories)
        stats = engine.grpo_update(trajectories, ppo_epochs=1)
        assert "loss" in stats
        assert torch.isfinite(torch.tensor(stats["loss"], device=torch.device(f"cuda:{rank}")))

        save_dir = os.path.join(os.path.dirname(model_path), f"weights_tp{tp}_pp{pp}_dp{dp}")
        engine.save_weights(save_dir, version=0)
        engine.load_weights(save_dir, version=0)

        if rank == 0:
            print(f"distributed smoke passed: tp={tp}, pp={pp}, dp={dp}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def run_distributed_case(tp: int, pp: int, dp: int):
    if not torch.cuda.is_available():
        print("SKIP: no CUDA available")
        return
    world_size = tp * pp * dp
    if torch.cuda.device_count() < world_size:
        print(
            f"SKIP: need {world_size} GPUs, only {torch.cuda.device_count()} available"
        )
        return

    tmpdir = tempfile.mkdtemp(prefix="megatron3d_dist_")
    try:
        model_path = build_tiny_qwen_checkpoint(tmpdir)
        port = _find_free_port()
        mp.spawn(
            _distributed_worker,
            args=(world_size, port, model_path, tp, pp, dp),
            nprocs=world_size,
            join=True,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["single", "distributed"], default="single")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    args = parser.parse_args()

    if args.case == "single":
        run_single_case()
    else:
        run_distributed_case(args.tp, args.pp, args.dp)


if __name__ == "__main__":
    main()

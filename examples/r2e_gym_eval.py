"""Evaluate R2E-Gym with a configured rollout engine."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer

from RL_Framework import parse_args_and_load_config
from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
from RL_Framework.engine.rollout_engine import MultiInstanceRolloutEngine
from RL_Framework.env.r2e_gym_reward import r2e_gym_reward_fn
from RL_Framework.workflow.r2e_gym import R2EGymWorkflow


def _normalize_modified_files(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return [str(value)]
    return parsed if isinstance(parsed, list) else [str(parsed)]


def _load_dataset():
    data_path = os.environ.get(
        "R2E_GYM_INDEX",
        str(_PROJECT_ROOT / "data" / "r2e_gym_v1" / "index.jsonl"),
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"R2E-Gym index not found: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(example):
        prompt = (example.get("prompt") or "").strip()
        target_issue = (
            example.get("task_text")
            or example.get("problem_statement")
            or ""
        ).strip()
        prompt_id = f"{example.get('repo_name', 'repo')}:{example.get('commit_hash', '')}"
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "task_text": target_issue,
            "target_issue": target_issue,
            "repo_name": example.get("repo_name", ""),
            "commit_hash": example.get("commit_hash", ""),
            "docker_image": example.get("docker_image", ""),
            "expected_output_json": example.get("expected_output_json", "{}"),
            "modified_files": _normalize_modified_files(example.get("modified_files")),
        }

    return dataset.map(preprocess)


def _build_engine(config):
    if getattr(config.heterogeneous_rollout, "enabled", False):
        engine = HeterogeneousRolloutEngine.from_config(config)
    else:
        endpoints = getattr(config, "vllm_endpoints", "") or os.environ.get(
            "VLLM_ENDPOINTS", ""
        )
        if endpoints:
            engine = MultiInstanceRolloutEngine(
                model_path=config.model_path,
                endpoints=[ep.strip() for ep in endpoints.split(",") if ep.strip()],
            )
        else:
            n_instances = int(getattr(config, "vllm_num_instances", 0) or 0)
            if n_instances <= 0:
                n_instances = max(
                    1,
                    int(getattr(config, "rollout_gpus", 1))
                    // max(1, int(getattr(config, "vllm_tp_size", 1))),
                )
            engine = MultiInstanceRolloutEngine(
                host=config.vllm_host,
                base_port=config.vllm_port,
                num_instances=n_instances,
                model_path=config.model_path,
            )
    engine.wait_for_ready(timeout=float(os.environ.get("VLLM_READY_TIMEOUT", "600")))
    return engine


async def _evaluate_and_close(workflow, engine, dataset, config):
    try:
        return await workflow.evaluate(
            engine,
            dataset,
            max_samples=int(os.environ.get("R2E_EVAL_MAX_SAMPLES", "4")),
            concurrency=int(os.environ.get("R2E_EVAL_CONCURRENCY", "1")),
            accuracy_threshold=float(os.environ.get("R2E_EVAL_ACCURACY_THRESHOLD", "0.5")),
            max_new_tokens=int(
                os.environ.get("R2E_EVAL_MAX_NEW_TOKENS", str(config.max_new_tokens))
            ),
        )
    finally:
        try:
            await engine.close()
        except Exception as exc:
            print(f"WARNING: failed to close rollout engine cleanly: {exc}", file=sys.stderr)


def main():
    config = parse_args_and_load_config()
    dataset = _load_dataset()
    repo_counts = Counter(dataset["repo_name"])
    print(f"Loaded R2E-Gym rows: {len(dataset)} repos={dict(sorted(repo_counts.items()))}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    workflow = R2EGymWorkflow(
        reward_fn=r2e_gym_reward_fn,
        tokenizer=tokenizer,
        max_turns=int(os.environ.get("R2E_MAX_TURNS", str(config.r2e_max_turns))),
        max_new_tokens=config.max_new_tokens,
        max_seq_length=config.max_seq_length,
        max_prompt_tokens=(
            int(os.environ["R2E_MAX_PROMPT_TOKENS"])
            if os.environ.get("R2E_MAX_PROMPT_TOKENS")
            else (config.r2e_max_prompt_tokens or None)
        ),
        temperature=config.temperature,
        top_p=config.top_p,
        n_samples=config.n_samples,
        stop_reward=config.r2e_stop_reward,
    )
    engine = _build_engine(config)
    stats = asyncio.run(_evaluate_and_close(workflow, engine, dataset, config))

    output_path = os.environ.get("R2E_EVAL_OUTPUT", "")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({k: v for k, v in stats.items() if k != "eval_records"}, indent=2))


if __name__ == "__main__":
    main()

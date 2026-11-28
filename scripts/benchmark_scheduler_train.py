"""Instrument the real DAPO trainer without replacing its training loop."""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from RL_Framework import AsyncRLTrainer, parse_args_and_load_config
try:
    from examples.dapo_math_async_rl import _preprocess
except ImportError:
    from RL_Framework.examples.dapo_math_async_rl import _preprocess
from RL_Framework.workflow.dapo_math import DAPOMathWorkflow

OUT = Path(os.environ['BENCH_RUN_DIR'])


def append(name, data):
    with (OUT / name).open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


class MeasuredEngine:
    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    async def generate(self, **kwargs):
        started = time.time()
        result = await self.engine.generate(**kwargs)
        append('generations.jsonl', dict(start=started, end=time.time(),
            route=result.get('_schedule_info', {}),
            offload=result.get('_cpu_offload_info', {})))
        return result


class MeasuredWorkflow(DAPOMathWorkflow):
    async def run_episode(self, engine, data, version=0, rollout_index=0):
        started = time.time()
        result = await super().run_episode(
            MeasuredEngine(engine),
            data,
            version,
            rollout_index=rollout_index,
        )
        append('episodes.jsonl', dict(start=started, end=time.time(), version=version,
            prompt_id=result['prompt_id'], input_tokens=result['input_len'],
            generated_tokens=result['output_len'], turns=result['n_turns'],
            reward=float(result['rewards'].item())))
        return result


class MeasuredTrainer(AsyncRLTrainer):
    def _record_history_step(self, step, batch, stats, *args, **kwargs):
        super()._record_history_step(step, batch, stats, *args, **kwargs)
        append('steps.jsonl', dict(step=step, timestamp=time.time(), stats=stats,
            generated_tokens=sum(int(t['loss_mask'].sum()) for t in batch),
            sequence_tokens=sum(int(t['input_ids'].numel()) for t in batch),
            prompt_ids=[t['prompt_id'] for t in batch],
            scheduler=self.rollout_engine.scheduler.get_stats().to_dict()))


def main():
    config = parse_args_and_load_config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    data_path = os.environ['DAPO_MATH_PATH']
    data_format = 'parquet' if data_path.endswith('.parquet') else 'json'
    dataset = load_dataset(data_format, data_files=data_path, split='train')
    dataset = dataset.map(_preprocess, with_indices=True)
    dataset = dataset.filter(lambda r: bool(r['question']) and bool(r['ground_truth']))
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path or config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    workflow = MeasuredWorkflow(
        tokenizer=tokenizer,
        max_turns=int(os.environ.get('DAPO_MAX_TURNS', '2')),
        max_new_tokens=config.max_new_tokens,
        max_seq_length=config.max_seq_length,
        max_prompt_tokens=int(os.environ.get('DAPO_MAX_PROMPT_TOKENS', '512')),
        temperature=config.temperature,
        top_p=config.top_p,
    )
    trainer = MeasuredTrainer(config)
    started = time.time()
    trainer.train(workflow, dataset)
    (OUT / 'completed.json').write_text(json.dumps(dict(start=started, end=time.time(),
        total_steps=config.total_steps, seed=config.seed)))


if __name__ == '__main__':
    main()

"""Instrument the real DAPO trainer without replacing its training loop."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from RL_Framework import AsyncRLTrainer, parse_args_and_load_config
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
        result = await super().run_episode(MeasuredEngine(engine), data, version)
        append('episodes.jsonl', dict(start=started, end=time.time(), version=version,
            prompt_id=result['prompt_id'], input_tokens=result['input_len'],
            generated_tokens=result['output_len'], turns=result['n_turns'],
            reward=float(result['rewards'].item())))
        return result


class MeasuredTrainer(AsyncRLTrainer):
    def _record_history_step(self, step, batch, stats, *args):
        super()._record_history_step(step, batch, stats, *args)
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
    dataset = load_dataset('json', data_files=os.environ['DAPO_MATH_PATH'], split='train')
    dataset = dataset.map(_preprocess, with_indices=True)
    dataset = dataset.filter(lambda r: bool(r['question']) and bool(r['ground_truth']))
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    workflow = MeasuredWorkflow(tokenizer=tokenizer, max_turns=2,
        max_new_tokens=config.max_new_tokens, max_seq_length=config.max_seq_length,
        max_prompt_tokens=512, temperature=config.temperature, top_p=config.top_p)
    trainer = MeasuredTrainer(config)
    started = time.time()
    trainer.train(workflow, dataset)
    (OUT / 'completed.json').write_text(json.dumps(dict(start=started, end=time.time(),
        total_steps=config.total_steps, seed=config.seed)))


if __name__ == '__main__':
    main()

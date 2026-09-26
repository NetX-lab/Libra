"""Summarize completed trials; incomplete/error trials are never ranked."""
import argparse
import json
from pathlib import Path
import statistics


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(root):
    results = []
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not (run/'completed.json').exists():
            continue
        completion = json.loads((run/'completed.json').read_text())
        steps = read_lines(run/'steps.jsonl')
        if len(steps) != completion['total_steps']:
            raise ValueError(f'incomplete step records: {run}')
        if not all(s['stats']['n_updates'] > 0 for s in steps):
            raise ValueError(f'missing optimizer updates: {run}')
        acks = list((run/'control').glob('ack_*.json'))
        if len(acks) < 2:
            raise ValueError(f'no confirmed rollout weight reload: {run}')
        policy, seed = run.name.rsplit('_', 1)
        seconds = sum(s['stats']['step_time'] for s in steps)
        samples = sum(s['stats']['n_trajectories'] for s in steps)
        tokens = sum(s['generated_tokens'] for s in steps)
        gens = read_lines(run/'generations.jsonl')
        imports = [g['offload'] for g in gens if g.get('offload', {}).get('imported_tokens',0)>0]
        warm = steps[1:]
        row = dict(policy=policy, seed=int(seed), steps=len(steps), samples=samples,
            optimizer_updates=sum(s['stats']['n_updates'] for s in steps),
            loop_seconds=seconds, process_seconds=completion['end']-completion['start'],
            samples_per_second=samples/seconds, generated_tokens=tokens,
            generated_tokens_per_second=tokens/seconds,
            after_first_step_samples_per_second=sum(s['stats']['n_trajectories'] for s in warm)/sum(s['stats']['step_time'] for s in warm),
            rollout_wait_seconds=sum(s['stats']['rollout_time'] for s in steps),
            train_seconds=sum(s['stats']['train_time'] for s in steps),
            recompute_seconds=sum(s['stats']['recompute_logprob_time'] for s in steps),
            sync_seconds=sum(s['stats']['weight_sync_time'] for s in steps),
            reward_mean=statistics.mean(s['stats']['reward_mean'] for s in steps),
            kv_import_generations=len(imports), imported_tokens=sum(i['imported_tokens'] for i in imports),
            kv_visible_wait_ms=sum(g.get('offload', {}).get('visible_wait_ms',0) for g in gens),
            generation_calls=len(gens), reload_acks=len(acks),
            route_counts={})
        for g in gens:
            key=str(g.get('route', {}).get('tp_degree','unknown'))
            row['route_counts'][key]=row['route_counts'].get(key,0)+1
        results.append(row)
    aggregates={}
    for policy in sorted({r['policy'] for r in results}):
        rows=[r for r in results if r['policy']==policy]
        aggregates[policy] = dict(runs=len(rows),
            samples_per_second=sum(r['samples'] for r in rows)/sum(r['loop_seconds'] for r in rows),
            generated_tokens_per_second=sum(r['generated_tokens'] for r in rows)/sum(r['loop_seconds'] for r in rows))
    improvements={}
    if 'cmlfq_cost' in aggregates:
        for baseline in ('round_robin','least_connections'):
            if baseline in aggregates:
                improvements[baseline]={metric:100*(aggregates['cmlfq_cost'][metric]/aggregates[baseline][metric]-1)
                    for metric in ('samples_per_second','generated_tokens_per_second')}
    return dict(runs=results, aggregate=aggregates, improvement_percent=improvements)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    result=summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

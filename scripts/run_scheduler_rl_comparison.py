"""Run controlled full-loop RL scheduler comparisons inside a four-GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs' / 'scheduler_rl' / os.environ['SLURM_JOB_ID']
MODEL = os.environ['MODEL_PATH']
GPUS = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
PYTHON = sys.executable


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_one(policy, seed):
    run = OUT / f'{policy}_{seed}'
    run.mkdir(parents=True)
    control = run / 'control'
    storage = Path('/dev/shm') / f'libra-rl-{os.environ["SLURM_JOB_ID"]}-{policy}-{seed}'
    storage.mkdir()
    env = dict(os.environ, PYTHONPATH=str(ROOT.parent), VLLM_USE_V1='1',
        VLLM_ATTENTION_BACKEND='FLASH_ATTN', OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
        LIBRA_KV_STORAGE_PATH=str(storage), FSDP_GRADIENT_CHECKPOINTING='1',
        FSDP_PROGRESS_DIR=str(run / 'progress'), BENCH_RUN_DIR=str(run),
        DAPO_MATH_PATH=str(OUT / 'dataset.jsonl'), NO_PROXY='localhost,127.0.0.1')
    cpu = policy == 'cmlfq_cost'
    schedule = dict(scheduler_type='cmlfq_cost' if cpu else 'load_balance',
        load_balance_strategy='round_robin' if policy == 'round_robin' else 'least_connections',
        max_queue_length=16, cmlfq_kv_backend='cpu_offload' if cpu else 'recompute',
        cmlfq_tree_path=str(run / 'tree.json'), cmlfq_rebuild_interval=8,
        cmlfq_kv_transfer_tp_pairs=[[1, 2], [2, 1]],
        cmlfq_buckets={'short': {'tp_degrees': [1], 'max_tokens': 512},
                      'long': {'tp_degrees': [2], 'max_tokens': 2048}})
    config = dict(model_path=MODEL, tokenizer_path=MODEL,
        hardware_config=str(ROOT/'configs/hardware_config/a800_80g.yaml'),
        model_arch_config=str(ROOT/'configs/model_arch_config/qwen3-4b.yaml'),
        train_backend='fsdp', train_gpus=1, rollout_gpus=3,
        train_tp_size=1, train_dp_size=1, batch_size=8, micro_batch_size=1,
        total_steps=int(os.environ.get('BENCH_STEPS', '8')), n_samples=2,
        max_concurrent_rollouts=16, max_head_offpolicyness=4, queue_size=32,
        max_new_tokens=512, max_seq_length=2048, temperature=0.8, top_p=0.95,
        learning_rate=1e-6, ppo_epochs=1, seed=seed, recompute_logprobs=True,
        sync_interval=4, rollout_sync_drain_lead_steps=1,
        weight_sync_mode='disk', sync_path=str(run/'weights'),
        rollout_weight_sync_mode='restart', rollout_weight_reload_method='inplace',
        rollout_weight_sync_control_dir=str(control), require_rollout_weight_sync=True,
        eval_interval=0, save_interval=0, log_dir=str(run),
        enable_history_collection=True, history_output_dir=str(run/'history'),
        history_flush_interval=1, phase_trace_enabled=True, phase_trace_dir=str(run/'phases'),
        heterogeneous_rollout=dict(enabled=True, total_gpus=3, max_model_len=2048,
            instances=[dict(instance_id='tp1', tp=1, gpus=[1], host='127.0.0.1', port=18920),
                       dict(instance_id='tp2', tp=2, gpus=[2,3], host='127.0.0.1', port=18921)],
            scheduling=schedule))
    (run/'config.yaml').write_text(yaml.safe_dump(config))
    processes = []
    logs = []
    try:
        for i, (tp, devices) in enumerate([(1, GPUS[1]), (2, ','.join(GPUS[2:4]))]):
            port = 18920+i
            command = [PYTHON, str(ROOT/'scripts/vllm_hot_reload_api_server.py'),
                '--model', '__MODEL_PATH__', '--host', '127.0.0.1', '--port', str(port),
                '--tensor-parallel-size', str(tp), '--dtype', 'bfloat16',
                '--max-model-len', '2048', '--gpu-memory-utilization', '0.65',
                '--max-num-seqs', '16', '--enforce-eager', '--disable-log-requests',
                '--no-enable-prefix-caching', '--seed', str(seed),
                '--worker-extension-cls', 'RL_Framework.vllm_hot_reload.InplaceReloadWorkerExtension']
            if cpu:
                command += ['--kv-transfer-config', json.dumps(dict(kv_connector='LibraCPUOffloadConnector',
                    kv_role='kv_both', kv_connector_module_path='RL_Framework.infra.scheduling.vllm_cpu_offload',
                    kv_connector_extra_config=dict(storage_path=str(storage))))]
            supervisor = [PYTHON, str(ROOT/'scripts/restartable_vllm_server.py'),
                '--instance-id', f'tp{tp}', '--control-dir', str(control),
                '--health-url', f'http://127.0.0.1:{port}/health', '--initial-model', MODEL,
                '--reload-method', 'inplace', '--', *command]
            log = (run/f'vllm_tp{tp}.log').open('w'); logs.append(log)
            processes.append(subprocess.Popen(supervisor, env=dict(env, CUDA_VISIBLE_DEVICES=devices),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic()+600
        for port in (18920, 18921):
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('rollout server exited; inspect logs')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                        break
                except Exception:
                    if time.monotonic()>deadline:
                        raise TimeoutError('rollout server startup timeout')
                    time.sleep(2)
        log = (run/'train.log').open('w'); logs.append(log)
        process = subprocess.Popen([PYTHON, str(ROOT/'scripts/benchmark_scheduler_train.py'),
            '--config', str(run/'config.yaml')], env=dict(env, CUDA_VISIBLE_DEVICES=GPUS[0]),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        deadline = time.monotonic() + 1800
        while process.poll() is None:
            time.sleep(5)
            if time.monotonic() > deadline:
                raise TimeoutError('training exceeded 1800 seconds')
            with (run/'train.log').open() as check:
                check.seek(max(0, (run/'train.log').stat().st_size-32768))
                recent = check.read()
            if 'ERROR: Rollout' in recent or 'Traceback (most recent call last)' in recent:
                raise RuntimeError(f'training error, aborting invalid trial: {run}')
        rc = process.returncode
        if rc:
            raise RuntimeError(f'training exited {rc}: {run}')
        print(f'COMPLETED {run}', flush=True)
    finally:
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()
        shutil.rmtree(storage)


def main():
    if len(GPUS)!=4:
        raise ValueError(f'Expected four allocated GPUs: {GPUS}')
    OUT.mkdir(parents=True)
    source = ROOT/'data/dapo_benchmark_en.jsonl'
    # Same fixed, real-data subset in every arm; no synthetic tools or delays.
    with source.open() as f:
        rows = [line for _, line in zip(range(256), f)]
    if len(rows) < 256 or rows[0].startswith('version https://git-lfs'):
        raise ValueError('Need materialized DAPO-Math dataset (at least 256 rows)')
    (OUT/'dataset.jsonl').write_text(''.join(rows))
    (OUT/'manifest.json').write_text(json.dumps(dict(model=MODEL, gpus=GPUS,
        hostname=os.uname().nodename, dataset_sha256=hashlib.sha256(''.join(rows).encode()).hexdigest(),
        started=time.time(), policies=['round_robin','least_connections','cmlfq_cost'])))
    for seed, policies in [(42, ['round_robin','least_connections','cmlfq_cost']),
                           (43, ['cmlfq_cost','least_connections','round_robin'])]:
        for policy in policies:
            print(f'START {policy} seed={seed}', flush=True)
            run_one(policy, seed)


if __name__ == '__main__':
    main()

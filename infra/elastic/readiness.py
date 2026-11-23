"""Validate actual per-model-lane EHP installation receipts."""
from __future__ import annotations

import json
from pathlib import Path


def snapshot_contract(snapshot_path, *, state_version=None, target_core_id=None):
    if not snapshot_path:
        raise ValueError('Hybrid readiness requires a captured state identity')
    if str(snapshot_path).startswith('ehp://'):
        if state_version is None:
            raise ValueError('direct live-state readiness requires its optimizer revision')
        return {'snapshot_path': str(snapshot_path), 'snapshot_id': None,
                'optimizer_revision': int(state_version), 'target_core_id': target_core_id}
    manifest = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
    if manifest.get('format') != 'libra-ehp-lanes-v1' or not manifest.get('snapshot_id'):
        raise ValueError('Hybrid readiness requires a live lane manifest')
    revision = int(manifest['optimizer_revision'])
    if state_version is not None and revision != int(state_version):
        raise ValueError('Hybrid readiness revision differs from its snapshot manifest')
    if target_core_id is not None and target_core_id not in manifest['lanes']:
        raise ValueError('Hybrid target core is missing from its state manifest')
    return {'snapshot_path': str(snapshot_path), 'snapshot_id': str(manifest['snapshot_id']),
            'optimizer_revision': revision, 'target_core_id': target_core_id,
            'lanes': manifest['lanes']}


def read_ready_receipts(task_dir, worker_id, world_size, contract):
    """Return a receipt only when every actual TP worker matches the contract.

    Old and partially written files are pending observations, never readiness.
    The process owner removes same-snapshot files before each new launch.
    """
    receipts = []
    revision = contract['optimizer_revision']
    for rank in range(int(world_size)):
        path = Path(task_dir) / f'{worker_id}.rank_{rank}.ready'
        try:
            rows = path.read_text(encoding='utf-8').splitlines()
            receipt = dict(row.split('=', 1) for row in rows if '=' in row)
            if receipt.get('acknowledged') != 'True':
                return None
            if int(receipt.get('rank', -1)) != rank or int(receipt.get('pid', -1)) <= 0:
                return None
            if receipt.get('snapshot') != contract['snapshot_path'] or not receipt.get('snapshot_id'):
                return None
            expected_identity = contract.get('snapshot_id')
            if expected_identity is not None and receipt['snapshot_id'] != expected_identity:
                return None
            if int(receipt.get('optimizer_revision', -1)) != revision or int(receipt.get('state_version', -1)) != revision:
                return None
            target = contract.get('target_core_id')
            if target is not None and receipt.get('target_core_id') != target:
                return None
            if expected_identity is not None:
                if not receipt.get('lane_snapshot_id'):
                    return None
                lane_core = target or receipt.get('target_core_id')
                expected_uri = contract['lanes'].get(lane_core, {}).get(str(rank))
                if not expected_uri or receipt.get('source_uri') != expected_uri:
                    return None
        except (OSError, ValueError, TypeError):
            return None
        receipts.append(receipt)
    if not receipts or len({receipt['pid'] for receipt in receipts}) != len(receipts):
        return None
    if len({receipt['snapshot_id'] for receipt in receipts}) != 1:
        return None
    targets = {receipt.get('target_core_id') for receipt in receipts}
    if len(targets) != 1:
        return None
    return {'acknowledged': True, 'snapshot_id': receipts[0]['snapshot_id'],
            'state_version': revision, 'optimizer_revision': revision,
            'target_core_id': receipts[0].get('target_core_id'), 'rank_receipts': receipts}


def clear_ready_receipts(task_dir, worker_id):
    directory = Path(task_dir)
    (directory / f'{worker_id}.ready').unlink(missing_ok=True)
    for path in directory.glob(f'{worker_id}.rank_*.ready'):
        path.unlink(missing_ok=True)

"""Trajectory weights remain fixed across uneven elastic microbatches."""
from types import SimpleNamespace

import pytest
import torch

from RL_Framework.engine import megatron_core_train_engine as engine_module
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.infra.elastic.batch_partition import partition_complete_grpo_groups
from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer


def _engine():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.train_tp_size = 1
    engine.clip_epsilon, engine.kl_coef = .2, .1
    return engine


def test_b12_n4_mb3_gradient_matches_each_trajectory_reference_with_or_without_hybrid():
    rows = [dict(grpo_group_id=str(i // 4), rollout_index=i % 4, index=i,
                 rewards=torch.tensor([float(i == 8)])) for i in range(12)]
    AsyncRLTrainer.__new__(AsyncRLTrainer)._compute_advantages(rows)
    engine = _engine()

    def gradient(participants, microbatch_size):
        parameter = torch.tensor(.1, requires_grad=True)
        assignments = partition_complete_grpo_groups(rows, 4, participants)
        for subset in assignments.values():
            for start in range(0, len(subset), microbatch_size):
                part = subset[start:start + microbatch_size]
                count = len(part)
                mask = torch.zeros(count, 11)
                mask[:, 1] = 1
                for index, row in enumerate(part):
                    if row['index'] in (6, 8):
                        mask[index, 1:] = 1
                # Distinct row features keep the reference gradient nonzero.
                features = torch.tensor([1 + row['index'] / 20 for row in part])[:, None]
                logits = torch.stack([(parameter * features).expand(count, 11),
                                      torch.zeros(count, 11)], -1)
                batch = dict(input_ids=torch.zeros(count, 11, dtype=torch.long),
                    old_logprobs=torch.full((count, 11), -torch.log(torch.tensor(2.))),
                    loss_mask=mask, advantages=torch.cat([row['advantages'] for row in part]),
                    rewards=torch.cat([row['rewards'] for row in part]))
                loss, _ = engine._loss(batch, logits)
                (loss * count / len(rows)).backward()
        return parameter.grad

    reference = gradient(['dp0'], 1)
    assert reference.abs() > 1e-4
    torch.testing.assert_close(gradient(['dp0'], 3), reference)
    torch.testing.assert_close(gradient(['dp0', 'hybrid'], 3), reference)


def test_masked_sample_losses_and_statistics_match_single_trajectory_reference(monkeypatch):
    engine = _engine()
    values = torch.tensor([.1, -.3, .8], requires_grad=True)
    logits = torch.stack([values[:, None].expand(3, 4), torch.zeros(3, 4)], -1)
    batch = dict(input_ids=torch.zeros(3, 4, dtype=torch.long),
        old_logprobs=torch.full((3, 4), -.7),
        loss_mask=torch.tensor([[0., 1., 0., 0.], [0., 1., 1., 1.], [0., 0., 0., 0.]]),
        advantages=torch.tensor([.7, -.4, 1.]), rewards=torch.tensor([1., 2., 3.]))

    def subset(start, stop):
        return {key: value[start:stop] for key, value in batch.items()}

    singles = [engine._loss(subset(i, i + 1), logits[i:i + 1]) for i in range(3)]
    reference_loss = torch.stack([loss for loss, _ in singles]).mean()
    actual_loss, actual_stats = engine._loss(batch, logits)
    torch.testing.assert_close(actual_loss, reference_loss)
    actual_gradient, = torch.autograd.grad(actual_loss, values, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference_loss, values, retain_graph=True)
    torch.testing.assert_close(actual_gradient, reference_gradient)
    assert actual_gradient[2] == 0
    assert singles[2][0] == 0 and singles[2][1]['n_samples'] == 1

    left = engine._loss(subset(0, 2), logits[:2])[1]
    right = singles[2][1]
    merged = engine._merge_stats([left, right])
    for key in ('loss', 'pg_loss', 'kl'):
        assert merged[key] == pytest.approx(actual_stats[key])
    assert merged['n_samples'] == 3 and merged['n_updates'] == 2

    # Exercise the real DP-stat encoder and decoder with a captured peer vector.
    engine._device = lambda: 'cpu'
    engine._parallel_state = lambda: SimpleNamespace(get_data_parallel_group=lambda **_: None)
    peer = []
    monkeypatch.setattr(engine_module.dist, 'all_reduce', lambda values, **_: peer.append(values.clone()))
    engine._reduce_stats(right)
    monkeypatch.setattr(engine_module.dist, 'all_reduce', lambda values, **_: values.add_(peer[0]))
    reduced = engine._reduce_stats(left)
    for key in ('loss', 'pg_loss', 'kl'):
        assert reduced[key] == pytest.approx(actual_stats[key])
    assert reduced['n_samples'] == 3 and reduced['n_updates'] == 2

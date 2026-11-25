"""Numerical and state contracts for the native-reduction decoupled objective."""

import math
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from RL_Framework.workflow.behavior import validate_anchor_rows, validate_behavior_rows
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


def engine(objective="decoupled", micro_batch_size=1):
    obj = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    obj.grpo_objective = objective
    obj.clip_epsilon, obj.kl_coef = .2, .1
    obj.optimizer_revision, obj.current_version = 0, 0
    obj.sequence_parallel, obj.train_tp_size, obj.world_size = False, 1, 1
    obj.micro_batch_size = obj.recompute_micro_batch_size = micro_batch_size
    obj._elastic_batch_partition, obj._elastic_step_hybrid_workers = None, ()
    obj.tokenizer = NS(pad_token_id=0)
    obj._device = lambda: torch.device("cpu")
    return obj


def trajectory(length=4, advantage=1., behavior=-1.5):
    return dict(input_ids=torch.arange(length).unsqueeze(0),
                attention_mask=torch.ones(1, length),
                logprobs=torch.full((1, length), -99.),
                behavior_logprobs=torch.full((1, length), behavior),
                loss_mask=torch.tensor([[0] + [1] * (length - 1)]),
                rewards=torch.tensor([1.]), advantages=torch.tensor([advantage]))


@pytest.fixture
def finalized(monkeypatch):
    module = ModuleType("megatron.core.distributed")
    module.finalize_model_grads = lambda _: None
    monkeypatch.setitem(sys.modules, module.__name__, module)


def trainable(monkeypatch, micro_batch_size=1):
    obj = engine(micro_batch_size=micro_batch_size)
    parameter = torch.nn.Parameter(torch.tensor(-1.1))
    optim = torch.optim.SGD([parameter], lr=.01)
    obj.model = []
    obj._set_train_mode = lambda enabled: None
    obj._forward = lambda batch: parameter.expand(batch["input_ids"].shape[0], batch["input_ids"].shape[1] - 1)
    obj._action_log_probs = lambda logits, ids, positions=None: logits if positions is None else logits[positions]
    obj._zero_grad = lambda: optim.zero_grad(set_to_none=True)
    obj._model_main_gradients = lambda: [parameter.grad]
    obj._apply_elastic_inter_replica_gradients = lambda: None
    obj._reduce_stats = lambda stats: stats
    obj.optimizer = NS(scale_loss=lambda value: value,
                       step=lambda: (optim.step() or True, 0., None))
    return obj, parameter


def real_loss(current, anchor, behavior, advantages, mask):
    obj = engine()
    obj._action_log_probs = lambda logits, ids, positions: current[positions]
    pad = lambda tensor: torch.nn.functional.pad(tensor, (1, 0))
    return obj._loss(dict(input_ids=torch.zeros(current.shape[0], current.shape[1] + 1, dtype=torch.long),
        old_logprobs=pad(anchor), behavior_logprobs=pad(behavior), loss_mask=pad(mask),
        advantages=advantages, rewards=torch.ones(current.shape[0])), None)


@pytest.mark.parametrize("advantage", [-1.2, 1.7])
def test_raw_weights_correct_both_pg_and_kl_without_backpropagating_to_policies(advantage):
    current = torch.tensor([[-.7, -1.4]], dtype=torch.float64, requires_grad=True)
    anchor = torch.tensor([[-.8, -1.2]], dtype=torch.float64, requires_grad=True)
    behavior = torch.tensor([[-1.5, -.9]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[math.exp(.7), math.exp(-.3)]], dtype=torch.float64)
    ratio = torch.exp(current - anchor.detach())
    expected_pg = (weights * torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.2))).mean()
    expected_kl = (weights * ((ratio - 1) - (current - anchor.detach()))).mean()
    actual, stats = real_loss(current, anchor, behavior,
        torch.tensor([advantage], dtype=torch.float64), torch.ones_like(current))
    expected = expected_pg + .1 * expected_kl
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(expected, current, retain_graph=True)[0],
                               torch.autograd.grad(actual, current, retain_graph=True)[0])
    actual.backward()
    assert anchor.grad is behavior.grad is None
    assert stats["pg_loss"] == pytest.approx(float(expected_pg))
    assert stats["kl"] == pytest.approx(float(expected_kl))


@pytest.mark.parametrize("micro_batch_size", [1, 2, 3])
def test_unit_weights_match_real_legacy_loss_and_microbatch_gradient(micro_batch_size):
    obj = engine(micro_batch_size=micro_batch_size)
    parameter = torch.tensor(-1.1, requires_grad=True)
    rows = [trajectory(length=length, advantage=adv) for length, adv in [(3, 1.), (7, -.4), (4, 2.)]]
    for row in rows:
        row["logprobs"] = row["behavior_logprobs"].clone()
    obj._action_log_probs = lambda logits, ids, positions: parameter.expand(
        ids.shape[0], ids.shape[1] - 1)[positions]
    losses = []
    for objective in ("decoupled", "legacy"):
        obj.grpo_objective = objective
        loss = sum(obj._loss(batch, None)[0] * (len(batch["trajectories"]) / len(rows))
                   for batch in obj._iter_micro_batches(rows))
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(losses[0], parameter, retain_graph=True)[0],
                               torch.autograd.grad(losses[1], parameter)[0], rtol=0, atol=0)


def test_reduction_gives_equal_weight_to_different_length_trajectories():
    current = torch.full((2, 3), -1.)
    behavior = current.clone()
    behavior[0, 0] -= math.log(2)
    _, stats = real_loss(current, current, behavior, torch.ones(2),
        torch.tensor([[1., 0., 0.], [1., 1., 1.]]))
    assert stats["pg_loss"] == pytest.approx(-1.5)


@pytest.mark.parametrize("current,anchor", [(-1., -25.), (-30., -1.)])
def test_extreme_proximal_ratio_keeps_main_clamp_and_gradient(current, anchor):
    obj = engine()
    parameter = torch.tensor([[current]], requires_grad=True)
    obj._action_log_probs = lambda logits, ids, positions: parameter[positions]
    reference = torch.tensor([[0., anchor]])
    batch = dict(input_ids=torch.tensor([[1, 2]]), old_logprobs=reference,
                 behavior_logprobs=reference,
                 loss_mask=torch.tensor([[0., 1.]]), advantages=torch.tensor([-1.]), rewards=torch.tensor([1.]))
    corrected, _ = obj._loss(batch, None)
    obj.grpo_objective = "legacy"
    original, _ = obj._loss(batch, None)
    torch.testing.assert_close(corrected, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(corrected, parameter, retain_graph=True)[0],
                               torch.autograd.grad(original, parameter)[0], rtol=0, atol=0)


def test_masked_nan_logits_have_no_value_or_gradient():
    obj = engine()
    logits = torch.tensor([[[0., 0.], [float("nan"), 0.], [0., 0.]]], requires_grad=True)
    reference = torch.tensor([[0., -math.log(2), float("nan")]])
    batch = dict(input_ids=torch.tensor([[0, 1, 0]]), old_logprobs=reference,
        behavior_logprobs=reference, loss_mask=torch.tensor([[0., 1., 0.]]),
        advantages=torch.ones(1), rewards=torch.ones(1))
    loss, _ = obj._loss(batch, logits)
    loss.backward()
    assert loss.item() == -1.
    torch.testing.assert_close(logits.grad, torch.tensor([[[.5, -.5], [0., 0.], [0., 0.]]]))


@pytest.mark.parametrize("anchor,behavior", [(-1., -1000.), (-1000., -1.)])
def test_importance_overflow_or_underflow_is_not_clipped(anchor, behavior):
    with pytest.raises(FloatingPointError, match="overflowed or underflowed"):
        real_loss(torch.tensor([[-1.]]), torch.tensor([[anchor]]),
            torch.tensor([[behavior]]), torch.ones(1), torch.ones(1, 1))


@pytest.mark.parametrize("current", [1e-3, float("nan"), float("inf"), -float("inf")])
def test_new_current_is_rejected_before_backward(current):
    value = torch.tensor([[current]], requires_grad=True)
    with pytest.raises(FloatingPointError, match="current logprobs"):
        real_loss(value, torch.tensor([[-1.]]), torch.tensor([[-1.]]),
            torch.ones(1), torch.ones(1, 1))
    assert value.grad is None


def test_nonfinite_loss_with_finite_current_and_raw_weights_is_rejected():
    current = torch.tensor([[-1e20]], requires_grad=True)
    with pytest.raises(FloatingPointError, match="loss is nonfinite"):
        real_loss(current, torch.tensor([[-1.]]), torch.tensor([[-70.]]),
            torch.ones(1), torch.ones(1, 1))
    assert current.grad is None


@pytest.mark.parametrize("mutation", ["missing", "truncated", "nan", "mask"])
def test_invalid_behavior_is_rejected(mutation):
    row = trajectory()
    if mutation == "missing":
        row.pop("behavior_logprobs")
    elif mutation == "truncated":
        row["behavior_logprobs"] = row["behavior_logprobs"][:, :-1]
    elif mutation == "nan":
        row["behavior_logprobs"][0, 1] = float("nan")
    else:
        row["loss_mask"][0, 1] = 2
    with pytest.raises((ValueError, FloatingPointError)):
        validate_behavior_rows([row])


def test_recompute_preserves_behavior_and_replaces_only_anchor(monkeypatch):
    obj, parameter = trainable(monkeypatch)
    row = trajectory()
    original = row["behavior_logprobs"]
    row.pop("logprobs")
    forwarded = []
    original_forward = obj._forward
    def forward(batch):
        forwarded.append(set(batch))
        return original_forward(batch)
    obj._forward = forward
    obj.optimizer_revision = 7
    obj.recompute_logprobs([row])
    assert row["behavior_logprobs"] is original
    assert not any({"old_logprobs", "anchor_logprobs", "behavior_logprobs"} & keys for keys in forwarded)
    assert "anchor_logprobs" not in row
    torch.testing.assert_close(row["logprobs"][:, 1:], parameter.detach().expand(1, 3))
    assert row["anchor_optimizer_revision"] == 7
    validate_anchor_rows([row], 7)
    with pytest.raises(ValueError, match="revision"):
        validate_anchor_rows([row], 8)


def test_alignment_keeps_original_sampler_storage_and_packs_probabilities(monkeypatch):
    obj = engine()
    obj.world_size = 2
    obj.sequence_parallel, obj.train_tp_size = True, 4
    row = trajectory(length=5)
    row["logprobs"] = torch.full((1, 5), -1.)
    original = row["behavior_logprobs"]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda lengths, **kwargs: lengths.fill_(7))
    obj.align_distributed_trajectories([row])
    assert row["input_ids"].shape == (1, 8)
    assert row["behavior_logprobs"] is original and original.shape == (1, 5)
    packed = obj._pad_micro_batch([row])
    for field in ("old_logprobs", "behavior_logprobs", "loss_mask"):
        assert packed[field].shape == (1, 8)
        assert packed[field][0, 5:].tolist() == [0., 0., 0.]
    short = obj._pad_micro_batch([trajectory() | dict(logprobs=torch.full((1, 4), -1.))])
    short["input_ids"] = torch.tensor([[0, 1, 2, 3, 4]])
    for field in ("old_logprobs", "behavior_logprobs", "loss_mask"):
        short[field] = torch.nn.functional.pad(short[field], (0, 1))
    obj._pad_micro_batch_for_sequence_parallel(short)
    assert all(short[field].shape == (1, 8) for field in ("input_ids", "old_logprobs", "behavior_logprobs", "loss_mask"))


def test_new_loss_recomputes_anchor_without_producer_logprobs(monkeypatch):
    obj, _ = trainable(monkeypatch)
    row = trajectory()
    row.pop("logprobs")
    obj.recompute_logprobs([row])
    batch = obj._pad_micro_batch([row])
    assert torch.isfinite(obj._loss(batch, obj._forward(batch))[0])


@pytest.mark.parametrize("hybrids", [(), ("hybrid",)])
def test_core_and_hybrid_reject_multiple_epochs_before_forward(monkeypatch, finalized, hybrids):
    obj, parameter = trainable(monkeypatch)
    rows = [trajectory()]
    obj.recompute_logprobs(rows)
    obj._elastic_step_hybrid_workers = hybrids
    obj._forward = lambda _: pytest.fail("invalid epoch count reached forward")
    with pytest.raises(ValueError, match="ppo_epochs=1"):
        obj.grpo_update(rows, ppo_epochs=2)
    assert obj.optimizer_revision == 0 and parameter.grad is None


def test_anchor_is_fixed_for_update_and_stale_reuse_fails(monkeypatch, finalized):
    obj, _ = trainable(monkeypatch)
    rows = [trajectory(length=length) for length in (3, 5)]
    obj.recompute_logprobs(rows)
    anchors = [row["logprobs"].clone() for row in rows]
    obj.grpo_update(rows)
    assert obj.optimizer_revision == 1
    for row, anchor in zip(rows, anchors):
        torch.testing.assert_close(row["logprobs"], anchor)
    with pytest.raises(ValueError, match="revision"):
        obj.grpo_update(rows)


def test_core_and_hybrid_payloads_keep_the_same_fixed_batch_gradient(monkeypatch, finalized):
    rows = [trajectory(length=3 + index % 4, advantage=(index - 3) / 5,
                       behavior=-1.3 - (index % 3) * .2) for index in range(8)]
    obj, parameter = trainable(monkeypatch)
    obj.recompute_logprobs(rows)
    original_behaviors = [row["behavior_logprobs"].clone() for row in rows]
    captured = []
    obj.optimizer.step = lambda: (captured.append(parameter.grad.clone()) or True, 0., None)
    obj.grpo_update(rows)
    combined = torch.zeros_like(parameter)
    worker, _ = trainable(monkeypatch)
    for start, count in ((0, 2), (2, 3), (5, 3)):
        shard = rows[start:start + count]
        worker.recompute_logprobs(shard)
        payload = worker.compute_elastic_gradient_payload(shard, worker_id=f"h{start}",
            target_core_id="dp0", step=0, state_version=0, membership_epoch=1, global_sample_count=8)
        combined += payload.tensors[0] * count / 8
        assert payload.sample_count == count and payload.global_sample_count == 8
    torch.testing.assert_close(combined, captured[0])
    for row, original in zip(rows, original_behaviors):
        torch.testing.assert_close(row["behavior_logprobs"], original)
    rows[0]["anchor_optimizer_revision"] = 4
    with pytest.raises(ValueError, match="revision"):
        worker.compute_elastic_gradient_payload(rows[:1], worker_id="h", target_core_id="dp0",
            step=0, state_version=0, membership_epoch=1)


def test_constructor_rejects_unknown_objective_before_initialization():
    with pytest.raises(ValueError, match="grpo_objective"):
        MegatronCoreTrainEngine(model_path="unused", grpo_objective="unknown")


def test_default_loss_returns_only_training_statistics_with_one_status_read(monkeypatch):
    reads = []
    original = torch.Tensor.tolist
    def read(tensor):
        if tensor.dtype is torch.bool:
            reads.append(tensor.numel())
        return original(tensor)
    monkeypatch.setattr(torch.Tensor, 'tolist', read)
    current = torch.tensor([[-1.]], requires_grad=True)
    loss, stats = real_loss(current, current.detach(), current.detach(), torch.ones(1), torch.ones(1, 1))
    assert set(stats) == {'loss', 'pg_loss', 'kl', 'reward_sum', 'n_samples', 'n_updates'}
    assert reads == [3]
    loss.backward()
    assert current.grad.tolist() == [[-1.]]


def test_microbatch_two_unequal_shards_match_hand_oracle_after_core_mean_and_elastic_sum(monkeypatch, finalized):
    # Five samples: two Core DP shards of 2/1 samples, one Hybrid shard of 2.
    # Action counts are 2,4 / 3 / 5,1. Each shard has microbatch_size=2.
    # With current=anchor, the trajectory mean losses are -.8, -1, -.405;
    # the sample-weighted global loss and derivative are both -.682.
    specs = [(3, 1., 2.), (5, -.4, 1.), (4, 2., .5), (6, .7, 1.5), (2, -.3, .8)]
    rows = [trajectory(length, advantage, behavior=-1.1 - math.log(weight))
            for length, advantage, weight in specs]
    local_losses, core_gradients = [], []
    for start, count in ((0, 2), (2, 1)):
        obj, parameter = trainable(monkeypatch, micro_batch_size=2)
        shard = rows[start:start + count]
        obj.recompute_logprobs(shard)
        obj.get_data_parallel_world_size = lambda: 2
        obj.set_elastic_batch_partition(count, 5, {'hybrid': 2})
        loss = sum(obj._loss(batch, obj._forward(batch))[0] * len(batch['trajectories']) / count
                   for batch in obj._iter_micro_batches(shard))
        local_losses.append(loss.detach() * count / 5)
        loss = loss * obj._elastic_core_loss_scale()
        core_gradients.append(torch.autograd.grad(loss, parameter)[0])
    worker, parameter = trainable(monkeypatch, micro_batch_size=2)
    shard = rows[3:]
    worker.recompute_logprobs(shard)
    batch = next(worker._iter_micro_batches(shard))
    local_losses.append(worker._loss(batch, worker._forward(batch))[0].detach() * 2 / 5)
    payload = worker.compute_elastic_gradient_payload(shard, worker_id='hybrid', target_core_id='dp0',
        step=0, state_version=0, membership_epoch=1, global_sample_count=5)
    # Finalize Core DP MEAN, then remove duplicate Core copies before elastic SUM.
    core_mean = sum(core_gradients) / 2
    elastic_sum = core_mean / 2 + core_mean / 2 + payload.tensors[0] * 2 / 5
    assert float(sum(local_losses)) == pytest.approx(-.682)
    assert float(elastic_sum) == pytest.approx(-.682)


def test_legacy_keeps_existing_fractional_mask_weights():
    obj = engine(objective='legacy')
    current = torch.tensor([[-1., -1.]], requires_grad=True)
    obj._action_log_probs = lambda logits, ids, positions: current[positions]
    batch = dict(input_ids=torch.tensor([[0, 1, 2]]), old_logprobs=torch.tensor([[0., -1., -1.]]),
        loss_mask=torch.tensor([[0., .25, 1.]]), advantages=torch.ones(1), rewards=torch.ones(1))
    loss, _ = obj._loss(batch, None)
    assert float(loss) == pytest.approx(-1.)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-.2, -.8]]))

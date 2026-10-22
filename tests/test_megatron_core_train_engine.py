from types import SimpleNamespace
import sys
import weakref

import pytest
import torch

from RL_Framework.engine.megatron_core_train_engine import (
    MegatronCoreTrainEngine,
)


def test_merge_stats_weights_losses_by_samples_and_grad_norm_by_updates():
    merged = MegatronCoreTrainEngine._merge_stats(
        [
            {
                "loss": 1.0,
                "pg_loss": 2.0,
                "kl": 0.1,
                "reward_sum": 2.0,
                "n_samples": 2.0,
                "n_updates": 1.0,
                "grad_norm": 3.0,
            },
            {
                "loss": 3.0,
                "pg_loss": 4.0,
                "kl": 0.3,
                "reward_sum": 4.0,
                "n_samples": 2.0,
                "n_updates": 3.0,
                "grad_norm": 5.0,
            },
        ]
    )

    assert merged["loss"] == pytest.approx(2.0)
    assert merged["pg_loss"] == pytest.approx(3.0)
    assert merged["kl"] == pytest.approx(0.2)
    assert merged["grad_norm"] == pytest.approx(4.5)
    assert merged["reward_mean"] == pytest.approx(1.5)


def test_loss_uses_non_negative_ppo_kl_approximation(monkeypatch):
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.clip_epsilon = 0.2
    engine.kl_coef = 0.1
    new_logprobs = torch.tensor([[0.3, -0.4]])
    monkeypatch.setattr(
        engine,
        "_action_log_probs",
        lambda logits, input_ids: new_logprobs,
    )
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "old_logprobs": torch.tensor([[0.0, 0.1, -0.2]]),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "advantages": torch.tensor([1.0]),
        "rewards": torch.tensor([1.0]),
    }

    _, stats = engine._loss(micro_batch, torch.empty(0))

    assert stats["kl"] >= 0.0


def test_sequence_parallel_padding_updates_training_tensors():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.sequence_parallel = True
    engine.train_tp_size = 4
    engine.tokenizer = SimpleNamespace(pad_token_id=99)
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "old_logprobs": torch.ones((1, 5)),
        "loss_mask": torch.ones((1, 5)),
    }

    padded = engine._pad_micro_batch_for_sequence_parallel(micro_batch)

    assert padded["input_ids"].shape == (1, 8)
    assert padded["old_logprobs"].shape == (1, 8)
    assert padded["loss_mask"].shape == (1, 8)
    assert padded["input_ids"].tolist()[0][-3:] == [99, 99, 99]
    assert padded["old_logprobs"].tolist()[0][-3:] == [0.0, 0.0, 0.0]
    assert padded["loss_mask"].tolist()[0][-3:] == [0.0, 0.0, 0.0]


def test_conversion_coverage_rejects_unmapped_local_parameters():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.model = [object()]
    engine.bridge = SimpleNamespace(
        get_conversion_tasks=lambda model: [
            SimpleNamespace(
                param_name="decoder.layers.0.mlp.experts.weight",
                megatron_module=None,
                mapping=None,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="1 local parameters unmapped"):
        engine._validate_conversion_coverage()


def test_megatron_core_device_uses_cuda():
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.local_rank = 3

    assert str(engine._device()) == "cuda:3"


def test_initialize_elastic_domain_creates_distinct_local_synchronized_group(
    monkeypatch,
):
    from RL_Framework.engine import megatron_core_train_engine as module

    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine._elastic_gradient_process_group = None
    engine._elastic_gradient_process_group_owned = False
    engine._elastic_gradient_group_ranks = ()
    core_group = object()
    elastic_group = object()
    calls = []
    monkeypatch.setattr(engine, "get_elastic_core_process_group", lambda: core_group)
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        module.dist,
        "get_process_group_ranks",
        lambda group: [0, 8, 16, 24] if group is core_group else [],
        raising=False,
    )
    monkeypatch.setattr(module.dist, "get_backend", lambda group: "nccl")

    def new_group(**kwargs):
        calls.append(kwargs)
        return elastic_group

    monkeypatch.setattr(module.dist, "new_group", new_group)

    created = engine.initialize_elastic_communication_domain()

    assert created is elastic_group
    assert created is not core_group
    assert calls == [
        {
            "ranks": [0, 8, 16, 24],
            "backend": "nccl",
            "use_local_synchronization": True,
        }
    ]
    assert engine._elastic_gradient_group_ranks == (0, 8, 16, 24)


def test_recompute_releases_logits_before_next_forward_and_preserves_logprobs(monkeypatch):
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.train_tp_size = 1
    weights = torch.arange(49, dtype=torch.float32).reshape(7, 7) / 10
    rows = [dict(input_ids=torch.tensor([[0, 2, 3, 1]]), logprobs=torch.zeros(1, 4)),
        dict(input_ids=torch.tensor([[4, 1, 5, 6]]), logprobs=torch.zeros(1, 4))]
    outputs, modes = [], []
    def forward(batch):
        assert all(output() is None for output in outputs)
        logits = weights[batch['input_ids']]
        outputs.append(weakref.ref(logits))
        return logits
    monkeypatch.setattr(engine, '_forward', forward)
    monkeypatch.setattr(engine, '_set_train_mode', modes.append)
    monkeypatch.setattr(engine, '_iter_recompute_micro_batches',
        lambda trajectories: (dict(input_ids=row['input_ids'], trajectories=[row]) for row in trajectories))
    engine.recompute_logprobs(rows)
    assert all(output() is None for output in outputs)
    assert modes == [False, True]
    for row in rows:
        expected = torch.log_softmax(weights[row['input_ids']][:, :-1], dim=-1)
        expected = expected.gather(-1, row['input_ids'][:, 1:, None]).squeeze(-1)
        torch.testing.assert_close(row['logprobs'][:, 1:], expected)
        assert row['logprobs'][0, 0] == 0



def test_grpo_releases_backward_outputs_before_next_microbatch_and_optimizer(monkeypatch):
    monkeypatch.setitem(sys.modules, 'megatron.core.distributed',
        SimpleNamespace(finalize_model_grads=lambda _: None))
    engine = MegatronCoreTrainEngine.__new__(MegatronCoreTrainEngine)
    engine.train_tp_size = 1
    engine.clip_epsilon, engine.kl_coef = .2, .1
    engine.current_version, engine.optimizer_revision = 3, 0
    initial = torch.arange(49, dtype=torch.float32).reshape(7, 7) / 10
    parameter = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.SGD([parameter], lr=.01)
    reference_optimizer = torch.optim.SGD([reference], lr=.01)
    rows = [dict(input_ids=torch.tensor([[0, 2, 3, 1]])),
        dict(input_ids=torch.tensor([[4, 1, 5, 6]]))]
    def batches(trajectories):
        for row in trajectories:
            yield dict(input_ids=row['input_ids'], old_logprobs=torch.full((1, 4), -2.),
                loss_mask=torch.tensor([[0., 1., 1., 1.]]), advantages=torch.tensor([.7]),
                rewards=torch.tensor([1.]), trajectories=[row])
    references = []
    def forward(batch):
        assert all(output() is None for output in references)
        logits = parameter[batch['input_ids']]
        references.append(weakref.ref(logits))
        return logits
    gradients = []
    def step():
        assert all(output() is None for output in references)
        gradients.append(parameter.grad.clone())
        optimizer.step()
        return True, parameter.grad.norm(), 0
    def scale_loss(value):
        assert all(output() is None for output in references)
        return value
    engine.model = []
    engine.optimizer = SimpleNamespace(scale_loss=scale_loss, step=step)
    monkeypatch.setattr(engine, '_iter_micro_batches', batches)
    monkeypatch.setattr(engine, '_forward', forward)
    monkeypatch.setattr(engine, '_zero_grad', lambda: optimizer.zero_grad(set_to_none=True))
    monkeypatch.setattr(engine, '_elastic_core_loss_scale', lambda: 1.)
    monkeypatch.setattr(engine, '_apply_elastic_inter_replica_gradients', lambda: None)
    monkeypatch.setattr(engine, '_reduce_stats', lambda stats: stats)
    engine.grpo_update(rows, ppo_epochs=2)
    for gradient in gradients:
        reference_optimizer.zero_grad(set_to_none=True)
        for batch in batches(rows):
            loss, _ = engine._loss(batch, reference[batch['input_ids']])
            (loss / len(rows)).backward()
        torch.testing.assert_close(gradient, reference.grad)
        reference_optimizer.step()
    torch.testing.assert_close(parameter, reference)
    assert engine.optimizer_revision == 2
    assert all(output() is None for output in references)

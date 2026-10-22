"""Rollback publication targets only newly restored services, preserving loans."""
from types import SimpleNamespace as NS

import pytest

from RL_Framework.infra.elastic.core_reconfiguration import _run_operation
from RL_Framework.infra.elastic.runtime_executor import RuntimeElasticExecutor


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('preexisting', [False, True])
def test_subset_publication_restores_temporary_scope_and_keeps_old_loan(fail, preexisting):
    borrowed = {'old-hybrid-engine'}
    unavailable = borrowed | {'returned-engine'}
    topology = object()
    engine = NS(optimizer_revision=7)
    trainer = NS(config=topology, _rollout_unavailable_ids=unavailable,
        rollout_engine=NS(scheduler=NS(_instances=[])))
    previous_targets, previous_pending = {'previous-target'}, {'previous-pending'}
    if preexisting:
        trainer._publication_target_ids = previous_targets
        trainer._restored_rollouts_pending_publication = previous_pending
    observed = []

    def sync(*, target_version):
        observed.append((target_version, set(trainer._publication_target_ids),
            set(trainer._restored_rollouts_pending_publication)))
        if fail:
            raise RuntimeError('receiver failed')
        # The real NCCL trainer invokes this helper after sender ACK.
        from RL_Framework.infra.elastic.rollout_lifecycle import reopen_after_publication
        reopen_after_publication(trainer)

    trainer._sync_weights = sync
    command = dict(operation='publish', target=dict(scope='returned_subset',
        instances=[dict(instance_id='returned-engine')]))
    if fail:
        with pytest.raises(RuntimeError, match='receiver failed'):
            _run_operation(engine, command, trainer)
    else:
        assert _run_operation(engine, command, trainer)['acknowledged']
    assert observed == [(7, {'returned-engine'}, {'returned-engine'})]
    assert trainer.config is topology
    assert trainer._rollout_unavailable_ids is unavailable
    if preexisting:
        assert trainer._publication_target_ids is previous_targets
        assert trainer._restored_rollouts_pending_publication is previous_pending
    else:
        assert not hasattr(trainer, '_publication_target_ids')
        assert not hasattr(trainer, '_restored_rollouts_pending_publication')


def test_runtime_subset_targets_are_selected_from_current_configuration(monkeypatch):
    from RL_Framework.infra.elastic import core_reconfiguration as commands
    runtime = object.__new__(RuntimeElasticExecutor)
    instances = [NS(instance_id=name, tp=1, gpus=[index], host='node', port=8000+index)
        for index, name in enumerate(('old-hybrid-engine', 'returned-engine', 'unaffected-engine'))]
    runtime.config = NS(heterogeneous_rollout=NS(instances=instances))
    calls = []
    monkeypatch.setattr(commands, 'dispatch_core_command',
        lambda *args, **kwargs: calls.append((args, kwargs)))
    runtime._publish_current_rollout_weights(instance_ids=['returned-engine'])
    target = calls[0][1]['target']
    assert target['scope'] == 'returned_subset'
    assert [entry['instance_id'] for entry in target['instances']] == ['returned-engine']
    with pytest.raises(ValueError, match='unknown instances'):
        runtime._publish_current_rollout_weights(instance_ids=['no-longer-configured'])
    assert len(calls) == 1

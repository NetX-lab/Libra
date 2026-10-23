from types import SimpleNamespace
import threading

import torch

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.engine.megatron_train_engine import Megatron3DTrainEngine
from RL_Framework.engine.train_engine import FSDPTrainEngine
from RL_Framework.infra.elastic.hybrid_pool import GradientPayload, InterReplicaGradientDomain


class _UnexpectedReduce:
    def reduce_core_gradients(self, **_kwargs):
        raise AssertionError("elastic reduce must not run without a hybrid payload")


def test_mcore_skips_elastic_collective_when_domain_is_not_configured():
    engine = object.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = None
    engine._pending_hybrid_gradients = []

    engine._apply_elastic_inter_replica_gradients()


def test_mcore_dp1_reuses_main_gradient_for_weighted_hybrid_update():
    domain = InterReplicaGradientDomain(core_replica_ids=['dp0'])
    domain.request_join('hybrid', 'dp0', membership_epoch=1)
    domain.mark_active('hybrid', membership_epoch=1)
    engine = object.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = domain
    engine._pending_hybrid_gradients = [GradientPayload(
        'hybrid', 'dp0', (torch.full((73,), 7.),),
        step=3, state_version=2, membership_epoch=1,
        sample_count=2, global_sample_count=4,
    )]
    engine._elastic_step_hybrid_workers = ('hybrid',)
    engine._elastic_active_gradient_timeout_s = 0
    engine._hybrid_gradient_condition = threading.Condition()
    engine._elastic_batch_partition = (2, 4, {'hybrid': 2})
    engine._elastic_training_step = 3
    engine._elastic_expected_state_version = 2
    # This is the already weighted, finalized Core contribution. Passing it
    # through an out-of-place divide-by-one would waste a full model buffer.
    grad = torch.full((73,), 3.)
    engine._model_main_gradients = lambda: [grad]
    engine.get_data_parallel_rank = lambda: 0
    reduce = domain.reduce_core_gradients

    def checked_reduce(**kwargs):
        assert kwargs['in_place'] is True
        assert kwargs['core_gradients']['dp0'][0].data_ptr() == grad.data_ptr()
        return reduce(**kwargs)

    domain.reduce_core_gradients = checked_reduce
    acknowledgements = []
    engine._elastic_gradient_update_callback = acknowledgements.append
    engine._apply_elastic_inter_replica_gradients()
    assert torch.equal(grad, torch.full((73,), 6.5))
    assert not engine._pending_hybrid_gradients
    assert len(acknowledgements) == 1
    assert acknowledgements[0].tensors[0].data_ptr() != grad.data_ptr()
    assert acknowledgements[0].state_version == 3
    grad.zero_()
    assert torch.equal(acknowledgements[0].tensors[0], torch.full((73,), 6.5))


def test_megatron3d_skips_elastic_collective_without_hybrid_payload():
    engine = object.__new__(Megatron3DTrainEngine)
    engine.elastic_gradient_domain = _UnexpectedReduce()
    engine.runtime = SimpleNamespace()
    engine._pending_hybrid_gradients = []

    engine._apply_elastic_inter_replica_gradients()


def test_fsdp_skips_elastic_collective_without_hybrid_payload():
    engine = object.__new__(FSDPTrainEngine)
    engine.elastic_gradient_domain = _UnexpectedReduce()
    engine._pending_hybrid_gradients = []

    engine._apply_elastic_inter_replica_gradients()

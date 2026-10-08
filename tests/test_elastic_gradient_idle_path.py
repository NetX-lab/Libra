from types import SimpleNamespace

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine
from RL_Framework.engine.megatron_train_engine import Megatron3DTrainEngine
from RL_Framework.engine.train_engine import FSDPTrainEngine


class _UnexpectedReduce:
    def reduce_core_gradients(self, **_kwargs):
        raise AssertionError("elastic reduce must not run without a hybrid payload")


def test_mcore_skips_elastic_collective_without_hybrid_payload():
    engine = object.__new__(MegatronCoreTrainEngine)
    engine.elastic_gradient_domain = _UnexpectedReduce()
    engine._pending_hybrid_gradients = []

    engine._apply_elastic_inter_replica_gradients()


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

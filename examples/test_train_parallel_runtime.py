"""Support code for Test train parallel runtime."""

import os
import sys
import types

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("RL_Framework")
_pkg.__path__ = [_project_root]
_pkg.__package__ = "RL_Framework"
sys.modules.setdefault("RL_Framework", _pkg)

from RL_Framework.engine.train_parallel import TrainParallelRuntime


def main():
    runtime = TrainParallelRuntime(
        rank=5,
        world_size=8,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2,
    )

    assert runtime.data_parallel_rank == 1
    assert runtime.pipeline_parallel_rank == 0
    assert runtime.tensor_parallel_rank == 1
    assert runtime.get_model_replica_ranks() == [4, 5, 6, 7]
    assert runtime.get_tp_ranks(dp_rank=1, pp_rank=0) == [4, 5]
    assert runtime.get_pp_ranks(dp_rank=1, tp_rank=1) == [5, 7]
    assert runtime.get_dp_ranks(pp_rank=1, tp_rank=0) == [2, 6]

    print("TrainParallelRuntime mapping test passed")


if __name__ == "__main__":
    main()

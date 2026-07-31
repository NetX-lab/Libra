from types import SimpleNamespace

from RL_Framework.engine.megatron_core_attention import (
    TorchSequenceParallelNorm,
)


def test_torch_norm_marks_replicated_parameters_for_sequence_parallel():
    config = SimpleNamespace(
        normalization="RMSNorm",
        sequence_parallel=True,
        layernorm_zero_centered_gamma=False,
        persist_layer_norm=False,
        memory_efficient_layer_norm=False,
    )

    norm = TorchSequenceParallelNorm(config, hidden_size=16, eps=1e-6)

    assert norm.weight.shape == (16,)
    assert norm.weight.sequence_parallel is True

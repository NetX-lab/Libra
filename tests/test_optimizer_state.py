"""Strict Adam metadata restoration must also preserve the next optimizer step."""
from types import SimpleNamespace

import pytest
import torch

from RL_Framework.engine.optimizer_state import optimizer_state_load_context


def test_restored_parameter_steps_do_not_share_mutable_metadata_tensor():
    restored = [torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))]
    reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in restored]
    actual = torch.optim.AdamW(restored, lr=.01)
    expected = torch.optim.AdamW(reference, lr=.01)
    shared = torch.tensor(3.)
    for optimizer, parameters in [(actual, restored), (expected, reference)]:
        for parameter in parameters:
            optimizer.state[parameter] = {
                'step': shared if optimizer is actual else shared.clone(),
                'exp_avg': torch.tensor([.1]), 'exp_avg_sq': torch.tensor([.2]),
            }
    wrapper = SimpleNamespace(optimizer=actual, model_param_group_index_map={
        parameter: (0, index) for index, parameter in enumerate(restored)})
    received = []
    wrapper._set_main_param_and_optimizer_states = lambda parameter, tensors: received.append(tensors)
    original = wrapper._set_main_param_and_optimizer_states
    with optimizer_state_load_context(wrapper):
        wrapper._set_main_param_and_optimizer_states(restored[0], {'exp_avg': torch.tensor([.1])})
    assert received[0]['step'].item() == 3
    assert wrapper._set_main_param_and_optimizer_states is original
    assert actual.state[restored[0]]['step'] is not actual.state[restored[1]]['step']
    for optimizer, parameters in [(actual, restored), (expected, reference)]:
        for parameter in parameters:
            parameter.grad = torch.tensor([.3])
        optimizer.step()
    for actual_parameter, expected_parameter in zip(restored, reference):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
        assert actual.state[actual_parameter]['step'].item() == 4


def test_missing_master_or_moment_state_is_not_invented():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter]['step'] = torch.tensor(3.)
    def strict_setter(parameter, tensors):
        return tensors['exp_avg']
    wrapper = SimpleNamespace(optimizer=optimizer, model_param_group_index_map={parameter: (0, 0)},
                              _set_main_param_and_optimizer_states=strict_setter)
    with pytest.raises(KeyError, match='exp_avg'):
        with optimizer_state_load_context(wrapper):
            wrapper._set_main_param_and_optimizer_states(parameter, {})
    assert wrapper._set_main_param_and_optimizer_states is strict_setter

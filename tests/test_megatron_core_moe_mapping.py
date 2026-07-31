import torch

from RL_Framework.engine.megatron_core_moe_mapping import (
    pack_grouped_weight1,
    pack_grouped_weight2,
)


def test_pack_grouped_weight1_preserves_expert_major_layout():
    gate0 = torch.arange(24).reshape(6, 4)
    up0 = gate0 + 100
    gate1 = gate0 + 200
    up1 = gate0 + 300

    packed = pack_grouped_weight1(
        [(gate0, up0), (gate1, up1)],
        etp_rank=1,
        etp_size=2,
    )
    blocks = packed.view(2, 4, 6)

    assert torch.equal(blocks[0, :, :3], gate0[3:].t())
    assert torch.equal(blocks[0, :, 3:], up0[3:].t())
    assert torch.equal(blocks[1, :, :3], gate1[3:].t())
    assert torch.equal(blocks[1, :, 3:], up1[3:].t())


def test_pack_grouped_weight2_preserves_expert_major_layout():
    down0 = torch.arange(24).reshape(4, 6)
    down1 = down0 + 100

    packed = pack_grouped_weight2(
        [down0, down1],
        etp_rank=0,
        etp_size=2,
    )
    blocks = packed.view(2, 3, 4)

    assert torch.equal(blocks[0], down0[:, :3].t())
    assert torch.equal(blocks[1], down1[:, :3].t())

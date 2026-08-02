import threading

import torch

from RL_Framework.infra.elastic.gradient_ipc import (
    ElasticGradientClient,
    ElasticGradientServer,
    GradientUpdate,
)
from RL_Framework.infra.elastic.hybrid_pool import GradientPayload


def test_bidirectional_gradient_exchange_returns_post_reduce_update():
    received = []
    payload_ready = threading.Event()

    def on_payload(payload):
        received.append(payload)
        payload_ready.set()

    server = ElasticGradientServer(
        host="127.0.0.1",
        port=0,
        authkey="secret",
        on_payload=on_payload,
        update_timeout=2.0,
    )
    endpoint = server.start()
    payload = GradientPayload(
        replica_id="hybrid0",
        target_core_id="dp0",
        tensors=(torch.tensor([2.0]),),
        step=7,
        state_version=7,
        membership_epoch=3,
    )
    result = {}

    def send_gradient():
        result["update"] = ElasticGradientClient(endpoint, timeout=2.0).send(
            payload,
            expect_update=True,
        )

    thread = threading.Thread(target=send_gradient)
    thread.start()
    assert payload_ready.wait(timeout=1.0)
    server.publish_update(
        GradientUpdate(
            replica_id="hybrid0",
            tensors=(torch.tensor([5.0]),),
            step=7,
            state_version=8,
            membership_epoch=3,
        )
    )
    thread.join(timeout=2.0)
    server.close()

    assert not thread.is_alive()
    assert len(received) == 1
    update = result["update"]
    assert update.state_version == 8
    assert torch.equal(update.tensors[0], torch.tensor([5.0]))

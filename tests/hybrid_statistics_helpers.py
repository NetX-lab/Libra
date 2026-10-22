"""Small CPU model fixture shared by payload-data and activation-integration tests."""
import sys
import threading
from types import SimpleNamespace as NS

import torch

from RL_Framework.engine import megatron_core_train_engine as module
from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine as Engine
from RL_Framework.infra.elastic.hybrid_pool import InterReplicaGradientDomain


def rows(start, count, zero_mask=False):
    return [dict(input_ids=torch.zeros(1, 2+i%3, dtype=torch.long),
                 logprobs=torch.full((1,2+i%3), -.7),
                 loss_mask=torch.zeros(1,2+i%3) if zero_mask else torch.ones(1,2+i%3),
                 rewards=torch.tensor([float(i)]), advantages=torch.tensor([1.+i/5]),
                 feature=float(i)/3)
            for i in range(start,start+count)]


def engine(monkeypatch, lane=0, width=1):
    # Only MCore finalization is replaced: the loss, microbatching, payload
    # compute, enqueue, gradient reduction and statistics methods are real.
    monkeypatch.setitem(sys.modules, 'megatron.core.distributed',
                        NS(finalize_model_grads=lambda model: None))
    ex=Engine.__new__(Engine)
    parameter=torch.tensor(.1,requires_grad=True)
    ex.train_tp_size=1
    ex.micro_batch_size=3
    ex.tokenizer=NS(pad_token_id=0)
    ex.clip_epsilon=.2
    ex.kl_coef=.1
    ex.optimizer_revision=2
    ex.current_version=2
    ex.model=[]
    ex._device=lambda:'cpu'
    ex._zero_grad=lambda:setattr(parameter,'grad',torch.zeros_like(parameter))
    ex._set_train_mode=lambda enabled:None
    def forward(batch):
        features=torch.tensor([r['feature'] for r in batch['trajectories']])[:,None]
        values=(parameter+features).expand_as(batch['input_ids'])
        return torch.stack((values,torch.zeros_like(values)),dim=-1)
    ex._forward=forward
    ex._model_main_gradients=lambda:[parameter.grad]
    ex.use_distributed_optimizer=False
    ex.get_data_parallel_rank=lambda:0
    ex.get_data_parallel_world_size=lambda:1
    ex._parallel_state=lambda:NS(get_data_parallel_group=lambda **kw:'dp-lane-'+str(lane))
    monkeypatch.setattr(module.dist,'all_reduce',lambda values,**kw:None)
    steps=[]
    ex.optimizer=NS(scale_loss=lambda loss:loss,
                    step=lambda:(steps.append(True) or True,7.,0))
    ex._pending_hybrid_gradients=[]
    ex._hybrid_gradient_condition=threading.Condition()
    ex._elastic_seen_payloads=set()
    ex._elastic_active_gradient_timeout_s=0
    ex._elastic_training_step=3
    ex._elastic_expected_state_version=2
    ex._elastic_gradient_update_callback=None
    domain=InterReplicaGradientDomain(core_replica_ids=['dp0'],
                                      replica_world_size=width,local_replica_rank=lane)
    domain.request_join('hybrid','dp0',membership_epoch=4)
    domain.mark_active('hybrid',membership_epoch=4)
    ex.elastic_gradient_domain=domain
    ex._elastic_step_hybrid_workers=('hybrid',)
    return ex,steps


def compute(monkeypatch, subset, lane=0, width=1):
    worker,_=engine(monkeypatch,lane,width)
    monkeypatch.setenv('RANK',str(lane))
    monkeypatch.setenv('WORLD_SIZE',str(width))
    payload=worker.compute_elastic_gradient_payload(subset,worker_id='hybrid',
        target_core_id='dp0',step=3,state_version=2,membership_epoch=4,
        global_sample_count=12)
    return worker,payload

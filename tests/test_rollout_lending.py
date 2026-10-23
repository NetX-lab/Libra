"""Loans grow without disturbing existing Hybrid ownership or exposing unsafe GPUs."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from RL_Framework.infra.elastic.rollout_lending import _choose_engines, lend_rollout, return_rollout
from RL_Framework.infra.elastic.rollout_lifecycle import unavailable_rollouts


@dataclass
class Meta:
    instance_id: str
    command: str
    pid: int
    gpus: list
    port: int
    host: str = 'node'
    tp: int = 1
    adopted: bool = False


def fixture(count=4, reserve=1):
    original = [Meta(chr(97+i), 'serve', 10+i, [i], 8000+i) for i in range(count)]
    handles = [NS(instance_id=m.instance_id, is_ready=True) for m in original]
    events = []
    dispatcher = NS(paused=False)
    dispatcher.is_paused = lambda: dispatcher.paused
    dispatcher.pause = lambda: setattr(dispatcher, 'paused', True)
    dispatcher.resume = lambda: setattr(dispatcher, 'paused', False)
    dispatcher.wait_until_idle = lambda **kw: events.append('drained')
    cfg = NS(train_tp_size=1, train_ep_size=1, train_pp_size=1, train_cp_size=1,
             global_resource_planner=NS(elastic_hybrid_min_rollout_gpus=reserve,
                                       runtime_drain_timeout_s=1))
    ex = NS(config=cfg, dispatcher=dispatcher, _process_meta={m.instance_id:m for m in original},
            _hybrid_worker_meta={}, rollout_engine=NS(scheduler=NS(_instances=handles),
                                                     wait_until_idle=lambda **kw: None))
    ex._adopt_existing_rollout_processes = lambda: None
    ex._elastic_replica_size_gpus = lambda: cfg.train_tp_size*cfg.train_ep_size*cfg.train_pp_size*cfg.train_cp_size
    def stop(identity):
        assert not next(h for h in handles if h.instance_id == identity).is_ready
        events.append(('stop',identity))
        return [ex._process_meta.pop(identity)]
    ex._stop_rollout_process = stop
    ex._command_for_instance_id = lambda identity,port: 'restart '+identity
    def start(meta):
        assert meta.instance_id not in ex._process_meta
        events.append(('start',meta.instance_id))
        ex._process_meta[meta.instance_id] = meta
        return meta
    ex._start_one_rollout_process = start
    ex._wait_rollout_process_start_ready = lambda meta: events.append(('ready',meta.instance_id))
    def publish(instance_ids=None):
        assert instance_ids is not None
        assert all(not h.is_ready for h in handles if h.instance_id in instance_ids)
        events.append(('publish',tuple(instance_ids)))
    ex._publish_current_rollout_weights = publish
    return ex,events,handles


def test_grow_total_loan_preserves_existing_slots_generation_and_hybrid():
    ex,events,handles = fixture()
    first = lend_rollout(ex,1)
    saved = ex._lent_rollout
    generation = saved['generation']
    hybrid = NS(host='node',gpus=[0])
    ex._hybrid_worker_meta['old-hybrid'] = hybrid
    second = lend_rollout(ex,2)
    assert ex._lent_rollout is saved and saved['generation'] == generation
    assert first == {'rollout0':{'host':'node','gpus':[0]}}
    assert second == {**first,'rollout1':{'host':'node','gpus':[1]}}
    assert ex._hybrid_worker_meta == {'old-hybrid':hybrid}
    assert [e for e in events if isinstance(e,tuple) and e[0]=='stop'] == [('stop','a'),('stop','b')]
    assert not handles[0].is_ready and not handles[1].is_ready
    assert lend_rollout(ex,2) == second


def test_reserve_counts_only_rollout_gpus_not_existing_loan():
    ex,events,_ = fixture(reserve=2)
    first = lend_rollout(ex,1)
    second = lend_rollout(ex,2)
    assert len(second) == 2
    before = list(events)
    assert lend_rollout(ex,3) == second
    assert events == before and len(ex._process_meta) == 2
    assert first.items() <= second.items()


def test_selection_uses_full_runtime_replica_width():
    ex,_,_ = fixture(count=3,reserve=0)
    ex.config.train_ep_size = 2
    for i,meta in enumerate(ex._process_meta.values()):
        meta.host = 'node'+str(i)
    assert _choose_engines(ex,2) == ()


def test_failed_expansion_restores_only_new_engines_after_targeted_ack():
    ex,events,handles = fixture(reserve=1)
    first = lend_rollout(ex,1)
    old = ex._lent_rollout
    snapshot = (dict(old['slots']),list(old['engines']),old['generation'])
    stop = ex._stop_rollout_process
    once = {'fail':True}
    def fail_second(identity):
        if identity=='c' and once['fail']:
            once['fail']=False
            raise RuntimeError('unconfirmed stop')
        return stop(identity)
    ex._stop_rollout_process = fail_second
    with pytest.raises(RuntimeError,match='unconfirmed stop'):
        lend_rollout(ex,3)
    assert ex._lent_rollout is old
    assert (old['slots'],old['engines'],old['generation']) == snapshot
    assert lend_rollout(ex,1)==first
    assert not handles[0].is_ready and all(h.is_ready for h in handles[1:])
    assert ('publish',('b','c')) in events
    assert not any(e in [('start','a'),('publish',('a',))] for e in events)
    assert getattr(ex,'_rollout_loan_rollback',None) is None


def test_failed_rollback_keeps_new_ownership_isolated_and_old_loan_usable():
    ex,events,handles = fixture()
    first = lend_rollout(ex,1)
    old = ex._lent_rollout
    def fail_publish(instance_ids=None):
        events.append(('publish-failed',tuple(instance_ids)))
        raise RuntimeError('current weights not acknowledged')
    ex._publish_current_rollout_weights=fail_publish
    stop=ex._stop_rollout_process
    once={'fail':True}
    def fail_stop(identity):
        if identity=='c' and once['fail']:
            once['fail']=False
            raise RuntimeError('stop lost acknowledgement')
        return stop(identity)
    ex._stop_rollout_process=fail_stop
    with pytest.raises(RuntimeError):
        lend_rollout(ex,3)
    assert ex._lent_rollout is old and old['slots']==first
    assert ex._rollout_loan_rollback
    assert set(unavailable_rollouts(ex))=={'a','b','c'}
    assert not any(h.is_ready for h in handles[:3])
    assert lend_rollout(ex,1)==first
    before=list(events)
    with pytest.raises(RuntimeError,match='incomplete'):
        lend_rollout(ex,3)
    assert events==before


def test_rollback_reopens_selected_services_that_were_never_stopped():
    ex,events,handles=fixture(count=6,reserve=1)
    lend_rollout(ex,1)
    stop=ex._stop_rollout_process
    once={'fail':True}
    def fail(identity):
        if identity=='c' and once['fail']:
            once['fail']=False
            raise RuntimeError('middle stop failed')
        return stop(identity)
    ex._stop_rollout_process=fail
    with pytest.raises(RuntimeError,match='middle stop failed'):
        lend_rollout(ex,4)
    assert not handles[0].is_ready
    assert all(h.is_ready for h in handles[1:])
    assert ('stop','d') not in events and ('start','d') not in events
    assert ('publish',('b','c')) in events


def test_return_readiness_retry_keeps_started_process_ownership(tmp_path):
    ex,events,_=fixture(count=2,reserve=0)
    lend_rollout(ex,1)
    ex._hybrid_worker_control_dir=lambda:tmp_path
    once={'fail':True}
    def wait(meta):
        if once['fail']:
            once['fail']=False
            raise RuntimeError('service still warming')
    ex._wait_rollout_process_start_ready=wait
    with pytest.raises(RuntimeError,match='still warming'):
        return_rollout(ex)
    owned=ex._lent_rollout['restored'][0]
    assert ex._process_meta['a'] is owned
    assert return_rollout(ex)==[owned]
    assert events.count(('start','a'))==1


def test_wide_replica_hands_off_only_after_every_engine_exit_ack():
    ex,events,_=fixture(count=4,reserve=0)
    ex.config.train_tp_size=2
    ex.config.train_pp_size=2
    stop=ex._stop_rollout_process
    def checked(identity):
        assert not getattr(ex,'_lent_rollout',None)
        return stop(identity)
    ex._stop_rollout_process=checked
    slots=lend_rollout(ex,4)
    assert len(slots)==4 and not ex._process_meta
    assert len([e for e in events if isinstance(e,tuple) and e[0]=='stop'])==4


def test_one_wide_engine_cannot_borrow_through_rollout_reserve():
    ex,events,_=fixture(count=1,reserve=2)
    ex.config.train_tp_size=2
    ex._process_meta['a'].gpus=[0,1,2,3]
    ex._process_meta['a'].tp=4
    assert lend_rollout(ex,2)=={}
    assert events==[]


def test_remote_exit_ack_is_bound_to_the_owned_request(tmp_path):
    import json
    from RL_Framework.infra.elastic.rollout_lending import wait_hybrid_stopped
    ex,_,_=fixture()
    ex._hybrid_worker_control_dir=lambda:tmp_path
    ex.config.global_resource_planner.hybrid_worker_stop_timeout_s=.01
    path=tmp_path/'.stop_worker.done'
    path.write_text(json.dumps(dict(worker_id='worker',request_id='old',exited=True)))
    with pytest.raises(TimeoutError):
        wait_hybrid_stopped(ex,'worker','current')
    path.write_text(json.dumps(dict(worker_id='worker',request_id='current',exited=True)))
    assert wait_hybrid_stopped(ex,'worker','current')['exited']

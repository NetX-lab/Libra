"""Support code for Test schedulers."""

import sys
import os
import time


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


import importlib.util


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_infra_dir = os.path.join(_project_root, "infra")
_scheduling_dir = os.path.join(_infra_dir, "scheduling")


import types as _types
_infra_pkg = _types.ModuleType("infra")
_infra_pkg.__path__ = [_infra_dir]
_infra_pkg.__package__ = "infra"
sys.modules["infra"] = _infra_pkg


_rl_pkg = _types.ModuleType("RL_Framework")
_rl_pkg.__path__ = [_project_root]
_rl_pkg.__package__ = "RL_Framework"
sys.modules.setdefault("RL_Framework", _rl_pkg)

_rl_infra_pkg = _types.ModuleType("RL_Framework.infra")
_rl_infra_pkg.__path__ = [_infra_dir]
_rl_infra_pkg.__package__ = "RL_Framework.infra"
sys.modules.setdefault("RL_Framework.infra", _rl_infra_pkg)

_rl_infra_scheduling_pkg = _types.ModuleType("RL_Framework.infra.scheduling")
_rl_infra_scheduling_pkg.__path__ = [_scheduling_dir]
_rl_infra_scheduling_pkg.__package__ = "RL_Framework.infra.scheduling"
sys.modules.setdefault("RL_Framework.infra.scheduling", _rl_infra_scheduling_pkg)

_scheduler_base = _load_module("infra.scheduling.base", os.path.join(_scheduling_dir, "base.py"))
_hetero_scheduler = _load_module("infra.scheduling.length_aware", os.path.join(_scheduling_dir, "length_aware.py"))
_la_mlfq_scheduler = _load_module("infra.scheduling.la_mlfq", os.path.join(_scheduling_dir, "la_mlfq.py"))
_load_balance_scheduler = _load_module("infra.scheduling.load_balance", os.path.join(_scheduling_dir, "load_balance.py"))
_scheduler_factory = _load_module("infra.scheduling.factory", os.path.join(_scheduling_dir, "factory.py"))

BaseScheduler = _scheduler_base.BaseScheduler
SchedulingResult = _scheduler_base.SchedulingResult
InstanceHandle = _scheduler_base.InstanceHandle
SchedulerStats = _scheduler_base.SchedulerStats
LoadBalanceStrategy = _scheduler_base.LoadBalanceStrategy
RoutingRule = _scheduler_base.RoutingRule

LengthAwareScheduler = _hetero_scheduler.LengthAwareScheduler
HeterogeneousScheduler = _hetero_scheduler.HeterogeneousScheduler

LAMLFQScheduler = _la_mlfq_scheduler.LAMLFQScheduler
HistoryTable = _la_mlfq_scheduler.HistoryTable
ScoutManager = _la_mlfq_scheduler.ScoutManager
ScoutStatus = _la_mlfq_scheduler.ScoutStatus
MigrationController = _la_mlfq_scheduler.MigrationController
MigrationDecision = _la_mlfq_scheduler.MigrationDecision
WaitingRequest = _la_mlfq_scheduler.WaitingRequest

LoadBalanceScheduler = _load_balance_scheduler.LoadBalanceScheduler
SchedulerFactory = _scheduler_factory.SchedulerFactory

from config import (
    HeterogeneousRolloutConfig,
    SchedulingConfig,
)


# =====================================================================

# =====================================================================

_pass_count = 0
_fail_count = 0


def check(name: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  [PASS] {name}")
    else:
        _fail_count += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _make_hetero_config(scheduler_type="length_aware", **sched_kwargs):
    """Make hetero config."""
    sched_dict = {
        "scheduler_type": scheduler_type,
        "length_thresholds": {"short": 1000, "medium": 3000, "long": 6000},
        "routing_rules": {
            "short": [1],
            "medium": [1, 2],
            "long": [2, 4],
            "extra_long": [4],
        },
        "load_balance_strategy": "least_connections",
        "max_queue_length": 50,
        "enable_fallback": True,
    }
    sched_dict.update(sched_kwargs)
    return HeterogeneousRolloutConfig.from_dict({
        "enabled": True,
        "instances": [
            {"instance_id": "tp1_0", "tp": 1, "gpus": [0]},
            {"instance_id": "tp1_1", "tp": 1, "gpus": [1]},
            {"instance_id": "tp2_0", "tp": 2, "gpus": [2, 3]},
            {"instance_id": "tp4_0", "tp": 4, "gpus": [4, 5, 6, 7]},
        ],
        "scheduling": sched_dict,
    })


def _register_standard_instances(scheduler):
    """Register standard instances."""
    scheduler.register_instance(0, "tp1_0", 1)
    scheduler.register_instance(1, "tp1_1", 1)
    scheduler.register_instance(2, "tp2_0", 2)
    scheduler.register_instance(3, "tp4_0", 4)


# =====================================================================

# =====================================================================

def test_length_aware_categorize():
    """Test length aware categorize."""
    section("1.1 LengthAwareScheduler length classification")
    sched = LengthAwareScheduler(
        length_thresholds={"short": 1000, "medium": 3000, "long": 6000}
    )
    check("500 -> short", sched.categorize(500) == "short")
    check("1000 -> short", sched.categorize(1000) == "short")
    check("1001 -> medium", sched.categorize(1001) == "medium")
    check("3000 -> medium", sched.categorize(3000) == "medium")
    check("5000 -> long", sched.categorize(5000) == "long")
    check("6000 -> long", sched.categorize(6000) == "long")
    check("10000 -> extra_long", sched.categorize(10000) == "extra_long")


def test_length_aware_routing():
    """Test length aware routing."""
    section("1.2 LengthAwareScheduler routing")
    sched = LengthAwareScheduler(
        routing_rules={"short": [1], "medium": [2], "long": [4], "extra_long": [4]},
        length_thresholds={"short": 1000, "medium": 3000, "long": 6000},
    )
    _register_standard_instances(sched)


    r = sched.schedule(500)
    check("Short sequence -> TP=1", r.tp_degree == 1, f"got TP={r.tp_degree}")
    check("Short sequence category=short", r.category == "short")
    check("Short sequence is not a fallback", r.is_fallback is False)
    sched.on_request_done(r.instance_index)


    r = sched.schedule(2000)
    check("Medium sequence -> TP=2", r.tp_degree == 2, f"got TP={r.tp_degree}")
    sched.on_request_done(r.instance_index)


    r = sched.schedule(5000)
    check("Long sequence -> TP=4", r.tp_degree == 4, f"got TP={r.tp_degree}")
    sched.on_request_done(r.instance_index)


def test_length_aware_fallback():
    """Test length aware fallback."""
    section("1.3 LengthAwareScheduler Fallback routing")

    sched = LengthAwareScheduler(
        routing_rules={"short": [1], "medium": [2]},
        length_thresholds={"short": 1000, "medium": 3000, "long": 6000},
    )
    sched.register_instance(0, "tp2_0", 2)

    r = sched.schedule(500)
    check("Fallback to TP=2", r.tp_degree == 2)
    check("is_fallback=True", r.is_fallback is True)
    sched.on_request_done(r.instance_index)


def test_length_aware_load_balance():
    """Test length aware load balance."""
    section("1.4 LengthAwareScheduler load balancing")
    sched = LengthAwareScheduler(
        routing_rules={"short": [1]},
        length_thresholds={"short": 1000, "medium": 3000, "long": 6000},
    )
    sched.register_instance(0, "tp1_0", 1)
    sched.register_instance(1, "tp1_1", 1)


    r1 = sched.schedule(500)



    r2 = sched.schedule(500)
    check("Two requests route to different instances", r1.instance_index != r2.instance_index,
          f"r1={r1.instance_index}, r2={r2.instance_index}")

    sched.on_request_done(r1.instance_index)
    sched.on_request_done(r2.instance_index)


def test_length_aware_stats():
    """Test length aware stats."""
    section("1.5 LengthAwareScheduler statistics")
    sched = LengthAwareScheduler()
    _register_standard_instances(sched)

    for _ in range(10):
        r = sched.schedule(500)
        sched.on_request_done(r.instance_index)
    for _ in range(5):
        r = sched.schedule(20000)
        sched.on_request_done(r.instance_index)

    stats = sched.get_stats()
    check("total_requests=15", stats.total_requests == 15)
    check("success rate=1.0", stats.success_rate == 1.0)
    check("category_counts contains short", "short" in stats.category_counts)
    check("category_counts contains extra_long", "extra_long" in stats.category_counts)


def test_length_aware_backward_compat():
    """Test length aware backward compat."""
    section("1.6 HeterogeneousScheduler alias compatibility")
    check("HeterogeneousScheduler is LengthAwareScheduler",
          HeterogeneousScheduler is LengthAwareScheduler)
    sched = HeterogeneousScheduler()
    check("HeterogeneousScheduler created successfully", isinstance(sched, LengthAwareScheduler))


def test_length_aware_from_config():
    """Test length aware from config."""
    section("1.7 LengthAwareScheduler.from_config")
    cfg = _make_hetero_config("length_aware")
    sched = LengthAwareScheduler.from_config(cfg)
    check("Created successfully", isinstance(sched, LengthAwareScheduler))
    check("name='LengthAware'", sched.name == "LengthAware")


# =====================================================================

# =====================================================================

# 2.1 HistoryTable

def test_history_table():
    """Test history table."""
    section("2.1 HistoryTable basic operations")
    ht = HistoryTable(ttl_epochs=3)
    check("Initial size=0", ht.size == 0)


    ht.update("prompt_A", "short", epoch=0)
    ht.update("prompt_B", "long", epoch=0)
    check("Size=2", ht.size == 2)


    check("Lookup prompt_A=short", ht.lookup("prompt_A") == "short")
    check("Lookup prompt_B=long", ht.lookup("prompt_B") == "long")
    check("prompt_C is absent", ht.lookup("prompt_C") is None)


    ht.on_epoch_end(epoch=5)
    check("prompt_A=None after expiration", ht.lookup("prompt_A", current_epoch=5) is None)


    ht.update("prompt_C", "long", epoch=4)
    check("New record prompt_C=long", ht.lookup("prompt_C", current_epoch=5) == "long")


# 2.2 ScoutManager

def test_scout_manager():
    """Test scout manager."""
    section("2.2 ScoutManager scout mechanism")
    sm = ScoutManager(scout_timeout=30.0)

    prompt_id = "prompt_X"


    check("should_scout=True on first request", sm.should_scout(prompt_id) is True)


    sm.register_scout(prompt_id, instance_index=0, bucket="short")
    check("should_scout=False with an existing scout", sm.should_scout(prompt_id) is False)
    check("active_scouts=1", sm.active_scouts == 1)
    check("is_scout identifies the scout", sm.is_scout(prompt_id, 0) is True)
    check("is_scout rejects the wrong instance", sm.is_scout(prompt_id, 1) is False)


    wr1 = WaitingRequest(prompt_id=prompt_id, input_tokens=500, n_samples=4, epoch=0, sample_index=1)
    wr2 = WaitingRequest(prompt_id=prompt_id, input_tokens=500, n_samples=4, epoch=0, sample_index=2)
    sm.add_waiting(wr1)
    sm.add_waiting(wr2)
    check("total_waiting=2", sm.total_waiting == 2)


    released = sm.on_scout_completed(prompt_id, "short")
    check("Release two waiters", len(released) == 2)
    check("total_waiting=0", sm.total_waiting == 0)


    sm.reset()
    check("active_scouts=0 after reset", sm.active_scouts == 0)


def test_scout_manager_migration():
    """Test scout manager migration."""
    section("2.3 ScoutManager scout migration")
    sm = ScoutManager()
    prompt_id = "prompt_Y"

    sm.register_scout(prompt_id, instance_index=0, bucket="short")
    wr = WaitingRequest(prompt_id=prompt_id, input_tokens=500, n_samples=2, epoch=0, sample_index=1)
    sm.add_waiting(wr)


    released = sm.on_scout_migrated(prompt_id, "long")
    check("Migration releases one waiter", len(released) == 1)
    check("Scout current bucket=long", sm.get_scout_bucket(prompt_id) == "long")
    check("Scout status=MIGRATED", sm.get_scout_status(prompt_id) == ScoutStatus.MIGRATED)


# 2.3 MigrationController

def test_migration_controller():
    """Test migration controller."""
    section("2.4 MigrationController migration decision")
    mc = MigrationController(bucket_thresholds={"short": 3000, "medium": 10000})


    mc.register_request("req_1", "prompt_A", 0, "short", is_scout=True)


    d1 = mc.check_migration("req_1", generated_tokens=1000)
    check("1000 tokens: no migration", d1.should_migrate is False)
    check("Reason=below_threshold", d1.reason == "below_threshold")


    d2 = mc.check_migration("req_1", generated_tokens=5000)
    check("5000 tokens: migrate", d2.should_migrate is True)
    check("Target bucket=long", d2.target_bucket == "long")
    check("Reason=threshold_exceeded", d2.reason == "threshold_exceeded")


    mc.on_migration_executed("req_1", "long")
    state = mc.get_request_state("req_1")
    check("current_bucket=long after migration", state.current_bucket == "long")


    d3 = mc.check_migration("req_1", generated_tokens=50000)
    check("Long bucket: no migration", d3.should_migrate is False)


    mc.on_request_done("req_1")
    stats = mc.get_stats()
    check("total_migrations=1", stats["total_migrations"] == 1)




def test_la_mlfq_default_route():
    """Test la mlfq default route."""
    section("2.5 LAMLFQScheduler default length routing")
    sched = LAMLFQScheduler(
        buckets={
            "short": {"tp_degrees": [1], "max_tokens": 3000},
            "long": {"tp_degrees": [2, 4], "max_tokens": 50000},
        },
        length_thresholds={"short": 1000, "medium": 3000, "long": 6000},
    )
    _register_standard_instances(sched)


    r = sched.schedule(input_tokens=500)
    check("Short sequence -> short bucket", r.category == "short")
    check("TP=1 instance", r.tp_degree == 1)
    sched.on_request_done(r.instance_index)

    r = sched.schedule(input_tokens=5000)
    check("Long sequence -> long bucket", r.category == "long")
    check("TP=2 or 4", r.tp_degree in [2, 4])
    sched.on_request_done(r.instance_index)


def test_la_mlfq_history_hit():
    """Test la mlfq history hit."""
    section("2.6 LAMLFQScheduler inter-epoch history hit")
    sched = LAMLFQScheduler(
        buckets={
            "short": {"tp_degrees": [1], "max_tokens": 3000},
            "long": {"tp_degrees": [2, 4], "max_tokens": 50000},
        },
        length_thresholds={"short": 1000},
    )
    _register_standard_instances(sched)


    sched.history_table.update("prompt_A", "long", epoch=0)


    sched.on_epoch_start(1)
    r = sched.schedule(input_tokens=500, prompt_id="prompt_A", epoch=1)
    check("History hit routes a short sequence to long", r.category == "long")
    check("reason contains history_hit", "history_hit" in r.reason)
    sched.on_request_done(r.instance_index, prompt_id="prompt_A", final_bucket="long")


def test_la_mlfq_scout_mechanism():
    """Test la mlfq scout mechanism."""
    section("2.7 LAMLFQScheduler scout mechanism")
    sched = LAMLFQScheduler(
        buckets={
            "short": {"tp_degrees": [1], "max_tokens": 3000},
            "long": {"tp_degrees": [2, 4], "max_tokens": 50000},
        },
    )
    _register_standard_instances(sched)

    prompt_id = "prompt_new"


    r1 = sched.schedule(input_tokens=500, prompt_id=prompt_id, n_samples=4)
    check("First request: not pending", r1.pending is False)
    check("First request: short bucket", r1.category == "short")
    check("First request: reason=scout", "scout" in r1.reason)


    r2 = sched.schedule(input_tokens=500, prompt_id=prompt_id, n_samples=4)
    check("Second request: pending=True", r2.pending is True)
    check("Second request: reason=waiting_for_scout", "waiting_for_scout" in r2.reason)


    sched.on_request_done(
        r1.instance_index, prompt_id=prompt_id, final_bucket="short"
    )


    r3 = sched.schedule(input_tokens=500, prompt_id=prompt_id, n_samples=4)
    check("Third request: follows the scout to short", r3.category == "short")
    check("Third request: not pending", r3.pending is False)
    sched.on_request_done(r3.instance_index, prompt_id=prompt_id, final_bucket="short")


def test_la_mlfq_scout_migration():
    """Test la mlfq scout migration."""
    section("2.8 LAMLFQScheduler scout migration")
    sched = LAMLFQScheduler(
        buckets={
            "short": {"tp_degrees": [1], "max_tokens": 3000},
            "long": {"tp_degrees": [2, 4], "max_tokens": 50000},
        },
    )
    _register_standard_instances(sched)

    prompt_id = "prompt_migrate"


    r1 = sched.schedule(input_tokens=500, prompt_id=prompt_id, n_samples=3)
    check("Scout: short bucket", r1.category == "short")


    scout_req_id = f"{prompt_id}_scout"
    decision = sched.check_and_migrate(scout_req_id, generated_tokens=5000)
    check("migration decision: should_migrate=True", decision.should_migrate is True)
    check("Migration target: long", decision.target_bucket == "long")


    stats = sched.get_stats()
    check("migrated_routes >= 1", stats.migrated_routes >= 1)


def test_la_mlfq_epoch_lifecycle():
    """Test la mlfq epoch lifecycle."""
    section("2.9 LAMLFQScheduler epoch lifecycle")
    sched = LAMLFQScheduler(history_ttl=2)
    _register_standard_instances(sched)

    # Epoch 0
    sched.on_epoch_start(0)
    sched.history_table.update("p1", "short", epoch=0)
    sched.history_table.update("p2", "long", epoch=0)
    sched.on_epoch_end(0)
    check("history_table.size=2 after epoch 0", sched.history_table.size == 2)


    sched.on_epoch_start(3)
    sched.on_epoch_end(3)
    r = sched.history_table.lookup("p1", current_epoch=3)
    check("p1 expires after epoch 3", r is None)


def test_la_mlfq_from_config():
    """Test la mlfq from config."""
    section("2.10 LAMLFQScheduler.from_config")
    cfg = _make_hetero_config(
        "la_mlfq",
        la_mlfq_migration_threshold=5000,
        la_mlfq_scout_timeout=45.0,
        la_mlfq_history_ttl=3,
    )
    sched = LAMLFQScheduler.from_config(cfg)
    check("Created successfully", isinstance(sched, LAMLFQScheduler))
    check("name='LA-MLFQ'", sched.name == "LA-MLFQ")


# =====================================================================

# =====================================================================

def test_load_balance_least_connections():
    """Test load balance least connections."""
    section("3.1 LoadBalanceScheduler Least Connections")
    sched = LoadBalanceScheduler(load_balance_strategy="least_connections")
    _register_standard_instances(sched)


    r1 = sched.schedule(500)
    check("Scheduling succeeds", r1.instance_index >= 0)


    r2 = sched.schedule(500)
    check("Selects different instances", r2.instance_index != r1.instance_index,
          f"r1={r1.instance_index}, r2={r2.instance_index}")

    sched.on_request_done(r1.instance_index)
    sched.on_request_done(r2.instance_index)


def test_load_balance_round_robin():
    """Test load balance round robin."""
    section("3.2 LoadBalanceScheduler Round Robin")
    sched = LoadBalanceScheduler(load_balance_strategy="round_robin")
    _register_standard_instances(sched)


    indices = set()
    for _ in range(4):
        r = sched.schedule(500)
        indices.add(r.instance_index)
        sched.on_request_done(r.instance_index)

    check("Round-robin covers all four instances", len(indices) == 4,
          f"got {len(indices)} unique indices")


def test_load_balance_weighted():
    """Test load balance weighted."""
    section("3.3 LoadBalanceScheduler Weighted")
    sched = LoadBalanceScheduler(
        load_balance_strategy="weighted",
        weights={1: 1.0, 2: 2.0, 4: 4.0},
    )
    _register_standard_instances(sched)

    r = sched.schedule(500)
    check("Weighted scheduling succeeds", r.instance_index >= 0)
    sched.on_request_done(r.instance_index)


def test_load_balance_ignores_length():
    """Test load balance ignores length."""
    section("3.4 LoadBalanceScheduler ignores length")
    sched = LoadBalanceScheduler()
    _register_standard_instances(sched)

    r_short = sched.schedule(100)
    sched.on_request_done(r_short.instance_index)
    r_long = sched.schedule(100000)
    sched.on_request_done(r_long.instance_index)

    check("category are all 'any'",
          r_short.category == "any" and r_long.category == "any")


def test_load_balance_from_config():
    """Test load balance from config."""
    section("3.5 LoadBalanceScheduler.from_config")
    cfg = _make_hetero_config("load_balance")
    sched = LoadBalanceScheduler.from_config(cfg)
    check("Created successfully", isinstance(sched, LoadBalanceScheduler))
    check("name='LoadBalance'", sched.name == "LoadBalance")


# =====================================================================

# =====================================================================

def test_factory_length_aware():
    """Test factory length aware."""
    section("4.1 SchedulerFactory → LengthAware")
    cfg = _make_hetero_config("length_aware")
    sched = SchedulerFactory.create("length_aware", cfg)
    check("Type is correct", type(sched).__name__ == "LengthAwareScheduler")
    check("name='LengthAware'", sched.name == "LengthAware")


def test_factory_la_mlfq():
    """Test factory la mlfq."""
    section("4.2 SchedulerFactory → LA-MLFQ")
    cfg = _make_hetero_config("la_mlfq")
    sched = SchedulerFactory.create("la_mlfq", cfg)
    check("Type is correct", type(sched).__name__ == "LAMLFQScheduler")


def test_factory_load_balance():
    """Test factory load balance."""
    section("4.3 SchedulerFactory → LoadBalance")
    cfg = _make_hetero_config("load_balance")
    sched = SchedulerFactory.create("load_balance", cfg)
    check("Type is correct", type(sched).__name__ == "LoadBalanceScheduler")


def test_factory_invalid_type():
    """Test factory invalid type."""
    section("4.4 SchedulerFactory invalid type")
    cfg = _make_hetero_config("unknown_type")
    try:
        SchedulerFactory.create("unknown_type", cfg)
        check("Should raise ValueError", False)
    except ValueError as e:
        check("Raises ValueError", True)
        check("Error lists available types", "length_aware" in str(e))


def test_factory_available_types():
    """Test factory available types."""
    section("4.5 SchedulerFactory available types")
    types = SchedulerFactory.available_types()
    check("length_aware is listed", "length_aware" in types)
    check("la_mlfq is listed", "la_mlfq" in types)
    check("load_balance is listed", "load_balance" in types)


# =====================================================================

# =====================================================================

def test_base_scheduler_instance_registration():
    """Test base scheduler instance registration."""
    section("5.1 BaseScheduler instance registration")
    sched = LengthAwareScheduler()
    _register_standard_instances(sched)

    h = sched.get_instance_handle(0)
    check("Lookup index=0", h is not None)
    check("instance_id='tp1_0'", h.instance_id == "tp1_0")
    check("tp_degree=1", h.tp_degree == 1)

    h3 = sched.get_instance_handle(3)
    check("Lookup index=3", h3 is not None)
    check("tp_degree=4", h3.tp_degree == 4)

    check("Lookup index=99 returns None", sched.get_instance_handle(99) is None)


def test_base_scheduler_stats_reset():
    """Test base scheduler stats reset."""
    section("5.2 BaseScheduler statistics reset")
    sched = LengthAwareScheduler()
    _register_standard_instances(sched)

    for _ in range(5):
        r = sched.schedule(500)
        sched.on_request_done(r.instance_index)

    check("Before reset total=5", sched.get_stats().total_requests == 5)
    sched.reset_stats()
    check("After reset total=0", sched.get_stats().total_requests == 0)


def test_scheduler_all_types_schedule_and_done():
    """Test scheduler all types schedule and done."""
    section("5.3 schedule/on_request_done cycle for all scheduler types")
    cfg = _make_hetero_config()
    for stype in ["length_aware", "la_mlfq", "load_balance"]:
        sched = SchedulerFactory.create(stype, cfg)
        _register_standard_instances(sched)

        for tokens in [100, 2000, 5000, 10000]:
            r = sched.schedule(input_tokens=tokens)
            if not r.pending:
                sched.on_request_done(r.instance_index)

        stats = sched.get_stats()
        check(f"{stype}: total_requests=4", stats.total_requests == 4)


# =====================================================================

# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("  Heterogeneous scheduler policy unit tests")
    print("=" * 60)

    # 1. LengthAwareScheduler
    test_length_aware_categorize()
    test_length_aware_routing()
    test_length_aware_fallback()
    test_length_aware_load_balance()
    test_length_aware_stats()
    test_length_aware_backward_compat()
    test_length_aware_from_config()

    # 2. LAMLFQScheduler
    test_history_table()
    test_scout_manager()
    test_scout_manager_migration()
    test_migration_controller()
    test_la_mlfq_default_route()
    test_la_mlfq_history_hit()
    test_la_mlfq_scout_mechanism()
    test_la_mlfq_scout_migration()
    test_la_mlfq_epoch_lifecycle()
    test_la_mlfq_from_config()

    # 3. LoadBalanceScheduler
    test_load_balance_least_connections()
    test_load_balance_round_robin()
    test_load_balance_weighted()
    test_load_balance_ignores_length()
    test_load_balance_from_config()

    # 4. SchedulerFactory
    test_factory_length_aware()
    test_factory_la_mlfq()
    test_factory_load_balance()
    test_factory_invalid_type()
    test_factory_available_types()


    test_base_scheduler_instance_registration()
    test_base_scheduler_stats_reset()
    test_scheduler_all_types_schedule_and_done()


    print(f"\n{'='*60}")
    total = _pass_count + _fail_count
    print(f"  Result: {_pass_count}/{total} passed, {_fail_count} failed")
    print(f"{'='*60}")

    if _fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

import itertools
import math
from types import SimpleNamespace

import pytest

from RL_Framework.config import AsyncRLConfig, HardwareConfig, ModelArchConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlan,
    GlobalResourcePlanner,
)
from RL_Framework.infra.cost_model.model import (
    CostModel,
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)
from RL_Framework.infra.cost_model.optimizer import TwoLevelNestedOptimizer


class TinyExactCostModel:
    """Transparent costs used to compare the planner with brute force."""

    def __init__(self, *, is_moe: bool = False):
        self.ma = SimpleNamespace(is_moe=is_moe, num_experts=4)

    def check_train_oom(self, config, B_global, L):
        return False

    def evaluate_training(self, config, B_global, lengths):
        seconds = (
            60.0 / config.n_gpus
            + 0.13 * config.tp
            + 0.17 * config.ep
            + 0.19 * config.pp
            + 0.01 * config.b_micro
        )
        return seconds, {}

    def evaluate_rollout_instance(self, tp, requests):
        if not requests:
            return 0.0, {}
        work = sum(request.total_length for request in requests)
        longest = max(request.total_length for request in requests)
        return work / (tp * 100.0) + longest / 1000.0 + 0.07, {}

    def evaluate_rollout(self, cluster, requests):
        # Only used by the explicit rollout-node-pattern compatibility path.
        return self.evaluate_rollout_instance(max(cluster.tp_list), requests)


def _ordered_tp_partitions(total, allowed):
    if total == 0:
        yield ()
        return
    for tp in allowed:
        if tp <= total:
            for suffix in _ordered_tp_partitions(total - tp, allowed):
                yield (tp, *suffix)


def _weak_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in _weak_compositions(total - head, parts - 1):
            yield (head, *tail)


def _brute_rollout(cost_model, gpu_budget, requests, allowed_tp):
    ordered = sorted(requests, key=lambda request: request.total_length)
    best = (float("inf"), None)
    for tp_list in _ordered_tp_partitions(gpu_budget, allowed_tp):
        for counts in _weak_compositions(len(ordered), len(tp_list)):
            cursor = 0
            times = []
            for tp, count in zip(tp_list, counts):
                segment = ordered[cursor : cursor + count]
                cursor += count
                times.append(cost_model.evaluate_rollout_instance(tp, segment)[0])
            candidate = max(times, default=0.0)
            if candidate < best[0]:
                best = (candidate, tp_list)
    return best


def _brute_training_configs(n_gpus, *, tp_values, ep_values, pp_values):
    configs = set()
    for tp, ep, pp in itertools.product(tp_values, ep_values, pp_values):
        denominator = tp * ep * pp
        if denominator <= n_gpus and n_gpus % denominator == 0:
            configs.add((tp, ep, pp, n_gpus // denominator, 1))
    return configs


def test_rollout_dp_matches_exhaustive_contiguous_partitioning():
    evaluator = TinyExactCostModel()
    optimizer = TwoLevelNestedOptimizer(
        evaluator,
        allowed_rollout_tp=[1, 2],
        max_rollout_instances=8,
    )
    requests = [
        RequestInfo(10, 10),
        RequestInfo(20, 40),
        RequestInfo(30, 90),
        RequestInfo(40, 180),
    ]

    config, makespan, details = optimizer.rollout_dp(4, requests)
    brute_time, _ = _brute_rollout(evaluator, 4, requests, [1, 2])

    assert config is not None
    assert makespan == pytest.approx(brute_time)
    assert sum(config.tp_list) == 4
    assert details["algorithm"] == "rollout_dp"
    assert sum(segment["num_requests"] for segment in details["segments"]) == 4


def test_training_decision_tree_matches_exhaustive_factorization_with_ep():
    evaluator = TinyExactCostModel(is_moe=True)
    optimizer = TwoLevelNestedOptimizer(
        evaluator,
        train_pipeline_bubble_ratio_threshold=1.0,
    )
    configs = optimizer.decision_tree_search(
        8,
        B_global=8,
        sequence_lengths=[128, 256, 512, 1024],
        allowed_train_tp=[1, 2],
        allowed_train_ep=[1, 2, 4],
        allowed_train_pp=[1, 2, 4],
        micro_batch_sizes=[1],
    )
    actual = {(c.tp, c.ep, c.pp, c.dp, c.b_micro) for c in configs}
    expected = _brute_training_configs(
        8,
        tp_values=[1, 2],
        ep_values=[1, 2, 4],
        pp_values=[1, 2, 4],
    )

    assert actual == expected
    assert any(config.ep > 1 for config in configs)
    assert all(config.n_gpus == 8 for config in configs)


def test_nested_optimizer_matches_brute_force_global_optimum():
    evaluator = TinyExactCostModel(is_moe=True)
    optimizer = TwoLevelNestedOptimizer(
        evaluator,
        allowed_rollout_tp=[1, 2],
        train_pipeline_bubble_ratio_threshold=1.0,
    )
    requests = [RequestInfo(10, 20), RequestInfo(20, 80), RequestInfo(30, 160)]

    result = optimizer.optimize(
        n_total_gpus=6,
        requests=requests,
        B_global=4,
        allowed_train_tp=[1, 2],
        allowed_train_ep=[1, 2],
        allowed_train_pp=[1, 2],
        micro_batch_sizes=[1],
        min_train_gpus=1,
        min_rollout_gpus=1,
    )

    brute_global = float("inf")
    for n_train in range(1, 6):
        train_configs = _brute_training_configs(
            n_train,
            tp_values=[1, 2],
            ep_values=[1, 2],
            pp_values=[1, 2],
        )
        for tp, ep, pp, dp, b_micro in train_configs:
            if 4 % (ep * dp * b_micro):
                continue
            train = TrainParallelConfig(
                tp=tp, ep=ep, pp=pp, dp=dp, b_micro=b_micro
            )
            train_time = evaluator.evaluate_training(train, 4, [110, 220, 330])[0]
            rollout_time, _ = _brute_rollout(
                evaluator, 6 - n_train, requests, [1, 2]
            )
            brute_global = min(brute_global, max(train_time, rollout_time))

    assert result.t_global == pytest.approx(brute_global)
    assert result.train_config is not None
    assert result.rollout_config is not None
    assert result.train_config.n_gpus + result.rollout_config.n_gpus == 6


def test_training_cost_uses_non_uniform_microbatch_lengths():
    model = CostModel(
        hardware=HardwareConfig(mem_capacity=80e9),
        model_arch=ModelArchConfig(
            num_params=1e8,
            d_model=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
        ),
        recompute_logprobs=False,
    )
    config = TrainParallelConfig(tp=1, ep=1, pp=2, dp=1, b_micro=1)
    lengths = [128, 2048, 256, 4096]

    mixed_time, details = model.evaluate_training(config, 4, lengths)
    average_time, _ = model.evaluate_training(
        config, 4, round(sum(lengths) / len(lengths))
    )

    schedule = details["replica_schedules"][0]
    assert schedule["micro_lengths"] == lengths
    assert schedule["dynamic_micro_bubble_s"] > 0
    assert math.isfinite(mixed_time)
    assert mixed_time != pytest.approx(average_time)


def test_decision_tree_prunes_tp_and_ep_communication_branches():
    model = CostModel(
        hardware=HardwareConfig(mem_capacity=80e9),
        model_arch=ModelArchConfig(
            num_params=1e8,
            d_model=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            is_moe=True,
            num_experts=4,
            num_activated_experts=2,
        ),
        recompute_logprobs=False,
    )
    optimizer = TwoLevelNestedOptimizer(
        model,
        train_comm_compute_ratio_threshold=0.0,
        train_ep_comm_compute_ratio_threshold=0.0,
        train_pipeline_bubble_ratio_threshold=1.0,
    )

    configs = optimizer.decision_tree_search(
        4,
        B_global=4,
        sequence_lengths=[128, 256, 512, 1024],
        allowed_train_tp=[1, 2],
        allowed_train_ep=[1, 2],
        allowed_train_pp=[1],
        micro_batch_sizes=[1],
    )

    assert configs
    assert all(config.tp == 1 for config in configs)
    assert all(config.ep == 1 for config in configs)


def test_ep_plan_maps_to_megatron_runtime_dp_without_losing_gpu_budget():
    config = AsyncRLConfig(
        model_path="/tmp/model",
        train_gpus=4,
        rollout_gpus=4,
        n_total_gpus=8,
        train_tp_size=1,
        train_dp_size=4,
        train_ep_size=1,
        batch_size=8,
        micro_batch_size=1,
        model_arch=ModelArchConfig(
            num_params=1e8,
            d_model=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            is_moe=True,
            num_experts=4,
            num_activated_experts=2,
        ),
    )
    planner = GlobalResourcePlanner.from_config(config)
    plan = GlobalResourcePlan(
        train_config=TrainParallelConfig(tp=1, ep=2, pp=1, dp=2, b_micro=1),
        rollout_config=RolloutClusterConfig(tp_list=[2, 2]),
        t_train=1.0,
        t_rollout=1.0,
        t_global=1.0,
        n_total_gpus=8,
        max_concurrent_rollouts=8,
    )

    planner.apply_plan_to_config(plan, config)

    assert config.train_ep_size == 2
    assert config.train_dp_size == 4
    assert config.train_tp_size * config.train_pp_size * config.train_dp_size == 4
    assert config.train_gpus + config.rollout_gpus == 8

"""Support code for Test cost model."""

import sys
import os
import json
import time
import numpy as np


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import types
_pkg = types.ModuleType("RL_Framework")
_pkg.__path__ = [_project_root]
_pkg.__package__ = "RL_Framework"
sys.modules["RL_Framework"] = _pkg

from RL_Framework.config import HardwareConfig, ModelArchConfig, ProfilingConfig
from RL_Framework.infra.cost_model.model import (
    CostModel,
    RolloutCostModel,
    TrainingCostModel,
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)
from RL_Framework.infra.cost_model.calibrator import (
    ShadowCalibrator,
    EpochRunData,
    CalibratedParams,
)
from RL_Framework.infra.cost_model.optimizer import (
    TwoLevelNestedOptimizer,
    OptimizationResult,
    generate_training_configs,
    generate_rollout_configs,
)
from RL_Framework.infra.cost_model.resource_alloc import CostModelAllocator


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_rollout_cost_model():
    separator("Test 1: RolloutCostModel (PagedAttention + Roofline + steady-state projection)")

    hw = HardwareConfig()
    ma = ModelArchConfig(num_params=7e9, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4)
    prof = ProfilingConfig()

    model = RolloutCostModel(
        num_params=ma.num_params,
        d_model=ma.d_model,
        n_layers=ma.n_layers,
        n_kv_heads=ma.n_kv_heads,
        head_dim=ma.head_dim,
        dtype_bytes=ma.dtype_bytes,
        flops_peak=hw.flops_peak,
        mem_bw=hw.mem_bw,
        mem_capacity=hw.mem_capacity,
        tp_comm_overhead=hw.tp_comm_overhead,
        prefill_mfu=prof.prefill_mfu,
        decode_bw_util=prof.decode_bw_util,
        prefill_chunk_size=prof.prefill_chunk_size,
        kv_frag_rate=prof.kv_frag_rate,
        act_workspace_bytes=prof.act_workspace_bytes,
    )


    print("\n  [memory model]")
    for tp in [1, 2, 4, 8]:
        s_token = model.compute_token_kv_bytes(tp)
        kv_pool = model.compute_kv_pool(tp)
        n_max = model.compute_token_capacity(tp)
        print(f"    TP={tp}: S_token_kv={s_token:.0f}B, "
              f"KV_pool={kv_pool/1e9:.1f}GB, N_max={n_max} tokens")


    print("\n  [Roofline step time]")
    for tp in [1, 2, 4]:
        t_prefill = model.compute_prefill_step_time(B_prefill=1, L_chunk=1024, tp=tp)
        t_decode_small = model.compute_decode_step_time(total_active_tokens=4096, tp=tp)
        t_decode_large = model.compute_decode_step_time(total_active_tokens=32768, tp=tp)
        print(f"    TP={tp}: prefill_chunk={t_prefill*1000:.2f}ms, "
              f"decode(4k tok)={t_decode_small*1000:.2f}ms, "
              f"decode(32k tok)={t_decode_large*1000:.2f}ms")


    print("\n  [steady-state projection]")
    for tp in [1, 2, 4]:
        b_bar = model.compute_steady_state_batch(tp, avg_prompt_len=300, avg_gen_len=500)
        print(f"    TP={tp}: steady_B={b_bar:.1f}")


    print("\n  [single-instance makespan]")
    requests = [
        RequestInfo(prompt_length=256, gen_length=300),
        RequestInfo(prompt_length=512, gen_length=200),
        RequestInfo(prompt_length=128, gen_length=800),
    ]
    for tp in [1, 2, 4]:
        t = model.compute_instance_makespan(requests, tp=tp)
        print(f"    TP={tp}, 3 reqs: {t:.4f}s")


    print("\n  [OOM check]")
    for seq_len, tp in [(4096, 1), (4096, 2), (16384, 1), (16384, 4), (32768, 8)]:
        oom = model.check_oom(seq_len, tp)
        n_max = model.compute_token_capacity(tp)
        print(f"    seq={seq_len}, TP={tp}, N_max={n_max}: {'OOM!' if oom else 'OK'}")


    print("\n  [cluster evaluation (greedy throughput allocation)]")
    cluster = RolloutClusterConfig(tp_list=[4, 2, 2])
    np.random.seed(42)
    requests = [
        RequestInfo(
            prompt_length=int(np.random.uniform(128, 512)),
            gen_length=int(np.random.uniform(64, 1024)),
        )
        for _ in range(50)
    ]
    makespan, details = model.evaluate_cluster(cluster, requests)
    print(f"    Cluster={cluster.tp_list}, 50 reqs")
    print(f"    Makespan: {makespan:.4f}s")
    print(f"    Instance times: {[f'{t:.4f}' for t in details.get('instance_times', [])]}")
    print(f"    Reqs/instance: {details.get('requests_per_instance', [])}")

    print("\n  [PASS] RolloutCostModel tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_training_cost_model():
    separator("Test 2: TrainingCostModel (memory model + Profiling + 1F1B)")

    hw = HardwareConfig()
    ma = ModelArchConfig(num_params=7e9, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4)
    prof = ProfilingConfig()

    model = TrainingCostModel(
        num_params=ma.num_params,
        d_model=ma.d_model,
        n_layers=ma.n_layers,
        n_kv_heads=ma.n_kv_heads,
        head_dim=ma.head_dim,
        dtype_bytes=ma.dtype_bytes,
        flops_peak=hw.flops_peak,
        mem_bw=hw.mem_bw,
        mem_capacity=hw.mem_capacity,
        bw_intra_node=hw.bw_intra_node,
        bw_inter_node=hw.bw_inter_node,
        gpus_per_node=hw.gpus_per_node,
        latency_inter_node=hw.latency_inter_node,
        alpha_mlp_fwd=prof.alpha_mlp_fwd,
        beta_mlp_fwd=prof.beta_mlp_fwd,
        alpha_mlp_bwd=prof.alpha_mlp_bwd,
        beta_mlp_bwd=prof.beta_mlp_bwd,
        alpha_attn_fwd=prof.alpha_attn_fwd,
        beta_attn_fwd=prof.beta_attn_fwd,
        gamma_attn_fwd=prof.gamma_attn_fwd,
        alpha_attn_bwd=prof.alpha_attn_bwd,
        beta_attn_bwd=prof.beta_attn_bwd,
        gamma_attn_bwd=prof.gamma_attn_bwd,
        rho_compute_bound=prof.rho_compute_bound,
        rho_memory_bound=prof.rho_memory_bound,
        theta_slowdown=prof.theta_slowdown,
        train_mem_frag_rate=prof.train_mem_frag_rate,
        train_workspace_bytes=prof.train_workspace_bytes,
    )


    print("\n  [Memory model and OOM pruning]")
    configs = [
        TrainParallelConfig(tp=1, pp=1, dp=4, b_micro=4),
        TrainParallelConfig(tp=2, pp=1, dp=2, b_micro=4),
        TrainParallelConfig(tp=2, pp=2, dp=1, b_micro=4),
        TrainParallelConfig(tp=4, pp=1, dp=1, b_micro=4),
        TrainParallelConfig(tp=1, pp=1, dp=1, b_micro=8),
    ]
    L = 2048
    for cfg in configs:
        mem = model.estimate_memory_per_gpu(cfg, cfg.b_micro, L)
        oom = model.check_memory_oom(cfg, cfg.b_micro, L)
        print(f"    TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} b={cfg.b_micro}: "
              f"mem={mem/1e9:.2f}GB {'OOM!' if oom else 'OK'}")


    print("\n  [communication model]")
    for cfg in configs[:4]:
        t_tp = model.compute_tp_comm(cfg.tp, cfg.b_micro, L)
        t_pp = model.compute_pp_comm(cfg, L)
        t_dp = model.compute_dp_comm(cfg)
        print(f"    TP={cfg.tp} PP={cfg.pp} DP={cfg.dp}: "
              f"TP_comm={t_tp*1000:.2f}ms, PP_comm={t_pp*1000:.2f}ms, "
              f"DP_comm={t_dp*1000:.2f}ms")


    print("\n  [Profiled compute time for one layer]")
    for L_test in [1024, 2048, 4096, 8192]:
        t_mlp_fwd = model.compute_pure_mlp_time(L_test, b_micro=4, tp=2, is_bwd=False)
        t_attn_fwd = model.compute_pure_attn_time(L_test, b_micro=4, tp=2, is_bwd=False)
        t_mlp_bwd = model.compute_pure_mlp_time(L_test, b_micro=4, tp=2, is_bwd=True)
        t_attn_bwd = model.compute_pure_attn_time(L_test, b_micro=4, tp=2, is_bwd=True)
        print(f"    L={L_test}: mlp_fwd={t_mlp_fwd*1000:.3f}ms, attn_fwd={t_attn_fwd*1000:.3f}ms, "
              f"mlp_bwd={t_mlp_bwd*1000:.3f}ms, attn_bwd={t_attn_bwd*1000:.3f}ms")


    print("\n  [1F1B pipeline timeline]")
    for cfg in configs[:4]:
        t_fwd = model.compute_stage_fwd_time(L, cfg)
        t_bwd = model.compute_stage_bwd_time(L, cfg)
        n_micro = max(1, 32 // cfg.dp // cfg.b_micro)
        t_pipeline = model.compute_pipeline_time(t_fwd, t_bwd, n_micro, cfg.pp)
        print(f"    TP={cfg.tp} PP={cfg.pp} DP={cfg.dp}: "
              f"fwd={t_fwd*1000:.2f}ms, bwd={t_bwd*1000:.2f}ms, "
              f"n_micro={n_micro}, pipeline={t_pipeline*1000:.1f}ms")


    print("\n  [Full iteration time]")
    B_global = 32
    for cfg in configs[:4]:
        t_iter, details = model.compute_iteration_time(cfg, B_global, L)
        print(f"    TP={cfg.tp} PP={cfg.pp} DP={cfg.dp}: "
              f"T_iter={t_iter*1000:.1f}ms "
              f"(pipeline={details['t_pipeline']*1000:.1f}ms, "
              f"dp_exposed={details['t_dp_exposed']*1000:.1f}ms)")

    print("\n  [PASS] TrainingCostModel tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_shadow_calibrator():
    separator("Test 3: ShadowCalibrator (EMA + change-point detection)")

    calibrator = ShadowCalibrator(
        alpha_default=0.1,
        alpha_emergency=0.8,
        change_point_threshold=3.0,
        history_window=20,
        min_epochs_for_calibration=2,
    )

    print("  Initial parameters:", calibrator.get_calibrated_params())


    print("\n  [Phase 1] normal operation:")
    for epoch in range(5):
        data = EpochRunData(
            t_train_real=10.0 + np.random.normal(0, 0.5),
            t_rollout_real=8.0 + np.random.normal(0, 0.3),
            t_train_predicted=10.0,
            t_rollout_predicted=8.0,
            gen_lengths=list(np.random.normal(500, 100, size=50).clip(min=16)),
            prompt_lengths=list(np.random.normal(200, 50, size=50).clip(min=16)),
            epoch_id=epoch,
        )
        params = calibrator.update(data)
        print(f"    Epoch {epoch}: eta_comp={params.eta_comp:.3f}, "
              f"eta_mem={params.eta_mem:.3f}, mu_gen={params.mu_gen:.0f}")


    print("\n  [Phase 2] capability emergence:")
    for epoch in range(5, 10):
        data = EpochRunData(
            t_train_real=15.0 + np.random.normal(0, 0.5),
            t_rollout_real=20.0 + np.random.normal(0, 1.0),
            t_train_predicted=10.0,
            t_rollout_predicted=8.0,
            gen_lengths=list(np.random.normal(1500, 300, size=50).clip(min=16)),
            prompt_lengths=list(np.random.normal(200, 50, size=50).clip(min=16)),
            epoch_id=epoch,
        )
        params = calibrator.update(data)
        print(f"    Epoch {epoch}: eta_comp={params.eta_comp:.3f}, "
              f"eta_mem={params.eta_mem:.3f}, mu_gen={params.mu_gen:.0f}, "
              f"change_point={calibrator._is_change_point}")

    stats = calibrator.get_statistics()
    print(f"\n  Final statistics: {stats}")
    print("\n  [PASS] ShadowCalibrator tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_config_generation():
    separator("Test 4: configuration-space generation")

    train_configs = generate_training_configs(
        n_total_gpus=8,
        allowed_tp=[1, 2, 4],
        allowed_pp=[1, 2],
        micro_batch_sizes=[4],
    )
    print(f"  Training configuration space (8 GPUs): {len(train_configs)} configurations")
    for cfg in train_configs[:10]:
        print(f"    TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} ({cfg.n_gpus} GPUs)")
    if len(train_configs) > 10:
        print(f"    ... total {len(train_configs)} configurations")

    rollout_configs = generate_rollout_configs(
        n_gpus=4, allowed_tp_sizes=[1, 2, 4],
    )
    print(f"\n  Rollout configuration space (4 GPUs, TP=[1,2,4]): {len(rollout_configs)} configurations")
    for cfg in rollout_configs:
        print(f"    {cfg.tp_list}")

    print("\n  [PASS] Configuration-space generation tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_nested_optimizer():
    separator("Test 5: TwoLevelNestedOptimizer (two-level nested optimization)")

    hw = HardwareConfig()
    ma = ModelArchConfig(num_params=7e9, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4)
    cost_model = CostModel(hardware=hw, model_arch=ma)

    optimizer = TwoLevelNestedOptimizer(
        cost_model=cost_model,
        allowed_rollout_tp=[1, 2, 4],
        verbose=True,
    )


    np.random.seed(42)
    n_requests = 100
    requests = [
        RequestInfo(
            prompt_length=int(np.random.uniform(150, 400)),
            gen_length=int(np.random.exponential(300) + 50),
        )
        for _ in range(n_requests)
    ]
    gen_lengths = [r.gen_length for r in requests]
    print(f"  Request distribution: n={n_requests}, "
          f"gen_mean={np.mean(gen_lengths):.0f}, "
          f"gen_p95={np.percentile(gen_lengths, 95):.0f}, "
          f"gen_max={max(gen_lengths)}")


    result = optimizer.optimize(
        n_total_gpus=8,
        requests=requests,
        B_global=32,
        allowed_train_tp=[1, 2, 4],
        allowed_train_pp=[1, 2],
        micro_batch_sizes=[4],
    )

    print(result.summary())


    assert result.t_global < float("inf"), "Optimizer did not find a feasible solution!"
    assert result.train_config is not None, "Missing training configuration!"
    assert result.rollout_config is not None, "Missing rollout configuration!"
    assert result.train_config.n_gpus + result.rollout_config.n_gpus <= 8, "GPU budget exceeded!"


    print("\n  --- Rollout-only optimization ---")
    rc, makespan, details = optimizer.optimize_rollout_only(
        n_rollout_gpus=4,
        requests=requests,
    )
    print(f"  Optimal rollout configuration: {rc.tp_list if rc else None}")
    print(f"  Makespan: {makespan:.6f}s")

    print("\n  [PASS] TwoLevelNestedOptimizer tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_cost_model_allocator():
    separator("Test 6: CostModelAllocator (end-to-end integration)")

    hw = HardwareConfig()
    ma = ModelArchConfig(num_params=7e9, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4)

    allocator = CostModelAllocator(
        hardware=hw,
        model_arch=ma,
        allowed_rollout_tp=[1, 2, 4],
        verbose=False,
    )


    np.random.seed(123)
    length_dist = [
        (int(np.random.uniform(150, 400)), int(np.random.exponential(300) + 50))
        for _ in range(80)
    ]

    result = allocator.allocate_resources(
        n_total_gpus=8,
        length_distribution=length_dist,
        B_global=32,
    )
    print(f"  Optimization result:")
    print(f"    Training: TP={result.train_config.tp}, PP={result.train_config.pp}, "
          f"DP={result.train_config.dp}" if result.train_config else "    Training: None")
    print(f"    Rollout: {result.rollout_config.tp_list}" if result.rollout_config else "    Rollout: None")
    print(f"    T_global: {result.t_global:.6f}s")


    for epoch in range(3):
        epoch_data = EpochRunData(
            t_train_real=result.t_train * (1 + np.random.normal(0, 0.1)),
            t_rollout_real=result.t_rollout * (1 + np.random.normal(0, 0.1)),
            t_train_predicted=result.t_train,
            t_rollout_predicted=result.t_rollout,
            gen_lengths=[gl for _, gl in length_dist],
            prompt_lengths=[pl for pl, _ in length_dist],
            epoch_id=epoch,
        )
        cal_params = allocator.update_calibration(epoch_data)
        print(f"    Calibration epoch {epoch}: eta_comp={cal_params['eta_comp']:.3f}, "
              f"eta_mem={cal_params['eta_mem']:.3f}")

    report = allocator.get_optimization_report()
    print(f"    Optimization count: {report['optimization_count']}")
    print(f"    Calibration confidence: {report['calibration_stats']['confidence']:.2f}")

    print("\n  [PASS] CostModelAllocator integration tests passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_performance_benchmark():
    separator("Test 7: optimizer benchmark")

    hw = HardwareConfig()
    ma = ModelArchConfig(num_params=7e9, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4)
    cost_model = CostModel(hardware=hw, model_arch=ma)
    optimizer = TwoLevelNestedOptimizer(
        cost_model=cost_model,
        allowed_rollout_tp=[1, 2, 4],
    )

    np.random.seed(0)
    requests = [
        RequestInfo(
            prompt_length=int(np.random.uniform(100, 500)),
            gen_length=int(np.random.exponential(400) + 30),
        )
        for _ in range(200)
    ]


    times = []
    for _ in range(5):
        start = time.perf_counter()
        result = optimizer.optimize(
            n_total_gpus=8,
            requests=requests,
            B_global=32,
            allowed_train_tp=[1, 2, 4],
            allowed_train_pp=[1, 2],
            micro_batch_sizes=[4],
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"  8-GPU, 200 requests, TP=[1,2,4], PP=[1,2]:")
    print(f"    Average optimization time: {avg_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"    Configurations explored: {result.n_configs_explored}")
    print(f"    OOM prunes: {result.n_configs_pruned_oom}")
    print(f"    Early-stop prunes: {result.n_configs_pruned_early_stop}")

    if avg_ms < 1000:
        print("    [OK] Optimization time < 1s; suitable for online execution")
    else:
        print("    [WARN] Optimization is slow; consider reducing the search space")

    print("\n  [PASS] Performance benchmark passed")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

_KNOWN_MODEL_ARCH = {
    "Qwen2.5-3B":  dict(num_params=3.0e+9, d_model=2048, n_layers=36, n_heads=16, n_kv_heads=2, vocab_size=151936, intermediate_size=11008),
    "Qwen2.5-7B":  dict(num_params=7.0e+9, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4, vocab_size=152064, intermediate_size=18944),
    "Qwen2.5-14B": dict(num_params=14.0e+9, d_model=5120, n_layers=40, n_heads=40, n_kv_heads=8, vocab_size=152064, intermediate_size=13824),
    "Qwen2.5-32B": dict(num_params=32.0e+9, d_model=5120, n_layers=64, n_heads=40, n_kv_heads=8, vocab_size=152064, intermediate_size=27648),
    "Qwen2.5-72B": dict(num_params=72.0e+9, d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=152064, intermediate_size=29568),
    "Qwen3-4B":    dict(num_params=4.0e+9, d_model=2560, n_layers=36, n_heads=32, n_kv_heads=4, vocab_size=151936, intermediate_size=9728),
    "Qwen3-8B":    dict(num_params=8.0e+9, d_model=4096, n_layers=36, n_heads=32, n_kv_heads=8, vocab_size=151936, intermediate_size=12288),
    "Qwen3-14B":   dict(num_params=14.0e+9, d_model=5120, n_layers=40, n_heads=40, n_kv_heads=8, vocab_size=151936, intermediate_size=17408),
    "Qwen3-32B":   dict(num_params=32.0e+9, d_model=5120, n_layers=64, n_heads=40, n_kv_heads=8, vocab_size=151936, intermediate_size=25600),
    "Qwen3-30B-A3B": dict(num_params=30.0e+9, d_model=2048, n_layers=48, n_heads=32, n_kv_heads=4, vocab_size=151936, intermediate_size=6144),
}


def _match_model_arch(model_name: str) -> dict:
    """Match model arch."""
    for key, params in _KNOWN_MODEL_ARCH.items():
        if key in model_name:
            return params
    print(f"  [WARN] Unknown model: {model_name[:60]}..., using default Qwen2.5-7B parameters")
    return _KNOWN_MODEL_ARCH["Qwen2.5-7B"]


def load_sequence_profile(profile_path: str) -> dict:
    """Load sequence profile."""
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def profile_to_requests(profile: dict) -> list[RequestInfo]:
    """Profile to requests."""
    requests = []
    for seq in profile.get("sequences", []):
        prompt_len = int(seq.get("input_tokens", 0))
        gen_len = int(seq.get("actual_output_tokens", seq.get("estimated_output_tokens", 0)))
        if prompt_len > 0 and gen_len > 0:
            requests.append(RequestInfo(prompt_length=prompt_len, gen_length=gen_len))
    return requests


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def test_real_profile():
    separator("Test 8: real-profile end-to-end test")

    profile_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile")
    if not os.path.isdir(profile_base):
        print(f"  [SKIP] Profile directory does not exist: {profile_base}")
        print("  [PASS] Skipping real-profile test (no data)")
        return

    profiles = []
    for dataset_dir in sorted(os.listdir(profile_base)):
        dataset_path = os.path.join(profile_base, dataset_dir)
        profile_file = os.path.join(dataset_path, "sequence_profile.json")
        if os.path.isdir(dataset_path) and os.path.isfile(profile_file):
            profiles.append((dataset_dir, profile_file))

    if not profiles:
        print("  [SKIP] No sequence_profile.json files found")
        print("  [PASS] Skipping real-profile test (no data)")
        return

    print(f"  Found {len(profiles)} profile datasets: {[p[0] for p in profiles]}")

    for dataset_name, profile_path in profiles:
        print(f"\n  {'─'*50}")
        print(f"  Dataset: {dataset_name}")
        print(f"  {'─'*50}")

        profile = load_sequence_profile(profile_path)
        model_name = profile.get("model_name", "unknown")
        arch_params = _match_model_arch(model_name)
        ma = ModelArchConfig(**arch_params)

        print(f"  Model: ...{model_name[-50:]}")
        print(f"  Architecture: P={ma.num_params:.1e}, d={ma.d_model}, L={ma.n_layers}, "
              f"kv_heads={ma.n_kv_heads}")

        requests = profile_to_requests(profile)
        if not requests:
            print("  [SKIP] No valid sequence data")
            continue

        gen_lengths = [r.gen_length for r in requests]
        prompt_lengths = [r.prompt_length for r in requests]
        total_lengths = [r.total_length for r in requests]

        print(f"  Sequences: {len(requests)}")
        print(f"  Prompt: mean={np.mean(prompt_lengths):.0f}, max={max(prompt_lengths)}")
        print(f"  Gen:    mean={np.mean(gen_lengths):.0f}, "
              f"p95={np.percentile(gen_lengths, 95):.0f}, max={max(gen_lengths)}")
        print(f"  Total:  mean={np.mean(total_lengths):.0f}, max={max(total_lengths)}")


        hw = HardwareConfig()
        cost_model = CostModel(hardware=hw, model_arch=ma)


        print("\n  [Rollout time prediction]")
        for tp in [1, 2, 4, 8]:
            oom = cost_model.rollout_model.check_oom(max(total_lengths), tp)
            if oom:
                print(f"    TP={tp}: OOM (N_max={cost_model.rollout_model.compute_token_capacity(tp)})")
                continue

            predicted_times = []
            for req in requests:
                t = cost_model.rollout_model.compute_instance_makespan([req], tp)
                predicted_times.append(t)
            pred_mean = np.mean(predicted_times)
            pred_p50 = np.percentile(predicted_times, 50)
            print(f"    TP={tp}: pred_mean={pred_mean:.3f}s, pred_p50={pred_p50:.3f}s")


        print(f"\n  [two-level optimizer]")
        if ma.num_params >= 30e9:
            n_total_gpus, allowed_tp, allowed_pp = 16, [1, 2, 4, 8], [1, 2, 4]
        elif ma.num_params >= 10e9:
            n_total_gpus, allowed_tp, allowed_pp = 16, [1, 2, 4, 8], [1, 2, 4]
        else:
            n_total_gpus, allowed_tp, allowed_pp = 16, [1, 2, 4, 8], [1, 2, 4]

        optimizer = TwoLevelNestedOptimizer(
            cost_model=cost_model, allowed_rollout_tp=allowed_tp, verbose=False,
        )

        start = time.perf_counter()

        result = optimizer.optimize(
            n_total_gpus=n_total_gpus, requests=requests, B_global=64,
            allowed_train_tp=allowed_tp, allowed_train_pp=allowed_pp,
            micro_batch_sizes=[16],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.train_config and result.rollout_config:
            print(f"    GPU={n_total_gpus}: "
                  f"Train(TP={result.train_config.tp},PP={result.train_config.pp},"
                  f"DP={result.train_config.dp}) + "
                  f"Rollout({result.rollout_config.tp_list})")
            print(f"    T_train={result.t_train:.4f}s, T_rollout={result.t_rollout:.4f}s, "
                  f"T_global={result.t_global:.4f}s")
            print(f"    Search: {result.n_configs_explored} configs, "
                  f"OOM={result.n_configs_pruned_oom}, "
                  f"early stop={result.n_configs_pruned_early_stop}")
            print(f"    Optimization time: {elapsed_ms:.1f}ms")
            total_used = result.train_config.n_gpus + result.rollout_config.n_gpus
            assert total_used <= n_total_gpus, f"GPU budget exceeded: {total_used} > {n_total_gpus}"
        else:
            print(f"    No feasible solution")

    print("\n  [PASS] Real-profile end-to-end test passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Cost Model v2 and two-level optimizer end-to-end tests")
    print("=" * 60)

    test_rollout_cost_model()
    test_training_cost_model()
    test_shadow_calibrator()
    test_config_generation()
    test_nested_optimizer()
    test_cost_model_allocator()
    test_performance_benchmark()
    test_real_profile()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

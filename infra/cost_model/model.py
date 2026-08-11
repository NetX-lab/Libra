"""Support code for Model."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class RequestInfo:
    """Request info implementation."""
    prompt_length: int
    gen_length: int

    @property
    def total_length(self) -> int:
        return self.prompt_length + self.gen_length


@dataclass
class InstanceConfig:
    """Instance config implementation."""
    tp: int
    token_capacity: int = 0
    assigned_requests: list[RequestInfo] = field(default_factory=list)


@dataclass
class TrainParallelConfig:
    """Train parallel config implementation."""
    tp: int = 1
    pp: int = 1
    dp: int = 1
    b_micro: int = 4       # micro batch size
    zero_level: int = 2
    cp: int = 1

    @property
    def n_gpus(self) -> int:
        return self.tp * self.pp * self.cp * self.dp


@dataclass
class RolloutClusterConfig:
    """Rollout cluster config implementation."""
    tp_list: list[int] = field(default_factory=list)  # e.g. [4, 2, 1, 1]

    @property
    def n_gpus(self) -> int:
        return sum(self.tp_list)

    @property
    def n_instances(self) -> int:
        return len(self.tp_list)


@dataclass
class CostModelResult:
    """Cost model result implementation."""
    t_train: float = 0.0
    t_rollout: float = 0.0
    t_global: float = 0.0          # max(t_train, t_rollout)
    is_oom: bool = False
    oom_source: str = ""            # "train" / "rollout" / ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class RolloutCostModel:
    """Rollout cost model implementation."""

    def __init__(
        self,
        num_params: float,
        d_model: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype_bytes: int,
        flops_peak: float,
        mem_bw: float,
        mem_capacity: float,
        tp_comm_overhead: float,
        prefill_mfu: float = 0.55,
        decode_bw_util: float = 0.85,
        prefill_chunk_size: int = 1024,
        kv_frag_rate: float = 0.04,
        act_workspace_bytes: float = 1.0e9,
        max_queue_limit: int = 256,
        effective_num_params: float | None = None,
    ):
        self.P = num_params
        self.P_active = effective_num_params or num_params
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.flops_peak = flops_peak
        self.mem_bw = mem_bw
        self.mem_capacity = mem_capacity
        self.tp_comm_overhead = tp_comm_overhead
        self.prefill_mfu = prefill_mfu
        self.decode_bw_util = decode_bw_util
        self.L_chunk = prefill_chunk_size
        self.kv_frag_rate = kv_frag_rate
        self.act_workspace = act_workspace_bytes
        self.max_queue_limit = max_queue_limit

    # ========================================================================

    # ========================================================================

    def compute_token_kv_bytes(self, tp: int) -> float:
        """Compute token kv bytes."""
        kv_heads_per_gpu = max(1, self.n_kv_heads // tp)
        return 2.0 * self.n_layers * kv_heads_per_gpu * self.head_dim * self.dtype_bytes

    def compute_kv_pool(self, tp: int) -> float:
        """Compute kv pool."""
        total_mem = tp * self.mem_capacity
        weights_total = self.P * self.dtype_bytes
        kv_pool = total_mem - weights_total - self.act_workspace
        return max(kv_pool, 0.0)

    def compute_token_capacity(self, tp: int) -> int:
        """Compute token capacity."""
        kv_pool = self.compute_kv_pool(tp)
        s_token_kv = self.compute_token_kv_bytes(tp)
        if s_token_kv <= 0:
            return 0
        raw_capacity = kv_pool / s_token_kv
        return int(raw_capacity * (1.0 - self.kv_frag_rate))

    # ========================================================================

    # ========================================================================

    def compute_prefill_step_time(self, B_prefill: int, L_chunk: int, tp: int) -> float:
        """Compute prefill step time."""
        d = self.d_model

        if self.P_active != self.P:
            linear_flops_per_layer = (
                2.0
                * self.P_active
                / max(self.n_layers, 1)
                * B_prefill
                * L_chunk
            )
        else:
            linear_flops_per_layer = (
                24.0 * B_prefill * L_chunk * (d ** 2)
            )

        attn_flops_per_layer = 4.0 * B_prefill * (L_chunk ** 2) * d

        total_flops = self.n_layers * (linear_flops_per_layer + attn_flops_per_layer)

        effective_flops = tp * self.flops_peak * max(self.prefill_mfu, 1e-6)
        t_comp = total_flops / effective_flops

        t_comm = 2.0 * self.n_layers * self.tp_comm_overhead if tp > 1 else 0.0
        return t_comp + t_comm

    def compute_decode_step_time(self, total_active_tokens: int, tp: int) -> float:
        """Compute decode step time."""

        weights_per_gpu = self.P_active * self.dtype_bytes / tp
        s_token_kv = self.compute_token_kv_bytes(tp)  # per-GPU per-token
        kv_read_per_gpu = total_active_tokens * s_token_kv

        d_read = weights_per_gpu + kv_read_per_gpu

        effective_bw = tp * self.mem_bw * max(self.decode_bw_util, 1e-6)
        t_mem = d_read / effective_bw

        t_comm = 2.0 * self.n_layers * self.tp_comm_overhead if tp > 1 else 0.0
        return t_mem + t_comm

    # ========================================================================

    # ========================================================================

    def compute_instance_throughput(self, tp: int, avg_prompt_len: float, avg_gen_len: float) -> float:
        """Compute instance throughput."""
        b_bar = self.compute_steady_state_batch(tp, avg_prompt_len, avg_gen_len)
        avg_active_tokens = b_bar * (avg_prompt_len + avg_gen_len / 2.0)
        t_step = self.compute_decode_step_time(int(avg_active_tokens), tp)
        if t_step <= 0:
            return float("inf")
        return b_bar / t_step

    def greedy_throughput_allocation(
        self,
        requests: list[RequestInfo],
        cluster: RolloutClusterConfig,
    ) -> list[InstanceConfig]:
        """Greedy throughput allocation."""
        instances = [InstanceConfig(tp=tp_val) for tp_val in cluster.tp_list]
        if not requests or not instances:
            return instances


        sorted_requests = sorted(requests, key=lambda r: r.total_length, reverse=True)


        avg_prompt = float(np.mean([r.prompt_length for r in requests]))
        avg_gen = float(np.mean([r.gen_length for r in requests]))

        throughputs = []
        for inst in instances:
            thru = self.compute_instance_throughput(inst.tp, avg_prompt, avg_gen)
            throughputs.append(thru)


        workloads = [0.0] * len(instances)

        for req in sorted_requests:
            best_idx = -1
            best_finish = float("inf")
            for i, inst in enumerate(instances):
                new_workload = workloads[i] + req.gen_length + req.prompt_length
                if throughputs[i] > 0:
                    finish = new_workload / throughputs[i]
                else:
                    finish = float("inf")
                if finish < best_finish:
                    best_finish = finish
                    best_idx = i
            instances[best_idx].assigned_requests.append(req)
            workloads[best_idx] += req.gen_length + req.prompt_length

        return instances

    # ========================================================================

    # ========================================================================

    def compute_steady_state_batch(
        self, tp: int, avg_prompt_len: float, avg_gen_len: float,
    ) -> float:
        """Compute steady state batch."""
        n_max = self.compute_token_capacity(tp)
        avg_occupancy = avg_prompt_len + avg_gen_len / 2.0
        if avg_occupancy <= 0:
            return 1.0
        steady_b = n_max / avg_occupancy
        return max(1.0, min(float(self.max_queue_limit), steady_b))

    def compute_instance_makespan(
        self, requests: list[RequestInfo], tp: int,
    ) -> float:
        """Compute instance makespan."""
        if not requests:
            return 0.0


        avg_prompt = float(np.mean([r.prompt_length for r in requests]))
        avg_gen = float(np.mean([r.gen_length for r in requests]))
        total_prompt_tokens = sum(r.prompt_length for r in requests)
        total_gen_tokens = sum(r.gen_length for r in requests)


        b_bar = self.compute_steady_state_batch(tp, avg_prompt, avg_gen)



        t_prefill_step = self.compute_prefill_step_time(
            B_prefill=1, L_chunk=self.L_chunk, tp=tp
        )
        n_prefill_steps = total_prompt_tokens / max(self.L_chunk, 1)
        total_prefill_time = n_prefill_steps * t_prefill_step



        avg_active_tokens_per_seq = avg_prompt + avg_gen / 2.0
        total_active_tokens = int(b_bar * avg_active_tokens_per_seq)
        t_decode_step = self.compute_decode_step_time(total_active_tokens, tp)


        n_decode_steps = total_gen_tokens / max(b_bar, 1.0)
        total_decode_time = n_decode_steps * t_decode_step

        return total_prefill_time + total_decode_time

    # ========================================================================

    # ========================================================================

    def check_oom(self, max_total_seq: int, tp: int) -> bool:
        """Check oom."""
        n_max = self.compute_token_capacity(tp)
        return max_total_seq > n_max

    def evaluate_cluster(
        self,
        cluster: RolloutClusterConfig,
        requests: list[RequestInfo],
    ) -> tuple[float, dict]:
        """Evaluate cluster."""
        if not cluster.tp_list:
            return float("inf"), {"error": "empty cluster"}


        if requests:
            max_seq = max(r.total_length for r in requests)
            max_tp = max(cluster.tp_list)
            if self.check_oom(max_seq, max_tp):
                return float("inf"), {
                    "oom": True, "max_seq": max_seq,
                    "max_tp": max_tp,
                    "token_capacity": self.compute_token_capacity(max_tp),
                }


        instances = self.greedy_throughput_allocation(requests, cluster)


        instance_times = []
        for inst in instances:
            t = self.compute_instance_makespan(inst.assigned_requests, inst.tp)
            instance_times.append(t)


        makespan = max(instance_times) if instance_times else 0.0
        details = {
            "instance_times": instance_times,
            "n_instances": len(instances),
            "requests_per_instance": [len(inst.assigned_requests) for inst in instances],
            "makespan": makespan,
            "token_capacities": [self.compute_token_capacity(inst.tp) for inst in instances],
        }
        return makespan, details


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class TrainingCostModel:
    """Training cost model implementation."""

    def __init__(
        self,
        num_params: float,
        d_model: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype_bytes: int,
        flops_peak: float,
        mem_bw: float,
        mem_capacity: float,
        bw_intra_node: float,
        bw_inter_node: float,
        gpus_per_node: int,
        latency_inter_node: float = 5e-6,

        alpha_mlp_fwd: float = 3.5e-8,
        beta_mlp_fwd: float = 1.0e-5,
        alpha_mlp_bwd: float = 7.0e-8,
        beta_mlp_bwd: float = 2.0e-5,
        alpha_attn_fwd: float = 1.5e-12,
        beta_attn_fwd: float = 2.5e-8,
        gamma_attn_fwd: float = 8.0e-6,
        alpha_attn_bwd: float = 3.0e-12,
        beta_attn_bwd: float = 5.0e-8,
        gamma_attn_bwd: float = 1.5e-5,

        rho_compute_bound: float = 0.08,
        rho_memory_bound: float = 0.45,
        theta_slowdown: float = 1.0,

        train_mem_frag_rate: float = 0.08,
        train_workspace_bytes: float = 0.5e9,
        effective_num_params: float | None = None,
    ):
        self.P = num_params
        self.P_active = effective_num_params or num_params
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.flops_peak = flops_peak
        self.mem_bw = mem_bw
        self.mem_capacity = mem_capacity
        self.bw_intra = bw_intra_node
        self.bw_inter = bw_inter_node
        self.gpus_per_node = gpus_per_node
        self.latency_inter = latency_inter_node

        self.alpha_mlp_fwd = alpha_mlp_fwd
        self.beta_mlp_fwd = beta_mlp_fwd
        self.alpha_mlp_bwd = alpha_mlp_bwd
        self.beta_mlp_bwd = beta_mlp_bwd
        self.alpha_attn_fwd = alpha_attn_fwd
        self.beta_attn_fwd = beta_attn_fwd
        self.gamma_attn_fwd = gamma_attn_fwd
        self.alpha_attn_bwd = alpha_attn_bwd
        self.beta_attn_bwd = beta_attn_bwd
        self.gamma_attn_bwd = gamma_attn_bwd

        self.rho_compute = rho_compute_bound
        self.rho_memory = rho_memory_bound
        self.theta = theta_slowdown

        self.mem_frag_rate = train_mem_frag_rate
        self.workspace = train_workspace_bytes



    def estimate_memory_per_gpu(
        self, config: TrainParallelConfig, b_micro: int, L: int,
    ) -> float:
        """Estimate memory per gpu."""
        tp, pp, dp = config.tp, config.pp, config.dp
        P = self.P

        if config.zero_level >= 2:
            shard = max(tp * pp * dp, 1)
            tp_pp = max(tp * pp, 1)
            m_weight = 2.0 * P / shard
            m_opt = 2.0 * P / shard
            m_grad = 2.0 * P / shard
            # FSDP all-gathers full layer shards during compute and PyTorch
            # keeps allocator/cache headroom around optimizer.step().
            m_fsdp_transient = 0.3 * 2.0 * P / tp_pp
            if P >= 20.0e9:
                m_fsdp_transient += 12.0e9
        else:
            tp_pp = max(tp * pp, 1)
            m_weight = 2.0 * P / tp_pp
            if config.zero_level >= 1:
                m_opt = 12.0 * P / max(tp * pp * dp, 1)
            else:
                m_opt = 12.0 * P / tp_pp
            m_grad = 2.0 * P / tp_pp
            m_fsdp_transient = 0.0


        # M_act = b_micro * L * d / TP * (n_layers / PP) * bytes_per_element
        layers_per_pp = self.n_layers / max(pp, 1)
        m_act = b_micro * L * self.d_model * self.dtype_bytes * layers_per_pp / tp

        return m_weight + m_opt + m_grad + m_fsdp_transient + m_act + self.workspace

    def check_memory_oom(
        self, config: TrainParallelConfig, b_micro: int, L: int,
    ) -> bool:
        """Check memory oom."""
        mem_per_gpu = self.estimate_memory_per_gpu(config, b_micro, L)
        available = self.mem_capacity * (1.0 - self.mem_frag_rate)
        return mem_per_gpu > available



    def compute_tp_comm(self, tp: int, b_micro: int, L: int) -> float:
        """Compute tp comm."""
        if tp <= 1:
            return 0.0

        s_act = b_micro * L * self.d_model * self.dtype_bytes
        # Ring AllReduce: 2*(TP-1)/TP * S
        allreduce_volume = 2.0 * (tp - 1) / tp * s_act

        t_per_layer = 2.0 * allreduce_volume / self.bw_intra
        return t_per_layer

    def compute_pp_comm(self, config: TrainParallelConfig, L: int) -> float:
        """Compute pp comm."""
        if config.pp <= 1:
            return 0.0
        s_activation = config.b_micro * L * self.d_model * self.dtype_bytes
        return s_activation / self.bw_inter + self.latency_inter

    def compute_dp_comm(self, config: TrainParallelConfig) -> float:
        """Compute dp comm."""
        if config.dp <= 1:
            return 0.0

        grad_per_rank = 2.0 * self.P / (config.tp * config.pp)
        s_grad_chunk = grad_per_rank


        tp_pp_per_node = config.tp * config.pp
        if tp_pp_per_node <= self.gpus_per_node:

            bw = self.bw_inter
        else:
            bw = self.bw_intra

        return (config.dp - 1) / config.dp * s_grad_chunk / bw



    def compute_pure_mlp_time(self, L: int, b_micro: int, tp: int, is_bwd: bool = False) -> float:
        """Compute pure mlp time."""
        if is_bwd:
            alpha, beta = self.alpha_mlp_bwd, self.beta_mlp_bwd
        else:
            alpha, beta = self.alpha_mlp_fwd, self.beta_mlp_fwd
        return (alpha * L + beta) * b_micro / max(tp, 1)

    def compute_pure_attn_time(self, L: int, b_micro: int, tp: int, is_bwd: bool = False) -> float:
        """Compute pure attn time."""
        if is_bwd:
            alpha, beta, gamma = self.alpha_attn_bwd, self.beta_attn_bwd, self.gamma_attn_bwd
        else:
            alpha, beta, gamma = self.alpha_attn_fwd, self.beta_attn_fwd, self.gamma_attn_fwd
        return (alpha * L * L + beta * L + gamma) * b_micro / max(tp, 1)

    def compute_actual_op_time(self, t_pure: float, is_compute_bound: bool, bg_bw_ratio: float) -> float:
        """Compute actual op time."""
        rho = self.rho_compute if is_compute_bound else self.rho_memory
        slowdown = 1.0 + rho * (bg_bw_ratio ** self.theta)
        return t_pure * slowdown

    def _determine_bg_bw_ratio(self, config: TrainParallelConfig) -> float:
        """Determine bg bw ratio."""
        if config.zero_level >= 2 and config.dp > 1:

            return 0.4
        elif config.zero_level >= 1 and config.dp > 1:
            return 0.2
        else:
            return 0.0

    def compute_stage_fwd_time(self, L: int, config: TrainParallelConfig) -> float:
        """Compute stage fwd time."""
        tp, pp = config.tp, config.pp
        b_micro = config.b_micro
        layers_per_pp = self.n_layers // max(pp, 1)
        bg_ratio = self._determine_bg_bw_ratio(config)


        t_attn = self.compute_pure_attn_time(L, b_micro, tp, is_bwd=False)
        t_mlp = self.compute_pure_mlp_time(L, b_micro, tp, is_bwd=False)


        t_attn_actual = self.compute_actual_op_time(t_attn, is_compute_bound=True, bg_bw_ratio=bg_ratio)
        t_mlp_actual = self.compute_actual_op_time(t_mlp, is_compute_bound=True, bg_bw_ratio=bg_ratio)


        t_tp_per_layer = self.compute_tp_comm(tp, b_micro, L)

        t_per_layer = t_attn_actual + t_mlp_actual + t_tp_per_layer
        return layers_per_pp * t_per_layer

    def compute_stage_bwd_time(self, L: int, config: TrainParallelConfig) -> float:
        """Compute stage bwd time."""
        tp, pp = config.tp, config.pp
        b_micro = config.b_micro
        layers_per_pp = self.n_layers // max(pp, 1)
        bg_ratio = self._determine_bg_bw_ratio(config)

        bg_ratio_bwd = min(bg_ratio * 1.5, 0.8)


        t_attn_recomp = self.compute_pure_attn_time(L, b_micro, tp, is_bwd=False)
        t_attn_recomp_actual = self.compute_actual_op_time(
            t_attn_recomp, is_compute_bound=True, bg_bw_ratio=bg_ratio_bwd
        )


        t_attn_bwd = self.compute_pure_attn_time(L, b_micro, tp, is_bwd=True)
        t_mlp_bwd = self.compute_pure_mlp_time(L, b_micro, tp, is_bwd=True)

        t_attn_bwd_actual = self.compute_actual_op_time(
            t_attn_bwd, is_compute_bound=True, bg_bw_ratio=bg_ratio_bwd
        )
        t_mlp_bwd_actual = self.compute_actual_op_time(
            t_mlp_bwd, is_compute_bound=True, bg_bw_ratio=bg_ratio_bwd
        )


        t_tp_per_layer = self.compute_tp_comm(tp, b_micro, L)

        t_per_layer = t_attn_recomp_actual + t_attn_bwd_actual + t_mlp_bwd_actual + t_tp_per_layer
        return layers_per_pp * t_per_layer



    def compute_pipeline_time(
        self, T_fwd: float, T_bwd: float, n_micro: int, pp: int,
    ) -> float:
        """Compute pipeline time."""
        if pp <= 1:

            return n_micro * (T_fwd + T_bwd)

        t_warmup = (pp - 1) * T_fwd
        t_steady = max(0, n_micro - pp + 1) * (T_fwd + T_bwd)
        t_cooldown = (pp - 1) * T_bwd
        return t_warmup + t_steady + t_cooldown



    def compute_iteration_time(
        self, config: TrainParallelConfig, B_global: int, L: int,
    ) -> tuple[float, dict]:
        """Compute iteration time."""
        tp, pp, dp = config.tp, config.pp, config.dp
        b_micro = config.b_micro



        b_per_dp = max(1, B_global // dp)
        n_micro = max(1, b_per_dp // b_micro)


        t_fwd = self.compute_stage_fwd_time(L, config)
        t_bwd = self.compute_stage_bwd_time(L, config)


        t_pipeline = self.compute_pipeline_time(t_fwd, t_bwd, n_micro, pp)


        t_pp = self.compute_pp_comm(config, L)

        t_pp_total = (pp - 1) * t_pp if pp > 1 else 0.0


        t_dp = self.compute_dp_comm(config)

        overlap_factor = 0.5 if pp > 1 else 0.3
        t_dp_exposed = t_dp * (1.0 - overlap_factor)


        t_iter = t_pipeline + t_pp_total + t_dp_exposed

        details = {
            "t_fwd_per_micro": t_fwd,
            "t_bwd_per_micro": t_bwd,
            "t_pipeline": t_pipeline,
            "t_pp_comm": t_pp_total,
            "t_dp_comm": t_dp,
            "t_dp_exposed": t_dp_exposed,
            "n_micro": n_micro,
            "b_per_dp": b_per_dp,
            "layers_per_pp": self.n_layers // max(pp, 1),
        }
        return t_iter, details


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class CostModel:
    """Cost model implementation."""

    def __init__(self, hardware=None, model_arch=None, profiling=None):
        from RL_Framework.config import HardwareConfig, ModelArchConfig, ProfilingConfig

        hw = hardware or HardwareConfig()
        ma = model_arch or ModelArchConfig()
        prof = profiling or ProfilingConfig()

        self.rollout_model = RolloutCostModel(
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
            effective_num_params=ma.effective_num_params,
        )

        self.training_model = TrainingCostModel(
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
            effective_num_params=ma.effective_num_params,
        )

        self.hw = hw
        self.ma = ma
        self.prof = prof

    def evaluate_rollout(
        self,
        cluster: RolloutClusterConfig,
        requests: list[RequestInfo],
    ) -> tuple[float, dict]:
        """Evaluate rollout."""
        return self.rollout_model.evaluate_cluster(cluster, requests)

    def evaluate_training(
        self,
        config: TrainParallelConfig,
        B_global: int,
        L: int,
    ) -> tuple[float, dict]:
        """Evaluate training."""
        return self.training_model.compute_iteration_time(config, B_global, L)

    def check_train_oom(
        self,
        config: TrainParallelConfig,
        B_global: int,
        L: int,
    ) -> bool:
        """Check train oom."""
        return self.training_model.check_memory_oom(config, config.b_micro, L)

    def evaluate(
        self,
        train_config: TrainParallelConfig,
        rollout_cluster: RolloutClusterConfig,
        requests: list[RequestInfo],
        B_global: int = 32,
    ) -> CostModelResult:
        """Evaluate."""
        result = CostModelResult()


        if requests:
            L_avg = int(np.mean([r.total_length for r in requests]))
        else:
            L_avg = 1024

        # ---- Training ----
        if self.check_train_oom(train_config, B_global, L_avg):
            result.is_oom = True
            result.oom_source = "train"
            result.t_train = float("inf")
            result.t_rollout = float("inf")
            result.t_global = float("inf")
            result.details["train_oom"] = True
            result.details["train_mem_per_gpu"] = (
                self.training_model.estimate_memory_per_gpu(train_config, train_config.b_micro, L_avg)
            )
            return result

        t_train, train_details = self.evaluate_training(train_config, B_global, L_avg)
        result.t_train = t_train
        result.details["training"] = train_details

        # ---- Rollout ----
        t_rollout, rollout_details = self.evaluate_rollout(rollout_cluster, requests)
        result.t_rollout = t_rollout
        result.details["rollout"] = rollout_details

        if t_rollout == float("inf"):
            result.is_oom = True
            result.oom_source = "rollout"
            result.t_global = float("inf")
            return result

        # ---- Global: max(T_train, T_rollout) ----
        result.t_global = max(result.t_train, result.t_rollout)
        result.details["t_train"] = result.t_train
        result.details["t_rollout"] = result.t_rollout
        result.details["train_config"] = {
            "tp": train_config.tp, "pp": train_config.pp,
            "dp": train_config.dp, "b_micro": train_config.b_micro,
        }
        result.details["rollout_config"] = rollout_cluster.tp_list

        return result

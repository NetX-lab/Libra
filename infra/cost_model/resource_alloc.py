"""Support code for Resource alloc."""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class ResourceConfig:
    """Resource config implementation."""
    max_concurrent_requests: int
    batch_size: int  # batch size
    target_rollout_time: float


class TrajectoryAwareAllocator:
    """Trajectory aware allocator implementation."""

    def __init__(
        self,
        train_time_estimate: float = 30.0,
        min_batch_size: int = 8,
        max_batch_size: int = 128,
    ):
        self.train_time_estimate = train_time_estimate
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size


        self.trajectory_lengths: List[float] = []


        self._cached_config: ResourceConfig | None = None
        self._update_interval = 10

    def update_trajectory_lengths(self, lengths: List[float]):
        """Update trajectory lengths."""
        self.trajectory_lengths.extend(lengths)


        if len(self.trajectory_lengths) > 1000:
            self.trajectory_lengths = self.trajectory_lengths[-1000:]


        self._cached_config = None

    def allocate_resources(
        self,
        num_gpus: int = 4,
    ) -> ResourceConfig:
        """Allocate resources."""

        if self._cached_config is not None:
            return self._cached_config


        if len(self.trajectory_lengths) == 0:
            config = ResourceConfig(
                max_concurrent_requests=num_gpus * 8,
                batch_size=self.min_batch_size,
                target_rollout_time=self.train_time_estimate * 0.8,
            )
            self._cached_config = config
            return config


        avg_length = np.mean(self.trajectory_lengths)
        p50_length = np.percentile(self.trajectory_lengths, 50)
        p95_length = np.percentile(self.trajectory_lengths, 95)
        max_length = np.max(self.trajectory_lengths)



        time_per_token = 0.01
        estimated_gen_time = p95_length * time_per_token


        target_rollout_time = self.train_time_estimate * 0.8




        if estimated_gen_time > target_rollout_time:

            batch_size = max(
                self.min_batch_size,
                int(self.min_batch_size * (target_rollout_time / estimated_gen_time)),
            )
        else:

            batch_size = min(
                self.max_batch_size,
                int(self.max_batch_size * (estimated_gen_time / target_rollout_time)),
            )



        max_concurrent = max(
            num_gpus * 4,
            batch_size * 2,
        )


        length_variance = np.var(self.trajectory_lengths)
        length_mean = np.mean(self.trajectory_lengths)
        cv = length_variance / (length_mean ** 2 + 1e-8)

        if cv > 0.5:

            max_concurrent = int(max_concurrent * 1.5)

        config = ResourceConfig(
            max_concurrent_requests=max_concurrent,
            batch_size=batch_size,
            target_rollout_time=target_rollout_time,
        )

        self._cached_config = config


        print("[ResourceAllocator] Resource allocation:")
        print(f"  Trajectory length: avg={avg_length:.0f}, p50={p50_length:.0f}, p95={p95_length:.0f}")
        print(f"  Estimated generation time: {estimated_gen_time:.1f}s")
        print(f"  Target rollout time: {target_rollout_time:.1f}s")
        print(f"  Batch size: {batch_size}")
        print(f"  Maximum concurrency: {max_concurrent}")

        return config

    def estimate_train_time(self) -> float:
        """Estimate train time."""


        return self.train_time_estimate

    def should_update_config(self, step: int) -> bool:
        """Should update config."""
        return step % self._update_interval == 0

    def get_statistics(self) -> dict:
        """Get statistics."""
        if len(self.trajectory_lengths) == 0:
            return {"count": 0}

        return {
            "count": len(self.trajectory_lengths),
            "mean": float(np.mean(self.trajectory_lengths)),
            "std": float(np.std(self.trajectory_lengths)),
            "min": float(np.min(self.trajectory_lengths)),
            "max": float(np.max(self.trajectory_lengths)),
            "p50": float(np.percentile(self.trajectory_lengths, 50)),
            "p95": float(np.percentile(self.trajectory_lengths, 95)),
        }


class SimpleTrajectoryAllocator(TrajectoryAwareAllocator):
    """Simple trajectory allocator implementation."""

    def __init__(
        self,
        fixed_batch_size: int = 32,
        fixed_concurrent: int = 32,
    ):
        super().__init__()
        self.fixed_batch_size = fixed_batch_size
        self.fixed_concurrent = fixed_concurrent

    def allocate_resources(self, num_gpus: int = 4) -> ResourceConfig:
        """Allocate resources."""
        return ResourceConfig(
            max_concurrent_requests=self.fixed_concurrent,
            batch_size=self.fixed_batch_size,
            target_rollout_time=self.train_time_estimate * 0.8,
        )


class CostModelAllocator:
    """Cost model allocator implementation."""

    def __init__(
        self,
        hardware=None,
        model_arch=None,
        allowed_rollout_tp: list[int] | None = None,
        verbose: bool = False,
    ):
        from .model import CostModel, RequestInfo
        from .calibrator import ShadowCalibrator, EpochRunData
        from .optimizer import TwoLevelNestedOptimizer, OptimizationResult

        self.cost_model = CostModel(hardware=hardware, model_arch=model_arch)
        self.calibrator = ShadowCalibrator()
        self.optimizer = TwoLevelNestedOptimizer(
            cost_model=self.cost_model,
            allowed_rollout_tp=allowed_rollout_tp or [1, 2, 4, 8],
            verbose=verbose,
        )


        self._last_result: OptimizationResult | None = None
        self._optimization_count = 0

    def allocate_resources(
        self,
        n_total_gpus: int,
        length_distribution: list[tuple[int, int]] | None = None,
        B_global: int = 32,
        allowed_train_tp: list[int] | None = None,
        allowed_train_pp: list[int] | None = None,
    ):
        """Allocate resources."""
        from .model import RequestInfo
        from .optimizer import OptimizationResult


        if length_distribution:
            requests = [
                RequestInfo(prompt_length=pl, gen_length=gl)
                for pl, gl in length_distribution
            ]
        else:

            params = self.calibrator.get_calibrated_params()
            n_requests = B_global * 4
            gen_lengths = np.random.normal(
                params.mu_gen, params.sigma_gen, size=n_requests
            ).clip(min=16).astype(int)
            prompt_lengths = np.random.normal(
                params.mu_prompt, params.mu_prompt * 0.3, size=n_requests
            ).clip(min=16).astype(int)
            requests = [
                RequestInfo(prompt_length=int(pl), gen_length=int(gl))
                for pl, gl in zip(prompt_lengths, gen_lengths)
            ]


        result = self.optimizer.optimize(
            n_total_gpus=n_total_gpus,
            requests=requests,
            B_global=B_global,
            allowed_train_tp=allowed_train_tp,
            allowed_train_pp=allowed_train_pp,
        )

        self._last_result = result
        self._optimization_count += 1

        return result

    def update_calibration(self, epoch_data) -> dict:
        """Update calibration."""
        params = self.calibrator.update(epoch_data)
        return {
            "eta_comp": params.eta_comp,
            "eta_mem": params.eta_mem,
            "gamma_overlap": params.gamma_overlap,
            "mu_gen": params.mu_gen,
            "sigma_gen": params.sigma_gen,
            "confidence": params.confidence,
        }

    def get_optimization_report(self) -> dict:
        """Get optimization report."""
        report = {
            "optimization_count": self._optimization_count,
            "calibration_stats": self.calibrator.get_statistics(),
        }
        if self._last_result is not None:
            r = self._last_result
            report["last_result"] = {
                "t_train": r.t_train,
                "t_rollout": r.t_rollout,
                "t_global": r.t_global,
                "train_config": {
                    "tp": r.train_config.tp, "pp": r.train_config.pp,
                    "dp": r.train_config.dp,
                } if r.train_config else None,
                "rollout_config": r.rollout_config.tp_list if r.rollout_config else None,
                "n_explored": r.n_configs_explored,
                "n_pruned_oom": r.n_configs_pruned_oom,
                "n_pruned_early_stop": r.n_configs_pruned_early_stop,
                "time_ms": r.optimization_time_ms,
            }
        return report

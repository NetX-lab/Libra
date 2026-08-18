"""Support code for Calibrator."""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpochRunData:
    """Epoch run data implementation."""
    t_train_real: float = 0.0
    t_rollout_real: float = 0.0
    t_train_predicted: float = 0.0
    t_rollout_predicted: float = 0.0
    gen_lengths: list[float] = field(default_factory=list)
    prompt_lengths: list[float] = field(default_factory=list)
    epoch_id: int = 0


@dataclass
class CalibratedParams:
    """Calibrated params implementation."""
    eta_comp: float = 0.5
    eta_mem: float = 0.5
    gamma_overlap: float = 0.3
    mu_gen: float = 512.0
    sigma_gen: float = 256.0
    mu_prompt: float = 256.0
    confidence: float = 0.0


class ShadowCalibrator:
    """Shadow calibrator implementation."""

    def __init__(
        self,
        alpha_default: float = 0.1,
        alpha_emergency: float = 0.8,
        change_point_threshold: float = 3.0,
        history_window: int = 20,
        min_epochs_for_calibration: int = 2,
    ):
        self.alpha_default = alpha_default
        self.alpha_emergency = alpha_emergency
        self.change_point_threshold = change_point_threshold
        self.history_window = history_window
        self.min_epochs = min_epochs_for_calibration


        self.params = CalibratedParams()


        self.history: deque[EpochRunData] = deque(maxlen=history_window)


        self._train_errors: deque[float] = deque(maxlen=history_window)
        self._rollout_errors: deque[float] = deque(maxlen=history_window)
        self._gen_length_means: deque[float] = deque(maxlen=history_window)


        self._is_change_point = False
        self._change_point_count = 0

    def update(self, epoch_data: EpochRunData) -> CalibratedParams:
        """Update."""
        self.history.append(epoch_data)


        self._detect_change_point(epoch_data)


        alpha = self.alpha_emergency if self._is_change_point else self.alpha_default


        self._update_eta_comp(epoch_data, alpha)


        self._update_eta_mem(epoch_data, alpha)


        self._update_gamma_overlap(epoch_data, alpha)


        self._update_sequence_stats(epoch_data, alpha)


        n = len(self.history)
        self.params.confidence = min(1.0, n / max(self.min_epochs * 2, 1))


        if epoch_data.t_train_predicted > 0:
            rel_error = (epoch_data.t_train_real - epoch_data.t_train_predicted) / epoch_data.t_train_predicted
            self._train_errors.append(rel_error)
        if epoch_data.t_rollout_predicted > 0:
            rel_error = (epoch_data.t_rollout_real - epoch_data.t_rollout_predicted) / epoch_data.t_rollout_predicted
            self._rollout_errors.append(rel_error)

        logger.info(
            f"[ShadowCalibrator] Epoch {epoch_data.epoch_id}: "
            f"eta_comp={self.params.eta_comp:.3f}, eta_mem={self.params.eta_mem:.3f}, "
            f"gamma={self.params.gamma_overlap:.3f}, mu_gen={self.params.mu_gen:.0f}, "
            f"change_point={self._is_change_point}, alpha={alpha:.2f}"
        )

        return self.params

    def _detect_change_point(self, epoch_data: EpochRunData):
        """Detect change point."""
        self._is_change_point = False

        if len(self.history) < self.min_epochs:
            return


        if epoch_data.t_train_predicted > 0 and len(self._train_errors) >= 3:
            errors = np.array(self._train_errors)
            mu_err = np.mean(errors)
            sigma_err = np.std(errors) + 1e-8
            current_err = (epoch_data.t_train_real - epoch_data.t_train_predicted) / epoch_data.t_train_predicted
            if abs(current_err - mu_err) > self.change_point_threshold * sigma_err:
                self._is_change_point = True
                logger.warning(
                    f"[ChangePoint] Training-time prediction error is too large: "
                    f"real={epoch_data.t_train_real:.2f}s, "
                    f"predicted={epoch_data.t_train_predicted:.2f}s, "
                    f"error={current_err:.3f}, threshold={self.change_point_threshold * sigma_err:.3f}"
                )


        if epoch_data.gen_lengths and len(self._gen_length_means) >= 3:
            current_mean = np.mean(epoch_data.gen_lengths)
            historical_means = np.array(self._gen_length_means)
            mu_len = np.mean(historical_means)
            sigma_len = np.std(historical_means) + 1e-8
            if abs(current_mean - mu_len) > self.change_point_threshold * sigma_len:
                self._is_change_point = True
                logger.warning(
                    f"[ChangePoint] Sequence-length step change: "
                    f"current_mean={current_mean:.0f}, "
                    f"historical_mean={mu_len:.0f}, "
                    f"threshold={self.change_point_threshold * sigma_len:.0f}"
                )

        if self._is_change_point:
            self._change_point_count += 1

    def _update_eta_comp(self, epoch_data: EpochRunData, alpha: float):
        """Update eta comp."""
        if epoch_data.t_train_predicted > 0 and epoch_data.t_train_real > 0:
            ratio = epoch_data.t_train_predicted / epoch_data.t_train_real
            eta_real = self.params.eta_comp * ratio
            eta_real = np.clip(eta_real, 0.05, 0.95)
            self.params.eta_comp = alpha * eta_real + (1 - alpha) * self.params.eta_comp

    def _update_eta_mem(self, epoch_data: EpochRunData, alpha: float):
        """Update eta mem."""
        if epoch_data.t_rollout_predicted > 0 and epoch_data.t_rollout_real > 0:
            ratio = epoch_data.t_rollout_predicted / epoch_data.t_rollout_real
            eta_real = self.params.eta_mem * ratio
            eta_real = np.clip(eta_real, 0.05, 0.95)
            self.params.eta_mem = alpha * eta_real + (1 - alpha) * self.params.eta_mem

    def _update_gamma_overlap(self, epoch_data: EpochRunData, alpha: float):
        """Update gamma overlap."""

        if epoch_data.t_train_predicted > 0 and epoch_data.t_train_real > 0:
            if epoch_data.t_train_real > epoch_data.t_train_predicted:

                adjustment = min(0.05, (epoch_data.t_train_real / epoch_data.t_train_predicted - 1))
                gamma_real = self.params.gamma_overlap + adjustment
            else:
                adjustment = min(0.05, (1 - epoch_data.t_train_real / epoch_data.t_train_predicted))
                gamma_real = self.params.gamma_overlap - adjustment
            gamma_real = np.clip(gamma_real, 0.0, 1.0)
            self.params.gamma_overlap = alpha * gamma_real + (1 - alpha) * self.params.gamma_overlap

    def _update_sequence_stats(self, epoch_data: EpochRunData, alpha: float):
        """Update sequence stats."""
        if epoch_data.gen_lengths:
            gen_arr = np.array(epoch_data.gen_lengths)
            mu_real = float(np.mean(gen_arr))
            sigma_real = float(np.std(gen_arr)) + 1e-8

            self.params.mu_gen = alpha * mu_real + (1 - alpha) * self.params.mu_gen
            self.params.sigma_gen = alpha * sigma_real + (1 - alpha) * self.params.sigma_gen

            self._gen_length_means.append(mu_real)

        if epoch_data.prompt_lengths:
            mu_prompt = float(np.mean(epoch_data.prompt_lengths))
            self.params.mu_prompt = alpha * mu_prompt + (1 - alpha) * self.params.mu_prompt

    def get_calibrated_params(self) -> CalibratedParams:
        """Get calibrated params."""
        return CalibratedParams(
            eta_comp=self.params.eta_comp,
            eta_mem=self.params.eta_mem,
            gamma_overlap=self.params.gamma_overlap,
            mu_gen=self.params.mu_gen,
            sigma_gen=self.params.sigma_gen,
            mu_prompt=self.params.mu_prompt,
            confidence=self.params.confidence,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics."""
        stats = {
            "n_epochs": len(self.history),
            "eta_comp": self.params.eta_comp,
            "eta_mem": self.params.eta_mem,
            "gamma_overlap": self.params.gamma_overlap,
            "mu_gen": self.params.mu_gen,
            "sigma_gen": self.params.sigma_gen,
            "confidence": self.params.confidence,
            "change_point_count": self._change_point_count,
            "is_change_point": self._is_change_point,
        }

        if self._train_errors:
            stats["train_error_mean"] = float(np.mean(self._train_errors))
            stats["train_error_std"] = float(np.std(self._train_errors))
        if self._rollout_errors:
            stats["rollout_error_mean"] = float(np.mean(self._rollout_errors))
            stats["rollout_error_std"] = float(np.std(self._rollout_errors))

        return stats

    def reset(self):
        """Reset the current state."""
        self.params = CalibratedParams()
        self.history.clear()
        self._train_errors.clear()
        self._rollout_errors.clear()
        self._gen_length_means.clear()
        self._is_change_point = False
        self._change_point_count = 0

"""Adaptive-equilibrium Task-LQR for the S6T5 development study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter
from uav_sway.evaluation.task_space_metrics import (
    HOLD_TIME_S,
    ORIENTATION_TOLERANCE_DEG,
    POSITION_TOLERANCE_M,
    TIP_SPEED_TOLERANCE_M_S,
)


@dataclass(frozen=True)
class AETSLQRDiagnostics:
    external_x_ref: float
    internal_x_ref: float
    equilibrium_bias_x: float
    filtered_tip_error_x: float
    bias_rate: float
    lqr_feedback_ax: float
    ax_cmd_raw: float
    ax_cmd_amplitude_limited: float
    ax_cmd_limited: float
    ax_saturated: bool
    ax_slew_limited: bool
    adaptation_held: bool
    task_ready: bool
    task_ready_timer_s: float
    task_locked: bool


class AdaptiveEquilibriumTaskLQR:
    """Frozen Task-LQR with a causal internal UAV equilibrium bias.

    The measured task error is external cutter-tip error.  The bias is an
    internal reference shift only; all formal task metrics remain external.
    """

    def __init__(self, gain: np.ndarray, k_b: float, tau_s: float,
                 bias_limit_m: float = 0.40, bias_rate_limit_m_s: float = 0.10,
                 command_holdoff_s: float = 1.0, dt: float = 0.05):
        self.gain = np.asarray(gain, dtype=float).reshape(1, 16)
        self.k_b = float(k_b); self.tau_s = float(tau_s)
        self.bias_limit_m = float(bias_limit_m)
        self.bias_rate_limit_m_s = float(bias_rate_limit_m_s)
        self.command_holdoff_s = float(command_holdoff_s); self.dt = float(dt)
        if self.k_b <= 0 or self.tau_s <= 0 or self.bias_limit_m <= 0 or self.bias_rate_limit_m_s <= 0:
            raise ValueError("adaptive-equilibrium parameters must be positive")
        self.limiter = AccelerationLimiter(-2.0, 2.0, 0.25)
        self.reset()

    def reset(self, state=None, reference=None):
        del state
        self.limiter.reset(0.0)
        self.bias_x = 0.0; self.filtered_error_x = 0.0
        self._clock = 0.0; self._hold_remaining = 0.0
        self._task_ready_timer_s = 0.0; self._task_locked = False
        self._previous_external_x = None if reference is None else float(reference.x_ref)
        self.diagnostics = AETSLQRDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, False, False, False,
                                              False, 0.0, False)

    def internal_reference(self, external_reference):
        from uav_sway.control.base import ReferenceState
        return ReferenceState(float(external_reference.x_ref) + self.bias_x,
                              float(external_reference.vx_ref), 0.0,
                              float(external_reference.y_ref),
                              float(external_reference.z_ref),
                              float(external_reference.yaw_ref))

    def command(self, state: np.ndarray, external_reference,
                measured_tip_error_x: float, dt: float = 0.05,
                task_position_error_m: float | None = None,
                task_orientation_error_deg: float | None = None,
                task_tip_speed_m_s: float | None = None) -> float:
        dt = float(dt)
        if dt <= 0 or not np.isfinite(measured_tip_error_x):
            raise ValueError("invalid adaptive-equilibrium sample")
        task_values = (task_position_error_m, task_orientation_error_deg, task_tip_speed_m_s)
        if all(value is not None for value in task_values):
            if not np.isfinite(np.asarray(task_values, dtype=float)).all():
                raise ValueError("invalid current task-space acquisition measurements")
            task_ready = bool(
                float(task_position_error_m) <= POSITION_TOLERANCE_M
                and float(task_orientation_error_deg) <= ORIENTATION_TOLERANCE_DEG
                and float(task_tip_speed_m_s) <= TIP_SPEED_TOLERANCE_M_S
            )
        else:
            # Compatibility for the pre-lock unit-level command API: without
            # the current task measurements the causal lock cannot engage.
            task_ready = False
        external_x = float(external_reference.x_ref)
        changed = self._previous_external_x is not None and abs(external_x - self._previous_external_x) > 1.0e-12
        if changed:
            # A command event is not a disturbance: retain learned bias, reset
            # only the filter, and protect adaptation for the frozen holdoff.
            self.filtered_error_x = 0.0
            self._hold_remaining = self.command_holdoff_s
            self._task_ready_timer_s = 0.0
            self._task_locked = False
        self._previous_external_x = external_x
        self._clock += dt
        if self._task_locked and not task_ready:
            self._task_locked = False
            self._task_ready_timer_s = 0.0
        elif task_ready:
            self._task_ready_timer_s += dt
            if self._task_ready_timer_s >= HOLD_TIME_S:
                self._task_locked = True
        else:
            self._task_ready_timer_s = 0.0
        held = self._hold_remaining > 0.0
        if held:
            self._hold_remaining = max(0.0, self._hold_remaining - dt)
            bias_rate = 0.0
        elif self._task_locked:
            alpha = float(np.exp(-dt / self.tau_s))
            self.filtered_error_x = alpha * self.filtered_error_x + (1.0 - alpha) * float(measured_tip_error_x)
            bias_rate = 0.0
        else:
            alpha = float(np.exp(-dt / self.tau_s))
            self.filtered_error_x = alpha * self.filtered_error_x + (1.0 - alpha) * float(measured_tip_error_x)
            bias_rate = float(np.clip(-self.k_b * self.filtered_error_x,
                                      -self.bias_rate_limit_m_s, self.bias_rate_limit_m_s))
            self.bias_x = float(np.clip(self.bias_x + dt * bias_rate,
                                        -self.bias_limit_m, self.bias_limit_m))
        internal = self.internal_reference(external_reference)
        x = np.asarray(state, dtype=float).reshape(16)
        feedback = float((-self.gain @ x.reshape(-1, 1))[0, 0])
        raw = feedback
        limited = self.limiter.limit(raw)
        diag = self.limiter.diagnostics
        self.diagnostics = AETSLQRDiagnostics(
            external_x, float(internal.x_ref), float(self.bias_x),
            float(self.filtered_error_x), float(bias_rate), feedback, raw,
            float(diag.amplitude_limited), float(limited), bool(diag.saturated),
            bool(diag.slew_limited), bool(held), task_ready,
            float(self._task_ready_timer_s), bool(self._task_locked))
        return float(limited)

"""Adaptive-equilibrium Task-LQR for the S6T5 development study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter


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
        self._previous_external_x = None if reference is None else float(reference.x_ref)
        self.diagnostics = AETSLQRDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, False, False, False)

    def internal_reference(self, external_reference):
        from uav_sway.control.base import ReferenceState
        return ReferenceState(float(external_reference.x_ref) + self.bias_x,
                              float(external_reference.vx_ref), 0.0,
                              float(external_reference.y_ref),
                              float(external_reference.z_ref),
                              float(external_reference.yaw_ref))

    def command(self, state: np.ndarray, external_reference,
                measured_tip_error_x: float, dt: float = 0.05) -> float:
        dt = float(dt)
        if dt <= 0 or not np.isfinite(measured_tip_error_x):
            raise ValueError("invalid adaptive-equilibrium sample")
        external_x = float(external_reference.x_ref)
        changed = self._previous_external_x is not None and abs(external_x - self._previous_external_x) > 1.0e-12
        if changed:
            # A command event is not a disturbance: retain learned bias, reset
            # only the filter, and protect adaptation for the frozen holdoff.
            self.filtered_error_x = 0.0
            self._hold_remaining = self.command_holdoff_s
        self._previous_external_x = external_x
        self._clock += dt
        held = self._hold_remaining > 0.0
        if held:
            self._hold_remaining = max(0.0, self._hold_remaining - dt)
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
            bool(diag.slew_limited), bool(held))
        return float(limited)


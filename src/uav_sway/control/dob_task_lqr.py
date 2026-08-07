"""Disturbance-observer Task-LQR for the S6T4 development study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter
from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver
from uav_sway.mpc.preview_model import PreviewModel


@dataclass(frozen=True)
class DOBTaskLQRDiagnostics:
    lqr_feedback_ax: float
    disturbance_hat: float
    ax_cmd_raw: float
    ax_cmd_amplitude_limited: float
    ax_cmd_limited: float
    ax_saturated: bool
    ax_slew_limited: bool
    observer_initialized: bool


class DOBTaskLQR:
    """Frozen Task-LQR plus matched scalar disturbance compensation.

    The observer sees only the measured 16D error state, the actually applied
    previous acceleration, and a causal reference-coordinate shift.
    """

    def __init__(self, gain: np.ndarray, A: np.ndarray, B: np.ndarray,
                 observer_gain: float, disturbance_limit: float = 2.0,
                 ax_min: float = -2.0, ax_max: float = 2.0,
                 slew_limit: float = 0.25, use_observer: bool = True):
        self.gain = np.asarray(gain, dtype=float).reshape(1, 16)
        self.A = np.asarray(A, dtype=float).reshape(16, 16)
        self.B = np.asarray(B, dtype=float).reshape(16)
        self.use_observer = bool(use_observer)
        self.observer = MatchedDisturbanceObserver(
            self.A, self.B, float(observer_gain), float(disturbance_limit))
        self.limiter = AccelerationLimiter(ax_min, ax_max, slew_limit)
        self._previous_reference = None
        self._observer_initialized = False
        self.diagnostics = DOBTaskLQRDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0,
                                                 False, False, False)

    def reset(self, state=None, reference=None) -> None:
        self.limiter.reset(0.0)
        self.observer.reset(state=np.zeros(16) if state is None else state)
        self._previous_reference = reference
        self._observer_initialized = False
        self.diagnostics = DOBTaskLQRDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0,
                                                 False, False, False)

    def command(self, state: np.ndarray, reference, dt: float = 0.05) -> float:
        del dt
        x = np.asarray(state, dtype=float).reshape(16)
        if self.use_observer:
            shift = None
            if self._previous_reference is not None:
                shift = PreviewModel.static_reference_shift(
                    self.A, self._previous_reference, reference)
            d_hat = self.observer.update(x, self.limiter.previous, shift)
            self._observer_initialized = bool(self.observer._initialized)
        else:
            d_hat = 0.0
        self._previous_reference = reference
        feedback = float((-self.gain @ x.reshape(-1, 1))[0, 0])
        raw = feedback - float(d_hat)
        limited = self.limiter.limit(raw)
        diag = self.limiter.diagnostics
        self.diagnostics = DOBTaskLQRDiagnostics(
            feedback, float(d_hat), raw, float(diag.amplitude_limited),
            float(limited), bool(diag.saturated), bool(diag.slew_limited),
            self._observer_initialized)
        return float(limited)


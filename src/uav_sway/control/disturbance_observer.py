"""Matched scalar disturbance observer using state and command history only."""

from __future__ import annotations

import numpy as np


class MatchedDisturbanceObserver:
    """17-state observer for x[k+1]=A x+B(u+d).

    The measured 16-state vector is accepted directly.  No wind, force, or
    profile object is part of this API; d_hat is updated from prediction error.
    """

    def __init__(self, A, B, gain: float = 0.15, limit: float = 2.0):
        self.A = np.asarray(A, dtype=float).reshape(16, 16)
        self.B = np.asarray(B, dtype=float).reshape(16)
        self.gain = float(gain); self.limit = float(limit)
        self.x_hat = np.zeros(16); self.d_hat = 0.0; self.previous_command = 0.0

    @property
    def dimension(self): return 17

    def reset(self, state=None, disturbance: float = 0.0, command: float = 0.0):
        self.x_hat = np.zeros(16) if state is None else np.asarray(state, dtype=float).copy()
        self.d_hat = float(disturbance); self.previous_command = float(command)

    def update(self, measured_state, command: float) -> float:
        y = np.asarray(measured_state, dtype=float).reshape(16)
        prediction = self.A @ self.x_hat + self.B * (self.previous_command + self.d_hat)
        innovation = y - prediction
        scale = float(self.B @ self.B) + 1.0e-12
        self.d_hat = float(np.clip(self.d_hat + self.gain * float(self.B @ innovation) / scale,
                                   -self.limit, self.limit))
        self.x_hat = y.copy()
        self.previous_command = float(command)
        return self.d_hat

    def augmented_state(self) -> np.ndarray:
        return np.r_[self.x_hat, self.d_hat]

"""Central finite-difference identification of the nonlinear closed-loop map."""

from __future__ import annotations

import numpy as np


def central_finite_difference(phi, state_eps: np.ndarray, input_eps: float) -> tuple[np.ndarray, np.ndarray]:
    state_eps = np.asarray(state_eps, dtype=float)
    n = len(state_eps)
    a = np.empty((n, n), dtype=float)
    zero = np.zeros(n, dtype=float)
    for i, epsilon in enumerate(state_eps):
        plus = zero.copy(); plus[i] = epsilon
        minus = zero.copy(); minus[i] = -epsilon
        a[:, i] = (np.asarray(phi(plus, 0.0)) - np.asarray(phi(minus, 0.0))) / (2.0 * epsilon)
    b = ((np.asarray(phi(zero, input_eps)) - np.asarray(phi(zero, -input_eps))) / (2.0 * input_eps)).reshape(n, 1)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("finite difference produced non-finite A/B")
    return a, b

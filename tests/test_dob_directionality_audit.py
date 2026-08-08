import numpy as np

from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver


def _run(d_true):
    A = np.eye(16) * 0.98
    B = np.zeros(16); B[0] = 0.05
    observer = MatchedDisturbanceObserver(A, B, gain=0.6, limit=2.0)
    state = np.zeros(16); observer.update(state, 0.0)
    for _ in range(250):
        state = A @ state + B * d_true
        observer.update(state, 0.0)
    return observer.d_hat


def test_matched_positive_and_negative_disturbance_signs():
    assert _run(0.10) > 0.0
    assert _run(-0.10) < 0.0


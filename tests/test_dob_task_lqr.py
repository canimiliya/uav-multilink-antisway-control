import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.control.dob_task_lqr import DOBTaskLQR


def test_observer_disabled_matches_task_lqr_feedback_and_limiter():
    A = np.eye(16); B = np.zeros(16); K = np.ones((1, 16)) * 0.01
    controller = DOBTaskLQR(K, A, B, 0.6, use_observer=False)
    controller.reset()
    state = np.zeros(16); state[0] = 0.5
    assert controller.command(state, ReferenceState(0, 0, 0, 0, 3.2, 0)) == -0.005
    assert controller.diagnostics.disturbance_hat == 0.0


def test_observer_first_sample_initializes_without_extra_compensation():
    A = np.eye(16); B = np.zeros(16); K = np.zeros((1, 16))
    controller = DOBTaskLQR(K, A, B, 0.6)
    controller.reset()
    controller.command(np.zeros(16), ReferenceState(0, 0, 0, 0, 3.2, 0))
    assert controller.diagnostics.disturbance_hat == 0.0
    assert controller.observer._initialized


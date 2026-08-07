import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver
from uav_sway.mpc.preview_model import PreviewModel


def test_reference_step_has_no_false_disturbance_spike():
    A = np.eye(16); B = np.ones(16) * 0.1
    observer = MatchedDisturbanceObserver(A, B, gain=0.6)
    first = ReferenceState(0, 0, 0, 0, 3.2, 0)
    second = ReferenceState(0.3, 0, 0, 0, 3.2, 0)
    observer.update(np.zeros(16), 0.0, None)
    shift = PreviewModel.static_reference_shift(A, first, second)
    measured_after_step = -shift
    observer.update(measured_after_step, 0.0, shift)
    assert abs(observer.d_hat) < 1e-12


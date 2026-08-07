import numpy as np

from uav_sway.paper_baseline.sep_nmpc_model import PlanarParameters
from uav_sway.paper_baseline.sep_nmpc_passivity import (
    acceleration_from_tracking_shaped_input,
    force_from_tracking_shaped_input,
    tracking_shaped_input,
    tracking_storage,
)


def test_perfect_moving_tracking_has_no_false_braking():
    parameters = PlanarParameters(9.74, 3.5, 2.57)
    state = np.zeros(4)
    assert tracking_shaped_input(parameters.m_T * 0.75, parameters.m_T * 0.75, 10.0, 0.0) == 0.0
    assert acceleration_from_tracking_shaped_input(0.0, 0.75, 10.0, 0.0, parameters) == 0.75
    assert force_from_tracking_shaped_input(0.0, 0.75, 10.0, 0.0, parameters) == parameters.m_T * 0.75
    assert tracking_storage(state, 0.0, 10.0, parameters) == 0.0


def test_stationary_form_parity():
    parameters = PlanarParameters(9.74, 3.5, 2.57)
    ex, ev, alpha, alpha_dot = 0.2, -0.3, 0.1, 0.04
    force = 1.2
    shaped = tracking_shaped_input(force, 0.0, 10.0, ex)
    assert shaped == force + 10.0 * ex
    assert np.isfinite(tracking_storage(np.array([ex, ev, alpha, alpha_dot]), ex, 10.0, parameters))

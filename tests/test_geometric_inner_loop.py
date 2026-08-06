import numpy as np

from uav_sway.control.base import ControlState, ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop


def test_m400_geometric_gains_and_hover_force():
    inner = GeometricInnerLoop(13.24, np.array([0.655826666666667, 0.966532666666667, 1.248343333333333]))
    assert np.allclose(inner.k_r, [10.493226666666672, 15.464522666666672, 19.973493333333328])
    assert np.allclose(inner.k_omega, [4.721952, 6.9590352, 8.988072])
    state = ControlState(np.array([0.0, 0.0, 3.2]), np.zeros(3), np.eye(3), np.zeros(3), np.zeros(5), np.zeros(5), 0.0)
    ref = ReferenceState(0.0, 0.0, 0.0, 0.0, 3.2, 0.0)
    force = inner.desired_force(state, ref, 0.0)
    assert np.allclose(force, [0.0, 0.0, 13.24 * 9.81])
    command = inner.compute(state, ref, 0.0)
    assert command["thrust_raw_N"] == np.float64(13.24 * 9.81)
    assert np.allclose(command["torque_raw_Nm"], 0.0)

import inspect

import numpy as np

from uav_sway.control.mppi import MuJoCoMPPI
from uav_sway.mppi.reference_horizon import make_reference_horizon


def test_horizon_uses_signal_index_plus_ten_and_holds_tail():
    ref = {name: np.arange(25, dtype=float) for name in ("time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}
    horizon = make_reference_horizon(ref, 20, 12)
    assert len(horizon) == 12
    assert horizon.action_count == 12
    assert len(horizon.boundary_samples) == 13
    assert horizon.boundary_indices[:2] == (20, 24)
    assert horizon.boundary_indices[-1] == 24
    assert horizon.action_reference(0) == horizon.boundary_samples[0]
    assert horizon.state_reference(0) == horizon.boundary_samples[1]
    assert horizon.state_reference(11) == horizon.boundary_samples[12]
    assert np.all(np.diff(horizon.boundary_times) >= 0)


def test_post_action_reference_alignment_has_no_artificial_half_second_error():
    time = np.arange(121, dtype=float) * 0.005
    ref = {name: np.zeros(time.size, dtype=float)
           for name in ("time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}
    ref["time"] = time
    ref["x_ref"] = time
    horizon = make_reference_horizon(ref, 0, 12)
    predicted_x_after_action = np.asarray([horizon.boundary_times[j + 1] for j in range(12)])
    old_error = predicted_x_after_action - np.asarray([horizon.action_reference(j).x_ref for j in range(12)])
    new_error = predicted_x_after_action - np.asarray([horizon.state_reference(j).x_ref for j in range(12)])
    assert np.allclose(old_error, 0.05)
    assert np.isclose(float(np.max(np.abs(old_error))), 0.05)
    assert float(np.max(np.abs(new_error))) == 0.0


def test_controller_solve_api_does_not_accept_wind_preview():
    names = set(inspect.signature(MuJoCoMPPI.solve).parameters)
    assert not names.intersection({"wind_future", "future_wind", "wind_profile", "wind_csv"})

import inspect

from uav_sway.control.mppi import MuJoCoMPPI
from uav_sway.mppi.reference_horizon import make_reference_horizon


def test_horizon_uses_signal_index_plus_ten_and_holds_tail():
    import numpy as np
    ref = {name: np.arange(25, dtype=float) for name in ("time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}
    horizon = make_reference_horizon(ref, 20, 12)
    assert len(horizon) == 12
    assert horizon.indices[:2] == (20, 24)
    assert horizon.indices[-1] == 24
    assert np.all(np.diff(horizon.times) >= 0)


def test_controller_solve_api_does_not_accept_wind_preview():
    names = set(inspect.signature(MuJoCoMPPI.solve).parameters)
    assert not names.intersection({"wind_future", "future_wind", "wind_profile", "wind_csv"})

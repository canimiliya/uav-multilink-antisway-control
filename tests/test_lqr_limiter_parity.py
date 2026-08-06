import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter
from uav_sway.control.base import ControlState, ReferenceState
from uav_sway.control.position_pid import PositionPID


def _state() -> ControlState:
    return ControlState(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), np.zeros(5), np.zeros(5), 0.0)


def test_lqr_limiter_matches_zero_gain_pid_final_command():
    pid = PositionPID(0.0, 0.0, 0.0, -2.0, 2.0, 0.25, 1.0)
    lqr = AccelerationLimiter(-2.0, 2.0, 0.25)
    state = _state(); pid.reset(state, ReferenceState(0, 0, 0, 0, 3.2, 0)); lqr.reset()
    commands = [0.0, 1.0, 2.0, -2.0, 0.5, 0.5]
    pid_values = []; lqr_values = []
    for value in commands:
        ref = ReferenceState(0, 0, value, 0, 3.2, 0)
        pid_values.append(pid.command(state, ref, 0.05))
        lqr_values.append(lqr.limit(value))
    assert np.array_equal(np.asarray(pid_values), np.asarray(lqr_values))

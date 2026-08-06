import numpy as np

from uav_sway.control.base import ControlState, ReferenceState
from uav_sway.control.position_pid import PositionPID


def test_pid_output_is_identical_when_only_sway_state_changes():
    reference = ReferenceState(1.0, 0.2, 0.1, 0.0, 3.2, 0.0)
    common_position = np.array([0.4, 0.0, 3.2])
    common_velocity = np.array([0.3, 0.0, 0.0])
    first = ControlState(common_position, common_velocity, np.eye(3), np.zeros(3), np.zeros(5), np.zeros(5), 0.0)
    second = ControlState(common_position, common_velocity, np.diag([1.0, -1.0, -1.0]), np.ones(3), np.full(5, 0.4), np.full(5, -0.3), 0.8)
    a = PositionPID(1.2, 1.8, 0.06)
    b = PositionPID(1.2, 1.8, 0.06)
    assert a.command(first, reference, 0.05) == b.command(second, reference, 0.05)

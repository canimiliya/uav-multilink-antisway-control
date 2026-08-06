from uav_sway.control.base import ControlState, ReferenceState
from uav_sway.control.position_pid import PositionPID
import numpy as np


def state(x=0.0, vx=0.0):
    return ControlState(np.array([x, 0.0, 3.2]), np.array([vx, 0.0, 0.0]), np.eye(3), np.zeros(3), np.zeros(5), np.zeros(5), 0.0)


REF = ReferenceState(0.0, 0.0, 0.0, 0.0, 3.2, 0.0)


def test_pid_zero_error_and_signs():
    controller = PositionPID(1.0, 2.0, 0.1)
    assert controller.command(state(), REF, 0.05) == 0.0
    assert controller.command(state(x=1.0), REF, 0.05) < 0.0
    controller.reset(state(), REF)
    assert controller.command(state(x=-1.0), REF, 0.05) > 0.0


def test_pid_reset_integral_and_limits():
    controller = PositionPID(0.0, 0.0, 1.0, ax_min=-2.0, ax_max=2.0, slew_limit=0.25, integral_limit=1.0)
    for _ in range(100):
        controller.command(state(x=-10.0), REF, 0.05)
    assert abs(controller.integral) <= 1.0
    assert abs(controller.diagnostics.ax_cmd_limited) <= 2.0
    assert abs(controller.diagnostics.ax_cmd_limited - controller.previous_command) < 1e-15
    controller.reset(state(), REF)
    assert controller.integral == 0.0
    assert controller.previous_command == 0.0


def test_pid_amplitude_and_slew_limits():
    controller = PositionPID(10.0, 0.0, 0.0, ax_min=-2.0, ax_max=2.0, slew_limit=0.25)
    first = controller.command(state(x=-10.0), REF, 0.05)
    second = controller.command(state(x=10.0), REF, 0.05)
    assert first == 0.25
    assert second == 0.0
    assert controller.diagnostics.ax_slew_limited is True


def test_anti_windup_freezes_when_pushing_into_saturation_and_recovers():
    controller = PositionPID(10.0, 0.0, 1.0, ax_min=-2.0, ax_max=2.0, slew_limit=100.0)
    for _ in range(20):
        controller.command(state(x=-10.0), REF, 0.05)
    frozen = controller.integral
    controller.command(state(x=10.0), REF, 0.05)
    assert frozen == controller.integral
    controller.command(state(x=0.0), REF, 0.05)
    assert abs(controller.integral) <= 1.0

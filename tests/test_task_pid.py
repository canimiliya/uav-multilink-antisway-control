from uav_sway.control.task_pid import TaskPID


def test_task_pid_uses_tip_error_and_frozen_limits():
    controller = TaskPID(2.0, 1.0, 0.0)
    controller.reset()
    command = controller.command(0.4, 0.0, 0.0, 0.0, 0.0, 0.05)
    assert command == -0.25
    assert controller.diagnostics.task_position_error_x == 0.4
    assert controller.diagnostics.ax_pid_feedback < 0.0


def test_task_pid_zero_error_preserves_reference_feedforward():
    controller = TaskPID(0.8, 1.0, 0.3)
    controller.reset()
    assert controller.command(0.0, 0.75, 0.0, 0.75, 0.2, 0.05) == 0.2

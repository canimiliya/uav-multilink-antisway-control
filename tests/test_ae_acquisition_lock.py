from uav_sway.control.adaptive_equilibrium_task_lqr import AdaptiveEquilibriumTaskLQR
from uav_sway.control.base import ReferenceState
import inspect


READY = dict(task_position_error_m=0.01, task_orientation_error_deg=1.0, task_tip_speed_m_s=0.01)
NOT_READY = dict(task_position_error_m=0.06, task_orientation_error_deg=1.0, task_tip_speed_m_s=0.01)


def controller():
    return AdaptiveEquilibriumTaskLQR([[0.0] * 16], 1.0, 0.5)


def test_ready_for_less_than_one_second_does_not_lock():
    c = controller(); r = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r)
    for _ in range(19):
        c.command([0.0] * 16, r, 0.2, 0.05, **READY)
    assert c.diagnostics.task_ready is True
    assert c.diagnostics.task_ready_timer_s < 1.0
    assert c.diagnostics.task_locked is False


def test_ready_for_one_second_locks():
    c = controller(); r = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r)
    for _ in range(20): c.command([0.0] * 16, r, 0.2, 0.05, **READY)
    assert c.diagnostics.task_locked is True
    assert c.diagnostics.task_ready_timer_s >= 1.0


def test_locked_bias_does_not_change():
    c = controller(); r = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r)
    for _ in range(20): c.command([0.0] * 16, r, 0.2, 0.05, **READY)
    bias = c.bias_x
    for _ in range(10): c.command([0.0] * 16, r, 10.0, 0.05, **READY)
    assert c.bias_x == bias
    assert c.diagnostics.bias_rate == 0.0


def test_invalid_task_immediately_unlocks_and_reentry_restarts_timer():
    c = controller(); r = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r)
    for _ in range(20): c.command([0.0] * 16, r, 0.2, 0.05, **READY)
    c.command([0.0] * 16, r, 0.2, 0.05, **NOT_READY)
    assert c.diagnostics.task_locked is False
    assert c.diagnostics.task_ready_timer_s == 0.0
    for _ in range(19): c.command([0.0] * 16, r, 0.2, 0.05, **READY)
    assert c.diagnostics.task_locked is False
    c.command([0.0] * 16, r, 0.2, 0.05, **READY)
    assert c.diagnostics.task_locked is True


def test_reference_change_unlocks_retains_bias_resets_filter_and_starts_holdoff():
    c = controller(); r0 = ReferenceState(0, 0, 0, 0, 3.2, 0); r1 = ReferenceState(0.3, 0, 0, 0, 3.2, 0)
    c.reset(reference=r0)
    c.command([0.0] * 16, r0, 0.3, 0.05, **NOT_READY)
    retained = c.bias_x
    c.filtered_error_x = 0.12
    c._task_locked = True
    c.command([0.0] * 16, r1, 0.3, 0.05, **READY)
    assert c.bias_x == retained
    assert c.diagnostics.task_locked is False
    assert c.diagnostics.task_ready_timer_s == 0.05
    assert c.filtered_error_x == 0.0
    assert c.diagnostics.adaptation_held is True


def test_external_target_does_not_move_with_bias():
    c = controller(); r = ReferenceState(0.525, 0, 0, 0, 0.39, 0); c.reset(reference=r)
    c.command([0.0] * 16, r, 0.3, 0.05, **NOT_READY)
    assert r.x_ref == 0.525
    assert c.internal_reference(r).x_ref != r.x_ref


def test_reference_change_retains_learned_bias():
    c = controller(); r0 = ReferenceState(0, 0, 0, 0, 3.2, 0); r1 = ReferenceState(0.3, 0, 0, 0, 3.2, 0); c.reset(reference=r0)
    c.command([0.0] * 16, r0, 0.3, 0.05, **NOT_READY); retained = c.bias_x
    c.command([0.0] * 16, r1, 0.3, 0.05, **READY)
    assert c.bias_x == retained


def test_reference_change_resets_filter_and_starts_one_second_holdoff():
    c = controller(); r0 = ReferenceState(0, 0, 0, 0, 3.2, 0); r1 = ReferenceState(0.3, 0, 0, 0, 3.2, 0); c.reset(reference=r0)
    c.command([0.0] * 16, r0, 0.3, 0.05, **NOT_READY); c.filtered_error_x = 0.2
    c.command([0.0] * 16, r1, 0.3, 0.05, **READY)
    assert c.filtered_error_x == 0.0 and c.diagnostics.adaptation_held is True


def test_lock_api_has_no_future_wind_or_target_inputs():
    names = set(inspect.signature(AdaptiveEquilibriumTaskLQR.command).parameters)
    assert not names.intersection({"future_wind", "wind_truth", "wind_force", "future_target", "target_horizon"})

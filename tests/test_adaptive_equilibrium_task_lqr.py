import numpy as np

from uav_sway.control.adaptive_equilibrium_task_lqr import AdaptiveEquilibriumTaskLQR
from uav_sway.control.base import ReferenceState


def _controller():
    return AdaptiveEquilibriumTaskLQR(np.zeros((1, 16)), 0.5, 0.5)


def test_positive_tip_error_moves_internal_bias_upwind():
    c = _controller(); r = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r)
    c.command(np.zeros(16), r, 0.2, 0.05)
    assert c.bias_x < 0.0 and c.diagnostics.bias_rate < 0.0


def test_bias_rate_and_magnitude_limits():
    c = _controller(); r = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r)
    for _ in range(1000): c.command(np.zeros(16), r, 10.0, 0.05)
    assert abs(c.bias_x) <= 0.40 + 1e-12
    assert abs(c.diagnostics.bias_rate) <= 0.10 + 1e-12


def test_setpoint_event_retains_bias_resets_filter_and_holds_adaptation():
    c = _controller(); r0 = ReferenceState(0, 0, 0, 0, 3.2, 0); c.reset(reference=r0)
    c.command(np.zeros(16), r0, 0.2, 0.05); old_bias = c.bias_x
    r1 = ReferenceState(0.3, 0, 0, 0, 3.2, 0)
    c.command(np.zeros(16), r1, 0.2, 0.05)
    assert c.bias_x == old_bias
    assert c.filtered_error_x == 0.0
    assert c.diagnostics.adaptation_held


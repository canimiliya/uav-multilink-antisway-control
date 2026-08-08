from uav_sway.control.adaptive_equilibrium_task_lqr import AdaptiveEquilibriumTaskLQR
from uav_sway.control.base import ReferenceState


def test_internal_bias_does_not_move_external_reference():
    c = AdaptiveEquilibriumTaskLQR([[0.0] * 16], 1.0, 0.5)
    reference = ReferenceState(0.0, 0, 0, 0, 3.2, 0); c.reset(reference=reference)
    c.command([0.0] * 16, reference, 0.3, 0.05)
    assert reference.x_ref == 0.0
    assert c.internal_reference(reference).x_ref <= reference.x_ref


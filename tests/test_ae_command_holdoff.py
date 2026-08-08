from uav_sway.control.adaptive_equilibrium_task_lqr import AdaptiveEquilibriumTaskLQR
from uav_sway.control.base import ReferenceState


def test_holdoff_expires_after_one_second():
    c = AdaptiveEquilibriumTaskLQR([[0.0] * 16], 1.0, 0.5)
    r0 = ReferenceState(0, 0, 0, 0, 3.2, 0); r1 = ReferenceState(0.3, 0, 0, 0, 3.2, 0)
    c.reset(reference=r0); c.command([0.0] * 16, r0, 0.3, 0.05)
    c.command([0.0] * 16, r1, 0.3, 0.05)
    for _ in range(20): c.command([0.0] * 16, r1, 0.3, 0.05)
    assert c.diagnostics.adaptation_held is False

import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.control.da_pmpc import LQRStabilizedDAPMPC
from uav_sway.mpc.preview_model import PreviewModel, reference_vector


def ref(x, vx, ax=0.0):
    return ReferenceState(x, vx, ax, 0.0, 3.2, 0.0)


def test_a_aware_shift_constant_velocity_has_zero_perfect_tracking_error():
    dt = 0.05
    A = np.eye(16)
    A[0, 1] = dt
    B = np.zeros((16, 1))
    B[0, 0] = 0.5 * dt * dt
    B[1, 0] = dt
    model = PreviewModel(A, B, np.eye(16), np.eye(16), np.zeros((1, 16)), 1)
    r0, r1 = ref(0.0, 0.75), ref(0.75 * dt, 0.75)
    c = model.reference_shift(r0, r1)
    predicted = B[:, 0] * 0.0 - c
    assert np.allclose(predicted[:2], 0.0, atol=1e-12)
    assert np.isclose(reference_vector(r1)[0] - reference_vector(r0)[0], 0.0375)


def test_a_aware_shift_constant_acceleration_has_zero_perfect_tracking_error():
    dt = 0.05
    acceleration = 0.5
    A = np.eye(16)
    A[0, 1] = dt
    B = np.zeros((16, 1))
    B[0, 0] = 0.5 * dt * dt
    B[1, 0] = dt
    model = PreviewModel(A, B, np.eye(16), np.eye(16), np.zeros((1, 16)), 1)
    r0 = ref(0.0, 0.0, acceleration)
    r1 = ref(0.5 * acceleration * dt * dt, acceleration * dt, acceleration)
    c = model.reference_shift(r0, r1)
    assert np.allclose(B[:, 0] * acceleration - c, 0.0, atol=1e-12)


def test_zero_residual_mode_is_exact_lqr_command():
    A = np.eye(16)
    B = np.ones((16, 1))
    Q = np.eye(16)
    P = np.eye(16)
    C = np.zeros((1, 16))
    K = np.arange(16, dtype=float).reshape(1, 16) / 100.0
    controller = LQRStabilizedDAPMPC(
        A, B, Q, P, C, K, 40.0, 8.0, solver=None,
        horizon_steps=1, observer_enabled=False, residual_enabled=False,
    )
    controller.reset()
    state = np.linspace(-0.02, 0.02, 16)
    horizon = type("Horizon", (), {
        "action_reference": lambda self, index: ref(0.0, 0.0, 0.2),
        "horizon_steps": 1,
    })()
    result = controller.command(state, horizon)
    expected = 0.2 - float((K @ state.reshape(-1, 1))[0, 0])
    # The limiter starts at zero, so the first executed command is slew limited;
    # raw command is the parity quantity.
    assert np.isclose(controller.diagnostics.ax_cmd_raw, expected, atol=1e-12)
    assert np.isclose(controller.diagnostics.residual_v, 0.0, atol=1e-12)
    assert np.isfinite(result)


def test_stabilized_qp_actual_input_constraint_uses_lqr_baseline():
    from uav_sway.mpc.qp_builder import build_stabilized_preview_qp

    A = np.eye(16)
    B = np.zeros((16, 1))
    B[0, 0] = 0.1
    K = np.zeros((1, 16))
    model = PreviewModel(A, B, np.eye(16), np.eye(16), np.zeros((1, 16)), 1)
    references = (ref(0.0, 0.0, 0.2), ref(0.0, 0.0, 0.2))
    qp = build_stabilized_preview_qp(model, np.zeros(16), references, K, 1.0, 20.0, 8.0)
    # a0 = 0.2 - v0, so v0 in [0.55,1.05] for previous_action=1.
    assert np.isclose(qp.lower[1], 0.55)
    assert np.isclose(qp.upper[1], 1.05)

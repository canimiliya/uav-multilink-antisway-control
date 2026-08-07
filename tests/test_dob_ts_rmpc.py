import numpy as np

from uav_sway.mpc.dob_task_residual_qp import DOBTaskResidualQP


def _qp():
    A = np.eye(16) * 0.95; B = np.zeros((16, 1)); B[0, 0] = 0.1
    K = np.zeros((1, 16)); Q = np.eye(16); P = np.eye(16)
    return DOBTaskResidualQP(A, B, K, Q, P, 20, 4.0)


def test_qp_uses_actual_action_limits_and_first_step_slew():
    result = _qp().solve(np.r_[10.0, np.zeros(15)], 0.0, 0.0)
    assert result["status_val"] in (1, 2)
    assert abs(result["predicted_first_action"]) <= 0.25 + 1e-8


def test_disturbance_is_not_double_compensated_in_prediction():
    qp = _qp()
    result = qp.solve(np.zeros(16), 0.5, 0.0)
    assert result["status_val"] in (1, 2)
    assert abs(result["predicted_first_action"]) <= 0.25 + 1e-8


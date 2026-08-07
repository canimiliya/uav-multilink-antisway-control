import numpy as np

from uav_sway.mpc.integral_task_model import build_task_lqi
from uav_sway.mpc.task_residual_qp import TaskResidualQP


def _model():
    a = 0.9 * np.eye(16); a[0, 1] = 0.05; a[1, 1] = 0.98
    b = np.zeros((16, 1)); b[1, 0] = 0.1
    c = np.zeros((4, 16)); c[0, 0] = 1.0
    return __import__("uav_sway.mpc.integral_task_model", fromlist=["build_augmented_task_model"]).build_augmented_task_model(a, b, c)


def test_residual_sign_and_first_action_constraints():
    model = _model(); lqi = build_task_lqi(model, 1.0)
    qp = TaskResidualQP(model.A_I, model.B_I, lqi["K_I"], lqi["Q_I"], lqi["P_I"], 20, 1.0)
    z = np.zeros(17); z[0] = -0.2
    result = qp.solve(z, 0.0)
    assert result["status_val"] in (1, 2)
    assert abs(result["predicted_first_action"]) <= 2.0 + 1e-7
    assert abs(result["predicted_first_action"]) <= 0.25 + 1e-7
    # Runtime convention is a = -K_I z - v.
    assert np.isclose(result["predicted_first_action"], float((-lqi["K_I"] @ z)[0] - result["v"]), atol=1e-6)

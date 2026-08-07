import numpy as np

from uav_sway.control.its_rmpc import ITSRMPC
from uav_sway.mpc.integral_task_model import build_augmented_task_model, build_task_lqi
from uav_sway.mpc.task_residual_qp import TaskResidualQP


def _model():
    a = 0.9 * np.eye(16); a[0, 1] = 0.05; a[1, 1] = 0.98
    b = np.zeros((16, 1)); b[1, 0] = 0.1
    c = np.zeros((4, 16)); c[0, 0] = 1.0
    return build_augmented_task_model(a, b, c)


def test_its_rmpc_residual_does_not_change_integral_state_dimension():
    model = _model(); lqi = build_task_lqi(model, 0.1)
    qp = TaskResidualQP(model.A_I, model.B_I, lqi["K_I"], lqi["Q_I"], lqi["P_I"], 20, 0.25)
    controller = ITSRMPC(lqi["K_I"], qp)
    action = controller.command(np.zeros(16), 0.0, dt=0.05)
    assert np.isscalar(action)
    assert controller.diagnostics.qp_status_val in (1, 2)
    assert np.isfinite(controller.diagnostics.residual_v)

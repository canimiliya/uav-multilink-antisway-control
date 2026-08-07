import numpy as np

from uav_sway.mpc.dob_task_residual_qp import DOBTaskResidualQP


def test_prediction_dimensions_and_constraints():
    qp = DOBTaskResidualQP(np.eye(16), np.ones((16, 1)) * 0.01,
                           np.zeros((1, 16)), np.eye(16), np.eye(16), 40, 1.0)
    assert qp.horizon_steps == 40
    assert qp._constraint_matrix.shape == (80, 40)


import numpy as np

from uav_sway.mpc.integral_task_model import build_augmented_task_model, build_task_lqi


def _model():
    a = np.array([[1.0, 0.05], [0.0, 0.98]])
    b = np.array([[0.0], [0.1]])
    c = np.zeros((4, 16)); c[0, 0] = 1.0
    aa = 0.9 * np.eye(16); aa[:2, :2] = a
    bb = np.zeros((16, 1)); bb[:2] = b
    return build_augmented_task_model(aa, bb, c)


def test_augmented_model_dimensions_and_integral_update():
    model = _model()
    assert model.A_I.shape == (17, 17)
    assert model.B_I.shape == (17, 1)
    assert np.allclose(model.A_I[-1, 0], 0.05)
    assert np.allclose(model.A_I[-1, -1], 1.0)


def test_task_lqi_solution_is_finite_and_stable():
    result = build_task_lqi(_model(), 1.0)
    assert result["K_I"].shape == (1, 17)
    assert np.isfinite(result["dare_residual_norm"])
    assert result["spectral_radius"] < 1.0

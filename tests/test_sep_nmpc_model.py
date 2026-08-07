import numpy as np

from uav_sway.paper_baseline.sep_nmpc_model import (
    PlanarParameters,
    planar_acceleration,
    planar_dynamics,
    planar_mass_matrix,
)


def test_planar_mass_matrix_and_vertical_equilibrium():
    parameters = PlanarParameters(9.74, 3.5, 2.57)
    matrix = planar_mass_matrix(0.0, parameters)
    np.testing.assert_allclose(matrix, [[13.24, 8.995], [8.995, 23.11715]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(planar_acceleration(0.0, 0.0, 0.0, parameters), [0.0, 0.0], atol=1e-12)


def test_nonlinear_tracking_dynamics_uses_reference_acceleration():
    parameters = PlanarParameters(9.74, 3.5, 2.57)
    derivative = planar_dynamics(np.array([0.0, 0.0, 0.0, 0.0]), 0.0, 0.75, parameters)
    np.testing.assert_allclose(derivative, [0.0, -0.75, 0.0, 0.0], atol=1e-12)

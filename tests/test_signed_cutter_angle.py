import numpy as np

from uav_sway.linearization.task_output import signed_cutter_planar_angle


def test_signed_cutter_angle_has_correct_zero_and_sign():
    assert np.isclose(signed_cutter_planar_angle([1.0, 0.0, 0.0]), 0.0)
    assert np.isclose(signed_cutter_planar_angle([np.cos(np.deg2rad(10.0)), 0.0, -np.sin(np.deg2rad(10.0))]), np.deg2rad(10.0))
    assert np.isclose(signed_cutter_planar_angle([np.cos(np.deg2rad(10.0)), 0.0, np.sin(np.deg2rad(10.0))]), -np.deg2rad(10.0))

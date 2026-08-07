import numpy as np

from uav_sway.mppi.sampler import stable_mppi_update


def test_mppi_weights_are_normalized_and_lower_cost_is_heavier():
    update = stable_mppi_update(np.zeros(3), np.eye(3), np.array([0.0, 1.0, 2.0]), 1.0)
    assert np.isclose(update.weight_sum, 1.0)
    assert update.weights[0] > update.weights[1] > update.weights[2]
    assert np.isclose(update.effective_sample_size, 1.0 / np.sum(update.weights ** 2))


def test_mppi_softmax_is_stable_for_large_cost_offset():
    update = stable_mppi_update(np.zeros(2), np.zeros((4, 2)), np.array([1e12, 1e12 + 1, 1e12 + 2, 1e12 + 3]), 5.0)
    assert np.isfinite(update.weights).all()
    assert np.isclose(update.weight_sum, 1.0)


def test_warm_start_shift_and_reset():
    nominal = np.array([1.0, 2.0, 3.0])
    shifted = np.r_[nominal[1:], 0.0]
    assert np.array_equal(shifted, [2.0, 3.0, 0.0])

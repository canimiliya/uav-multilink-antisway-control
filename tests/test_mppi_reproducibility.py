import numpy as np

from uav_sway.mppi.sampler import stable_mppi_update


def test_pcg64_same_seed_reproduces_noise_and_update():
    a = np.random.Generator(np.random.PCG64(20260810))
    b = np.random.Generator(np.random.PCG64(20260810))
    na = a.normal(size=(64, 12)); nb = b.normal(size=(64, 12))
    assert np.array_equal(na, nb)
    ua = stable_mppi_update(np.zeros(12), na, np.sum(na**2, axis=1), 1.0)
    ub = stable_mppi_update(np.zeros(12), nb, np.sum(nb**2, axis=1), 1.0)
    assert np.array_equal(ua.weights, ub.weights)
    assert np.array_equal(ua.updated_sequence, ub.updated_sequence)


def test_pcg64_different_seed_changes_noise():
    a = np.random.Generator(np.random.PCG64(20260810)).normal(size=(64, 12))
    b = np.random.Generator(np.random.PCG64(20260811)).normal(size=(64, 12))
    assert not np.array_equal(a, b)

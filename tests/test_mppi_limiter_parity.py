import numpy as np

from uav_sway.control.acceleration_limiter import AccelerationLimiter


def test_mppi_uses_frozen_amplitude_then_slew_limiter_contract():
    limiter = AccelerationLimiter(-2.0, 2.0, 0.25)
    output = [limiter.limit(value) for value in (2.0, 2.0, -2.0)]
    assert np.allclose(output, [0.25, 0.5, 0.25])
    assert limiter.diagnostics.amplitude_limited == -2.0

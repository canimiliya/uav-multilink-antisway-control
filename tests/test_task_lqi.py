import numpy as np

from uav_sway.control.its_rmpc import TaskLQI


def test_integral_clamp_and_lqi_output_is_scalar():
    controller = TaskLQI(np.ones((1, 17)))
    for _ in range(100):
        value = controller.command(np.zeros(16), 10.0, dt=0.05)
    assert isinstance(value, float)
    assert controller.eta == 1.0
    assert abs(value) <= 2.0

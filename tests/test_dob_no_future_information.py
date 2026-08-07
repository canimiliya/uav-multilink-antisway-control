import inspect

from uav_sway.control.dob_task_lqr import DOBTaskLQR
from uav_sway.control.dob_ts_rmpc import DOBTSRMPC


def test_controller_apis_do_not_accept_wind_or_future_target():
    for cls in (DOBTaskLQR, DOBTSRMPC):
        names = set(inspect.signature(cls.command).parameters)
        assert not names.intersection({"wind", "wind_profile", "wind_force", "future_wind", "future_target", "reference_horizon"})


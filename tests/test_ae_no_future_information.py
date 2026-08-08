import inspect

from uav_sway.control.adaptive_equilibrium_task_lqr import AdaptiveEquilibriumTaskLQR


def test_ae_api_has_no_wind_or_future_target_argument():
    names = set(inspect.signature(AdaptiveEquilibriumTaskLQR.command).parameters)
    assert not names.intersection({"wind", "wind_force", "wind_profile", "future_wind", "future_target", "reference_horizon"})


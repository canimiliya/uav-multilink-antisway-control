import inspect

from uav_sway.control.its_rmpc import ITSRMPC, TaskLQI
from uav_sway.evaluation.its_rmpc_runner import run_its_scenario


def test_controller_and_runner_have_no_future_wind_or_target_argument():
    for function in (TaskLQI.command, ITSRMPC.command, run_its_scenario):
        names = set(inspect.signature(function).parameters)
        assert not names.intersection({"future_wind", "wind_truth", "future_target", "target_preview"})

from pathlib import Path

import pytest


def test_formal_controller_api_has_no_future_wind_argument():
    from uav_sway.paper_baseline.sep_nmpc_controller import FormalSEPController

    assert "future_wind" not in FormalSEPController.command.__code__.co_varnames
    assert "wind_csv" not in FormalSEPController.command.__code__.co_varnames


def test_runtime_module_is_present():
    assert Path("src/uav_sway/paper_baseline/sep_nmpc_runtime.py").exists()

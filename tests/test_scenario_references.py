from pathlib import Path

import numpy as np
import pytest

from uav_sway.scenarios.reference_profiles import generate_reference
from uav_sway.scenarios.scenario_config import load_scenario_config


ROOT = Path(__file__).resolve().parents[1]


def test_reference_sample_count_and_frozen_keypoints():
    config = load_scenario_config(ROOT / "configs/scenarios.yaml")
    time = np.arange(2401, dtype=float) * 0.005
    approach = generate_reference("approach_stop", time, config)
    for t, x, vx in [(0, 0, 0), (1, 0, 0), (2, 0.375, 0.75), (5, 2.625, 0.75), (6, 3, 0), (12, 3, 0)]:
        index = int(round(t / 0.005))
        assert approach["x_ref"][index] == pytest.approx(x, abs=1e-12)
        assert approach["vx_ref"][index] == pytest.approx(vx, abs=1e-12)
    gust = generate_reference("gust_micro_adjust", time, config)
    assert gust["x_ref"][600] == pytest.approx(0.0)
    assert gust["x_ref"][1000] == pytest.approx(0.30, abs=1e-12)
    assert gust["vx_ref"][600] == pytest.approx(0.0)
    assert gust["vx_ref"][1000] == pytest.approx(0.0)
    assert np.isfinite(approach["x_ref"]).all()

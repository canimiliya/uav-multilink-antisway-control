from pathlib import Path

import numpy as np
import pytest

from uav_sway.scenarios.reference_profiles import generate_reference
from uav_sway.scenarios.reference_profiles import quintic_boundary_segment
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


def test_quintic_boundary_segment_satisfies_six_endpoint_conditions():
    time = np.asarray([1.0, 2.0])
    x, v, a = quintic_boundary_segment(time, 1.0, 2.0, 0.0, 0.0, 0.0, 0.375, 0.75, 0.0)
    assert x == pytest.approx([0.0, 0.375], abs=1e-12)
    assert v == pytest.approx([0.0, 0.75], abs=1e-12)
    assert a == pytest.approx([0.0, 0.0], abs=1e-12)


def test_approach_stop_is_continuous_at_all_boundaries():
    config = load_scenario_config(ROOT / "configs/scenarios.yaml")
    epsilon = 1e-7
    for boundary in (1.0, 2.0, 5.0, 6.0):
        reference = generate_reference("approach_stop", np.asarray([boundary - epsilon, boundary, boundary + epsilon]), config)
        for key, threshold in (("x_ref", 1e-6), ("vx_ref", 1e-5), ("ax_ref", 1e-4)):
            assert np.max(np.abs(np.diff(reference[key]))) < threshold

    time = np.arange(2401, dtype=float) * 0.005
    reference = generate_reference("approach_stop", time, config)
    assert abs(reference["vx_ref"][400] - reference["vx_ref"][399]) < 1e-3
    assert abs(reference["vx_ref"][1200] - reference["vx_ref"][1199]) < 1e-3

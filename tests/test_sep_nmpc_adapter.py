from pathlib import Path

import numpy as np

from uav_sway.paper_baseline.sep_nmpc_adapter import (
    equivalent_sway_angle,
    equivalent_sway_rate,
    measure_equivalent_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def test_equivalent_angle_sign_and_equilibrium():
    suspension = np.array([0.0, 0.0, 2.96])
    cutter = np.array([0.0, 0.0, 0.39])
    assert equivalent_sway_angle(suspension, cutter) == 0.0
    assert equivalent_sway_angle(suspension, cutter + [0.01, 0.0, 0.0]) > 0
    assert equivalent_sway_angle(suspension, cutter + [-0.01, 0.0, 0.0]) < 0
    assert np.isfinite(equivalent_sway_rate(0.1, 0.2, 0.05))


def test_equivalent_parameters_are_measured_from_frozen_model():
    result = measure_equivalent_parameters(ROOT / "configs/model_5link.yaml")
    assert result["m_Q"] == 9.74
    assert result["m_L"] == 3.5
    assert result["l_eq"] > 2.5
    assert result["mapping"] == "OUR FAIR ADAPTATION"

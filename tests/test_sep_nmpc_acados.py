from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uav_sway.paper_baseline.sep_nmpc_model import PlanarParameters
from uav_sway.paper_baseline.sep_nmpc_ocp import SEPTrackingConfig, frozen_parameter_grid
from uav_sway.paper_baseline.sep_nmpc_passivity import acceleration_from_tracking_shaped_input, passivity_residual


def test_frozen_grid_is_exactly_eight_and_no_holdout_names():
    grid = frozen_parameter_grid()
    assert len(grid) == 8
    assert {(c.k_e, c.rho, c.epsilon) for c in grid} == {
        (10.0, 0.05, 0.005), (10.0, 0.05, 0.02), (10.0, 0.2, 0.005), (10.0, 0.2, 0.02),
        (40.0, 0.05, 0.005), (40.0, 0.05, 0.02), (40.0, 0.2, 0.005), (40.0, 0.2, 0.02),
    }


def test_actual_acceleration_mapping_and_passivity():
    p = PlanarParameters(9.74, 3.5, 2.57)
    c = SEPTrackingConfig(k_e=40.0, rho=0.2, epsilon=0.02)
    ax = acceleration_from_tracking_shaped_input(0.5, 0.75, c.k_e, -0.01, p)
    assert ax == pytest.approx(0.75 + (0.5 + 0.4) / p.m_T)
    assert passivity_residual(0.0, 0.0, c.rho, c.epsilon, 0.0) == 0.0


def test_formal_acados_dimensions_and_smoke(tmp_path: Path):
    pytest.importorskip("acados_template")
    from uav_sway.paper_baseline.sep_nmpc_controller import FormalSEPController

    controller = FormalSEPController(PlanarParameters(9.74, 3.5, 2.57), SEPTrackingConfig(), tmp_path / "generated")
    assert controller.config.shooting_nodes == 40
    assert controller.config.horizon_seconds == 2.0
    assert int(controller.model.x.shape[0]) == 4
    assert int(controller.model.u.shape[0]) == 2
    reference = {name: np.zeros(41) for name in ("x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}
    controller.reset(np.zeros(4))
    ax = controller.command(np.zeros(4), reference, 0.0)
    assert np.isfinite(ax)
    assert controller.diagnostics.acados_status == 0


def test_parameter_preregistration_is_unchanged():
    path = Path("artifacts/s5d1/parameter_preregistration.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["grid_size"] == 8
    assert data["holdout_tuning_allowed"] is False

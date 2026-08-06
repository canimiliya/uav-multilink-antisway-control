from pathlib import Path

import mujoco
import numpy as np

from uav_sway.disturbances.aerodynamics import box_projected_area, link_projected_area, quadratic_wind_force
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.models.model_config import load_model_config


ROOT = Path(__file__).resolve().parents[1]


def test_quadratic_force_zero_reverse_and_square_scaling():
    assert quadratic_wind_force(0.0, 0.0, 1.225, 1.0, 1.0) == 0.0
    assert quadratic_wind_force(-2.0, 0.0, 1.0, 1.0, 1.0) < 0.0
    one = abs(quadratic_wind_force(1.0, 0.0, 1.0, 1.0, 1.0))
    two = abs(quadratic_wind_force(2.0, 0.0, 1.0, 1.0, 1.0))
    assert two == 4.0 * one
    assert quadratic_wind_force(2.0, 2.0, 1.0, 1.0, 1.0) == 0.0


def test_each_s2_body_gets_independent_force_and_buffer_is_cleared():
    model_config = load_model_config(ROOT / "configs/model_5link.yaml")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s1/generated/model_5link.xml"))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 3.2, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    aero = load_aerodynamic_config(ROOT / "configs/aerodynamics.yaml")
    data.xfrc_applied[:] = 123.0
    zero = clear_and_apply_wind(model, data, model_config, aero, 0.0)
    assert np.allclose(data.xfrc_applied, 0.0)
    assert zero["total_x"] == 0.0
    forces = clear_and_apply_wind(model, data, model_config, aero, 3.0)
    assert all(forces[f"{name}_x"] > 0.0 for name in ["quadrotor", *[f"link_{i}" for i in range(1, 6)], "cutter"])
    assert np.isfinite(list(forces.values())).all()
    assert box_projected_area((0.98, 0.76, 0.48), np.eye(3), np.array([1, 0, 0])) > 0.0
    assert link_projected_area(0.5, 0.05, np.eye(3), np.array([1, 0, 0])) > 0.0

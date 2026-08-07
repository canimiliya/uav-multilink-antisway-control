"""Fair five-link measurement adapter for SEP-NMPC-adapted."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.model_config import ModelConfig, load_model_config


def equivalent_sway_angle(suspension_position: np.ndarray, cutter_com_position: np.ndarray) -> float:
    """Return atan2(dx, downward dz), with +x sway mapped to +angle."""

    suspension = np.asarray(suspension_position, dtype=float)
    cutter = np.asarray(cutter_com_position, dtype=float)
    if suspension.shape != (3,) or cutter.shape != (3,):
        raise ValueError("positions must be three-dimensional")
    return float(np.arctan2(cutter[0] - suspension[0], suspension[2] - cutter[2]))


def equivalent_sway_rate(previous_angle: float, current_angle: float, dt: float) -> float:
    """Causal finite-difference angular rate; no future state is used."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    return float((float(current_angle) - float(previous_angle)) / float(dt))


def equivalent_cutter_com_and_suspension(model, data, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Read the current attachment and cutter COM from MuJoCo kinematics."""

    quad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_1")
    cutter_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cutter")
    if min(quad_id, link_id, cutter_id) < 0:
        raise KeyError("quadrotor, link_1, or cutter body missing")
    # link_1 xpos is the generated suspension attachment point.  xipos is the
    # mass COM; unlike the tip site it is the correct paper payload analogue.
    suspension = np.asarray(data.xpos[link_id], dtype=float).copy()
    cutter_com = np.asarray(data.xipos[cutter_id], dtype=float).copy()
    return cutter_com, suspension


def measure_equivalent_parameters(config_path: str | Path) -> dict:
    """Build the frozen five-link model and measure its zero-sway geometry."""

    config_path = Path(config_path).resolve()
    config = load_model_config(config_path)
    with tempfile.TemporaryDirectory(prefix="sep_s5d1_") as temporary:
        xml_path = Path(temporary) / "model_5link.xml"
        build_planar_chain_model(config_path, xml_path)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        data.eq_active[:] = 0
        mujoco.mj_forward(model, data)
        cutter_com, suspension = equivalent_cutter_com_and_suspension(model, data, config)
    length = float(np.linalg.norm(cutter_com - suspension))
    expected_mass = float(config.total_link_mass_kg + config.payload.mass_kg)
    return {
        "m_Q": float(config.airframe.mass_kg),
        "m_L": expected_mass,
        "l_eq": length,
        "g": 9.81,
        "mapping": "OUR FAIR ADAPTATION",
        "source_config": str(config_path),
        "n_links": int(config.n_links),
        "total_link_mass_kg": float(config.total_link_mass_kg),
        "cutter_mass_kg": float(config.payload.mass_kg),
        "suspension_position_m": suspension.tolist(),
        "cutter_com_position_m": cutter_com.tolist(),
        "measurement_definition": "equilibrium suspension attachment point to cutter rigid-body COM",
        "future_wind_used": False,
    }


def save_equivalent_parameters(config_path: str | Path, output_path: str | Path) -> dict:
    result = measure_equivalent_parameters(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result

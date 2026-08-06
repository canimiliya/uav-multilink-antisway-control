from pathlib import Path

import mujoco
import numpy as np
import pytest

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.model_config import load_model_config


ROOT = Path(__file__).resolve().parents[1]


def test_m400_configuration_and_payload_limits(tmp_path):
    config = load_model_config(ROOT / "configs/model_5link.yaml")
    airframe = config.airframe
    payload = config.payload
    assert airframe.model == "Matrice 400"
    assert airframe.mass_kg == pytest.approx(9.74, abs=1e-9)
    assert airframe.dimensions_m == (0.98, 0.76, 0.48)
    assert airframe.diagonal_wheelbase_m == pytest.approx(1.07)
    assert airframe.propeller_diameter_m == pytest.approx(0.635)
    assert airframe.inertia_diagonal_kg_m2 == pytest.approx(
        (0.655826666666667, 0.966532666666667, 1.248343333333333)
    )
    assert all(value > 0 for value in airframe.inertia_diagonal_kg_m2)
    assert payload.mass_kg == pytest.approx(2.5, abs=1e-9)
    assert payload.dimensions_xyz_m == (0.45, 0.16, 0.14)
    assert payload.half_extents_xyz_m == (0.225, 0.08, 0.07)
    assert payload.attachment_local_position_m == (0.0, 0.0, 0.0)
    assert payload.geom_center_local_position_m == (0.0, 0.0, -0.07)
    assert payload.tip_local_position_m == (0.225, 0.0, -0.07)
    external_payload = config.total_link_mass_kg + payload.mass_kg
    total_takeoff_mass = airframe.mass_kg + external_payload
    assert external_payload <= airframe.max_payload_kg
    assert total_takeoff_mass <= airframe.max_takeoff_mass_kg
    assert external_payload == pytest.approx(3.5)
    assert total_takeoff_mass == pytest.approx(13.24)

    xml_path = tmp_path / "model_5link.xml"
    build_planar_chain_model(ROOT / "configs/model_5link.yaml", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    quad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    cutter_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cutter")
    assert model.body_mass[quad_id] == pytest.approx(9.74, abs=1e-9)
    assert model.body_mass[cutter_id] == pytest.approx(2.5, abs=1e-9)
    assert np.allclose(model.body_inertia[quad_id], airframe.inertia_diagonal_kg_m2)
    free_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "quadrotor_free")
    assert model.jnt_type[free_joint] == mujoco.mjtJoint.mjJNT_FREE
    expected_rotors = (
        (airframe.rotor_xy_coordinate_abs_m, airframe.rotor_xy_coordinate_abs_m),
        (-airframe.rotor_xy_coordinate_abs_m, airframe.rotor_xy_coordinate_abs_m),
        (-airframe.rotor_xy_coordinate_abs_m, -airframe.rotor_xy_coordinate_abs_m),
        (airframe.rotor_xy_coordinate_abs_m, -airframe.rotor_xy_coordinate_abs_m),
    )
    for index, (x, y) in enumerate(expected_rotors):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"quadrotor_rotor_prop_geom_{index}")
        assert np.allclose(model.geom_pos[geom_id][:2], [x, y])
        assert model.geom_size[geom_id][0] == pytest.approx(0.3175)
    assert np.isfinite(model.body_mass).all()
    assert np.allclose(model.body_inertia[cutter_id], [0.00941666666667, 0.0462708333333, 0.0475208333333])
    assert np.allclose(model.body_pos[cutter_id], [0.0, 0.0, -0.5])
    assert np.allclose(model.body_ipos[cutter_id], [0.0, 0.0, -0.07])
    cutter_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cutter_geom")
    assert np.allclose(model.geom_pos[cutter_geom_id], [0.0, 0.0, -0.07])
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    assert np.allclose(model.site_pos[tip_id], [0.225, 0.0, -0.07])
    assert not config.airframe.show_dimension_envelope

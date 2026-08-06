from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.model_config import load_model_config


ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path, n_links):
    config_path = ROOT / "configs" / f"model_{n_links}link.yaml"
    xml_path = tmp_path / f"model_{n_links}link.xml"
    build_planar_chain_model(config_path, xml_path)
    return mujoco.MjModel.from_xml_path(str(xml_path))


def test_all_frozen_chain_sizes_load_with_expected_topology(tmp_path):
    for n_links in (4, 5, 6):
        model = _build(tmp_path, n_links)
        assert (model.nq, model.nv) == (7 + n_links, 6 + n_links)
        assert np.isfinite(model.body_mass).all()
        assert model.njnt == n_links + 1
        types = list(model.jnt_type)
        assert types.count(mujoco.mjtJoint.mjJNT_FREE) == 1
        assert types.count(mujoco.mjtJoint.mjJNT_HINGE) == n_links
        assert mujoco.mjtJoint.mjJNT_BALL not in types
        assert mujoco.mjtJoint.mjJNT_SLIDE not in types
        for index in range(1, n_links + 1):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{index}")
            assert np.allclose(model.jnt_axis[joint_id], [0, 1, 0])
            dof_id = model.jnt_dofadr[joint_id]
            assert model.dof_damping[dof_id] == 0.05
            assert model.dof_frictionloss[dof_id] == 0.005
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cutter") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip") >= 0


def test_mass_and_geometry_configuration_are_frozen(tmp_path):
    model = _build(tmp_path, 5)
    config = load_model_config(ROOT / "configs" / "model_5link.yaml")
    quad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    cutter = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cutter")
    link_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}") for i in range(1, 6)]
    assert model.body_mass[quad] == pytest.approx(9.74, abs=1e-9)
    assert model.body_mass[cutter] == pytest.approx(2.5, abs=1e-9)
    assert sum(model.body_mass[i] for i in link_ids) == pytest.approx(1.0, abs=1e-9)
    assert config.link_length == 0.5
    cutter_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cutter_geom")
    assert model.geom_type[cutter_geom] == mujoco.mjtGeom.mjGEOM_BOX
    assert np.allclose(model.geom_size[cutter_geom], [0.225, 0.08, 0.07])
    assert model.geom_size[cutter_geom][0] > model.geom_size[cutter_geom][1]
    assert model.geom_size[cutter_geom][0] > model.geom_size[cutter_geom][2]
    assert np.allclose(model.geom_pos[cutter_geom], [0.0, 0.0, -0.07])


def test_visual_geometry_is_massless_collisionless_and_topology_preserving(tmp_path):
    model = _build(tmp_path, 5)
    xml_root = ET.parse(tmp_path / "model_5link.xml").getroot()
    geoms_by_name = {
        geom.attrib["name"]: geom
        for geom in xml_root.iter("geom")
        if "name" in geom.attrib
    }
    envelope = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m400_dimension_envelope_debug")
    assert envelope < 0
    for name in ("m400_fuselage", "m400_lower_body"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0
        assert geoms_by_name[name].attrib["mass"] == "0"
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0
    marker_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"joint_marker_{i}") for i in range(1, 6)]
    assert len(set(marker_ids)) == 5
    assert all(marker_id >= 0 for marker_id in marker_ids)
    assert all(geoms_by_name[f"joint_marker_{i}"].attrib["mass"] == "0" for i in range(1, 6))
    assert all(model.geom_contype[marker_id] == 0 for marker_id in marker_ids)
    assert all(model.geom_conaffinity[marker_id] == 0 for marker_id in marker_ids)
    assert all(model.geom_group[marker_id] == 2 for marker_id in marker_ids)
    expected_colors = np.asarray([
        [0.95, 0.20, 0.20, 1.0],
        [0.95, 0.75, 0.15, 1.0],
        [0.20, 0.80, 0.35, 1.0],
        [0.10, 0.75, 0.90, 1.0],
        [0.25, 0.40, 0.95, 1.0],
    ])
    assert np.allclose(model.geom_rgba[marker_ids], expected_colors)

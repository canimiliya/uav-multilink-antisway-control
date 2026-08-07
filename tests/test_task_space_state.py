from pathlib import Path

import mujoco
import numpy as np

from uav_sway.task_space.state import CutterTaskSpaceReader


ROOT = Path(__file__).resolve().parents[1]


def _equilibrium():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def test_equilibrium_axis_is_unit_and_state_finite():
    model, data = _equilibrium()
    state = CutterTaskSpaceReader(model).read(model, data)
    assert np.isclose(np.linalg.norm(state.cutter_axis_world), 1.0)
    assert np.isfinite(state.tip_position_world).all()
    assert np.isfinite(state.tip_velocity_world).all()


def test_tip_velocity_matches_mujoco_jacobian():
    model, data = _equilibrium()
    data.qvel[:] = np.linspace(-0.2, 0.2, model.nv)
    mujoco.mj_forward(model, data)
    reader = CutterTaskSpaceReader(model)
    state = reader.read(model, data)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    tip_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip"))
    mujoco.mj_jacSite(model, data, jacp, jacr, tip_id)
    assert np.allclose(state.tip_velocity_world, jacp @ data.qvel)


def test_rotation_axis_uses_local_cutter_x_axis():
    model, data = _equilibrium()
    reader = CutterTaskSpaceReader(model)
    state = reader.read(model, data)
    assert np.allclose(state.cutter_axis_world, state.cutter_rotation_world @ np.array([1.0, 0.0, 0.0]))

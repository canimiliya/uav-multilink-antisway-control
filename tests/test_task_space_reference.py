from pathlib import Path

import mujoco
import numpy as np

from uav_sway.task_space.reference import build_equilibrium_task_pose, task_reference_at


ROOT = Path(__file__).resolve().parents[1]


def test_equilibrium_reference_is_measured_and_axis_is_normalized():
    path = ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"
    model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
    data.qpos[:] = 0.0; data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]; mujoco.mj_forward(model, data)
    pose = build_equilibrium_task_pose(model, data, path)
    ref = task_reference_at({"x_ref": 2.0, "y_ref": 0.5, "z_ref": 4.0}, pose)
    assert np.isclose(pose.axis_norm, 1.0)
    assert np.isclose(np.linalg.norm(ref.cutter_axis_world), 1.0)
    assert np.allclose(ref.tip_position_world, np.array([2.0, 0.5, 4.0]) + pose.tip_relative_position_m)
    assert len(pose.model_sha256) == 64

from pathlib import Path

import mujoco
import numpy as np

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.passive_sim import _set_initial_state
from uav_sway.models.state_io import capture_state, restore_state


ROOT = Path(__file__).resolve().parents[1]


def test_state_restore_replays_identically(tmp_path):
    xml_path = tmp_path / "model_5link.xml"
    build_planar_chain_model(ROOT / "configs/model_5link.yaml", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_initial_state(model, data, 10.0)
    for _ in range(250):
        mujoco.mj_step(model, data)
    snapshot = capture_state(model, data)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    for _ in range(1000):
        mujoco.mj_step(model, data)
    qpos_b = data.qpos.copy()
    qvel_b = data.qvel.copy()
    tip_b = data.site_xpos[tip_id].copy()
    restore_state(model, data, snapshot)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.max(np.abs(data.qpos - qpos_b)) < 1e-9
    assert np.max(np.abs(data.qvel - qvel_b)) < 1e-9
    assert np.max(np.abs(data.site_xpos[tip_id] - tip_b)) < 1e-9

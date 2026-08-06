from pathlib import Path

import mujoco
import numpy as np

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.passive_sim import _set_initial_state


ROOT = Path(__file__).resolve().parents[1]


def test_five_link_static_hanging_equilibrium(tmp_path):
    xml_path = tmp_path / "model_5link.xml"
    build_planar_chain_model(ROOT / "configs/model_5link.yaml", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_initial_state(model, data, 0.0)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    initial_tip = data.site_xpos[tip_id].copy()
    for _ in range(5000):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert np.max(np.abs(data.qpos[7:])) < 1e-3
    assert np.max(np.abs(data.qvel[6:])) < 1e-3
    assert np.linalg.norm(data.site_xpos[tip_id][:2] - initial_tip[:2]) < 1e-3

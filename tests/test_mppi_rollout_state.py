from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.mppi import MuJoCoMPPI
from uav_sway.mppi.reference_horizon import make_reference_horizon
from uav_sway.models.state_io import capture_state


ROOT = Path(__file__).resolve().parents[1]


def test_mppi_solve_does_not_modify_real_plant():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"))
    data = mujoco.MjData(model); data.qpos[:7] = [0, 0, 3.2, 1, 0, 0, 0]; data.qvel[:] = 0; data.ctrl[:] = 0; data.eq_active[:] = 0; mujoco.mj_forward(model, data)
    snapshot = capture_state(model, data)
    cfg = yaml.safe_load((ROOT / "configs/mppi.yaml").read_text(encoding="utf-8"))
    q = np.load(ROOT / "artifacts/s4/lqr/Q.npy"); r = np.load(ROOT / "artifacts/s4/lqr/R.npy")
    mppi = MuJoCoMPPI(model, q, r, float(np.sum(model.body_mass)), model.body_inertia[1], 5, 0.0, 1.0, .2, 20260810, num_rollouts=2)
    ref = {name: np.zeros(2401) for name in ("time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}; ref["z_ref"][:] = 3.2
    mppi.solve(data, make_reference_horizon(ref, 0, 12))
    assert np.array_equal(data.qpos, snapshot.qpos)
    assert np.array_equal(data.qvel, snapshot.qvel)
    assert data.time == snapshot.time
    assert np.array_equal(data.ctrl, snapshot.ctrl)

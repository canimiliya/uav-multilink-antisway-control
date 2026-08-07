"""Tip-output identification from the frozen MuJoCo equilibrium."""

from __future__ import annotations

import mujoco
import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.linearization.reduced_state import ReducedStateLayout
from uav_sway.models.state_io import MujocoStateSnapshot, restore_state


def identify_tip_output(model, snapshot: MujocoStateSnapshot, epsilon: np.ndarray,
                        equilibrium_relative_x: float) -> np.ndarray:
    """Return the 1x16 centre-difference Jacobian of horizontal tip displacement."""
    layout = ReducedStateLayout(model)
    data = mujoco.MjData(model)
    reference = ReferenceState(0.0, 0.0, 0.0, 0.0, 3.2, 0.0)
    tip_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip"))
    quad_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor"))
    result = np.zeros(16, dtype=float)
    for i, h in enumerate(np.asarray(epsilon, dtype=float)):
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError("tip-output epsilon must be positive and finite")
        plus = np.zeros(16, dtype=float); plus[i] = h
        minus = np.zeros(16, dtype=float); minus[i] = -h
        restore_state(model, data, snapshot)
        layout.inject(model, data, snapshot, plus, reference)
        e_plus = float(data.site_xpos[tip_id, 0] - data.xpos[quad_id, 0] - equilibrium_relative_x)
        restore_state(model, data, snapshot)
        layout.inject(model, data, snapshot, minus, reference)
        e_minus = float(data.site_xpos[tip_id, 0] - data.xpos[quad_id, 0] - equilibrium_relative_x)
        result[i] = (e_plus - e_minus) / (2.0 * h)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite tip output Jacobian")
    return result.reshape(1, 16)

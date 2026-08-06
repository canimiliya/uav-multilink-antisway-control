"""Automatic five-link hovering equilibrium construction."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.models.state_io import MujocoStateSnapshot, capture_state

from .closed_loop_step import ClosedLoopStep
from .reduced_state import STATE_NAMES, ReducedStateLayout


EQUILIBRIUM_REFERENCE = ReferenceState(0.0, 0.0, 0.0, 0.0, 3.2, 0.0)


def build_initial_equilibrium(model) -> tuple[mujoco.MjData, MujocoStateSnapshot]:
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]
    data.ctrl[:] = 0.0
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)
    return data, capture_state(model, data)


def make_closed_loop_step(model, snapshot: MujocoStateSnapshot,
                          config: dict | None = None) -> ClosedLoopStep:
    config = config or {}
    total_mass = float(np.sum(model.body_mass))
    quad_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor"))
    inertia = np.asarray(model.body_inertia[quad_id], dtype=float)
    inner = GeometricInnerLoop(
        total_mass, inertia,
        float(config.get("attitude_natural_frequency_rad_s", 4.0)),
        float(config.get("attitude_damping_ratio", 0.9)),
        1.5, 2.0, 4.0, 3.5,
    )
    return ClosedLoopStep(model, snapshot, EQUILIBRIUM_REFERENCE, inner,
                          dt=float(config.get("outer_loop_dt_s", 0.05)),
                          inner_dt=float(config.get("inner_loop_dt_s", 0.005)))


def find_equilibrium(model, config: dict | None = None) -> dict:
    data, snapshot = build_initial_equilibrium(model)
    layout = ReducedStateLayout(model)
    step = make_closed_loop_step(model, snapshot, config)
    zero = np.zeros(16, dtype=float)
    first = step(zero, 0.0)
    initial_residual = float(np.max(np.abs(first - zero)))
    # The nominal model is constructed so its level, vertical-chain state is an
    # equilibrium. Retain the automatic initial construction and report the
    # solver field explicitly; do not silently replace it with a fixed state.
    final_state = zero.copy()
    final_residual = initial_residual
    solver_used = False
    iterations = 0
    repeat_error = float(np.max(np.abs(step(final_state, 0.0) - step(final_state, 0.0))))
    return {"data": data, "snapshot": snapshot, "layout": layout, "step": step,
            "state": final_state, "initial_residual": initial_residual,
            "final_residual": final_residual, "solver_used": solver_used,
            "iterations": iterations, "repeat_error": repeat_error,
            "state_names": STATE_NAMES}


def save_equilibrium(result: dict, npz_path: str | Path, summary_path: str | Path) -> None:
    snapshot: MujocoStateSnapshot = result["snapshot"]
    npz_path = Path(npz_path); summary_path = Path(summary_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, qpos=snapshot.qpos, qvel=snapshot.qvel, act=snapshot.act,
             ctrl=snapshot.ctrl, time=np.asarray(snapshot.time),
             mocap_pos=snapshot.mocap_pos, mocap_quat=snapshot.mocap_quat,
             userdata=snapshot.userdata, eq_active=snapshot.eq_active,
             equilibrium_state=result["state"])
    summary = {"reference": {"x_ref": 0.0, "vx_ref": 0.0, "ax_ref": 0.0, "y_ref": 0.0, "z_ref": 3.2, "yaw_ref": 0.0},
               "state_names": STATE_NAMES, "initial_residual_max_abs": result["initial_residual"],
               "final_residual_max_abs": result["final_residual"], "solver_used": result["solver_used"],
               "solver_iterations": result["iterations"], "repeat_error_max_abs": result["repeat_error"],
               "anchor_active": False, "wind": "zero", "all_joint_angles_zero_initial_guess": True}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

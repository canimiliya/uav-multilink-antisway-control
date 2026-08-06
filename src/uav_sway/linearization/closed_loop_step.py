"""Deterministic 0.05 s nonlinear closed-loop map used for identification."""

from __future__ import annotations

import mujoco
import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.models.state_io import MujocoStateSnapshot, restore_state

from .reduced_state import ReducedStateLayout


class ClosedLoopStep:
    def __init__(self, model, equilibrium_snapshot: MujocoStateSnapshot,
                 reference: ReferenceState, inner_loop: GeometricInnerLoop,
                 dt: float = 0.05, inner_dt: float = 0.005):
        self.model = model
        self.snapshot = equilibrium_snapshot
        self.reference = reference
        self.inner = inner_loop
        self.dt = float(dt)
        self.inner_dt = float(inner_dt)
        self.physics_steps = int(round(self.dt / float(model.opt.timestep)))
        self.inner_steps = int(round(self.inner_dt / float(model.opt.timestep)))
        if self.physics_steps != 50 or self.inner_steps != 5:
            raise ValueError("S4 closed-loop schedule must be 50/5 physics steps")
        self.layout = ReducedStateLayout(model)
        self.quad_id = self.layout.quad_body_id
        self.actuator_ids = {
            name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
            for name in ("rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3", "thrust_motor", "mx_motor", "my_motor", "mz_motor")
        }

    def __call__(self, state: np.ndarray, u: float) -> np.ndarray:
        data = mujoco.MjData(self.model)
        restore_state(self.model, data, self.snapshot)
        self.layout.inject(self.model, data, self.snapshot, state, self.reference)
        for step in range(self.physics_steps):
            data.xfrc_applied[:] = 0.0
            if step % self.inner_steps == 0:
                from uav_sway.control.state_reader import StateReader
                quad_x = float(data.xpos[self.quad_id, 0])
                tip_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip"))
                reader = StateReader(self.model, 5, float(data.site_xpos[tip_id, 0] - quad_x))
                control_state = reader.read(self.model, data)
                result = self.inner.compute(control_state, self.reference, float(u))
                thrust = float(result["thrust_raw_N"])
                torque = np.asarray(result["torque_raw_Nm"], dtype=float)
                data.ctrl[:] = 0.0
                data.ctrl[self.actuator_ids["thrust_motor"]] = np.clip(thrust, *self.model.actuator_ctrlrange[self.actuator_ids["thrust_motor"]])
                for i, name in enumerate(("mx_motor", "my_motor", "mz_motor")):
                    data.ctrl[self.actuator_ids[name]] = np.clip(torque[i], *self.model.actuator_ctrlrange[self.actuator_ids[name]])
            mujoco.mj_step(self.model, data)
        return self.layout.extract(self.model, data, self.reference)

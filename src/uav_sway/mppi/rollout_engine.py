"""Independent-data MuJoCo nonlinear rollout engine for S5."""

from __future__ import annotations

import mujoco
import numpy as np

from uav_sway.control.base import ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.state_reader import StateReader
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.linearization.reduced_state import ReducedStateLayout
from uav_sway.models.state_io import MujocoStateSnapshot, restore_state
from uav_sway.mppi.cost import candidate_acceleration, mppi_candidate_cost
from uav_sway.mppi.reference_horizon import ReferenceHorizon


def _id(model, object_type, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise KeyError(name)
    return value


class NonlinearRolloutEngine:
    def __init__(self, model, q, r, layout: ReducedStateLayout,
                 reader: StateReader, inner: GeometricInnerLoop, n_links: int,
                 equilibrium_relative_x: float, tip_weight: float,
                 terminal_multiplier: float, ax_min: float, ax_max: float,
                 slew_limit: float, model_config=None, aerodynamic_config=None):
        self.model = model
        self.q = np.asarray(q, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.layout = layout
        self.reader = reader
        self.inner = inner
        self.n_links = n_links
        self.equilibrium_relative_x = float(equilibrium_relative_x)
        self.tip_weight = float(tip_weight)
        self.terminal_multiplier = float(terminal_multiplier)
        self.ax_min, self.ax_max, self.slew_limit = float(ax_min), float(ax_max), float(slew_limit)
        self.model_config = model_config
        self.aerodynamic_config = aerodynamic_config
        self.data = mujoco.MjData(model)
        self.quad_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self.tip_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
        self.joint_qpos_addresses = []
        for i in range(1, n_links + 1):
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}")
            self.joint_qpos_addresses.append(int(model.jnt_qposadr[jid]))
        self.actuator_ids = {name: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                             for name in ("rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3", "thrust_motor", "mx_motor", "my_motor", "mz_motor")}
        self.inner_steps = int(round(0.005 / float(model.opt.timestep)))
        self.outer_steps = int(round(0.05 / float(model.opt.timestep)))
        self.physics_steps_per_rollout = self.outer_steps * 12

    @staticmethod
    def _pitch(rotation: np.ndarray) -> float:
        return float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))

    def _unsafe(self) -> bool:
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            return True
        if float(self.data.xpos[self.quad_id, 2]) <= 0.05 or float(self.data.site_xpos[self.tip_id, 2]) <= 0.05:
            return True
        if abs(self._pitch(np.asarray(self.data.xmat[self.quad_id]).reshape(3, 3))) >= np.deg2rad(25.0):
            return True
        for address in self.joint_qpos_addresses:
            if abs(float(self.data.qpos[address])) >= np.deg2rad(100.0):
                return True
        return False

    def _apply_controls(self, reference: ReferenceState, ax: float) -> None:
        state = self.reader.read(self.model, self.data)
        inner_result = self.inner.compute(state, reference, ax)
        thrust = float(inner_result["thrust_raw_N"])
        torque = np.asarray(inner_result["torque_raw_Nm"], dtype=float)
        self.data.ctrl[:] = 0.0
        thrust_id = self.actuator_ids["thrust_motor"]
        self.data.ctrl[thrust_id] = np.clip(thrust, *self.model.actuator_ctrlrange[thrust_id])
        for idx, name in enumerate(("mx_motor", "my_motor", "mz_motor")):
            aid = self.actuator_ids[name]
            self.data.ctrl[aid] = np.clip(torque[idx], *self.model.actuator_ctrlrange[aid])

    def rollout(self, snapshot: MujocoStateSnapshot, delta_sequence: np.ndarray,
                horizon: ReferenceHorizon, previous_ax: float) -> tuple[float, bool]:
        restore_state(self.model, self.data, snapshot)
        limiter_previous = float(previous_ax)
        state_values: list[np.ndarray] = []
        tip_values: list[float] = []
        actual_deltas = []
        for action_index, delta in enumerate(np.asarray(delta_sequence, dtype=float)):
            if action_index >= horizon.action_count:
                raise ValueError("delta sequence exceeds reference action horizon")
            action_reference = horizon.action_reference(action_index)
            state_reference = horizon.state_reference(action_index)
            amplitude = float(np.clip(candidate_acceleration(action_reference.ax_ref, delta), self.ax_min, self.ax_max))
            action_ax = limiter_previous + float(np.clip(amplitude - limiter_previous,
                                                          -self.slew_limit, self.slew_limit))
            action_ax = float(np.clip(action_ax, self.ax_min, self.ax_max))
            limiter_previous = action_ax
            actual_deltas.append(float(delta))
            for _ in range(self.outer_steps // self.inner_steps):
                self._apply_controls(action_reference, action_ax)
                for _ in range(self.inner_steps):
                    if self.model_config is not None and self.aerodynamic_config is not None:
                        clear_and_apply_wind(
                            self.model, self.data, self.model_config,
                            self.aerodynamic_config, wind_x=0.0,
                        )
                    else:
                        self.data.xfrc_applied[:] = 0.0
                    mujoco.mj_step(self.model, self.data)
                    if self._unsafe():
                        return 1.0e12, False
            state = self.layout.extract(self.model, self.data, state_reference)
            tip = float(self.data.site_xpos[self.tip_id, 0] - self.data.xpos[self.quad_id, 0] - self.equilibrium_relative_x)
            if not np.isfinite(state).all() or not np.isfinite(tip):
                return 1.0e12, False
            state_values.append(state)
            tip_values.append(tip)
        states = np.asarray(state_values, dtype=float)
        tips = np.asarray(tip_values, dtype=float)
        return mppi_candidate_cost(states, tips, np.asarray(actual_deltas), self.q, self.r,
                                   self.tip_weight, self.terminal_multiplier), True

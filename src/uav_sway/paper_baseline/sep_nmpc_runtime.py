"""Same-process MuJoCo runtime for S5D2 SEP-NMPC development runs."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.acceleration_limiter import AccelerationLimiter
from uav_sway.control.base import ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.state_reader import StateReader
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.models.model_config import load_model_config
from uav_sway.scenarios.scenario_config import load_scenario_config

from .sep_nmpc_adapter import equivalent_cutter_com_and_suspension, equivalent_sway_angle, equivalent_sway_rate
from .sep_nmpc_controller import FormalSEPController
from .sep_nmpc_reference import load_reference, preview


def _id(model, object_type, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise KeyError(name)
    return value


def _rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    return (float(np.arctan2(rotation[2, 1], rotation[2, 2])), float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))), float(np.arctan2(rotation[1, 0], rotation[0, 0])))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no runtime rows")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("true" if value is True else "false" if value is False else value if isinstance(value, str) else format(float(value), ".17g")) for key, value in row.items()})


def run_sep_scene(model_config_path: str | Path, sep_config_path: str | Path, scenario: str, wind_path: str | Path, reference_path: str | Path, output_csv: str | Path, controller: FormalSEPController, duration_s: float = 12.0) -> dict:
    root = Path(model_config_path).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(str(root / "artifacts/s3/runtime/model_5link_controlled.xml"))
    model_cfg = load_model_config(model_config_path)
    scenario_cfg = load_scenario_config(root / "configs/scenarios.yaml")
    sep_cfg = yaml.safe_load(Path(sep_config_path).read_text(encoding="utf-8"))
    aero = load_aerodynamic_config(root / "configs/aerodynamics.yaml")
    reference = load_reference(reference_path)
    wind = read_wind_csv(wind_path)
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)
    quad_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    tip_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    total_mass = float(np.sum(model.body_mass))
    inner_cfg = yaml.safe_load((root / "configs/s3_pid.yaml").read_text(encoding="utf-8"))
    inner = GeometricInnerLoop(total_mass, np.asarray(model.body_inertia[quad_id], dtype=float), inner_cfg["attitude_natural_frequency_rad_s"], inner_cfg["attitude_damping_ratio"], *inner_cfg["position_gains_y"], *inner_cfg["position_gains_z"])
    reader = StateReader(model, model_cfg.n_links, float(data.site_xpos[tip_id, 0] - data.xpos[quad_id, 0]))
    limiter = AccelerationLimiter(float(sep_cfg["ax_min_m_s2"]), float(sep_cfg["ax_max_m_s2"]), float(sep_cfg["ax_slew_limit_m_s2_per_update"]))
    state = reader.read(model, data)
    cutter_com, suspension = equivalent_cutter_com_and_suspension(model, data, model_cfg)
    previous_alpha = equivalent_sway_angle(suspension, cutter_com)
    last_alpha_dot = 0.0
    controller.reset(np.zeros(4))
    actuator_ids = {name: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ("rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3", "thrust_motor", "mx_motor", "my_motor", "mz_motor")}
    physics_dt = float(model.opt.timestep)
    signal_steps = int(round(0.005 / physics_dt))
    outer_steps = int(round(0.05 / physics_dt))
    physics_steps = int(round(duration_s / physics_dt))
    rows: list[dict] = []
    previous_applied_ax = 0.0
    last_ax_raw = 0.0
    last_ax_limited = 0.0
    last_diag = controller.diagnostics
    last_inner = {"thrust_raw_N": total_mass * 9.81, "torque_raw_Nm": np.zeros(3)}
    last_limited = {"thrust": total_mass * 9.81, "torque": np.zeros(3)}
    for step in range(physics_steps + 1):
        index = min(step // signal_steps, len(wind["time"]) - 1)
        force = clear_and_apply_wind(model, data, model_cfg, aero, float(wind["wind_x"][index]))
        ref = ReferenceState(*(float(reference[name][index]) for name in ("x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")))
        if step % outer_steps == 0:
            current = reader.read(model, data)
            cutter_com, suspension = equivalent_cutter_com_and_suspension(model, data, model_cfg)
            alpha = equivalent_sway_angle(suspension, cutter_com)
            alpha_dot = equivalent_sway_rate(previous_alpha, alpha, 0.05)
            z = np.array([current.position[0] - ref.x_ref, current.velocity[0] - ref.vx_ref, alpha, alpha_dot], dtype=float)
            reference_window = preview(reference, index, controller.config.shooting_nodes)
            try:
                last_ax_raw = controller.command(z, reference_window, previous_applied_ax)
            except Exception:
                _write_rows(Path(output_csv), rows)
                raise
            last_diag = controller.diagnostics
            last_ax_limited = limiter.limit(last_ax_raw)
            previous_applied_ax = last_ax_limited
            previous_alpha = alpha
            last_alpha_dot = alpha_dot
        if step % signal_steps == 0:
            state = reader.read(model, data)
            last_inner = inner.compute(state, ref, last_ax_limited)
            thrust_raw = float(last_inner["thrust_raw_N"])
            torque_raw = np.asarray(last_inner["torque_raw_Nm"], dtype=float)
            thrust_lim = float(np.clip(thrust_raw, *model.actuator_ctrlrange[actuator_ids["thrust_motor"]]))
            torque_lim = np.asarray([np.clip(torque_raw[i], *model.actuator_ctrlrange[actuator_ids[name]]) for i, name in enumerate(("mx_motor", "my_motor", "mz_motor"))])
            data.ctrl[:] = 0.0
            data.ctrl[actuator_ids["thrust_motor"]] = thrust_lim
            for i, name in enumerate(("mx_motor", "my_motor", "mz_motor")):
                data.ctrl[actuator_ids[name]] = torque_lim[i]
            roll, pitch, yaw = _rpy(state.rotation)
            cutter_com, suspension = equivalent_cutter_com_and_suspension(model, data, model_cfg)
            alpha = equivalent_sway_angle(suspension, cutter_com)
            rows.append({
                "time": float(wind["time"][index]), "scenario": scenario, "candidate": controller.config.k_e,
                "K_e": controller.config.k_e, "rho": controller.config.rho, "epsilon": controller.config.epsilon,
                "x_ref": ref.x_ref, "vx_ref": ref.vx_ref, "ax_ref": ref.ax_ref,
                "y_ref": ref.y_ref, "z_ref": ref.z_ref, "yaw_ref": ref.yaw_ref,
                "uav_x": state.position[0], "uav_vx": state.velocity[0], "uav_y": state.position[1], "uav_z": state.position[2],
                "uav_vy": state.velocity[1], "uav_vz": state.velocity[2],
                "protocol_mode": "free_flight_controlled", "controller": "sep_nmpc_adapted", "controller_mode": "formal_acados",
                "tip_x": data.site_xpos[tip_id, 0], "tip_y": data.site_xpos[tip_id, 1], "tip_z": data.site_xpos[tip_id, 2], "tip_displacement": state.tip_displacement,
                "alpha_eq": alpha, "alpha_dot_eq": last_alpha_dot,
                "u_ae": last_diag.u_ae, "passivity_slack": last_diag.slack, "passivity_residual": last_diag.passivity_residual,
                "ax_cmd_raw": last_ax_raw, "ax_cmd_limited": last_ax_limited, "ax_slew_limited": bool(limiter.diagnostics.slew_limited), "ax_saturated": bool(limiter.diagnostics.saturated),
                "solve_time_ms": last_diag.solve_time_ms, "acados_status": last_diag.acados_status, "qp_iterations": last_diag.qp_iterations,
                "anchor_active": False, "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
                "thrust_cmd_raw_N": thrust_raw, "thrust_cmd_limited_N": thrust_lim, "mx_cmd_raw_Nm": torque_raw[0], "my_cmd_raw_Nm": torque_raw[1], "mz_cmd_raw_Nm": torque_raw[2],
                "mx_cmd_limited_Nm": torque_lim[0], "my_cmd_limited_Nm": torque_lim[1], "mz_cmd_limited_Nm": torque_lim[2], "rotor_motor_max_abs_cmd": 0.0,
                **{f"joint_{i}_angle": state.joint_angles[i - 1] for i in range(1, model_cfg.n_links + 1)},
                **{f"joint_{i}_velocity": state.joint_velocities[i - 1] for i in range(1, model_cfg.n_links + 1)},
                "wind_x": wind["wind_x"][index], "wind_force_total_x": force["total_x"],
            })
        if step < physics_steps:
            mujoco.mj_step(model, data)
    _write_rows(Path(output_csv), rows)
    del scenario_cfg
    return {"sample_count": len(rows), "duration_s": rows[-1]["time"] - rows[0]["time"], "solver_failure_count": int(sum(row["acados_status"] != 0 for row in rows))}

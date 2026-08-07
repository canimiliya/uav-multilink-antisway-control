"""Free-flight S5 runner using independent nonlinear MuJoCo MPPI rollouts."""

from __future__ import annotations

import csv
import json
import time as wall_time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.base import ReferenceState
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.mppi import MuJoCoMPPI
from uav_sway.control.state_reader import StateReader
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.mppi_schema import mppi_schema_columns
from uav_sway.models.model_config import load_model_config
from uav_sway.scenarios.scenario_config import load_scenario_config
from uav_sway.mppi.reference_horizon import make_reference_horizon


def _id(model, object_type, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise KeyError(name)
    return value


def _read_reference(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty reference")
    result = {key: np.asarray([float(row[key]) for row in rows], dtype=float)
              for key in ("time", "x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")}
    result["event"] = np.asarray([row["event"] for row in rows], dtype=object)
    return result


def _reference_at(ref: dict[str, np.ndarray], index: int) -> ReferenceState:
    return ReferenceState(*(float(ref[name][index]) for name in
                            ("x_ref", "vx_ref", "ax_ref", "y_ref", "z_ref", "yaw_ref")))


def _rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    return (float(np.arctan2(rotation[2, 1], rotation[2, 2])),
            float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))),
            float(np.arctan2(rotation[1, 0], rotation[0, 0])))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in columns:
                value = row[key]
                if isinstance(value, (bool, np.bool_)):
                    encoded[key] = "true" if value else "false"
                elif isinstance(value, str):
                    encoded[key] = value
                else:
                    encoded[key] = format(float(value), ".17g")
            writer.writerow(encoded)


def run_mppi_scenario(model_config_path: str | Path, mppi_config: dict,
                      scenario: str, disturbance_path: str | Path,
                      reference_csv: str | Path, output_csv: str | Path,
                      repo_root: str | Path | None = None, seed: int | None = None,
                      temperature: float | None = None,
                      noise_sigma: float | None = None,
                      duration_s: float = 12.0) -> dict:
    root = Path(repo_root or Path(model_config_path).resolve().parents[1])
    model = mujoco.MjModel.from_xml_path(str(root / "artifacts/s3/runtime/model_5link_controlled.xml"))
    model_cfg = load_model_config(model_config_path)
    s3 = yaml.safe_load((root / "configs/s3_pid.yaml").read_text(encoding="utf-8"))
    scenario_cfg = load_scenario_config(root / "configs/scenarios.yaml")
    aero = load_aerodynamic_config(root / "configs/aerodynamics.yaml")
    wind = read_wind_csv(disturbance_path)
    ref = _read_reference(reference_csv)
    if len(wind["time"]) != len(ref["time"]):
        raise ValueError("wind/reference sample counts differ")
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)
    quad_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    tip_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    equilibrium_relative_x = float(data.site_xpos[tip_id, 0] - data.xpos[quad_id, 0])
    reader = StateReader(model, model_cfg.n_links, equilibrium_relative_x)
    total_mass = float(np.sum(model.body_mass))
    inner = GeometricInnerLoop(total_mass, np.asarray(model.body_inertia[quad_id], dtype=float),
                               s3["attitude_natural_frequency_rad_s"], s3["attitude_damping_ratio"],
                               *s3["position_gains_y"], *s3["position_gains_z"])
    q = np.load(root / "artifacts/s4/lqr/Q.npy")
    r = np.load(root / "artifacts/s4/lqr/R.npy")
    mppi = MuJoCoMPPI(
        model, q, r, total_mass, np.asarray(model.body_inertia[quad_id], dtype=float),
        model_cfg.n_links, equilibrium_relative_x,
        float(mppi_config["temperature"] if temperature is None else temperature),
        float(mppi_config["noise_sigma"] if noise_sigma is None else noise_sigma),
        int(mppi_config["formal_seed"] if seed is None else seed),
        int(mppi_config["horizon_steps"]), int(mppi_config["num_rollouts"]),
        float(mppi_config["tip_displacement_weight"]), float(mppi_config["terminal_multiplier"]),
        float(mppi_config["ax_min_m_s2"]), float(mppi_config["ax_max_m_s2"]),
        float(mppi_config["ax_slew_limit_m_s2_per_update"]),
        model_config=model_cfg, aerodynamic_config=aero,
    )
    mppi.reset(0.0)
    actuator_ids = {name: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in (
        "rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3",
        "thrust_motor", "mx_motor", "my_motor", "mz_motor")}
    physics_dt = float(model.opt.timestep)
    inner_steps = int(round(float(mppi_config["inner_dt_s"]) / physics_dt))
    outer_steps = int(round(float(mppi_config["outer_dt_s"]) / physics_dt))
    physics_steps = int(round(duration_s / physics_dt))
    rows: list[dict] = []
    last_ax = 0.0
    last_solve_ms = 0.0
    last_inner = {"thrust_raw_N": total_mass * 9.81, "torque_raw_Nm": np.zeros(3)}
    last_limited = {"thrust": total_mass * 9.81, "torque": np.zeros(3)}
    last_force = {"quadrotor_x": 0.0, "cutter_x": 0.0, "total_x": 0.0,
                  **{f"link_{i}_x": 0.0 for i in range(1, model_cfg.n_links + 1)}}
    outer_calls = inner_calls = log_calls = wind_calls = 0
    for step in range(physics_steps + 1):
        signal_index = min(step // inner_steps, len(wind["time"]) - 1)
        last_force = clear_and_apply_wind(model, data, model_cfg, aero,
                                          float(wind["wind_x"][signal_index]))
        wind_calls += 1
        reference = _reference_at(ref, signal_index)
        if step % outer_steps == 0:
            horizon = make_reference_horizon(ref, signal_index, int(mppi_config["horizon_steps"]))
            started = wall_time.perf_counter_ns()
            last_ax = mppi.solve(data, horizon)
            last_solve_ms = (wall_time.perf_counter_ns() - started) / 1.0e6
            outer_calls += 1
        if step % inner_steps == 0:
            state = reader.read(model, data)
            last_inner = inner.compute(state, reference, last_ax)
            thrust_raw = float(last_inner["thrust_raw_N"])
            torque_raw = np.asarray(last_inner["torque_raw_Nm"], dtype=float)
            thrust_lim = float(np.clip(thrust_raw, *model.actuator_ctrlrange[actuator_ids["thrust_motor"]]))
            torque_lim = np.asarray([np.clip(torque_raw[i], *model.actuator_ctrlrange[actuator_ids[name]])
                                     for i, name in enumerate(("mx_motor", "my_motor", "mz_motor"))])
            last_limited = {"thrust": thrust_lim, "torque": torque_lim}
            data.ctrl[:] = 0.0
            data.ctrl[actuator_ids["thrust_motor"]] = thrust_lim
            for i, name in enumerate(("mx_motor", "my_motor", "mz_motor")):
                data.ctrl[actuator_ids[name]] = torque_lim[i]
            inner_calls += 1
            if step % inner_steps == 0:
                d = mppi.limiter.diagnostics
                roll, pitch, yaw = _rpy(np.asarray(data.xmat[quad_id]).reshape(3, 3))
                state = reader.read(model, data)
                md = mppi.diagnostics
                rows.append({
                    "time": float(wind["time"][signal_index]), "scenario": scenario,
                    "seed": -1 if wind["seed"] is None else int(wind["seed"]),
                    "protocol_mode": "free_flight_controlled", "wind_x": float(wind["wind_x"][signal_index]),
                    "wind_y": 0.0, "wind_z": 0.0, "x_ref": reference.x_ref,
                    "vx_ref": reference.vx_ref, "ax_ref": reference.ax_ref,
                    "y_ref": reference.y_ref, "z_ref": reference.z_ref, "yaw_ref": reference.yaw_ref,
                    "uav_x": state.position[0], "uav_y": state.position[1], "uav_z": state.position[2],
                    "uav_vx": state.velocity[0], "uav_vy": state.velocity[1], "uav_vz": state.velocity[2],
                    "uav_qw": data.xquat[quad_id, 0], "uav_qx": data.xquat[quad_id, 1],
                    "uav_qy": data.xquat[quad_id, 2], "uav_qz": data.xquat[quad_id, 3],
                    **{f"joint_{i}_angle": state.joint_angles[i - 1] for i in range(1, model_cfg.n_links + 1)},
                    **{f"joint_{i}_velocity": state.joint_velocities[i - 1] for i in range(1, model_cfg.n_links + 1)},
                    "tip_x": data.site_xpos[tip_id, 0], "tip_y": data.site_xpos[tip_id, 1],
                    "tip_z": data.site_xpos[tip_id, 2], "tip_relative_x": data.site_xpos[tip_id, 0] - state.position[0],
                    "tip_equilibrium_relative_x": equilibrium_relative_x, "tip_displacement": state.tip_displacement,
                    "wind_force_quad_x": last_force["quadrotor_x"],
                    **{f"wind_force_link_{i}_x": last_force[f"link_{i}_x"] for i in range(1, model_cfg.n_links + 1)},
                    "wind_force_cutter_x": last_force["cutter_x"], "wind_force_total_x": last_force["total_x"],
                    "ax_cmd_raw": d.raw, "ax_cmd_limited": d.limited, "ax_saturated": d.saturated,
                    "solve_time_ms": last_solve_ms, "controller": "mppi", "anchor_active": False,
                    "position_error_x": state.position[0] - reference.x_ref,
                    "velocity_error_x": state.velocity[0] - reference.vx_ref, "pid_integral_x": 0.0,
                    "ax_reference_feedforward": reference.ax_ref, "ax_pid_feedback": 0.0,
                    "ax_cmd_amplitude_limited": d.amplitude_limited, "ax_slew_limited": d.slew_limited,
                    "thrust_cmd_raw_N": thrust_raw, "thrust_cmd_limited_N": thrust_lim,
                    "mx_cmd_raw_Nm": torque_raw[0], "my_cmd_raw_Nm": torque_raw[1], "mz_cmd_raw_Nm": torque_raw[2],
                    "mx_cmd_limited_Nm": torque_lim[0], "my_cmd_limited_Nm": torque_lim[1], "mz_cmd_limited_Nm": torque_lim[2],
                    "inner_loop_saturated": bool(thrust_raw != thrust_lim or np.any(torque_raw != torque_lim)),
                    "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
                    "mppi_seed": int(mppi_config["formal_seed"] if seed is None else seed),
                    "mppi_horizon_steps": mppi.horizon_steps, "mppi_num_rollouts": mppi.num_rollouts,
                    "mppi_temperature": mppi.temperature, "mppi_noise_sigma": mppi.noise_sigma,
                    "mppi_nominal_first": md.nominal_first, "mppi_cost_min": md.cost_min,
                    "mppi_cost_mean": md.cost_mean, "mppi_cost_std": md.cost_std,
                    "mppi_weight_max": md.weight_max, "mppi_effective_sample_size": md.effective_sample_size,
                    "mppi_invalid_rollouts": md.invalid_rollouts,
                    "mppi_rollout_physics_steps": md.rollout_physics_steps,
                    "mppi_rollout_calls": md.rollout_calls,
                    "rotor_motor_max_abs_cmd": float(np.max(np.abs(data.ctrl[[actuator_ids[f"rotor_motor_{i}"] for i in range(4)]]))),
                })
                log_calls += 1
        if step < physics_steps:
            mujoco.mj_step(model, data)
    output_csv = Path(output_csv)
    _write_csv(output_csv, rows, mppi_schema_columns(model_cfg.n_links))
    metrics = compute_controlled_metrics(output_csv, float(mppi_config["settling_start_s"][scenario]))
    metrics.update({"physics_intervals": physics_steps, "formal_log_samples": log_calls,
                    "outer_control_updates": outer_calls, "inner_loop_updates": inner_calls,
                    "wind_force_calls": wind_calls, "anchor_active": False, "controller": "mppi",
                    "mppi_seed": int(mppi_config["formal_seed"] if seed is None else seed),
                    "mppi_temperature": mppi.temperature, "mppi_noise_sigma": mppi.noise_sigma,
                    "mppi_horizon_steps": mppi.horizon_steps, "mppi_num_rollouts": mppi.num_rollouts})
    return metrics

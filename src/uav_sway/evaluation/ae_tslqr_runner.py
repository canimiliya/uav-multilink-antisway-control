"""S6T5 closed-loop runner for Adaptive-Equilibrium Task-LQR."""

from __future__ import annotations

import json
import time as wall_time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.adaptive_equilibrium_task_lqr import AdaptiveEquilibriumTaskLQR
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.state_reader import StateReader
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.task_baseline_runner import _id, _rpy, _write_csv, make_task_output_map, read_reference, reference_at, sha256_file
from uav_sway.evaluation.task_space_metrics import compute_task_metrics
from uav_sway.linearization.reduced_state import ReducedStateLayout
from uav_sway.models.model_config import load_model_config
from uav_sway.control.task_lqr import build_task_lqr


SCENES = ("task_acquire_calm", "task_acquire_crosswind")


def run_scenario(candidate, scenario, wind_csv, reference_csv, output_csv, root, duration_s=12.0):
    root = Path(root)
    model, data, task_map, runtime_xml = make_task_output_map(root)
    s3 = yaml.safe_load((root / "configs/s3_pid.yaml").read_text(encoding="utf-8"))
    model_cfg = load_model_config(root / "configs/model_5link.yaml")
    aero = load_aerodynamic_config(root / "configs/aerodynamics.yaml")
    wind = read_wind_csv(wind_csv); ref = read_reference(reference_csv)
    if len(wind["time"]) != len(ref["time"]): raise ValueError("wind/reference sample counts differ")
    A = np.load(root / "artifacts/s4/linearization/A.npy")
    B = np.load(root / "artifacts/s4/linearization/B.npy")
    C = np.load(root / "artifacts/s6_taskspace/t1/task_lqr/C_task.npy")
    lqr = build_task_lqr(A, B, C, 80.0, 5.0, 1.0)
    controller = AdaptiveEquilibriumTaskLQR(lqr["K"], candidate["k_b"], candidate["tau_s"], 0.40, 0.10, 1.0, 0.05)
    quad_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    tip_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    equilibrium_relative_x = float(data.site_xpos[tip_id, 0] - data.xpos[quad_id, 0])
    reader = StateReader(model, model_cfg.n_links, equilibrium_relative_x); layout = ReducedStateLayout(model)
    total_mass = float(np.sum(model.body_mass))
    inner = GeometricInnerLoop(total_mass, np.asarray(model.body_inertia[quad_id], dtype=float), s3["attitude_natural_frequency_rad_s"], s3["attitude_damping_ratio"], *s3["position_gains_y"], *s3["position_gains_z"])
    controller.reset(reference=reference_at(ref, 0))
    actuator_ids = {name: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ("rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3", "thrust_motor", "mx_motor", "my_motor", "mz_motor")}
    physics_dt = float(model.opt.timestep); signal_steps = int(round(float(s3["wind_dt_s"]) / physics_dt)); outer_steps = int(round(float(s3["outer_dt_s"]) / physics_dt)); physics_steps = int(round(duration_s / physics_dt))
    rows = []; outer_calls = inner_calls = wind_calls = 0; last_solve = 0.0; force = {"quadrotor_x": 0.0, "cutter_x": 0.0, "total_x": 0.0, **{f"link_{i}_x": 0.0 for i in range(1, model_cfg.n_links + 1)}}
    for step in range(physics_steps + 1):
        index = min(step // signal_steps, len(wind["time"]) - 1); force = clear_and_apply_wind(model, data, model_cfg, aero, float(wind["wind_x"][index])); wind_calls += 1
        external = reference_at(ref, index)
        if step % outer_steps == 0:
            task_y = task_map.from_mujoco(data, external); measured_tip_error = float(task_y[0])
            internal = controller.internal_reference(external); state = layout.extract(model, data, internal)
            started = wall_time.perf_counter_ns(); controller.command(state, external, measured_tip_error, 0.05); last_solve = (wall_time.perf_counter_ns() - started) / 1.0e6; outer_calls += 1
        if step % signal_steps == 0:
            control_state = reader.read(model, data); task_state = task_map.from_mujoco(data, external); internal = controller.internal_reference(external); last_inner = inner.compute(control_state, internal, controller.diagnostics.ax_cmd_limited)
            thrust_raw = float(last_inner["thrust_raw_N"]); torque_raw = np.asarray(last_inner["torque_raw_Nm"], dtype=float); thrust_lim = float(np.clip(thrust_raw, *model.actuator_ctrlrange[actuator_ids["thrust_motor"]])); torque_lim = np.asarray([np.clip(torque_raw[i], *model.actuator_ctrlrange[actuator_ids[n]]) for i, n in enumerate(("mx_motor", "my_motor", "mz_motor"))]); data.ctrl[:] = 0.0; data.ctrl[actuator_ids["thrust_motor"]] = thrust_lim
            for i, n in enumerate(("mx_motor", "my_motor", "mz_motor")): data.ctrl[actuator_ids[n]] = torque_lim[i]
            tip_pos, tip_vel, axis, _ = task_map.kinematics_from_mujoco(data); task_ref = task_map.pose.tip_relative_position_m + np.asarray([external.x_ref, external.y_ref, external.z_ref]); position_error = tip_pos - task_ref; dot = float(np.clip(np.dot(axis, task_map.pose.cutter_axis_world), -1.0, 1.0)); roll, pitch, yaw = _rpy(control_state.rotation); d = controller.diagnostics
            rows.append({"time": float(wind["time"][index]), "scenario": scenario, "seed": -1 if wind["seed"] is None else int(wind["seed"]), "protocol_mode": "free_flight_controlled", "wind_x": float(wind["wind_x"][index]), "wind_y": 0.0, "wind_z": 0.0, "x_ref": external.x_ref, "vx_ref": external.vx_ref, "ax_ref": external.ax_ref, "y_ref": external.y_ref, "z_ref": external.z_ref, "yaw_ref": external.yaw_ref, "reference_event": str(ref["event"][index]), "uav_x": control_state.position[0], "uav_y": control_state.position[1], "uav_z": control_state.position[2], "uav_vx": control_state.velocity[0], "uav_vy": control_state.velocity[1], "uav_vz": control_state.velocity[2], "uav_qw": data.xquat[quad_id, 0], "uav_qx": data.xquat[quad_id, 1], "uav_qy": data.xquat[quad_id, 2], "uav_qz": data.xquat[quad_id, 3], **{f"joint_{i}_angle": control_state.joint_angles[i - 1] for i in range(1, model_cfg.n_links + 1)}, **{f"joint_{i}_velocity": control_state.joint_velocities[i - 1] for i in range(1, model_cfg.n_links + 1)}, "tip_x": tip_pos[0], "tip_y": tip_pos[1], "tip_z": tip_pos[2], "tip_relative_x": tip_pos[0] - control_state.position[0], "tip_equilibrium_relative_x": equilibrium_relative_x, "tip_displacement": control_state.tip_displacement, "wind_force_quad_x": force["quadrotor_x"], **{f"wind_force_link_{i}_x": force[f"link_{i}_x"] for i in range(1, model_cfg.n_links + 1)}, "wind_force_cutter_x": force["cutter_x"], "wind_force_total_x": force["total_x"], "ax_cmd_raw": d.ax_cmd_raw, "ax_cmd_limited": d.ax_cmd_limited, "ax_saturated": d.ax_saturated, "solve_time_ms": last_solve, "controller": "AE-TSLQR", "anchor_active": False, "position_error_x": state[0], "velocity_error_x": state[1], "pid_integral_x": 0.0, "ax_reference_feedforward": 0.0, "ax_pid_feedback": 0.0, "ax_cmd_amplitude_limited": d.ax_cmd_amplitude_limited, "ax_slew_limited": d.ax_slew_limited, "thrust_cmd_raw_N": thrust_raw, "thrust_cmd_limited_N": thrust_lim, "mx_cmd_raw_Nm": torque_raw[0], "my_cmd_raw_Nm": torque_raw[1], "mz_cmd_raw_Nm": torque_raw[2], "mx_cmd_limited_Nm": torque_lim[0], "my_cmd_limited_Nm": torque_lim[1], "mz_cmd_limited_Nm": torque_lim[2], "inner_loop_saturated": bool(thrust_raw != thrust_lim or np.any(torque_raw != torque_lim)), "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw, "task_position_error_x": task_state[0], "task_velocity_error_x": task_state[1], "theta_cutter_signed_rad": task_state[2], "omega_cutter_y": task_state[3], "task_pid_integral_x": 0.0, "task_lqr_feedback_ax": d.lqr_feedback_ax, "task_lqr_state_norm": float(np.linalg.norm(state)), "tip_velocity_world_x": tip_vel[0], "tip_velocity_world_y": tip_vel[1], "tip_velocity_world_z": tip_vel[2], "cutter_axis_world_x": axis[0], "cutter_axis_world_y": axis[1], "cutter_axis_world_z": axis[2], "task_reference_tip_x": task_ref[0], "task_reference_tip_y": task_ref[1], "task_reference_tip_z": task_ref[2], "task_reference_axis_x": task_map.pose.cutter_axis_world[0], "task_reference_axis_y": task_map.pose.cutter_axis_world[1], "task_reference_axis_z": task_map.pose.cutter_axis_world[2], "tip_task_x_error_m": position_error[0], "tip_task_z_error_m": position_error[2], "task_position_error_xz_m": np.linalg.norm(position_error[[0, 2]]), "position_error_3d_m": np.linalg.norm(position_error), "tip_speed_m_s": np.linalg.norm(tip_vel - np.asarray([external.vx_ref, 0.0, 0.0])), "orientation_error_rad": np.arccos(dot), "orientation_error_deg": np.rad2deg(np.arccos(dot)), "external_x_ref": external.x_ref, "internal_x_ref": d.internal_x_ref, "equilibrium_bias_x": d.equilibrium_bias_x, "filtered_tip_error_x": d.filtered_tip_error_x, "bias_rate": d.bias_rate, "adaptation_held": 1.0 if d.adaptation_held else 0.0})
            inner_calls += 1
        if step < physics_steps: mujoco.mj_step(model, data)
    path = Path(output_csv); _write_csv(path, rows, list(rows[0].keys())); metrics = compute_task_metrics(path); bias = np.asarray([float(r["equilibrium_bias_x"]) for r in rows]); settle = np.asarray([float(r["time"]) >= 8.0 for r in rows])
    metrics.update({"controller": "AE-TSLQR", "candidate_id": candidate["candidate_id"], "physics_intervals": physics_steps, "formal_log_samples": len(rows), "outer_control_updates": outer_calls, "inner_loop_updates": inner_calls, "wind_force_calls": wind_calls, "rotor_motor_max_abs_cmd": 0.0, "saturation_rate": float(np.mean([str(r["ax_saturated"]).lower() == "true" for r in rows])), "final_tip_speed_m_s": float(rows[-1]["tip_speed_m_s"]), "bias_final": float(bias[-1]), "bias_mean_after_settle": float(np.mean(bias[settle])), "bias_std_after_settle": float(np.std(bias[settle])), "max_abs_bias": float(np.max(np.abs(bias)),), "max_abs_bias_rate": float(np.max(np.abs([float(r["bias_rate"]) for r in rows]))), "runtime_model_sha256": sha256_file(runtime_xml), "reference_sha256": sha256_file(reference_csv), "wind_sha256": sha256_file(wind_csv), "A_sha256": sha256_file(root / "artifacts/s4/linearization/A.npy"), "B_sha256": sha256_file(root / "artifacts/s4/linearization/B.npy"), "C_task_sha256": sha256_file(root / "artifacts/s6_taskspace/t1/task_lqr/C_task.npy"), "scheduled_physics_hz": 1000, "scheduled_inner_hz": 200, "scheduled_outer_hz": 20, "future_target_preview": False, "provenance_complete": True})
    return metrics

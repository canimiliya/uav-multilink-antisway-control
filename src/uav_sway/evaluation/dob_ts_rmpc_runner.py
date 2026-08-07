"""S6T4 closed-loop development runner for DOB task-space residual MPC."""

from __future__ import annotations

import csv
import json
import time as wall_time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.base import ReferenceState
from uav_sway.control.dob_task_lqr import DOBTaskLQR
from uav_sway.control.dob_ts_rmpc import DOBTSRMPC
from uav_sway.control.geometric_inner_loop import GeometricInnerLoop
from uav_sway.control.runtime_model import create_runtime_model
from uav_sway.control.state_reader import StateReader
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.task_baseline_runner import (
    _id, _rpy, _write_csv, make_task_output_map, read_reference,
    reference_at, sha256_file,
)
from uav_sway.evaluation.task_space_metrics import compute_task_metrics
from uav_sway.linearization.reduced_state import ReducedStateLayout
from uav_sway.models.model_config import load_model_config
from uav_sway.mpc.dob_task_residual_qp import DOBTaskResidualQP
from uav_sway.mpc.integral_task_model import Q_S4
from uav_sway.control.task_lqr import build_task_lqr
from uav_sway.task_space.reference import task_reference_at
from uav_sway.linearization.task_output import TaskOutputMap


ROOT = Path(__file__).resolve().parents[3]
SCENES = ("task_acquire_calm", "task_acquire_crosswind")


def _make_controller(kind: str, candidate: dict, root: Path):
    A = np.load(root / "artifacts/s4/linearization/A.npy")
    B = np.load(root / "artifacts/s4/linearization/B.npy")
    C = np.load(root / "artifacts/s6_taskspace/t1/task_lqr/C_task.npy")
    lqr = build_task_lqr(A, B, C, 80.0, 5.0, 1.0)
    use_observer = kind != "TS-RMPC-noDOB"
    if kind == "DOB-Task-LQR":
        return DOBTaskLQR(lqr["K"], A, B, candidate["observer_gain"], 2.0,
                          use_observer=use_observer), lqr
    weights = np.diag([80.0 * candidate["alpha_p"],
                       20.0 * candidate["alpha_p"],
                       5.0 * candidate["alpha_theta"],
                       1.25 * candidate["alpha_theta"]])
    Q = 0.01 * Q_S4 + C.T @ weights @ C
    qp = DOBTaskResidualQP(A, B, lqr["K"], Q, lqr["P"],
                           candidate["horizon_steps"], candidate["residual_weight"])
    return DOBTSRMPC(lqr["K"], A, B, qp, candidate["observer_gain"], 2.0,
                     use_observer=use_observer), lqr


def run_scenario(kind: str, candidate: dict, scenario: str, wind_csv,
                 reference_csv, output_csv, root: str | Path = ROOT,
                 duration_s: float = 12.0) -> dict:
    root = Path(root)
    model, data, task_map, runtime_xml = make_task_output_map(root)
    s3 = yaml.safe_load((root / "configs/s3_pid.yaml").read_text(encoding="utf-8"))
    model_cfg = load_model_config(root / "configs/model_5link.yaml")
    aero = load_aerodynamic_config(root / "configs/aerodynamics.yaml")
    wind = read_wind_csv(wind_csv); ref = read_reference(reference_csv)
    if len(wind["time"]) != len(ref["time"]):
        raise ValueError("wind/reference sample counts differ")
    quad_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
    tip_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    equilibrium_relative_x = float(data.site_xpos[tip_id, 0] - data.xpos[quad_id, 0])
    reader = StateReader(model, model_cfg.n_links, equilibrium_relative_x)
    layout = ReducedStateLayout(model)
    total_mass = float(np.sum(model.body_mass))
    inner = GeometricInnerLoop(total_mass, np.asarray(model.body_inertia[quad_id], dtype=float),
                               s3["attitude_natural_frequency_rad_s"], s3["attitude_damping_ratio"],
                               *s3["position_gains_y"], *s3["position_gains_z"])
    controller, lqr = _make_controller(kind, candidate, root)
    controller.reset()
    actuator_ids = {name: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in
                    ("rotor_motor_0", "rotor_motor_1", "rotor_motor_2", "rotor_motor_3",
                     "thrust_motor", "mx_motor", "my_motor", "mz_motor")}
    physics_dt = float(model.opt.timestep)
    signal_steps = int(round(float(s3["wind_dt_s"]) / physics_dt))
    outer_steps = int(round(float(s3["outer_dt_s"]) / physics_dt))
    physics_steps = int(round(duration_s / physics_dt))
    rows = []; outer_calls = inner_calls = wind_calls = 0; qp_failures = 0
    last_solve_ms = 0.0
    force = {"quadrotor_x": 0.0, "cutter_x": 0.0, "total_x": 0.0,
             **{f"link_{i}_x": 0.0 for i in range(1, model_cfg.n_links + 1)}}
    for step in range(physics_steps + 1):
        index = min(step // signal_steps, len(wind["time"]) - 1)
        force = clear_and_apply_wind(model, data, model_cfg, aero, float(wind["wind_x"][index])); wind_calls += 1
        reference = reference_at(ref, index)
        if step % outer_steps == 0:
            state = layout.extract(model, data, reference)
            started = wall_time.perf_counter_ns()
            controller.command(state, reference, 0.05)
            last_solve_ms = (wall_time.perf_counter_ns() - started) / 1.0e6
            outer_calls += 1
            if getattr(controller.diagnostics, "qp_status_val", 0) not in (0, 1, 2):
                qp_failures += 1
        if step % signal_steps == 0:
            control_state = reader.read(model, data)
            task_state = task_map.from_mujoco(data, reference)
            task_ref = task_reference_at({k: float(ref[k][index]) for k in ("x_ref", "y_ref", "z_ref")}, task_map.pose)
            last_inner = inner.compute(control_state, reference, controller.diagnostics.ax_cmd_limited)
            thrust_raw = float(last_inner["thrust_raw_N"]); torque_raw = np.asarray(last_inner["torque_raw_Nm"], dtype=float)
            thrust_lim = float(np.clip(thrust_raw, *model.actuator_ctrlrange[actuator_ids["thrust_motor"]]))
            torque_lim = np.asarray([np.clip(torque_raw[i], *model.actuator_ctrlrange[actuator_ids[name]]) for i, name in enumerate(("mx_motor", "my_motor", "mz_motor"))])
            data.ctrl[:] = 0.0; data.ctrl[actuator_ids["thrust_motor"]] = thrust_lim
            for i, name in enumerate(("mx_motor", "my_motor", "mz_motor")): data.ctrl[actuator_ids[name]] = torque_lim[i]
            tip_position, tip_velocity, axis, angular_velocity = task_map.kinematics_from_mujoco(data)
            position_error = tip_position - task_ref.tip_position_world
            dot = float(np.clip(np.dot(axis, task_ref.cutter_axis_world), -1.0, 1.0))
            diag = controller.diagnostics
            row = {
                "time": float(wind["time"][index]), "scenario": scenario, "seed": -1 if wind["seed"] is None else int(wind["seed"]),
                "protocol_mode": "free_flight_controlled", "wind_x": float(wind["wind_x"][index]), "wind_y": 0.0, "wind_z": 0.0,
                "x_ref": reference.x_ref, "vx_ref": reference.vx_ref, "ax_ref": reference.ax_ref, "y_ref": reference.y_ref, "z_ref": reference.z_ref, "yaw_ref": reference.yaw_ref,
                "reference_event": str(ref["event"][index]), "uav_x": control_state.position[0], "uav_y": control_state.position[1], "uav_z": control_state.position[2],
                "uav_vx": control_state.velocity[0], "uav_vy": control_state.velocity[1], "uav_vz": control_state.velocity[2],
                "uav_qw": data.xquat[quad_id, 0], "uav_qx": data.xquat[quad_id, 1], "uav_qy": data.xquat[quad_id, 2], "uav_qz": data.xquat[quad_id, 3],
                **{f"joint_{i}_angle": control_state.joint_angles[i - 1] for i in range(1, model_cfg.n_links + 1)},
                **{f"joint_{i}_velocity": control_state.joint_velocities[i - 1] for i in range(1, model_cfg.n_links + 1)},
                "tip_x": tip_position[0], "tip_y": tip_position[1], "tip_z": tip_position[2], "tip_relative_x": tip_position[0] - control_state.position[0], "tip_equilibrium_relative_x": equilibrium_relative_x, "tip_displacement": control_state.tip_displacement,
                "wind_force_quad_x": force["quadrotor_x"], **{f"wind_force_link_{i}_x": force[f"link_{i}_x"] for i in range(1, model_cfg.n_links + 1)}, "wind_force_cutter_x": force["cutter_x"], "wind_force_total_x": force["total_x"],
                "ax_cmd_raw": diag.ax_cmd_raw, "ax_cmd_limited": diag.ax_cmd_limited, "ax_saturated": diag.ax_saturated, "solve_time_ms": last_solve_ms, "controller": kind, "anchor_active": False,
                "position_error_x": task_state[0], "velocity_error_x": task_state[1], "pid_integral_x": 0.0, "ax_reference_feedforward": 0.0, "ax_pid_feedback": 0.0, "ax_cmd_amplitude_limited": diag.ax_cmd_amplitude_limited, "ax_slew_limited": diag.ax_slew_limited,
                "thrust_cmd_raw_N": thrust_raw, "thrust_cmd_limited_N": thrust_lim, "mx_cmd_raw_Nm": torque_raw[0], "my_cmd_raw_Nm": torque_raw[1], "mz_cmd_raw_Nm": torque_raw[2], "mx_cmd_limited_Nm": torque_lim[0], "my_cmd_limited_Nm": torque_lim[1], "mz_cmd_limited_Nm": torque_lim[2], "inner_loop_saturated": bool(thrust_raw != thrust_lim or np.any(torque_raw != torque_lim)),
                "roll_rad": _rpy(control_state.rotation)[0], "pitch_rad": _rpy(control_state.rotation)[1], "yaw_rad": _rpy(control_state.rotation)[2],
                "task_position_error_x": task_state[0], "task_velocity_error_x": task_state[1], "theta_cutter_signed_rad": task_state[2], "omega_cutter_y": task_state[3], "task_pid_integral_x": 0.0, "task_lqr_feedback_ax": diag.lqr_feedback_ax, "task_lqr_state_norm": float(np.linalg.norm(state)),
                "tip_velocity_world_x": tip_velocity[0], "tip_velocity_world_y": tip_velocity[1], "tip_velocity_world_z": tip_velocity[2], "cutter_axis_world_x": axis[0], "cutter_axis_world_y": axis[1], "cutter_axis_world_z": axis[2],
                "task_reference_tip_x": task_ref.tip_position_world[0], "task_reference_tip_y": task_ref.tip_position_world[1], "task_reference_tip_z": task_ref.tip_position_world[2], "task_reference_axis_x": task_ref.cutter_axis_world[0], "task_reference_axis_y": task_ref.cutter_axis_world[1], "task_reference_axis_z": task_ref.cutter_axis_world[2],
                "tip_task_x_error_m": position_error[0], "tip_task_z_error_m": position_error[2], "task_position_error_xz_m": np.linalg.norm(position_error[[0, 2]]), "position_error_3d_m": np.linalg.norm(position_error), "tip_speed_m_s": np.linalg.norm(tip_velocity - np.asarray([reference.vx_ref, 0.0, 0.0])), "orientation_error_rad": np.arccos(dot), "orientation_error_deg": np.rad2deg(np.arccos(dot)),
                "disturbance_hat": diag.disturbance_hat, "residual_v": getattr(diag, "residual_v", 0.0), "qp_status_val": getattr(diag, "qp_status_val", 0), "qp_predicted_first_action": getattr(diag, "predicted_first_action", diag.ax_cmd_limited), "qp_first_action_mismatch": getattr(diag, "first_action_mismatch", 0.0), "qp_failures": qp_failures,
            }
            rows.append(row); inner_calls += 1
        if step < physics_steps: mujoco.mj_step(model, data)
    path = Path(output_csv); _write_csv(path, rows, list(rows[0].keys()))
    metrics = compute_task_metrics(path)
    solve = np.asarray([float(r["solve_time_ms"]) for r in rows]); d_hat = np.asarray([float(r["disturbance_hat"]) for r in rows])
    settle = np.asarray([float(r["time"]) >= 8.0 for r in rows])
    qp_rows = solve if kind != "DOB-Task-LQR" else np.zeros(1)
    metrics.update({
        "controller": kind, "candidate_id": candidate.get("candidate_id", kind), "physics_intervals": physics_steps, "formal_log_samples": len(rows), "outer_control_updates": outer_calls, "inner_loop_updates": inner_calls, "wind_force_calls": wind_calls, "qp_status_nonzero_count": qp_failures,
        "qp_first_action_mismatch_max": float(np.max(np.abs([float(r["qp_first_action_mismatch"]) for r in rows]))), "solve_time_max_ms": float(np.max(solve)), "solve_time_mean_ms": float(np.mean(solve)), "solve_time_p95_ms": float(np.percentile(solve, 95)), "qp_solve_time_mean_ms": float(np.mean(qp_rows)), "qp_solve_time_p95_ms": float(np.percentile(qp_rows, 95)), "qp_solve_time_max_ms": float(np.max(qp_rows)),
        "saturation_rate": float(np.mean([str(r["ax_saturated"]).lower() == "true" for r in rows])), "rotor_motor_max_abs_cmd": 0.0, "d_hat_final": float(d_hat[-1]), "d_hat_mean_after_settle": float(np.mean(d_hat[settle])), "d_hat_std_after_settle": float(np.std(d_hat[settle])), "reference_sha256": sha256_file(reference_csv), "wind_sha256": sha256_file(wind_csv), "runtime_model_sha256": sha256_file(runtime_xml), "A_sha256": sha256_file(root / "artifacts/s4/linearization/A.npy"), "B_sha256": sha256_file(root / "artifacts/s4/linearization/B.npy"), "C_task_sha256": sha256_file(root / "artifacts/s6_taskspace/t1/task_lqr/C_task.npy"), "scheduled_physics_hz": 1000, "scheduled_inner_hz": 200, "scheduled_outer_hz": 20, "future_target_preview": False, "provenance_complete": True,
    })
    metrics["final_tip_speed_m_s"] = float(rows[-1]["tip_speed_m_s"])
    return metrics

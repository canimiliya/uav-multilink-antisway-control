"""Safety, competence, scoring, and MPC-contribution gates for S6T3."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


SCENES = ("task_acquire_calm", "task_acquire_crosswind")


def _read_numeric(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); rows = list(reader); columns = list(reader.fieldnames or [])
    values = {}
    for key in columns:
        try:
            values[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
        except (KeyError, ValueError):
            continue
    return columns, values


def final_tip_speed_from_csv(path: str | Path) -> float:
    """Return the instantaneous tip speed at the final logged sample."""
    _columns, values = _read_numeric(path)
    if "tip_speed_m_s" not in values or values["tip_speed_m_s"].size == 0:
        raise ValueError(f"missing final tip speed in {path}")
    return float(values["tip_speed_m_s"][-1])


def legacy_mpc_contribution(its: dict[str, dict], lqi: dict[str, dict]) -> dict:
    """Reproduce the pre-audit null semantics for validity comparison only."""
    acquisition = [its[s]["task_acquisition_time_s"] for s in SCENES]
    lqi_acquisition = [lqi[s]["task_acquisition_time_s"] for s in SCENES]
    if any(value is None for value in acquisition + lqi_acquisition):
        return {"pass": False, "reason": "legacy null acquisition rejection"}
    pos_its = float(np.mean([its[s]["tip_task_position_rmse_m"] for s in SCENES]))
    pos_lqi = float(np.mean([lqi[s]["tip_task_position_rmse_m"] for s in SCENES]))
    acq_its = float(np.mean(acquisition)); acq_lqi = float(np.mean(lqi_acquisition))
    acquisition_improvement = (acq_lqi - acq_its) / max(acq_lqi, 1.0e-12)
    position_improvement = (pos_lqi - pos_its) / max(pos_lqi, 1.0e-12)
    no_degradation = all(its[s]["tip_task_position_rmse_m"] <= 1.10 * lqi[s]["tip_task_position_rmse_m"] and its[s]["cutter_orientation_rmse_deg"] <= 1.10 * lqi[s]["cutter_orientation_rmse_deg"] for s in SCENES)
    return {"pass": bool(no_degradation and (acquisition_improvement >= 0.05 or position_improvement >= 0.05)), "reason": "legacy comparable acquisition rule"}


def safety_audit(path: str | Path, metrics: dict, require_qp: bool) -> dict:
    columns, values = _read_numeric(path)
    joints = [key for key in values if key.startswith("joint_") and key.endswith("_angle")]
    checks = {
        "finite_outputs": bool(metrics.get("finite_outputs", False)),
        "anchor_false": not bool(np.any(values.get("anchor_active", np.asarray([0.0])))),
        "uav_height_safe": float(np.min(values["uav_z"])) > 0.05,
        "tip_height_safe": float(np.min(values["tip_z"])) > 0.05,
        "joint_safe": bool(max(np.max(np.abs(values[key])) for key in joints) < np.deg2rad(100.0)),
        "roll_safe": float(np.max(np.abs(values["roll_rad"]))) < np.deg2rad(25.0),
        "pitch_safe": float(np.max(np.abs(values["pitch_rad"]))) < np.deg2rad(25.0),
        "acceleration_safe": float(np.max(np.abs(values["ax_cmd_limited"]))) <= 2.0 + 1.0e-12,
        "slew_safe": float(np.max(np.abs(np.diff(values["ax_cmd_limited"])))) <= 0.25 + 1.0e-12,
        "thrust_safe": float(np.max(values["thrust_cmd_limited_N"])) <= 285.74568 + 1.0e-12 and float(np.min(values["thrust_cmd_limited_N"])) >= -1.0e-12,
        "torque_safe": max(float(np.max(np.abs(values[key]))) for key in ("mx_cmd_limited_Nm", "my_cmd_limited_Nm")) <= 25.0 + 1.0e-12 and float(np.max(np.abs(values["mz_cmd_limited_Nm"]))) <= 12.0 + 1.0e-12,
        "rotor_motors_zero": float(metrics.get("rotor_motor_max_abs_cmd", 0.0)) == 0.0,
        "sample_count_2401": len(values["time"]) == 2401,
        "physics_intervals_12000": int(metrics.get("physics_intervals", -1)) == 12000,
    }
    if require_qp:
        checks["qp_status_solved"] = int(metrics.get("qp_status_nonzero_count", 1)) == 0
        checks["first_action_matches_limiter"] = float(metrics.get("qp_first_action_mismatch_max", np.inf)) <= 1.0e-5
    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "pass": bool(all(checks.values())),
        "max_abs_ax": float(np.max(np.abs(values["ax_cmd_limited"]))),
        "max_ax_step": float(np.max(np.abs(np.diff(values["ax_cmd_limited"]))),),
        "required_columns_present": all(key in columns for key in ("time", "ax_cmd_limited", "tip_x", "tip_z")),
    }


def competence_gate(metrics: dict[str, dict], traditional: dict[str, dict]) -> dict:
    checks = {}
    for scene in SCENES:
        best_position = traditional[scene]["best_position_rmse"]
        best_orientation = traditional[scene]["best_orientation_rmse"]
        current = metrics[scene]
        checks[f"{scene}_acquired"] = bool(current["task_acquired"])
        checks[f"{scene}_position"] = current["tip_task_position_rmse_m"] <= 1.10 * best_position
        checks[f"{scene}_orientation"] = current["cutter_orientation_rmse_deg"] <= 1.25 * best_orientation
        checks[f"{scene}_final_position"] = current["final_tip_position_error_m"] <= 0.05
        checks[f"{scene}_final_orientation"] = current["final_orientation_error_deg"] <= 5.0
        checks[f"{scene}_tip_speed"] = current.get("final_tip_speed_m_s", np.inf) <= 0.10
    return {"checks": checks, "pass": bool(all(checks.values()))}


def candidate_score(metrics: dict[str, dict], traditional: dict[str, dict], old_lqr: dict[str, dict]) -> dict:
    positions = []; orientations = []; acquisitions = []; rates = []
    for scene in SCENES:
        m = metrics[scene]; ref = traditional[scene]
        positions.append(m["tip_task_position_rmse_m"] / ref["best_position_rmse"])
        orientations.append(m["cutter_orientation_rmse_deg"] / ref["best_orientation_rmse"])
        acquisitions.append(float(m["task_acquisition_time_s"]) / ref["available_task_time_s"])
        rates.append(m["control_rate_proxy"] / max(old_lqr[scene]["control_rate_proxy"], 1.0e-12))
    score = float(np.mean(positions) + 0.25 * np.mean(orientations) + 0.50 * np.mean(acquisitions) + 0.05 * np.mean(rates))
    return {"score": score, "position_ratio_mean": float(np.mean(positions)), "orientation_ratio_mean": float(np.mean(orientations)), "acquisition_norm_mean": float(np.mean(acquisitions)), "control_rate_ratio_mean": float(np.mean(rates))}


def mpc_contribution(its: dict[str, dict], lqi: dict[str, dict]) -> dict:
    scene_acquisition = {}
    comparable_its = []
    comparable_lqi = []
    dominance = []
    for scene in SCENES:
        its_acquired = bool(its[scene]["task_acquired"])
        lqi_acquired = bool(lqi[scene]["task_acquired"])
        if lqi_acquired and its_acquired:
            comparable_its.append(float(its[scene]["task_acquisition_time_s"]))
            comparable_lqi.append(float(lqi[scene]["task_acquisition_time_s"]))
            scene_acquisition[scene] = "both_acquired"
            dominance.append(None)
        elif not lqi_acquired and its_acquired:
            scene_acquisition[scene] = "its_acquisition_dominance"
            dominance.append(True)
        elif lqi_acquired and not its_acquired:
            scene_acquisition[scene] = "its_acquisition_loss"
            dominance.append(False)
        else:
            scene_acquisition[scene] = "both_not_acquired"
            dominance.append(None)

    pos_its = float(np.mean([its[s]["tip_task_position_rmse_m"] for s in SCENES]))
    pos_lqi = float(np.mean([lqi[s]["tip_task_position_rmse_m"] for s in SCENES]))
    if comparable_its:
        acq_its = float(np.mean(comparable_its)); acq_lqi = float(np.mean(comparable_lqi))
        acquisition_improvement = (acq_lqi - acq_its) / max(acq_lqi, 1.0e-12)
    else:
        acquisition_improvement = None
    position_improvement = (pos_lqi - pos_its) / max(pos_lqi, 1.0e-12)
    no_degradation = all(its[s]["tip_task_position_rmse_m"] <= 1.10 * lqi[s]["tip_task_position_rmse_m"] and its[s]["cutter_orientation_rmse_deg"] <= 1.10 * lqi[s]["cutter_orientation_rmse_deg"] for s in SCENES)
    acquisition_dominance = bool(any(value is True for value in dominance) and not any(value is False for value in dominance))
    acquisition_time_pass = bool(acquisition_improvement is not None and acquisition_improvement >= 0.05)
    position_pass = bool(position_improvement >= 0.05)
    pass_gate = bool(no_degradation and (acquisition_time_pass or position_pass or acquisition_dominance))
    return {
        "acquisition_improvement_vs_lqi": None if acquisition_improvement is None else float(acquisition_improvement),
        "position_improvement_vs_lqi": float(position_improvement),
        "no_more_than_10_percent_degradation": bool(no_degradation),
        "scene_acquisition_semantics": scene_acquisition,
        "acquisition_dominance": acquisition_dominance,
        "acquisition_time_comparison_available": bool(comparable_its),
        "acquisition_time_pass": acquisition_time_pass,
        "position_improvement_pass": position_pass,
        "pass": pass_gate,
        "reason": "acquisition dominance, comparable acquisition-time improvement, or position improvement",
    }

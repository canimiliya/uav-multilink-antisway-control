"""Raw CSV gates for the S5A development pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .controlled_metrics import compute_controlled_metrics, load_controlled_csv


def gate_scene(path, lqr_path, scene, config):
    _, values = load_controlled_csv(path)
    reasons=[]
    numeric=[v for v in values.values() if v.dtype != object and v.dtype != bool]
    if any(not np.isfinite(v).all() for v in numeric): reasons.append("finite")
    if bool(np.any(values["anchor_active"])): reasons.append("anchor_active")
    if float(np.min(values["uav_z"])) <= .05: reasons.append("uav_z")
    if float(np.min(values["tip_z"])) <= .05: reasons.append("tip_z")
    joint=[c for c in values if c.startswith("joint_") and c.endswith("_angle")]
    if joint and max(float(np.max(np.abs(values[c]))) for c in joint) >= np.deg2rad(100): reasons.append("joint_range")
    for c in ("roll_rad","pitch_rad"):
        if float(np.max(np.abs(values[c]))) >= np.deg2rad(25): reasons.append(c)
    if float(np.max(np.abs(values["ax_cmd_limited"]))) > 2+1e-12: reasons.append("ax_limit")
    if float(np.max(np.abs(np.diff(values["ax_cmd_limited"])))) > .25+1e-12: reasons.append("ax_slew")
    if float(np.min(values["thrust_cmd_limited_N"])) < -1e-12 or float(np.max(values["thrust_cmd_limited_N"])) > 285.74568+1e-12: reasons.append("thrust")
    for c,lim in (("mx_cmd_limited_Nm",25),("my_cmd_limited_Nm",25),("mz_cmd_limited_Nm",12)):
        if float(np.max(np.abs(values[c]))) > lim+1e-12: reasons.append(c)
    if float(np.max(np.abs(values["rotor_motor_max_abs_cmd"]))) != 0: reasons.append("rotor_motors")
    if "qp_limiter_mismatch" in values and float(np.max(values["qp_limiter_mismatch"])) >= 1e-4: reasons.append("qp_limiter_parity")
    metric=compute_controlled_metrics(path, float(config["settling_start_s"][scene]))
    if "disturbance_hat" in values:
        metric.update({
            "final_d_hat": float(values["disturbance_hat"][-1]),
            "max_abs_d_hat": float(np.max(np.abs(values["disturbance_hat"]))),
            "mean_abs_raw_ax": float(np.mean(np.abs(values["ax_cmd_raw"]))),
            "mean_abs_limited_ax": float(np.mean(np.abs(values["ax_cmd_limited"]))),
            "qp_limiter_mismatch_max": float(np.max(values.get("qp_limiter_mismatch", np.zeros(len(values["time"]))))),
        })
    lqr=compute_controlled_metrics(lqr_path, float(config["settling_start_s"][scene]))
    if scene == "approach_stop":
        if metric["x_position_rmse_m"] > 1.05*lqr["x_position_rmse_m"]: reasons.append("approach_position")
        if metric["tip_rms_m"] > .90*lqr["tip_rms_m"]: reasons.append("approach_tip")
    else:
        if not metric["x_position_rmse_m"] < lqr["x_position_rmse_m"]: reasons.append("crosswind_position")
        if metric["tip_rms_m"] > lqr["tip_rms_m"]: reasons.append("crosswind_tip")
    return {"pass": not reasons, "failure_reasons": reasons, "metric": metric, "lqr_metric": lqr}


def raw_da_pmpc_gate(paths: dict[str, Path], lqr_paths: dict[str, Path], config: dict) -> dict:
    scenes={s: gate_scene(paths[s], lqr_paths[s], s, config) for s in paths}
    return {"source":"independent_raw_csv_recomputation", "pass": bool(all(x["pass"] for x in scenes.values())), "scenarios": scenes}

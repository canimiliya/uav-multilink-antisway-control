"""S6T5 AE-TSLQR safety, competence, contribution and score gates."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from uav_sway.evaluation.its_rmpc_gate import safety_audit

SCENES = ("task_acquire_calm", "task_acquire_crosswind")


def safety_for_run(path, metrics):
    result = safety_audit(path, metrics, False)
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    bias = np.asarray([float(row["equilibrium_bias_x"]) for row in rows])
    rate = np.asarray([float(row["bias_rate"]) for row in rows])
    result["checks"]["bias_safe"] = bool(np.max(np.abs(bias)) <= 0.40 + 1e-12 and np.max(np.abs(rate)) <= 0.10 + 1e-12)
    result["pass"] = bool(result["pass"] and result["checks"]["bias_safe"])
    result["max_abs_bias"] = float(np.max(np.abs(bias))); result["max_abs_bias_rate"] = float(np.max(np.abs(rate)))
    return result


def competence_gate(metrics, traditional, safety):
    checks = {}
    for scene in SCENES:
        m = metrics[scene]; b = traditional[scene]
        checks[f"{scene}_acquired"] = bool(m["task_acquired"])
        checks[f"{scene}_position"] = m["tip_task_position_rmse_m"] <= 1.10 * b["best_position_rmse"]
        checks[f"{scene}_orientation"] = m["cutter_orientation_rmse_deg"] <= 1.25 * b["best_orientation_rmse"]
        checks[f"{scene}_final_position"] = m["final_tip_position_error_m"] <= 0.05
        checks[f"{scene}_final_orientation"] = m["final_orientation_error_deg"] <= 5.0
        checks[f"{scene}_final_tip_speed"] = m["final_tip_speed_m_s"] <= 0.10
        checks[f"{scene}_safety"] = bool(safety[scene]["pass"])
    return {"checks": checks, "pass": bool(all(checks.values()))}


def calm_preservation(metrics, task_lqr):
    m = metrics["task_acquire_calm"]; base = task_lqr["task_acquire_calm"]
    checks = {"acquired": bool(m["task_acquired"]), "acquisition_time": bool(m["task_acquisition_time_s"] is not None and m["task_acquisition_time_s"] <= 1.10 * base["task_acquisition_time_s"]), "position": bool(m["tip_task_position_rmse_m"] <= 1.10 * base["tip_task_position_rmse_m"]), "orientation": bool(m["cutter_orientation_rmse_deg"] <= 1.10 * base["cutter_orientation_rmse_deg"])}
    return {"checks": checks, "pass": bool(all(checks.values()))}


def adaptation_contribution(metrics, task_lqr):
    calm = calm_preservation(metrics, task_lqr)
    cross = metrics["task_acquire_crosswind"]; base = task_lqr["task_acquire_crosswind"]
    dominance = bool(not base["task_acquired"] and cross["task_acquired"])
    improvement = float((base["tip_task_position_rmse_m"] - cross["tip_task_position_rmse_m"]) / max(base["tip_task_position_rmse_m"], 1e-12))
    passed = bool(calm["pass"] and (dominance or improvement >= 0.10))
    return {"calm_preservation": calm, "crosswind_acquisition_dominance": dominance, "crosswind_position_improvement": improvement, "pass": passed}


def selection_score(metrics, traditional, old_lqr):
    positions = [metrics[s]["tip_task_position_rmse_m"] / traditional[s]["best_position_rmse"] for s in SCENES]
    orientations = [metrics[s]["cutter_orientation_rmse_deg"] / traditional[s]["best_orientation_rmse"] for s in SCENES]
    acquisitions = [metrics[s]["task_acquisition_time_s"] / traditional[s]["available_task_time_s"] for s in SCENES]
    rates = [metrics[s]["control_rate_proxy"] / max(old_lqr[s]["control_rate_proxy"], 1e-12) for s in SCENES]
    score = float(np.mean(positions) + 0.25 * np.mean(orientations) + 0.50 * np.mean(acquisitions) + 0.05 * np.mean(rates))
    return {"score": score, "position_ratio_mean": float(np.mean(positions)), "orientation_ratio_mean": float(np.mean(orientations)), "acquisition_norm_mean": float(np.mean(acquisitions)), "control_rate_ratio_mean": float(np.mean(rates))}


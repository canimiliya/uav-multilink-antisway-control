"""Gates and frozen scoring rules for S6T4."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from uav_sway.evaluation.its_rmpc_gate import safety_audit

SCENES = ("task_acquire_calm", "task_acquire_crosswind")


def observer_contribution(dob, task_lqr, old_lqr, traditional) -> dict:
    calm_no_deg = (dob["task_acquire_calm"]["tip_task_position_rmse_m"] <= 1.10 * task_lqr["task_acquire_calm"]["tip_task_position_rmse_m"] and
                   dob["task_acquire_calm"]["cutter_orientation_rmse_deg"] <= 1.10 * task_lqr["task_acquire_calm"]["cutter_orientation_rmse_deg"])
    pos_improvement = (old_lqr["task_acquire_crosswind"]["tip_task_position_rmse_m"] - dob["task_acquire_crosswind"]["tip_task_position_rmse_m"]) / max(old_lqr["task_acquire_crosswind"]["tip_task_position_rmse_m"], 1e-12)
    final_improvement = (old_lqr["task_acquire_crosswind"]["final_tip_position_error_m"] - dob["task_acquire_crosswind"]["final_tip_position_error_m"]) / max(old_lqr["task_acquire_crosswind"]["final_tip_position_error_m"], 1e-12)
    dominance = (not task_lqr["task_acquire_crosswind"]["task_acquired"] and dob["task_acquire_crosswind"]["task_acquired"])
    passed = bool(calm_no_deg and (pos_improvement >= 0.05 or final_improvement >= 0.20 or dominance))
    return {"calm_no_degradation": bool(calm_no_deg), "crosswind_position_improvement": float(pos_improvement), "crosswind_final_tip_improvement": float(final_improvement), "crosswind_acquisition_dominance": bool(dominance), "pass": passed}


def competence_gate(metrics, traditional, safety_by_scene, solve_p95_limit=50.0) -> dict:
    checks = {}
    for scene in SCENES:
        current = metrics[scene]; base = traditional[scene]
        checks[f"{scene}_acquired"] = bool(current["task_acquired"])
        checks[f"{scene}_position"] = current["tip_task_position_rmse_m"] <= 1.10 * base["best_position_rmse"]
        checks[f"{scene}_orientation"] = current["cutter_orientation_rmse_deg"] <= 1.25 * base["best_orientation_rmse"]
        checks[f"{scene}_final_position"] = current["final_tip_position_error_m"] <= 0.05
        checks[f"{scene}_final_orientation"] = current["final_orientation_error_deg"] <= 5.0
        checks[f"{scene}_final_tip_speed"] = float(current.get("final_tip_speed_m_s", _final_speed(metrics[scene].get("source_csv")))) <= 0.10
        checks[f"{scene}_safety"] = bool(safety_by_scene[scene]["pass"])
        checks[f"{scene}_solve_p95"] = float(current.get("solve_time_p95_ms", np.inf)) < solve_p95_limit
    return {"checks": checks, "pass": bool(all(checks.values()))}


def _final_speed(path):
    if not path:
        return np.inf
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return float(rows[-1]["tip_speed_m_s"]) if rows else np.inf


def mpc_contribution(full, dob_lqi) -> dict:
    no_deg = all(full[s]["tip_task_position_rmse_m"] <= 1.10 * dob_lqi[s]["tip_task_position_rmse_m"] and full[s]["cutter_orientation_rmse_deg"] <= 1.10 * dob_lqi[s]["cutter_orientation_rmse_deg"] for s in SCENES)
    comparable = [s for s in SCENES if full[s]["task_acquired"] and dob_lqi[s]["task_acquired"]]
    if comparable:
        acq_full = np.mean([full[s]["task_acquisition_time_s"] for s in comparable]); acq_dob = np.mean([dob_lqi[s]["task_acquisition_time_s"] for s in comparable])
        acq_improvement = float((acq_dob - acq_full) / max(acq_dob, 1e-12))
    else:
        acq_improvement = None
    position_improvement = float((np.mean([dob_lqi[s]["tip_task_position_rmse_m"] for s in SCENES]) - np.mean([full[s]["tip_task_position_rmse_m"] for s in SCENES])) / max(np.mean([dob_lqi[s]["tip_task_position_rmse_m"] for s in SCENES]), 1e-12))
    dominance = any(full[s]["task_acquired"] and not dob_lqi[s]["task_acquired"] for s in SCENES) and not any(dob_lqi[s]["task_acquired"] and not full[s]["task_acquired"] for s in SCENES)
    passed = bool(no_deg and ((acq_improvement is not None and acq_improvement >= 0.05) or position_improvement >= 0.05 or dominance))
    return {"no_more_than_10_percent_degradation": bool(no_deg), "acquisition_improvement": acq_improvement, "position_improvement": position_improvement, "acquisition_dominance": bool(dominance), "pass": passed}


def safety_for_run(path, metrics, require_qp):
    return safety_audit(path, metrics, require_qp)


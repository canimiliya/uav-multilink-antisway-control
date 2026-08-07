"""Frozen S6T1 development scoring and competence gates."""

from __future__ import annotations

import numpy as np


def task_baseline_score(position_ratios, orientation_ratios, acquisition_norms,
                        control_rate_ratios) -> float:
    arrays = [np.asarray(v, dtype=float) for v in (position_ratios, orientation_ratios, acquisition_norms, control_rate_ratios)]
    if any(v.size == 0 or not np.isfinite(v).all() for v in arrays):
        raise ValueError("task baseline score inputs must be non-empty and finite")
    return float(np.mean(arrays[0]) + 0.25 * np.mean(arrays[1]) + 0.25 * np.mean(arrays[2]) + 0.05 * np.mean(arrays[3]))


def competence_gate(candidate, old, position_limit=1.10, orientation_limit=1.25) -> dict:
    checks = {
        "position_approach": float(candidate["approach_stop"]["tip_task_position_rmse_m"]) <= position_limit * float(old["approach_stop"]["tip_task_position_rmse_m"]),
        "position_crosswind": float(candidate["crosswind_hover"]["tip_task_position_rmse_m"]) <= position_limit * float(old["crosswind_hover"]["tip_task_position_rmse_m"]),
        "orientation_approach": float(candidate["approach_stop"]["cutter_orientation_rmse_deg"]) <= orientation_limit * float(old["approach_stop"]["cutter_orientation_rmse_deg"]),
        "orientation_crosswind": float(candidate["crosswind_hover"]["cutter_orientation_rmse_deg"]) <= orientation_limit * float(old["crosswind_hover"]["cutter_orientation_rmse_deg"]),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}

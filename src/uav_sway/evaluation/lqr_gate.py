"""Independent raw CSV gate and PID comparison for S4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .controlled_metrics import compute_controlled_metrics, load_controlled_csv


THRESHOLDS = {"approach_stop": (0.25, 0.45, 0.15, 3.0), "crosswind_hover": (0.30, 0.35, 0.15, 0.0), "gust_micro_adjust": (0.20, 0.30, 0.15, 0.30)}


def raw_lqr_gate(run_paths: dict[str, Path], pid_paths: dict[str, Path], include_global: bool = True) -> dict:
    results = {}; all_pass = True; residual = False
    for scenario, path in run_paths.items():
        _, v = load_controlled_csv(path); final_target, max_rmse, max_z, _ = THRESHOLDS[scenario]
        joints = [c for c in v if c.startswith("joint_") and c.endswith("_angle")]
        max_joint = float(max(np.max(np.abs(v[c])) for c in joints))
        residual = residual or float(np.max(np.abs(v["tip_displacement"]))) > 0.02 or max_joint > 0.005
        checks = {
            "finite": bool(all(np.isfinite(a).all() for a in v.values() if a.dtype != object and a.dtype != bool)),
            "anchor_inactive": not bool(np.any(v["anchor_active"])), "minimum_uav_height": float(np.min(v["uav_z"])) > 0.05, "minimum_tip_height": float(np.min(v["tip_z"])) > 0.05,
            "joint_range": max_joint < np.deg2rad(100.0), "roll_limit": float(np.max(np.abs(v["roll_rad"]))) < np.deg2rad(25.0), "pitch_limit": float(np.max(np.abs(v["pitch_rad"]))) < np.deg2rad(25.0),
            "ax_limit": float(np.max(np.abs(v["ax_cmd_limited"]))) <= 2.0 + 1e-12, "ax_slew_limit": float(np.max(np.abs(np.diff(v["ax_cmd_limited"])))) <= 0.25 + 1e-12,
            "thrust_limit": bool(np.all(v["thrust_cmd_limited_N"] >= -1e-12) and np.all(v["thrust_cmd_limited_N"] <= 285.74568 + 1e-12)), "torque_limit": bool(np.all(np.abs(v["mx_cmd_limited_Nm"]) <= 25.0 + 1e-12) and np.all(np.abs(v["my_cmd_limited_Nm"]) <= 25.0 + 1e-12) and np.all(np.abs(v["mz_cmd_limited_Nm"]) <= 12.0 + 1e-12)),
            "final_x_error": abs(float(v["uav_x"][-1] - v["x_ref"][-1])) <= final_target, "x_rmse": compute_controlled_metrics(path)["x_position_rmse_m"] <= max_rmse, "z_rmse": compute_controlled_metrics(path)["z_position_rmse_m"] <= max_z,
        }
        pid = compute_controlled_metrics(pid_paths[scenario]); lqr = compute_controlled_metrics(path)
        checks["position_fairness"] = lqr["x_position_rmse_m"] <= 1.10 * pid["x_position_rmse_m"]
        if scenario == "approach_stop": checks["approach_tip_improvement"] = lqr["tip_rms_m"] <= 0.95 * pid["tip_rms_m"]
        checks = {k: bool(x) for k, x in checks.items()}; results[scenario] = {"checks": checks, "pass": bool(all(checks.values())), "metrics": lqr, "pid_metrics": pid}; all_pass = all_pass and results[scenario]["pass"]
    gate = {"source": "independent_raw_csv_recomputation", "residual_sway_confirmed": bool(residual), "scenarios": results}
    if include_global:
        root = Path(__file__).resolve().parents[3]
        local_path = root / "artifacts/s4/linearization/local_validation.json"
        operating_path = root / "artifacts/s4/linearization/operating_region_validation.json"
        score_path = root / "artifacts/s4/repair/scoring_formula_audit.json"
        grid_path = root / "artifacts/s4/tuning/lqr_grid_repair_64.csv"
        selection_path = root / "artifacts/s4/tuning/lqr_selection.json"
        local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
        score_audit = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        grid_rows = sum(1 for _ in grid_path.open("r", encoding="utf-8")) - 1 if grid_path.exists() else 0
        selected = selection.get("selected")
        gate["local_linearization_pass"] = bool(local.get("pass", False) and local.get("local_validation_reference") == "10x_epsilon")
        gate["operating_region_validation_reported"] = bool(operating_path.exists())
        gate["scoring_formula_correct"] = bool(score_audit.get("pass", False))
        gate["grid_size"] = int(grid_rows)
        gate["selected_candidate_is_safe"] = bool(selection.get("selection_status") == "SELECTED_SAFE_CANDIDATE" and isinstance(selected, dict) and selected.get("safe_gate", False))
        gate["pass"] = bool(all_pass and residual and gate["local_linearization_pass"] and gate["operating_region_validation_reported"] and gate["scoring_formula_correct"] and grid_rows == 64 and gate["selected_candidate_is_safe"])
    else:
        gate["pass"] = bool(all_pass)
    return gate

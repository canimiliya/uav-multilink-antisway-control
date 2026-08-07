"""Independent raw CSV gate for S5 MPPI evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .controlled_metrics import compute_controlled_metrics, load_controlled_csv
from uav_sway.mppi.cost import mppi_candidate_score


def candidate_gate_reasons(path, baseline_path, config: dict, scene: str) -> list[str]:
    """Return raw-CSV safety and development-fairness failures for one candidate."""
    columns, values = load_controlled_csv(path)
    reasons: list[str] = []
    numeric = [v for v in values.values() if v.dtype != object and v.dtype != bool]
    if any(not np.isfinite(v).all() for v in numeric):
        reasons.append("finite")
    if "anchor_active" not in values or bool(np.any(values["anchor_active"])):
        reasons.append("anchor_active")
    if float(np.min(values["uav_z"])) <= 0.05:
        reasons.append("uav_z")
    if float(np.min(values["tip_z"])) <= 0.05:
        reasons.append("tip_z")
    joint_columns = sorted((c for c in columns if c.startswith("joint_") and c.endswith("_angle")),
                           key=lambda c: int(c.split("_")[1]))
    if joint_columns and max(float(np.max(np.abs(values[c]))) for c in joint_columns) >= np.deg2rad(100.0):
        reasons.append("joint_range")
    for name, limit in (("roll_rad", np.deg2rad(25.0)), ("pitch_rad", np.deg2rad(25.0))):
        if float(np.max(np.abs(values[name]))) >= limit:
            reasons.append(name)
    if float(np.max(np.abs(values["ax_cmd_limited"]))) > 2.0 + 1e-12:
        reasons.append("ax_limit")
    if float(np.max(np.abs(np.diff(values["ax_cmd_limited"])))) > 0.25 + 1e-12:
        reasons.append("ax_slew")
    if float(np.min(values["thrust_cmd_limited_N"])) < -1e-12 or float(np.max(values["thrust_cmd_limited_N"])) > 285.74568 + 1e-12:
        reasons.append("thrust")
    for name, limit in (("mx_cmd_limited_Nm", 25.0), ("my_cmd_limited_Nm", 25.0), ("mz_cmd_limited_Nm", 12.0)):
        if float(np.max(np.abs(values[name]))) > limit + 1e-12:
            reasons.append(name.replace("_cmd_limited_Nm", "torque"))
    if "rotor_motor_max_abs_cmd" not in values or float(np.max(np.abs(values["rotor_motor_max_abs_cmd"]))) != 0.0:
        reasons.append("rotor_motors")
    if str(values["controller"][0]) != "mppi":
        reasons.append("controller")
    baseline = compute_controlled_metrics(baseline_path, float(config["settling_start_s"][scene]))
    metric = compute_controlled_metrics(path, float(config["settling_start_s"][scene]))
    if metric["x_position_rmse_m"] > 1.10 * baseline["x_position_rmse_m"]:
        reasons.append("position_fairness")
    tip_limit = 0.95 if scene == "approach_stop" else 1.10
    if metric["tip_rms_m"] > tip_limit * baseline["tip_rms_m"]:
        reasons.append("tip_fairness")
    return reasons


def raw_mppi_gate(mppi_paths: dict[str, str | Path], lqr_paths: dict[str, str | Path],
                  config: dict, pid_paths: dict[str, str | Path] | None = None) -> dict:
    scenarios = {}
    all_safe = True
    for scene, path in mppi_paths.items():
        columns, values = load_controlled_csv(path)
        lqr = compute_controlled_metrics(lqr_paths[scene], float(config["settling_start_s"][scene]))
        mppi = compute_controlled_metrics(path, float(config["settling_start_s"][scene]))
        finite = all(np.isfinite(v).all() for v in values.values() if v.dtype != object and v.dtype != bool)
        reasons = candidate_gate_reasons(path, lqr_paths[scene], config, scene)
        safe = not reasons
        ratio = float(mppi["x_position_rmse_m"] / lqr["x_position_rmse_m"])
        tip_ratio = float(mppi["tip_rms_m"] / lqr["tip_rms_m"])
        scenarios[scene] = {"finite": finite, "safe": safe, "failure_reasons": reasons, "mppi": mppi,
                            "lqr": lqr, "position_ratio": ratio,
                            "tip_ratio": tip_ratio,
                            "tip_improvement_percent": 100.0 * (lqr["tip_rms_m"] - mppi["tip_rms_m"]) / lqr["tip_rms_m"]}
        if pid_paths is not None:
            scenarios[scene]["pid"] = compute_controlled_metrics(pid_paths[scene], float(config["settling_start_s"][scene]))
        all_safe = all_safe and safe
    position_fair = all(v["position_ratio"] <= 1.10 for v in scenarios.values())
    tip_gate = (scenarios["approach_stop"]["tip_ratio"] <= 0.95
                and scenarios["crosswind_hover"]["tip_ratio"] <= 1.10
                and scenarios["gust_micro_adjust"]["tip_ratio"] <= 1.10)
    not_worse = sum(v["tip_ratio"] <= 1.0 for v in scenarios.values()) >= 2
    score = mppi_candidate_score([v["tip_ratio"] for v in scenarios.values()],
                                 [v["position_ratio"] for v in scenarios.values()],
                                 [v["mppi"]["control_rate_proxy"] / max(v["lqr"]["control_rate_proxy"], 1e-9) for v in scenarios.values()],
                                 [v["mppi"]["saturation_rate"] for v in scenarios.values()])
    return {"source": "independent_raw_csv_recomputation", "pass": bool(all_safe and position_fair and tip_gate and not_worse),
            "scenarios": scenarios, "position_rmse_within_lqr_110_percent": position_fair,
            "tip_requirements": tip_gate, "at_least_two_scenarios_not_worse_than_lqr": not_worse,
            "candidate_score": score, "settling_start_s": config["settling_start_s"]}

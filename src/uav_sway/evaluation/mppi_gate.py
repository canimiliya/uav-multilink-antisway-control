"""Independent raw CSV gate for S5 MPPI evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .controlled_metrics import compute_controlled_metrics, load_controlled_csv
from uav_sway.mppi.cost import mppi_candidate_score


def raw_mppi_gate(mppi_paths: dict[str, str | Path], lqr_paths: dict[str, str | Path],
                  config: dict, pid_paths: dict[str, str | Path] | None = None) -> dict:
    scenarios = {}
    all_safe = True
    for scene, path in mppi_paths.items():
        columns, values = load_controlled_csv(path)
        lqr = compute_controlled_metrics(lqr_paths[scene], float(config["settling_start_s"][scene]))
        mppi = compute_controlled_metrics(path, float(config["settling_start_s"][scene]))
        finite = all(np.isfinite(v).all() for v in values.values() if v.dtype != object and v.dtype != bool)
        safe = (finite and not bool(np.any(values["anchor_active"])) and float(np.min(values["tip_z"])) > 0.05
                and float(np.min(values["uav_z"])) > 0.05
                and float(np.max(np.abs(values["ax_cmd_limited"]))) <= 2.0 + 1e-12
                and float(np.max(np.abs(np.diff(values["ax_cmd_limited"])))) <= 0.25 + 1e-12
                and float(np.max(np.abs(values["pitch_rad"]))) < np.deg2rad(25.0)
                and float(np.max(np.abs(values["roll_rad"]))) < np.deg2rad(25.0)
                and bool(np.all(values["controller"] == "mppi")))
        ratio = float(mppi["x_position_rmse_m"] / lqr["x_position_rmse_m"])
        tip_ratio = float(mppi["tip_rms_m"] / lqr["tip_rms_m"])
        scenarios[scene] = {"finite": finite, "safe": safe, "mppi": mppi,
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

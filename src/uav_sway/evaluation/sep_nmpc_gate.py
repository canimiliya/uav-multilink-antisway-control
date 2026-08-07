"""Independent development gates and frozen candidate score for SEP-NMPC."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics, load_controlled_csv
from uav_sway.evaluation.s5b_holdout import safety_check


def load_metrics(path: str | Path, settling_start_s: float) -> dict:
    metrics = compute_controlled_metrics(path, settling_start_s)
    _, values = load_controlled_csv(path)
    metrics.update({
        "thrust_min_N": float(np.min(values["thrust_cmd_limited_N"])),
        "thrust_max_N": float(np.max(values["thrust_cmd_limited_N"])),
        "torque_max_abs_Nm": float(np.max(np.abs(np.column_stack([
            values["mx_cmd_limited_Nm"], values["my_cmd_limited_Nm"], values["mz_cmd_limited_Nm"]
        ])))),
        "rotor_motor_max_abs_cmd": float(np.max(np.abs(values["rotor_motor_max_abs_cmd"]))),
        "solver_failure_count": int(np.count_nonzero(values["acados_status"] != 0)),
        "first_action_limiter_mismatch_max": float(np.max(np.abs(values["ax_cmd_raw"] - values["ax_cmd_limited"]))),
    })
    safe, reasons = safety_check(metrics)
    if metrics["solver_failure_count"] != 0:
        reasons.append("acados_solver_failure")
    if metrics["first_action_limiter_mismatch_max"] > 1e-5:
        reasons.append("first_action_limiter_mismatch")
    metrics["safe"] = bool(safe and not reasons)
    metrics["safety_reasons"] = reasons
    return metrics


def passivity_summary(run_path: str | Path, prediction_slacks: list[float], prediction_residuals: list[float], slack_max: float = 5.0) -> dict:
    _, values = load_controlled_csv(run_path)
    slacks = np.asarray(prediction_slacks, dtype=float)
    residuals = np.asarray(prediction_residuals, dtype=float)
    if slacks.size == 0:
        slacks = np.asarray(values["passivity_slack"], dtype=float)
    if residuals.size == 0:
        residuals = np.asarray(values["passivity_residual"], dtype=float)
    return {
        "passivity_residual_max": float(np.max(residuals)),
        "slack_max": float(np.max(slacks)),
        "slack_mean": float(np.mean(slacks)),
        "slack_rms": float(np.sqrt(np.mean(slacks**2))),
        "slack_saturation_rate": float(np.mean(slacks >= slack_max - 1e-8)),
        "prediction_node_count": int(slacks.size),
    }


def candidate_score(scene_metrics: dict[str, dict], lqr_metrics: dict[str, dict], passivity: dict) -> float:
    tip_ratios = [scene_metrics[s]["tip_rms_m"] / lqr_metrics[s]["tip_rms_m"] for s in scene_metrics]
    x_ratios = [scene_metrics[s]["x_position_rmse_m"] / lqr_metrics[s]["x_position_rmse_m"] for s in scene_metrics]
    return float(np.mean(tip_ratios) - 0.5 * np.mean(x_ratios) - 0.1 * (passivity["slack_rms"] / 5.0) - 0.05 * np.mean([scene_metrics[s]["saturation_rate"] for s in scene_metrics]))


def evaluate_candidate(scene_metrics: dict[str, dict], lqr_metrics: dict[str, dict], passivity: dict) -> dict:
    safe = all(scene_metrics[s]["safe"] for s in scene_metrics)
    usable = all(
        scene_metrics[s]["x_position_rmse_m"] <= 1.5 * lqr_metrics[s]["x_position_rmse_m"]
        and scene_metrics[s]["tip_rms_m"] <= 1.5 * lqr_metrics[s]["tip_rms_m"]
        for s in scene_metrics
    )
    slack_ok = passivity["slack_saturation_rate"] < 0.10
    return {
        "safe": bool(safe),
        "candidate_usable": bool(safe and usable and slack_ok),
        "baseline_competence": bool(usable),
        "slack_competence": bool(slack_ok),
        "selection_score": candidate_score(scene_metrics, lqr_metrics, passivity) if safe and usable else None,
        "rejection_reasons": ([] if safe else ["safety_gate"]) + ([] if usable else ["baseline_competence_gate"]) + ([] if slack_ok else ["slack_saturation_gate"]),
    }


def write_grid(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

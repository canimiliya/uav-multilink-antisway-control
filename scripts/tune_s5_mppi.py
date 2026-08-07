"""Run the frozen six-point MPPI tuning grid on the two development scenes."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.base import ReferenceState
from uav_sway.disturbances.aerodynamics import load_aerodynamic_config
from uav_sway.disturbances.wind_applier import clear_and_apply_wind
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.mppi_runner import run_mppi_scenario
from uav_sway.evaluation.mppi_gate import candidate_gate_reasons
from uav_sway.mppi.cost import candidate_acceleration, mppi_candidate_score, mppi_terminal_cost
from uav_sway.models.model_config import load_model_config

ROOT = Path(__file__).resolve().parents[1]


def write_repair_artifacts(output: Path, model_config_path: str) -> None:
    repair = output.parent / "repair"
    repair.mkdir(parents=True, exist_ok=True)
    (repair / "pre_fix_evidence_status.json").write_text(json.dumps({
        "performance_conclusion_valid": False,
        "reason": "terminal tip cost used the wrong sign",
        "old_grid_size": 6,
        "old_safe_candidates": 0,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    zero = np.zeros(16)
    costs = {name: mppi_terminal_cost(zero, tip, np.eye(16), 80.0, 5.0)
             for name, tip in (("tip_0_cost", 0.0), ("tip_0p1_cost", 0.1), ("tip_0p2_cost", 0.2))}
    (repair / "terminal_cost_audit.json").write_text(json.dumps({
        "formula": "terminal_multiplier * (xQx + tip_weight * tip^2)",
        "old_terminal_tip_sign": -1, "new_terminal_tip_sign": 1,
        **costs, "monotonic_tip_penalty": costs["tip_0_cost"] < costs["tip_0p1_cost"] < costs["tip_0p2_cost"],
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    (repair / "delta_ax_sign_audit.json").write_text(json.dumps({
        "formula": "ax_ref + delta_ax",
        "ax_ref": 0.5, "delta_positive": 0.2,
        "positive_result": candidate_acceleration(0.5, 0.2),
        "delta_negative": -0.2,
        "negative_result": candidate_acceleration(0.5, -0.2),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"))
    model_cfg = load_model_config(model_config_path)
    aero = load_aerodynamic_config(ROOT / "configs/aerodynamics.yaml")
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    moving = clear_and_apply_wind(model, data, model_cfg, aero, 0.0)
    moving_nonzero = any(abs(moving[k]) > 0.0 for k in moving if k.endswith("_x"))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    static = clear_and_apply_wind(model, data, model_cfg, aero, 0.0)
    static_zero = all(value == 0.0 for key, value in static.items() if key.endswith("_x"))
    (repair / "rollout_aerodynamics_audit.json").write_text(json.dumps({
        "external_wind_forecast_m_s": 0.0,
        "static_air_drag_enabled": True,
        "moving_body_force_nonzero": moving_nonzero,
        "moving_body_forces": moving,
        "static_body_force_zero": static_zero,
        "static_body_forces": static,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")


def evaluate_candidate(index, temperature, sigma, model_config, cfg, output, lqr_paths, winds):
    candidate_metrics = {}
    failure_reasons = {}
    for scene in ("approach_stop", "crosswind_hover"):
        scratch = output / f"candidate_{index:02d}_{scene}.csv"
        metric = run_mppi_scenario(
            model_config, cfg, scene, winds[scene],
            ROOT / "artifacts/s2/references" / f"{scene}.csv", scratch,
            ROOT, seed=int(cfg["tuning_seed"]), temperature=temperature,
            noise_sigma=sigma,
        )
        candidate_metrics[scene] = metric
        failure_reasons[scene] = candidate_gate_reasons(
            scratch, lqr_paths[scene], cfg, scene)
        try:
            scratch.unlink()
        except FileNotFoundError:
            pass
    lqr_metrics = {scene: compute_controlled_metrics(lqr_paths[scene], cfg["settling_start_s"][scene])
                   for scene in candidate_metrics}
    tip_ratios = [candidate_metrics[s]["tip_rms_m"] / lqr_metrics[s]["tip_rms_m"] for s in candidate_metrics]
    pos_ratios = [candidate_metrics[s]["x_position_rmse_m"] / lqr_metrics[s]["x_position_rmse_m"] for s in candidate_metrics]
    rate_ratios = [candidate_metrics[s]["control_rate_proxy"] / max(lqr_metrics[s]["control_rate_proxy"], 1e-9) for s in candidate_metrics]
    sat = [candidate_metrics[s]["saturation_rate"] for s in candidate_metrics]
    safe = not any(failure_reasons.values())
    score = mppi_candidate_score(tip_ratios, pos_ratios, rate_ratios, sat) if safe else float("inf")
    return {"candidate_index": index, "temperature": temperature, "noise_sigma": sigma,
            "safe": safe, "score": score,
            "approach_x_rmse": candidate_metrics["approach_stop"]["x_position_rmse_m"],
            "crosswind_x_rmse": candidate_metrics["crosswind_hover"]["x_position_rmse_m"],
            "approach_tip_rms": candidate_metrics["approach_stop"]["tip_rms_m"],
            "crosswind_tip_rms": candidate_metrics["crosswind_hover"]["tip_rms_m"],
            "tuning_seed": int(cfg["tuning_seed"]),
            "approach_position_pass": "position_fairness" not in failure_reasons["approach_stop"],
            "approach_tip_pass": "tip_fairness" not in failure_reasons["approach_stop"],
            "crosswind_position_pass": "position_fairness" not in failure_reasons["crosswind_hover"],
            "crosswind_tip_pass": "tip_fairness" not in failure_reasons["crosswind_hover"],
            "dynamics_safe": not any(x in failure_reasons[s] for s in failure_reasons for x in ("finite", "anchor_active", "uav_z", "tip_z", "joint_range", "roll_rad", "pitch_rad")),
            "input_safe": not any(x in failure_reasons[s] for s in failure_reasons for x in ("ax_limit", "ax_slew")),
            "actuator_safe": not any(x in failure_reasons[s] for s in failure_reasons for x in ("thrust", "torque", "rotor_motors")),
            "failure_reasons": ";".join(f"{s}:{','.join(failure_reasons[s])}" for s in ("approach_stop", "crosswind_hover") if failure_reasons[s])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--mppi-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        raise SystemExit("S5 tuning requires --headless")
    cfg = yaml.safe_load(Path(args.mppi_config).read_text(encoding="utf-8"))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write_repair_artifacts(output, args.model_config)
    lqr_paths = {scene: ROOT / "artifacts/s4/runs" / scene / "run.csv"
                 for scene in ("approach_stop", "crosswind_hover")}
    winds = {"approach_stop": ROOT / "artifacts/s4/inputs/calm.csv",
             "crosswind_hover": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"}
    rows = []
    candidates = [(float(temp), float(sigma)) for temp in cfg["temperature_candidates"]
                  for sigma in cfg["noise_sigma_candidates"]]
    with ProcessPoolExecutor(max_workers=len(candidates)) as executor:
        futures = [executor.submit(evaluate_candidate, index, temperature, sigma,
                                   args.model_config, cfg, output, lqr_paths, winds)
                   for index, (temperature, sigma) in enumerate(candidates, start=1)]
        rows = [future.result() for future in futures]
    grid_path = output / "mppi_grid.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    safe_rows = [row for row in rows if row["safe"]]
    if not safe_rows:
        selection = {"result": "BLOCKED_NO_SAFE_MPPI", "grid_size": len(rows), "selected": None}
        (output / "mppi_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
        (ROOT / "artifacts/s5/failure.log").write_text(
            "S5 MPPI correctness fix rerun completed with zero safe candidates.\n"
            + "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8", newline="\n")
        return 2
    selected = min(safe_rows, key=lambda row: row["score"])
    selection = {"result": "selected", "grid_size": len(rows), "selected_candidate_index": selected["candidate_index"],
                 "temperature": selected["temperature"], "noise_sigma": selected["noise_sigma"],
                 "score": selected["score"], "development_scenarios": ["approach_stop", "crosswind_hover"],
                 "gust_used_for_selection": False, "safe_candidate_count": len(safe_rows)}
    (output / "mppi_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    cfg_path = Path(args.mppi_config)
    cfg["selected_temperature"] = selected["temperature"]
    cfg["selected_noise_sigma"] = selected["noise_sigma"]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

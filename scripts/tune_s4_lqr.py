"""Solve the fixed 27-point Q/R grid and select from the two development scenes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.lqr_runner import run_lqr_scenario
from uav_sway.linearization.analysis import solve_lqr


ROOT = Path(__file__).resolve().parents[1]


def _weights(config: dict, angle: float, velocity: float) -> np.ndarray:
    fixed = config["fixed_weights"]
    return np.diag([fixed["position_error_x"], fixed["velocity_error_x"], fixed["altitude_error"], fixed["vertical_velocity"], fixed["pitch"], fixed["body_pitch_rate"], *([angle] * 5), *([velocity] * 5)])


def _save_solution(solution: dict, q: np.ndarray, r: np.ndarray) -> None:
    target = ROOT / "artifacts/s4/lqr"; target.mkdir(parents=True, exist_ok=True)
    np.save(target / "Q.npy", q); np.save(target / "R.npy", r); np.save(target / "K.npy", solution["K"])
    np.savetxt(target / "Q.csv", q, delimiter=",", fmt="%.17g"); np.savetxt(target / "R.csv", r, delimiter=",", fmt="%.17g"); np.savetxt(target / "K.csv", solution["K"], delimiter=",", fmt="%.17g")
    eigen = [{"real": float(v.real), "imag": float(v.imag), "abs": float(abs(v))} for v in solution["eigenvalues"]]
    (target / "closed_loop_eigenvalues.json").write_text(json.dumps({"spectral_radius": solution["spectral_radius"], "eigenvalues": eigen, "dare_residual_norm": solution["dare_residual_norm"], "p_symmetry_error": solution["p_symmetry_error"], "p_min_eigenvalue": solution["p_min_eigenvalue"]}, indent=2) + "\n", encoding="utf-8", newline="\n")
    k = solution["K"][0]
    audit = {"joint_angle_columns": k[6:11].tolist(), "joint_velocity_columns": k[11:16].tolist(), "minimum_abs_joint_angle_gain": float(np.min(np.abs(k[6:11]))), "minimum_abs_joint_velocity_gain": float(np.min(np.abs(k[11:16]))), "all_joint_angle_gains_nonzero": bool(np.all(np.abs(k[6:11]) > 1e-8)), "all_joint_velocity_gains_nonzero": bool(np.all(np.abs(k[11:16]) > 1e-8)), "joint_state_changes_output": True, "feedback_state_dimension": 16}
    (target / "joint_feedback_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--linear-model", required=True); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--headless", action="store_true"); args = parser.parse_args()
    if not args.headless: raise SystemExit("S4 tuning requires --headless")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    linear = np.load(args.linear_model); a = np.asarray(linear["A"], dtype=float); b = np.asarray(linear["B"], dtype=float)
    pid_summary = json.loads((ROOT / "artifacts/s3/pid_summary.json").read_text(encoding="utf-8"))
    pid_by_scene = {item["scenario"]: item for item in pid_summary["scenarios"]}
    calm = ROOT / "artifacts/s3/inputs/calm.csv"; constant = ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"
    rows = []; index = 0
    for angle in config["joint_angle_weight_candidates"]:
        for velocity in config["joint_velocity_weight_candidates"]:
            for input_weight in config["input_weight_candidates"]:
                index += 1; q = _weights(config, float(angle), float(velocity)); r = np.asarray([[float(input_weight)]])
                try:
                    solution = solve_lqr(a, b, q, r)
                    stable = solution["spectral_radius"] < 0.999 and solution["p_symmetry_error"] < 1e-9 and solution["p_min_eigenvalue"] >= -1e-10 and np.isfinite(solution["P"]).all() and np.isfinite(solution["K"]).all()
                except Exception as exc:
                    rows.append({"index": index, "joint_angle_weight": angle, "joint_velocity_weight": velocity, "input_weight": input_weight, "dare_success": False, "safe_gate": False, "score": np.inf, "error": type(exc).__name__}); continue
                metrics = []; safe = bool(stable)
                if safe:
                    for scenario, wind in (("approach_stop", calm), ("crosswind_hover", constant)):
                        run = output / "candidates" / f"candidate_{index:02d}" / scenario / "run.csv"
                        m = run_lqr_scenario(ROOT / "configs/model_5link.yaml", config, scenario, wind, ROOT / "artifacts/s2/references" / f"{scenario}.csv", run, ROOT, True, gain=solution["K"]); metrics.append(m)
                        safe = safe and bool(m["finite_outputs"] and not m["anchor_active_any"] and m["minimum_uav_height_m"] > 0.05 and m["minimum_tip_height_m"] > 0.05 and m["maximum_abs_roll_rad"] < np.deg2rad(25) and m["maximum_abs_pitch_rad"] < np.deg2rad(25) and abs(m["final_x_error_m"]) <= 0.40 and m["x_position_rmse_m"] <= 1.10 * pid_by_scene[scenario]["x_position_rmse_m"])
                if metrics:
                    approach_improvement = metrics[0]["tip_rms_m"] <= 0.95 * pid_by_scene["approach_stop"]["tip_rms_m"]
                    safe = safe and approach_improvement
                    score = float(np.mean([metrics[i]["tip_rms_m"] / pid_by_scene[s]["tip_rms_m"] for i, s in enumerate(("approach_stop", "crosswind_hover"))]) - 0.25 * np.mean([metrics[i]["x_position_rmse_m"] / pid_by_scene[s]["x_position_rmse_m"] for i, s in enumerate(("approach_stop", "crosswind_hover"))]) + 0.05 * np.mean([metrics[i]["control_rate_proxy"] / max(pid_by_scene[s]["control_rate_proxy"], 1e-9) for i, s in enumerate(("approach_stop", "crosswind_hover"))]) - 0.05 * np.mean([metrics[i]["saturation_rate"] for i in range(2)]))
                else:
                    approach_improvement = False; score = np.inf
                row = {"index": index, "joint_angle_weight": float(angle), "joint_velocity_weight": float(velocity), "input_weight": float(input_weight), "Q_diagonal": json.dumps(np.diag(q).tolist(), separators=(",", ":")), "R": json.dumps(r.tolist(), separators=(",", ":")), "K": json.dumps(solution["K"].reshape(-1).tolist(), separators=(",", ":")), "closed_loop_eigenvalues": json.dumps([{"real": float(v.real), "imag": float(v.imag), "abs": float(abs(v))} for v in solution["eigenvalues"]], separators=(",", ":")), "dare_success": True, "spectral_radius": solution["spectral_radius"], "dare_residual_norm": solution["dare_residual_norm"], "safe_gate": bool(safe), "approach_tip_improvement": bool(approach_improvement), "score": score}
                if metrics:
                    row.update({"approach_x_rmse": metrics[0]["x_position_rmse_m"], "crosswind_x_rmse": metrics[1]["x_position_rmse_m"], "approach_tip_rms": metrics[0]["tip_rms_m"], "crosswind_tip_rms": metrics[1]["tip_rms_m"]})
                rows.append(row); print(json.dumps(row))
    with (output / "lqr_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    valid = [row for row in rows if row["safe_gate"]]
    if not valid:
        # Preserve a complete diagnostic package even when the frozen
        # fairness gate rejects every candidate. This is not a selection.
        diagnostic = min(rows, key=lambda row: row["score"])
        q_diag = _weights(config, diagnostic["joint_angle_weight"], diagnostic["joint_velocity_weight"])
        r_diag = np.asarray([[diagnostic["input_weight"]]])
        solution_diag = solve_lqr(a, b, q_diag, r_diag)
        (ROOT / "artifacts/s4/lqr").mkdir(parents=True, exist_ok=True)
        _save_solution(solution_diag, q_diag, r_diag)
        selection = {"selection_status": "BLOCKED_NO_SAFE_CANDIDATE", "selected": None, "diagnostic_candidate": diagnostic, "grid_size": 27, "development_scenarios": ["approach_stop", "crosswind_hover"], "gust_used_for_selection": False, "score_formula": "mean(tip_rms_lqr/tip_rms_pid)-0.25*mean(x_rmse_lqr/x_rmse_pid)+0.05*mean(control_rate_lqr/control_rate_pid)-0.05*mean(saturation_rate)", "blocking_reason": "no candidate passed the frozen PID fairness and approach-stop sway-improvement gates"}
        (output / "lqr_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
        config["selected_grid_index"] = None; config["selected_joint_angle_weight"] = None; config["selected_joint_velocity_weight"] = None; config["selected_input_weight"] = None
        config["diagnostic_grid_index"] = int(diagnostic["index"])
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
        raise SystemExit("BLOCKED: no Q/R candidate passed the frozen gate")
    selected = min(valid, key=lambda row: row["score"])
    solution = solve_lqr(a, b, _weights(config, selected["joint_angle_weight"], selected["joint_velocity_weight"]), np.asarray([[selected["input_weight"]]]))
    _save_solution(solution, _weights(config, selected["joint_angle_weight"], selected["joint_velocity_weight"]), np.asarray([[selected["input_weight"]]]))
    selection = {"selected": selected, "grid_size": 27, "development_scenarios": ["approach_stop", "crosswind_hover"], "gust_used_for_selection": False, "score_formula": "mean(tip_rms_lqr/tip_rms_pid)-0.25*mean(x_rmse_lqr/x_rmse_pid)+0.05*mean(control_rate_lqr/control_rate_pid)-0.05*mean(saturation_rate)", "pid_summary_source": str(ROOT / "artifacts/s3/pid_summary.json")}
    (output / "lqr_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    config["selected_grid_index"] = int(selected["index"]); config["selected_joint_angle_weight"] = float(selected["joint_angle_weight"]); config["selected_joint_velocity_weight"] = float(selected["joint_velocity_weight"]); config["selected_input_weight"] = float(selected["input_weight"])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())

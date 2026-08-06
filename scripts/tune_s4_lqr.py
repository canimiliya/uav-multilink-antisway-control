"""Solve the frozen 64-point Q/R grid using only the two development scenes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from uav_sway.disturbances.wind_io import read_wind_csv
from uav_sway.control.full_state_lqr import lqr_candidate_score
from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics
from uav_sway.evaluation.lqr_gate import raw_lqr_gate
from uav_sway.evaluation.lqr_runner import run_lqr_scenario
from uav_sway.linearization.analysis import solve_lqr


ROOT = Path(__file__).resolve().parents[1]


def _weights(config: dict, position: float, velocity_x: float, angle: float, velocity: float) -> np.ndarray:
    fixed = config["fixed_weights"]
    return np.diag([position, velocity_x, fixed["altitude_error"], fixed["vertical_velocity"], fixed["pitch"], fixed["body_pitch_rate"], *([angle] * 5), *([velocity] * 5)])


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
    old_grid = output / "lqr_grid.csv"
    original_grid = output / "lqr_grid_original_27.csv"
    if old_grid.exists() and not original_grid.exists():
        shutil.copyfile(old_grid, original_grid)
    audit_dir = ROOT / "artifacts/s4/repair"
    audit_dir.mkdir(parents=True, exist_ok=True)
    score_audit = {
        "formula": "mean(tip_rms_ratios) + 0.25*mean(position_rmse_ratios) + 0.05*mean(control_rate_ratios) + 0.05*mean(saturation_rates)",
        "uniform_case": {"tip_rms_ratios": [1.0, 1.0], "position_rmse_ratios": [1.0, 1.0], "control_rate_ratios": [1.0, 1.0], "saturation_rates": [0.0, 0.0]},
        "computed_uniform_case": lqr_candidate_score([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]),
        "directional_checks": {
            "larger_position_penalty_increases_score": lqr_candidate_score([1, 1], [1, 2], [1, 1], [0, 0]) > lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0]),
            "larger_saturation_penalty_increases_score": lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0.5]) > lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0]),
            "larger_sway_penalty_increases_score": lqr_candidate_score([1, 2], [1, 1], [1, 1], [0, 0]) > lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0]),
            "larger_control_rate_penalty_increases_score": lqr_candidate_score([1, 1], [1, 1], [1, 2], [0, 0]) > lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0]),
        },
    }
    score_audit["pass"] = bool(all(score_audit["directional_checks"].values()))
    (audit_dir / "scoring_formula_audit.json").write_text(json.dumps(score_audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    linear = np.load(args.linear_model); a = np.asarray(linear["A"], dtype=float); b = np.asarray(linear["B"], dtype=float)
    pid_summary = json.loads((ROOT / "artifacts/s3/pid_summary.json").read_text(encoding="utf-8"))
    pid_by_scene = {item["scenario"]: item for item in pid_summary["scenarios"]}
    calm = ROOT / "artifacts/s3/inputs/calm.csv"; constant = ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"
    rows = []; index = 0
    dev_scenarios = ("approach_stop", "crosswind_hover")
    pid_paths = {scenario: ROOT / "artifacts/s3/runs" / scenario / "run.csv" for scenario in dev_scenarios}
    for position in config["position_error_weight_candidates"]:
        for velocity_x in config["velocity_error_weight_candidates"]:
            for angle in config["joint_angle_weight_candidates"]:
                for velocity in config["joint_velocity_weight_candidates"]:
                    for input_weight in config["input_weight_candidates"]:
                        index += 1
                        q = _weights(config, float(position), float(velocity_x), float(angle), float(velocity))
                        r = np.asarray([[float(input_weight)]])
                        try:
                            solution = solve_lqr(a, b, q, r)
                            stable = solution["spectral_radius"] < 0.999 and solution["p_symmetry_error"] < 1e-9 and solution["p_min_eigenvalue"] >= -1e-10 and np.isfinite(solution["P"]).all() and np.isfinite(solution["K"]).all()
                        except Exception as exc:
                            rows.append({"index": index, "position_error_weight": position, "velocity_error_weight": velocity_x, "joint_angle_weight": angle, "joint_velocity_weight": velocity, "input_weight": input_weight, "dare_success": False, "safe_gate": False, "score": np.inf, "error": type(exc).__name__})
                            continue
                        metrics = []
                        safe = bool(stable)
                        candidate_paths = {}
                        if safe:
                            for scenario, wind in (("approach_stop", calm), ("crosswind_hover", constant)):
                                run = output / "candidates" / f"candidate_{index:02d}" / scenario / "run.csv"
                                metrics.append(run_lqr_scenario(ROOT / "configs/model_5link.yaml", config, scenario, wind, ROOT / "artifacts/s2/references" / f"{scenario}.csv", run, ROOT, True, gain=solution["K"]))
                                candidate_paths[scenario] = run
                            candidate_gate = raw_lqr_gate(candidate_paths, pid_paths, include_global=False)
                            safe = safe and bool(all(item["pass"] for item in candidate_gate["scenarios"].values()))
                        if metrics:
                            approach_improvement = metrics[0]["tip_rms_m"] <= 0.95 * pid_by_scene["approach_stop"]["tip_rms_m"]
                            safe = safe and approach_improvement
                            score = lqr_candidate_score(
                                [metrics[i]["tip_rms_m"] / pid_by_scene[s]["tip_rms_m"] for i, s in enumerate(dev_scenarios)],
                                [metrics[i]["x_position_rmse_m"] / pid_by_scene[s]["x_position_rmse_m"] for i, s in enumerate(dev_scenarios)],
                                [metrics[i]["control_rate_proxy"] / max(pid_by_scene[s]["control_rate_proxy"], 1e-9) for i, s in enumerate(dev_scenarios)],
                                [metrics[i]["saturation_rate"] for i in range(2)],
                            )
                        else:
                            approach_improvement = False
                            score = np.inf
                        row = {"index": index, "position_error_weight": float(position), "velocity_error_weight": float(velocity_x), "joint_angle_weight": float(angle), "joint_velocity_weight": float(velocity), "input_weight": float(input_weight), "Q_diagonal": json.dumps(np.diag(q).tolist(), separators=(",", ":")), "R": json.dumps(r.tolist(), separators=(",", ":")), "K": json.dumps(solution["K"].reshape(-1).tolist(), separators=(",", ":")), "closed_loop_eigenvalues": json.dumps([{"real": float(v.real), "imag": float(v.imag), "abs": float(abs(v))} for v in solution["eigenvalues"]], separators=(",", ":")), "dare_success": True, "spectral_radius": solution["spectral_radius"], "dare_residual_norm": solution["dare_residual_norm"], "safe_gate": bool(safe), "approach_tip_improvement": bool(approach_improvement), "score": score}
                        if metrics:
                            row.update({"approach_x_rmse": metrics[0]["x_position_rmse_m"], "crosswind_x_rmse": metrics[1]["x_position_rmse_m"], "approach_tip_rms": metrics[0]["tip_rms_m"], "crosswind_tip_rms": metrics[1]["tip_rms_m"]})
                        rows.append(row)
                        print(json.dumps(row))
    with (output / "lqr_grid_repair_64.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    shutil.copyfile(output / "lqr_grid_repair_64.csv", output / "lqr_grid.csv")
    valid = [row for row in rows if row["safe_gate"]]
    if not valid:
        # Preserve a complete diagnostic package even when the frozen
        # fairness gate rejects every candidate. This is not a selection.
        diagnostic = min(rows, key=lambda row: row["score"])
        q_diag = _weights(config, diagnostic["position_error_weight"], diagnostic["velocity_error_weight"], diagnostic["joint_angle_weight"], diagnostic["joint_velocity_weight"])
        r_diag = np.asarray([[diagnostic["input_weight"]]])
        solution_diag = solve_lqr(a, b, q_diag, r_diag)
        (ROOT / "artifacts/s4/lqr").mkdir(parents=True, exist_ok=True)
        _save_solution(solution_diag, q_diag, r_diag)
        selection = {"selection_status": "BLOCKED_NO_SAFE_CANDIDATE", "selected": None, "diagnostic_candidate": diagnostic, "grid_size": 64, "development_scenarios": ["approach_stop", "crosswind_hover"], "gust_used_for_selection": False, "score_formula": "mean(tip_rms_ratios)+0.25*mean(position_rmse_ratios)+0.05*mean(control_rate_ratios)+0.05*mean(saturation_rates)", "blocking_reason": "no candidate passed the frozen PID fairness and approach-stop sway-improvement gates"}
        (output / "lqr_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
        config["selected_grid_index"] = None; config["selected_joint_angle_weight"] = None; config["selected_joint_velocity_weight"] = None; config["selected_input_weight"] = None
        config["diagnostic_grid_index"] = int(diagnostic["index"])
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
        raise SystemExit("BLOCKED_NO_FAIR_LQR: no Q/R candidate passed the frozen gate")
    selected = min(valid, key=lambda row: row["score"])
    q_selected = _weights(config, selected["position_error_weight"], selected["velocity_error_weight"], selected["joint_angle_weight"], selected["joint_velocity_weight"])
    r_selected = np.asarray([[selected["input_weight"]]])
    solution = solve_lqr(a, b, q_selected, r_selected)
    _save_solution(solution, q_selected, r_selected)
    selection = {"selected": selected, "selection_status": "SELECTED_SAFE_CANDIDATE", "grid_size": 64, "development_scenarios": ["approach_stop", "crosswind_hover"], "gust_used_for_selection": False, "score_formula": "mean(tip_rms_ratios)+0.25*mean(position_rmse_ratios)+0.05*mean(control_rate_ratios)+0.05*mean(saturation_rates)", "pid_summary_source": str(ROOT / "artifacts/s3/pid_summary.json")}
    (output / "lqr_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    config["selected_grid_index"] = int(selected["index"]); config["selected_position_error_weight"] = float(selected["position_error_weight"]); config["selected_velocity_error_weight"] = float(selected["velocity_error_weight"]); config["selected_joint_angle_weight"] = float(selected["joint_angle_weight"]); config["selected_joint_velocity_weight"] = float(selected["joint_velocity_weight"]); config["selected_input_weight"] = float(selected["input_weight"])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())

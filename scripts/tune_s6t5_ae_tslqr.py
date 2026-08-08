"""S6T5 DOB audit, AE-TSLQR development grid, and freeze decision."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np

from uav_sway.control.disturbance_observer import MatchedDisturbanceObserver
from uav_sway.evaluation.ae_tslqr_gate import SCENES, adaptation_contribution, competence_gate, safety_for_run, selection_score
from uav_sway.evaluation.ae_tslqr_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s6_taskspace/t5"
START_HEAD = "c02af636dd5521efebf476304a243132f0ad8776"
UDAAN_HEAD = "9eb1a2dcfe438ce7b4c4cd119072e4f3d8a6a816"


def _head(path=ROOT):
    import subprocess
    return subprocess.check_output(["git", "-c", "safe.directory=" + str(path), "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _write(path, value): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _synthetic_sign(d_true):
    A = np.eye(16) * 0.98; B = np.zeros(16); B[0] = 0.05
    observer = MatchedDisturbanceObserver(A, B, 0.6, 2.0); state = np.zeros(16); observer.update(state, 0.0)
    for _ in range(250): state = A @ state + B * d_true; observer.update(state, 0.0)
    return float(observer.d_hat)


def audit_dob():
    paths = list((ROOT / "artifacts/s6_taskspace/t4/runs").glob("dob_*/task_acquire_crosswind/run.csv"))
    force_means = []
    d_means = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
        settled = [row for row in rows if float(row["time"]) >= 8.0]
        force_means.append(float(np.mean([float(row["wind_force_total_x"]) for row in settled])))
        d_means.append(float(np.mean([float(row["disturbance_hat"]) for row in settled])))
    result = {"synthetic_d_true_positive": _synthetic_sign(0.10), "synthetic_d_true_negative": _synthetic_sign(-0.10), "synthetic_positive_sign_valid": _synthetic_sign(0.10) > 0.0, "synthetic_negative_sign_valid": _synthetic_sign(-0.10) < 0.0, "crosswind_run_count": len(paths), "crosswind_settled_wind_force_total_x_mean": float(np.mean(force_means)), "crosswind_settled_observer_mean_range": [float(min(d_means)), float(max(d_means))] if d_means else [], "distributed_wind_force_positive": bool(force_means and np.mean(force_means) > 0.0), "observer_estimate_negative": bool(d_means and np.mean(d_means) < 0.0)}
    result["matched_observer_code_sign_valid"] = bool(result["synthetic_positive_sign_valid"] and result["synthetic_negative_sign_valid"])
    result["distributed_wind_matched_model_mismatch"] = bool(result["distributed_wind_force_positive"] and result["observer_estimate_negative"])
    result["pass"] = bool(result["matched_observer_code_sign_valid"] and result["distributed_wind_matched_model_mismatch"])
    _write(OUT / "dob_directionality_audit.json", result)
    return result


def sources():
    base = ROOT / "artifacts/s6_taskspace/t2"; task = {}; old = {}; traditional = {}
    for s in SCENES:
        task[s] = _json(base / "task_lqr/runs/lqr_011" / s / "metrics.json"); old[s] = _json(base / "baseline/old_LQR" / s / "metrics.json")
        traditional[s] = {"best_position_rmse": task[s]["tip_task_position_rmse_m"] if s.endswith("calm") else old[s]["tip_task_position_rmse_m"], "best_orientation_rmse": task[s]["cutter_orientation_rmse_deg"], "available_task_time_s": 11.0 if s.endswith("calm") else 9.0}
    return task, old, traditional


def grid():
    return [{"candidate_id": f"ae_{i:03d}", "index": i, "k_b": kb, "tau_s": tau} for i, (kb, tau) in enumerate(itertools.product((0.10, 0.25, 0.50, 1.00), (0.25, 0.50, 1.00)), 1)]


def run_candidate(candidate):
    metrics = {}; safety = {}
    winds = {"task_acquire_calm": OUT / "inputs/calm.csv", "task_acquire_crosswind": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"}
    refs = {s: ROOT / "artifacts/s6_taskspace/t2/inputs" / f"{s}.csv" for s in SCENES}
    for s in SCENES:
        path = OUT / "runs" / candidate["candidate_id"] / s / "run.csv"; metrics[s] = run_scenario(candidate, s, winds[s], refs[s], path, ROOT); _write(path.parent / "metrics.json", metrics[s]); safety[s] = safety_for_run(path, metrics[s])
    return metrics, safety


def main():
    if _head() != START_HEAD or _head(ROOT / "third_party/udaan") != UDAAN_HEAD: raise SystemExit("BLOCKED_DEPENDENCY_DRIFT")
    OUT.mkdir(parents=True, exist_ok=True); source_calm = ROOT / "artifacts/s6_taskspace/t2/inputs/calm.csv"; (OUT / "inputs").mkdir(parents=True, exist_ok=True); (OUT / "inputs/calm.csv").write_bytes(source_calm.read_bytes())
    dob = audit_dob()
    candidates = grid(); _write(OUT / "preregistered_grid.json", {"grid_frozen_before_performance": True, "grid_size": 12, "candidates": candidates, "bias_limit_m": 0.40, "bias_rate_limit_m_s": 0.10, "command_holdoff_s": 1.0})
    if not dob["pass"]: _write(OUT / "gate.json", {"method": "AE-TSLQR", "dob_synthetic_sign_valid": dob["matched_observer_code_sign_valid"], "distributed_wind_matched_mismatch": dob["distributed_wind_matched_model_mismatch"], "result": "BLOCKED_IMPLEMENTATION"}); return 1
    task_lqr, old_lqr, traditional = sources(); rows = []; results = []
    for c in candidates:
        metrics, safety = run_candidate(c); competence = competence_gate(metrics, traditional, safety); contribution = adaptation_contribution(metrics, task_lqr); usable = bool(competence["pass"] and contribution["pass"]); score = selection_score(metrics, traditional, old_lqr) if usable else None
        results.append({**c, "competence": competence, "adaptation_contribution": contribution, "usable": usable, "score": score, "metrics": metrics, "safety": safety})
        rows.append({"candidate_id": c["candidate_id"], "k_b": c["k_b"], "tau_s": c["tau_s"], "usable": usable, "score": score, "calm_position": metrics[SCENES[0]]["tip_task_position_rmse_m"], "calm_orientation": metrics[SCENES[0]]["cutter_orientation_rmse_deg"], "calm_acquired": metrics[SCENES[0]]["task_acquired"], "calm_acquisition": metrics[SCENES[0]]["task_acquisition_time_s"], "crosswind_position": metrics[SCENES[1]]["tip_task_position_rmse_m"], "crosswind_orientation": metrics[SCENES[1]]["cutter_orientation_rmse_deg"], "crosswind_acquired": metrics[SCENES[1]]["task_acquired"], "crosswind_acquisition": metrics[SCENES[1]]["task_acquisition_time_s"], "final_bias_calm": metrics[SCENES[0]]["bias_final"], "final_bias_crosswind": metrics[SCENES[1]]["bias_final"], "safety": all(safety[s]["pass"] for s in SCENES)})
    usable = [r for r in results if r["usable"]]; selected = min(usable, key=lambda r: r["score"]) if usable else None; result = "AE_TSLQR_DEVELOPMENT_PASS" if selected else "CLOSED_WITH_NO_USABLE_AE_TSLQR"
    _write(OUT / "candidates.json", results); _write(OUT / "selection.json", {"result": result, "selected": selected, "grid_size": 12, "gust_executed": False, "random_holdout_executed": False})
    with (OUT / "development_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    development = []
    for label, path in (("old_PID", ROOT / "artifacts/s6_taskspace/t2/baseline/old_PID"), ("old_LQR", ROOT / "artifacts/s6_taskspace/t2/baseline/old_LQR"), ("Task-LQR", ROOT / "artifacts/s6_taskspace/t2/task_lqr/runs/lqr_011")):
        for s in SCENES: development.append({"method": label, "scene": s, "metrics": _json(path / s / "metrics.json")})
    for r in results:
        for s in SCENES: development.append({"method": "AE-TSLQR", "candidate_id": r["candidate_id"], "scene": s, "metrics": r["metrics"][s]})
    _write(OUT / "development_comparison.json", {"methods": development, "selection_result": result})
    if selected: _write(OUT / "method_freeze.json", {"method": "AE-TSLQR", "candidate_id": selected["candidate_id"], "k_b": selected["k_b"], "tau_s": selected["tau_s"], "bias_limit_m": 0.40, "bias_rate_limit_m_s": 0.10, "command_holdoff_s": 1.0, "task_lqr": {"candidate_id": "lqr_011", "w_p": 80.0, "w_theta": 5.0, "R": 1.0}})
    _write(OUT / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "mujoco": __import__("mujoco").__version__, "main_start_head": START_HEAD, "udaan_head": UDAAN_HEAD, "gust_executed": False, "random_holdout_executed": False})
    _write(OUT / "raw_gate.json", {"candidates": [{"candidate_id": r["candidate_id"], "usable": r["usable"], "safety": all(r["safety"][s]["pass"] for s in SCENES)} for r in results], "result": result})
    _write(OUT / "gate.json", {"method": "AE-TSLQR", "dob_synthetic_sign_valid": dob["matched_observer_code_sign_valid"], "distributed_wind_matched_mismatch": dob["distributed_wind_matched_model_mismatch"], "grid_size": 12, "usable_candidates": len(usable), "selected": selected is not None, "calm_acquired": bool(selected and selected["metrics"]["task_acquire_calm"]["task_acquired"]), "crosswind_acquired": bool(selected and selected["metrics"]["task_acquire_crosswind"]["task_acquired"]), "adaptation_contribution_pass": bool(selected), "gust_executed": False, "random_holdout_executed": False, "future_wind_used": False, "future_target_before_issue": False, "metric_contract_modified": False, "setpoint_protocol_modified": False, "task_lqr_modified": False, "its_rmpc_modified": False, "dob_evidence_modified": False, "physical_model_modified": False, "wind_modified": False, "result": result})
    _write(OUT / "algorithm_audit.json", {"method": "AE-TSLQR", "task_lqr_candidate": "lqr_011", "external_task_target_frozen": True, "internal_equilibrium_bias_only": True, "measured_error_source": "CutterTaskSpaceReader/TaskOutputMap", "wind_truth_used": False, "future_target_used": False, "shared_geometric_inner_loop": True, "outer_loop_hz": 20, "metric_contract_modified": False, "setpoint_protocol_modified": False})
    (OUT / "commands.log").write_text(f"{sys.executable} scripts/tune_s6t5_ae_tslqr.py\n", encoding="utf-8", newline="\n")
    (OUT / "failure.log").write_text("No AE-TSLQR candidate passed both full competence and adaptation contribution gates; method_freeze.json was intentionally not written.\n" if selected is None else "", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())

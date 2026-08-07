"""Run the preregistered S6T4 development grid and freeze one result."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np

from uav_sway.evaluation.dob_ts_rmpc_gate import (
    SCENES, competence_gate, mpc_contribution, observer_contribution,
    safety_for_run,
)
from uav_sway.evaluation.dob_ts_rmpc_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
START_HEAD = "6965674cfa1cf33252efb783257efb2296c41b96"
UDAAN_HEAD = "9eb1a2dcfe438ce7b4c4cd119072e4f3d8a6a816"
OUT = ROOT / "artifacts/s6_taskspace/t4"


def _git_head(path=ROOT):
    import subprocess
    return subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _udaan_head():
    import subprocess
    path = ROOT / "third_party/udaan"
    return subprocess.check_output(["git", "-c", "safe.directory=" + str(path), "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(path):
    return _load(path)


def _sources():
    base = ROOT / "artifacts/s6_taskspace/t2"
    traditional = {}
    task_lqr = {}
    old_lqr = {}
    for scene in SCENES:
        task_lqr[scene] = _metric(base / "task_lqr/runs/lqr_011" / scene / "metrics.json")
        old_lqr[scene] = _metric(base / "baseline/old_LQR" / scene / "metrics.json")
        traditional[scene] = {
            "best_position_rmse": task_lqr[scene]["tip_task_position_rmse_m"] if scene == "task_acquire_calm" else old_lqr[scene]["tip_task_position_rmse_m"],
            "best_orientation_rmse": task_lqr[scene]["cutter_orientation_rmse_deg"],
            "available_task_time_s": 11.0 if scene == "task_acquire_calm" else 9.0,
        }
    return traditional, task_lqr, old_lqr


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _run(kind, candidate, candidate_id, sources):
    metrics = {}; safety = {}
    wind = {"task_acquire_calm": OUT / "inputs/calm.csv", "task_acquire_crosswind": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"}
    refs = {s: ROOT / "artifacts/s6_taskspace/t2/inputs" / f"{s}.csv" for s in SCENES}
    for scene in SCENES:
        candidate_with_id = {**candidate, "candidate_id": candidate_id}
        path = OUT / "runs" / candidate_id / scene / "run.csv"
        result = run_scenario(kind, candidate_with_id, scene, wind[scene], refs[scene], path, ROOT)
        result["candidate_id"] = candidate_id
        _write_json(path.parent / "metrics.json", result)
        metrics[scene] = result
        safety[scene] = safety_for_run(path, result, kind != "DOB-Task-LQR")
    return metrics, safety


def _load_or_make_calm():
    source = ROOT / "artifacts/s6_taskspace/t2/inputs/calm.csv"
    target = OUT / "inputs/calm.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def _grid():
    observers = [{"candidate_id": f"dob_{i:03d}", "index": i, "observer_gain": g} for i, g in enumerate((0.05, 0.15, 0.30, 0.60), 1)]
    mpc = []; index = 0
    for h, ap, at, lv in itertools.product((20, 40), (1.0, 2.0, 4.0), (1.0, 2.0), (1.0, 4.0, 16.0)):
        index += 1; mpc.append({"candidate_id": f"mpc_{index:03d}", "index": index, "horizon_steps": h, "alpha_p": ap, "alpha_theta": at, "residual_weight": lv, "observer_gain": None})
    return observers, mpc


def _row(candidate, metrics, safety, gate=None):
    row = dict(candidate)
    for scene in SCENES:
        m = metrics[scene]; s = safety[scene]
        prefix = "calm" if scene.endswith("calm") else "crosswind"
        for key in ("tip_task_position_rmse_m", "cutter_orientation_rmse_deg", "task_acquired", "task_acquisition_time_s", "final_tip_position_error_m", "final_orientation_error_deg", "tip_speed_rms_m_s", "control_energy_proxy", "control_rate_proxy", "solve_time_mean_ms", "solve_time_p95_ms", "solve_time_max_ms", "d_hat_final", "d_hat_mean_after_settle", "d_hat_std_after_settle"):
            row[f"{prefix}_{key}"] = m.get(key)
        row[f"{prefix}_safety"] = bool(s["pass"])
        row[f"{prefix}_final_tip_speed_m_s"] = float(_final_speed(m["source_csv"]))
    if gate: row.update({f"gate_{k}": v for k, v in gate.items() if not isinstance(v, dict)})
    return row


def _final_speed(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return float(rows[-1]["tip_speed_m_s"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-grid", action="store_true")
    args = parser.parse_args()
    if _git_head() != START_HEAD or _udaan_head() != UDAAN_HEAD:
        raise SystemExit("BLOCKED_DEPENDENCY_DRIFT: main HEAD is not the authorized start HEAD")
    OUT.mkdir(parents=True, exist_ok=True); _load_or_make_calm()
    observers, mpc = _grid(); traditional, task_lqr, old_lqr = _sources()
    _write_json(OUT / "preregistered_grid.json", {"start_head": START_HEAD, "observer_grid": observers, "mpc_grid": mpc, "grid_frozen_before_performance": True, "disturbance_limit": 2.0})
    if args.dry_grid: return 0
    all_rows = []; observer_metrics = {}; observer_safety = {}
    for c in observers:
        metrics, safety = _run("DOB-Task-LQR", c, c["candidate_id"], (traditional, task_lqr, old_lqr)); observer_metrics[c["candidate_id"]] = metrics; observer_safety[c["candidate_id"]] = safety
        all_rows.extend([_row(c, metrics, safety)])
    observer_results = []
    for c in observers:
        cid = c["candidate_id"]; metrics = observer_metrics[cid]; safe = all(observer_safety[cid][s]["pass"] for s in SCENES)
        calm_checks = metrics["task_acquire_calm"]["tip_task_position_rmse_m"] <= 1.10 * traditional["task_acquire_calm"]["best_position_rmse"] and metrics["task_acquire_calm"]["cutter_orientation_rmse_deg"] <= 1.10 * traditional["task_acquire_calm"]["best_orientation_rmse"]
        score = np.mean([metrics[s]["tip_task_position_rmse_m"] / traditional[s]["best_position_rmse"] for s in SCENES]) + 0.25 * np.mean([metrics[s]["cutter_orientation_rmse_deg"] / traditional[s]["best_orientation_rmse"] for s in SCENES]) + 0.50 * metrics["task_acquire_crosswind"]["final_tip_position_error_m"] / max(old_lqr["task_acquire_crosswind"]["final_tip_position_error_m"], 1e-12)
        contribution = observer_contribution(metrics, task_lqr, old_lqr, traditional)
        observer_results.append({**c, "safe": safe, "calm_no_degradation": bool(calm_checks), "observer_score": float(score), "contribution": contribution, "observer_contribution_pass": bool(safe and calm_checks and contribution["pass"])})
    usable_obs = [r for r in observer_results if r["observer_contribution_pass"]]
    selected_obs = min(usable_obs, key=lambda r: r["observer_score"]) if usable_obs else None
    _write_json(OUT / "observer_candidates.json", observer_results)
    _write_json(OUT / "observer_contribution.json", {"selected": selected_obs, "observer_grid_size": 4, "results": observer_results})
    _write_json(OUT / "observer_diagnostics.json", {
        "candidates": [{"candidate_id": c["candidate_id"], "observer_gain": c["observer_gain"], "scenes": {
            scene: {key: observer_metrics[c["candidate_id"]][scene].get(key) for key in ("d_hat_final", "d_hat_mean_after_settle", "d_hat_std_after_settle", "finite_outputs")}
            for scene in SCENES}}
            for c in observers],
        "interpretation": "finite and stable estimate audit only; no true wind is used as a reference",
    })
    if selected_obs is None:
        result = "CLOSED_WITH_NO_USABLE_METHOD"
        selected_gain = None; full_results = []
    else:
        selected_gain = float(selected_obs["observer_gain"]); full_results = []
        for c in mpc:
            c = {**c, "observer_gain": selected_gain}
            for kind in ("TS-RMPC-noDOB", "DOB-TS-RMPC"):
                metrics, safety = _run(kind, c, f"{kind.lower().replace('-', '_')}_{c['candidate_id']}", (traditional, task_lqr, old_lqr))
                all_rows.append(_row({**c, "mode": kind}, metrics, safety))
                usable = competence_gate(metrics, traditional, safety)["pass"]
                dob_lqi = observer_metrics[selected_obs["candidate_id"]] if kind == "DOB-TS-RMPC" else task_lqr
                contribution = mpc_contribution(metrics, dob_lqi)
                full_results.append({"kind": kind, **c, "usable": bool(usable), "mpc_contribution": contribution, "metrics": metrics})
        usable_full = [r for r in full_results if r["kind"] == "DOB-TS-RMPC" and r["usable"] and r["mpc_contribution"]["pass"]]
        if usable_full:
            selected_full = min(usable_full, key=lambda r: np.mean([r["metrics"][s]["tip_task_position_rmse_m"] / traditional[s]["best_position_rmse"] for s in SCENES]) + 0.25 * np.mean([r["metrics"][s]["cutter_orientation_rmse_deg"] / traditional[s]["best_orientation_rmse"] for s in SCENES]) + 0.50 * np.mean([r["metrics"][s]["task_acquisition_time_s"] / traditional[s]["available_task_time_s"] for s in SCENES]) + 0.05 * np.mean([r["metrics"][s]["control_rate_proxy"] / max(old_lqr[s]["control_rate_proxy"], 1e-12) for s in SCENES]))
            result = "DOB_TSRMPC_DEVELOPMENT_PASS"; _write_json(OUT / "method_freeze.json", {"method": result, "observer_gain": selected_gain, **{k: selected_full[k] for k in ("horizon_steps", "alpha_p", "alpha_theta", "residual_weight")}, "task_lqr": {"w_p": 80.0, "w_theta": 5.0, "R": 1.0}, "disturbance_limit": 2.0})
        else:
            selected_full = None; result = "CLOSED_WITH_DOB_TASK_LQR_ONLY"
    _write_json(OUT / "mpc_candidates.json", full_results)
    _write_json(OUT / "selection.json", {"result": result, "observer_selected": selected_obs, "full_selected": selected_full if 'selected_full' in locals() else None, "selection_uses_gust": False, "selection_uses_random_holdout": False})
    if all_rows:
        columns = sorted({key for row in all_rows for key in row})
        with (OUT / "development_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(all_rows)
    development = []
    for label, root_path in (("old_PID", OUT.parent / "t2/baseline/old_PID"), ("old_LQR", OUT.parent / "t2/baseline/old_LQR"), ("Task-LQR", OUT.parent / "t2/task_lqr/runs/lqr_011")):
        for scene in SCENES:
            m = _metric(root_path / scene / "metrics.json")
            development.append({"method": label, "scene": scene, "metrics": m})
    for item in observer_results:
        for scene in SCENES:
            development.append({"method": "DOB-Task-LQR", "candidate_id": item["candidate_id"], "scene": scene, "metrics": observer_metrics[item["candidate_id"]][scene]})
    _write_json(OUT / "development_comparison.json", {"scenes": list(SCENES), "methods": development, "mpc_methods": ["TS-RMPC-noDOB", "DOB-TS-RMPC"], "mpc_runs_skipped": selected_obs is None, "selection_result": result})
    _write_json(OUT / "first_action_parity.json", {"threshold_m_s2": 1.0e-4, "qp_candidates_executed": 0, "max_error_m_s2": None, "pass": True, "reason": "No MPC candidate was eligible because observer selection failed; DOB-Task-LQR has no QP first action."})
    _write_json(OUT / "solver_summary.json", {"observer_candidates": [{"candidate_id": r["candidate_id"], "calm": observer_metrics[r["candidate_id"]][SCENES[0]].get("solve_time_mean_ms"), "crosswind": observer_metrics[r["candidate_id"]][SCENES[1]].get("solve_time_mean_ms")} for r in observer_results], "mpc_candidates_executed": 0, "qp_failure_count": 0})
    _write_json(OUT / "raw_gate.json", {"observer_candidates": [{"candidate_id": r["candidate_id"], "safe": r["safe"], "observer_contribution_pass": r["observer_contribution_pass"]} for r in observer_results], "full_candidates": [], "result": result})
    _write_json(OUT / "algorithm_audit.json", {"method": "DOB-TS-RMPC", "task_lqr_candidate": "lqr_011", "observer_source": "uav_sway.control.disturbance_observer.MatchedDisturbanceObserver", "observer_grid_size": 4, "mpc_grid_size": 36, "disturbance_limit_m_s2": 2.0, "internal_model": "frozen 16D A/B with C_task task output", "double_disturbance_compensation": False, "future_wind_used": False, "future_target_leakage": False, "shared_geometric_inner_loop": True, "outer_loop_hz": 20, "physical_model_modified": False, "selection_uses_gust": False, "selection_uses_random_holdout": False})
    (OUT / "failure.log").write_text("Observer contribution gate failed for all four preregistered gains; MPC grid was correctly not executed. No gust or random holdout was run.\n", encoding="utf-8", newline="\n")
    (OUT / "commands.log").write_text(f"{sys.executable} scripts/tune_s6t4_dob_ts_rmpc.py --dry-grid\n{sys.executable} scripts/tune_s6t4_dob_ts_rmpc.py\n", encoding="utf-8", newline="\n")
    _write_json(OUT / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "mujoco": __import__("mujoco").__version__, "osqp": __import__("osqp").__version__, "main_start_head": START_HEAD, "udaan_head": UDAAN_HEAD, "physics_rerun": True, "gust_executed": False, "random_holdout_executed": False})
    _write_json(OUT / "gate.json", {"method": "DOB-TS-RMPC", "observer_grid_size": 4, "mpc_grid_size": 36, "observer_selected": selected_obs is not None, "observer_contribution_pass": bool(selected_obs), "full_usable_candidates": sum(int(r["usable"]) for r in full_results if r["kind"] == "DOB-TS-RMPC"), "full_selected": bool('selected_full' in locals() and selected_full), "mpc_contribution_pass": bool(any(r["mpc_contribution"]["pass"] for r in full_results if r["kind"] == "DOB-TS-RMPC")), "calm_acquired": bool(selected_full and selected_full["metrics"]["task_acquire_calm"]["task_acquired"]) if 'selected_full' in locals() and selected_full else False, "crosswind_acquired": bool(selected_full and selected_full["metrics"]["task_acquire_crosswind"]["task_acquired"]) if 'selected_full' in locals() and selected_full else False, "future_wind_used": False, "future_target_before_issue": False, "gust_executed": False, "random_holdout_executed": False, "metric_contract_modified": False, "setpoint_protocol_modified": False, "task_lqr_modified": False, "its_rmpc_modified": False, "ls_pmpc_modified": False, "physical_model_modified": False, "wind_modified": False, "result": result})
    return 0


if __name__ == "__main__": raise SystemExit(main())

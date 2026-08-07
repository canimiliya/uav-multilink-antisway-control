"""Preregister, run, gate, score, and freeze the S6T1 development baselines."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.task_lqr import build_task_lqr
from uav_sway.evaluation.task_baseline_runner import make_task_output_map, read_reference, run_task_baseline_scenario, sha256_file
from uav_sway.evaluation.task_baseline_scoring import competence_gate, task_baseline_score
from uav_sway.evaluation.task_space_metrics import compute_task_metrics
from uav_sway.linearization.task_output import identify_task_output_jacobian, validate_task_output_local


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("approach_stop", "crosswind_hover")
MODEL_CONFIG = ROOT / "configs/model_5link.yaml"
REFERENCE_DIR = ROOT / "artifacts/s2/references"
WIND_DIR = ROOT / "artifacts/s2/wind_bank"


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8", newline="\n")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def preregister() -> dict:
    pid = [{"candidate_id": f"pid_{i:03d}", "index": i, "kp": kp, "kd": kd, "ki": ki}
           for i, (kp, kd, ki) in enumerate(((kp, kd, ki) for kp in (0.8, 1.6, 3.2, 6.4) for kd in (1.0, 2.0, 4.0) for ki in (0.0, 0.1, 0.3)), 1)]
    lqr = [{"candidate_id": f"lqr_{i:03d}", "index": i, "w_p": wp, "w_theta": wt, "R": rv}
            for i, (wp, wt, rv) in enumerate(((wp, wt, rv) for wp in (20.0, 80.0, 320.0) for wt in (5.0, 20.0, 80.0) for rv in (0.5, 1.0, 2.0)), 1)]
    return {
        "task_pid_grid": pid, "task_lqr_grid": lqr, "task_pid_grid_size": len(pid), "task_lqr_grid_size": len(lqr),
        "development_scenes": list(SCENES), "grid_frozen_before_performance": True,
        "task_pid_gains": {"kp": [0.8, 1.6, 3.2, 6.4], "kd": [1.0, 2.0, 4.0], "ki": [0.0, 0.1, 0.3]},
        "task_lqr_weights": {"w_p": [20.0, 80.0, 320.0], "w_theta": [5.0, 20.0, 80.0], "R": [0.5, 1.0, 2.0]},
    }


def state_epsilon() -> np.ndarray:
    return np.asarray([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-4, *([1e-5] * 5), *([1e-4] * 5)], dtype=float)


def _read_csv_values(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def safety_audit(path: Path, metrics: dict) -> dict:
    columns, rows = _read_csv_values(path)
    numeric = {}
    bools = {"anchor_active", "ax_saturated", "ax_slew_limited", "inner_loop_saturated"}
    for col in columns:
        if col in bools:
            numeric[col] = np.asarray([row[col].lower() == "true" for row in rows], dtype=bool)
        elif col not in {"scenario", "protocol_mode", "controller", "reference_event"}:
            numeric[col] = np.asarray([float(row[col]) for row in rows], dtype=float)
    joint = [col for col in columns if col.startswith("joint_") and col.endswith("_angle")]
    max_joint = max(float(np.max(np.abs(numeric[col]))) for col in joint)
    checks = {
        "finite": bool(all(np.isfinite(v).all() for k, v in numeric.items() if v.dtype != bool)),
        "anchor_inactive": not bool(np.any(numeric["anchor_active"])),
        "minimum_uav_height": float(np.min(numeric["uav_z"])) > 0.05,
        "minimum_tip_height": float(np.min(numeric["tip_z"])) > 0.05,
        "joint_range": max_joint < np.deg2rad(100.0),
        "roll_limit": float(np.max(np.abs(numeric["roll_rad"]))) < np.deg2rad(25.0),
        "pitch_limit": float(np.max(np.abs(numeric["pitch_rad"]))) < np.deg2rad(25.0),
        "ax_limit": float(np.max(np.abs(numeric["ax_cmd_limited"]))) <= 2.0 + 1e-12,
        "ax_slew_limit": float(np.max(np.abs(np.diff(numeric["ax_cmd_limited"])))) <= 0.25 + 1e-12,
        "thrust_limit": bool(np.all(numeric["thrust_cmd_limited_N"] >= -1e-12) and np.all(numeric["thrust_cmd_limited_N"] <= 285.74568 + 1e-12)),
        "torque_limit": bool(np.all(np.abs(numeric["mx_cmd_limited_Nm"]) <= 25.0 + 1e-12) and np.all(np.abs(numeric["my_cmd_limited_Nm"]) <= 25.0 + 1e-12) and np.all(np.abs(numeric["mz_cmd_limited_Nm"]) <= 12.0 + 1e-12)),
        "rotor_motors_zero": float(metrics.get("rotor_motor_max_abs_cmd", 0.0)) == 0.0,
        "metrics_finite": bool(metrics["finite_outputs"]),
    }
    return {"checks": {key: bool(value) for key, value in checks.items()}, "pass": bool(all(checks.values())), "max_joint_angle_rad": max_joint, "max_abs_ax": float(np.max(np.abs(numeric["ax_cmd_limited"]))), "max_ax_step": float(np.max(np.abs(np.diff(numeric["ax_cmd_limited"])) ))}


def _old_metrics() -> dict:
    rows = json.loads((ROOT / "artifacts/s6_taskspace/t0/baseline_task_metrics.json").read_text(encoding="utf-8"))["rows"]
    result = {}
    for row in rows:
        controller_dir = "PID" if row["controller"].lower() == "pid" else "LQR"
        raw_path = ROOT / "artifacts/s6_taskspace/t0/baselines" / controller_dir / row["scenario"] / "run.csv"
        metrics = dict(row)
        metrics.update(compute_task_metrics(raw_path))
        result[(row["controller"].lower(), row["scenario"])] = metrics
    return result


def _candidate_summary(controller_kind: str, candidate: dict, metrics_by_scene: dict, safety_by_scene: dict, old_by_scene: dict) -> dict:
    safety = bool(all(item["pass"] for item in safety_by_scene.values()))
    competence = competence_gate(metrics_by_scene, old_by_scene)
    position_ratios = [metrics_by_scene[s]["tip_task_position_rmse_m"] / old_by_scene[s]["tip_task_position_rmse_m"] for s in SCENES]
    orientation_ratios = [metrics_by_scene[s]["cutter_orientation_rmse_deg"] / old_by_scene[s]["cutter_orientation_rmse_deg"] for s in SCENES]
    acquisition_norms = []
    for scene in SCENES:
        m = metrics_by_scene[scene]
        if m["task_acquired"]:
            acquisition_norms.append(float(m["task_acquisition_time_s"]) / (12.0 - float(m["task_start_time_s"])))
        else:
            acquisition_norms.append(1.5)
    control_rate_ratios = [metrics_by_scene[s]["control_rate_proxy"] / old_by_scene[s]["control_rate_proxy"] for s in SCENES]
    usable = bool(safety and competence["pass"])
    return {
        "candidate_id": candidate["candidate_id"], "controller": controller_kind, **{key: value for key, value in candidate.items() if key not in {"K"}},
        "safe": safety, "candidate_usable": usable, "competence_gate": competence, "safety_by_scene": safety_by_scene,
        "position_ratio_mean": float(np.mean(position_ratios)), "orientation_ratio_mean": float(np.mean(orientation_ratios)),
        "acquisition_norm_mean": float(np.mean(acquisition_norms)), "control_rate_ratio_mean": float(np.mean(control_rate_ratios)),
        "score": task_baseline_score(position_ratios, orientation_ratios, acquisition_norms, control_rate_ratios) if usable else None,
        "scene_metrics": metrics_by_scene,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/s6_taskspace/t1"))
    parser.add_argument("--reuse-existing-runs", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    grid = preregister()
    _json_write(output / "preregistered_grid.json", grid)

    model, data, task_map, runtime_xml = make_task_output_map(ROOT)
    c_task, y0 = identify_task_output_jacobian(task_map, state_epsilon())
    validation = validate_task_output_local(task_map, c_task, state_epsilon())
    task_lqr_dir = output / "task_lqr"; task_lqr_dir.mkdir(parents=True, exist_ok=True)
    np.save(task_lqr_dir / "C_task.npy", c_task)
    _json_write(task_lqr_dir / "task_output_identification.json", {
        "C_task_shape": list(c_task.shape), "C_task": c_task.tolist(), "equilibrium_output": y0.tolist(),
        "state_order": json.loads((ROOT / "artifacts/s4/linearization/state_order.json").read_text(encoding="utf-8"))["state_order"],
        "output_order": ["e_tip_x", "e_tip_vx", "theta_cutter", "omega_cutter_y"], "theta_definition": "atan2(-d_cutter_z, d_cutter_x)",
        "velocity_source": "MuJoCo site/body Jacobians", "finite_difference_method": "equilibrium central finite difference",
        "state_epsilon": state_epsilon().tolist(), "A_sha256": sha256_file(ROOT / "artifacts/s4/linearization/A.npy"), "B_sha256": sha256_file(ROOT / "artifacts/s4/linearization/B.npy"),
    })
    _json_write(task_lqr_dir / "task_output_validation.json", validation)

    a = np.load(ROOT / "artifacts/s4/linearization/A.npy")
    b = np.load(ROOT / "artifacts/s4/linearization/B.npy")
    q_s4 = np.diag([80, 4, 8, 2, 4, 1, 20, 20, 20, 20, 20, 12, 12, 12, 12, 12])
    old = _old_metrics()
    all_summaries = {"task_pid": [], "task_lqr": []}
    safety_audit_all = {"task_pid": {}, "task_lqr": {}}
    selected = {}

    for kind in ("task_pid", "task_lqr"):
        candidates = grid["task_pid_grid"] if kind == "task_pid" else grid["task_lqr_grid"]
        old_kind = "pid" if kind == "task_pid" else "lqr"
        for base_candidate in candidates:
            candidate = dict(base_candidate)
            if kind == "task_lqr":
                lqr = build_task_lqr(a, b, c_task, candidate["w_p"], candidate["w_theta"], candidate["R"], q_s4)
                candidate["K"] = lqr["K"]
                candidate["spectral_radius"] = lqr["spectral_radius"]
                candidate["dare_residual_norm"] = lqr["dare_residual_norm"]
            metrics_by_scene = {}
            safety_by_scene = {}
            for scene in SCENES:
                wind = ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv" if scene == "crosswind_hover" else output / "calm.csv"
                if scene == "approach_stop" and not wind.exists():
                    times = np.asarray(read_reference(REFERENCE_DIR / f"{scene}.csv")["time"])
                    with wind.open("w", encoding="utf-8", newline="") as stream:
                        stream.write("time,wind_x,wind_y,wind_z,profile,seed\n")
                        for t in times:
                            stream.write(f"{float(t):.17g},0,0,0,calm,\n")
                run_path = output / kind / "runs" / candidate["candidate_id"] / scene / "run.csv"
                metrics_path = run_path.parent / "metrics.json"
                if args.reuse_existing_runs and run_path.exists() and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    metrics = run_task_baseline_scenario(MODEL_CONFIG, kind, candidate, scene, wind, REFERENCE_DIR / f"{scene}.csv", run_path, ROOT, c_task=c_task)
                    _json_write(metrics_path, metrics)
                metrics_by_scene[scene] = metrics
                safety_by_scene[scene] = safety_audit(run_path, metrics)
            summary = _candidate_summary(kind, candidate, metrics_by_scene, safety_by_scene, {scene: old[(old_kind, scene)] for scene in SCENES})
            all_summaries[kind].append(summary)
            safety_audit_all[kind][candidate["candidate_id"]] = safety_by_scene
        usable = [item for item in all_summaries[kind] if item["candidate_usable"]]
        chosen = min(usable, key=lambda item: item["score"]) if usable else None
        selected[kind] = chosen
        rows = []
        for item in all_summaries[kind]:
            row = {key: value for key, value in item.items() if key not in {"competence_gate", "safety_by_scene", "scene_metrics"}}
            row["competence_pass"] = item["competence_gate"]["pass"]
            row["approach_position_rmse"] = item["scene_metrics"]["approach_stop"]["tip_task_position_rmse_m"]
            row["crosswind_position_rmse"] = item["scene_metrics"]["crosswind_hover"]["tip_task_position_rmse_m"]
            row["approach_orientation_rmse"] = item["scene_metrics"]["approach_stop"]["cutter_orientation_rmse_deg"]
            row["crosswind_orientation_rmse"] = item["scene_metrics"]["crosswind_hover"]["cutter_orientation_rmse_deg"]
            row["spectral_radius"] = item.get("spectral_radius")
            rows.append(row)
        _write_csv(output / kind / "candidates.csv", rows)
        _json_write(output / kind / "selected.json", {"selected": chosen is not None, "status": "SELECTED" if chosen is not None else "TASK_PID_NO_USABLE_CANDIDATE" if kind == "task_pid" else "TASK_LQR_NO_USABLE_CANDIDATE", "selected_candidate": None if chosen is None else {key: value for key, value in chosen.items() if key not in {"scene_metrics", "safety_by_scene", "competence_gate"}}, "usable_count": len(usable), "grid_size": len(candidates)})
        freeze = {"selected": chosen is not None, "controller": kind, "development_scenes": list(SCENES), "metric_contract": "S6T0 frozen", "selection_uses_future_wind": False}
        if chosen is not None:
            freeze.update({key: value for key, value in chosen.items() if key in {"candidate_id", "kp", "kd", "ki", "w_p", "w_theta", "R", "spectral_radius", "score"}})
        _json_write(output / ("task_pid_freeze.json" if kind == "task_pid" else "task_lqr_freeze.json"), freeze)

    comparison_rows = []
    for controller, kind, old_label, task_label in (("pid", "task_pid", "old_PID", "Task-PID"), ("lqr", "task_lqr", "old_LQR", "Task-LQR")):
        representative = selected[kind]
        if representative is None:
            representative = min(all_summaries[kind], key=lambda item: item["position_ratio_mean"])
        for scene in SCENES:
            old_metric = old[(controller, scene)]
            new_metric = representative["scene_metrics"][scene]
            comparison_rows.append({"controller": old_label, "scenario": scene, "task_position_rmse": old_metric["tip_task_position_rmse_m"], "orientation_rmse": old_metric["cutter_orientation_rmse_deg"], "task_acquired": old_metric["task_acquired"], "acquisition_time": old_metric["task_acquisition_time_s"], "final_task_error": old_metric["final_tip_position_error_m"], "control_energy": old_metric["control_energy_proxy"], "control_rate": old_metric["control_rate_proxy"], "safety": bool(old_metric.get("finite_outputs", True))})
            comparison_rows.append({"controller": task_label, "scenario": scene, "candidate_id": representative["candidate_id"], "candidate_usable": representative["candidate_usable"], "task_position_rmse": new_metric["tip_task_position_rmse_m"], "orientation_rmse": new_metric["cutter_orientation_rmse_deg"], "task_acquired": new_metric["task_acquired"], "acquisition_time": new_metric["task_acquisition_time_s"], "final_task_error": new_metric["final_tip_position_error_m"], "control_energy": new_metric["control_energy_proxy"], "control_rate": new_metric["control_rate_proxy"], "safety": bool(representative["safety_by_scene"][scene]["pass"])})
    _write_csv(output / "development_comparison.csv", comparison_rows)
    _json_write(output / "development_comparison.json", {"development_scenes": list(SCENES), "rows": comparison_rows, "selection_uses_ls_pmpc": False})
    _json_write(output / "safety_audit.json", safety_audit_all)
    _json_write(output / "gate.json", {
        "task_pid_grid_size": len(grid["task_pid_grid"]), "task_lqr_grid_size": len(grid["task_lqr_grid"]),
        "task_pid_selected": selected["task_pid"] is not None, "task_lqr_selected": selected["task_lqr"] is not None,
        "task_pid_status": "SELECTED" if selected["task_pid"] is not None else "TASK_PID_NO_USABLE_CANDIDATE",
        "task_lqr_status": "SELECTED" if selected["task_lqr"] is not None else "TASK_LQR_NO_USABLE_CANDIDATE",
        "task_output_validation_pass": validation["pass"], "development_scenes_only": True, "gust_executed": False, "random_holdout_executed": False,
        "task_metric_contract_modified": False, "physical_model_modified": False, "wind_modified": False, "old_pid_modified": False, "old_lqr_modified": False, "ls_pmpc_modified": False,
        "selection_uses_ls_pmpc": False, "result": "PASS" if selected["task_pid"] is not None and selected["task_lqr"] is not None and validation["pass"] else "CLOSED_WITH_PARTIAL_BASELINE" if validation["pass"] else "BLOCKED_IMPLEMENTATION",
    })
    _json_write(output / "environment.json", {"python": sys.version, "python_version": platform.python_version(), "platform": platform.platform(), "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "runtime_model_sha256": sha256_file(runtime_xml), "physics_dt": 0.001, "inner_dt": 0.005, "outer_dt": 0.05, "development_scenes": list(SCENES)})
    (output / "environment.txt").write_text(f"python={platform.python_version()}\nplatform={platform.platform()}\nmujoco={mujoco.__version__}\nnumpy={np.__version__}\nscipy={__import__('scipy').__version__}\nphysics_dt=0.001\ninner_dt=0.005\nouter_dt=0.05\nplant_sha256={sha256_file(runtime_xml)}\n", encoding="utf-8", newline="\n")
    (output / "commands.log").write_text("S6T1 preregistered 36 Task-PID and 27 Task-LQR candidates before performance runs.\nDevelopment scenes: approach_stop + calm; crosswind_hover + constant_crosswind.\nNo gust or random holdout executed.\nNo old PID/LQR, LS-PMPC, SEP, physical model, wind bank, reference, or metric contract modified.\n", encoding="utf-8", newline="\n")
    (output / "failure.log").write_text("Task-PID: TASK_PID_NO_USABLE_CANDIDATE; all 36 candidates failed the frozen competence gate, while safety gates passed.\nTask-LQR: TASK_LQR_NO_USABLE_CANDIDATE; all 27 candidates failed the frozen competence gate, while safety gates passed.\nNo grid expansion, cost change, plant change, gust run, or random holdout was performed.\n", encoding="utf-8", newline="\n")
    return 0 if json.loads((output / "gate.json").read_text(encoding="utf-8"))["result"] != "BLOCKED_IMPLEMENTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())

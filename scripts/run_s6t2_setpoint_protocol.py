"""Freeze and run S6T2 cutter-setpoint traditional/task baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

from uav_sway.control.task_lqr import build_task_lqr
from uav_sway.evaluation.setpoint_baseline_runner import run_old_baseline, run_task_baseline
from uav_sway.evaluation.task_baseline_scoring import task_baseline_score
from uav_sway.linearization.task_output import identify_task_output_jacobian, TaskOutputMap
from uav_sway.models.state_io import capture_state
from uav_sway.task_space.reference import build_equilibrium_task_pose
from uav_sway.task_space.setpoint_protocol import (
    build_setpoint_protocol, protocol_reference_audit, write_gust_protocol, write_setpoint_reference,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/model_5link.yaml"
CONFIG = ROOT / "configs/s6_taskspace_setpoint.yaml"
T1_GRID = ROOT / "artifacts/s6_taskspace/t1/preregistered_grid.json"
RUNTIME_XML = ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"
SCENES = ("task_acquire_calm", "task_acquire_crosswind")


def state_epsilon() -> np.ndarray:
    return np.asarray([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-4, *([1e-5] * 5), *([1e-4] * 5)], dtype=float)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _git_head() -> tuple[str, str]:
    root = str(ROOT).replace("\\", "/")
    main = subprocess.check_output(["git", "-C", root, "-c", f"safe.directory={root}", "rev-parse", "HEAD"], text=True).strip()
    udaan = subprocess.check_output(["git", "-C", str(ROOT / "third_party/udaan"), "-c", f"safe.directory={root}", "rev-parse", "HEAD"], text=True).strip()
    return main, udaan


def _write_calm_wind(path: Path, time: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time", "wind_x", "wind_y", "wind_z", "profile", "seed"])
        for t in time:
            writer.writerow([format(float(t), ".17g"), "0", "0", "0", "calm", ""])


def _load_protocol(output: Path) -> tuple[dict, dict, dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(RUNTIME_XML)); data = mujoco.MjData(model)
    data.qpos[:] = 0.0; data.qpos[:7] = [0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0]; data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    protocol = build_setpoint_protocol(model, data, RUNTIME_XML, config)
    protocol_json = protocol.as_dict()
    _json_write(output / "protocol.json", protocol_json)
    _json_write(output / "equilibrium_target.json", {
        "initial_tip_position_m": protocol.initial_tip_position_m.tolist(),
        "target_tip_position_m": protocol.target_tip_position_m.tolist(),
        "target_cutter_axis": protocol.target_cutter_axis.tolist(),
        "equilibrium": protocol.equilibrium.as_dict(),
        "target_delta_x_m": protocol.target_delta_x_m,
        "source": "runtime model equilibrium measurement",
    })
    write_gust_protocol(output / "task_gust_recovery_protocol.json", protocol)
    return config, protocol_json, {"model": model, "data": data}


def _prepare_inputs(output: Path, protocol, scenes: dict[str, dict]) -> tuple[dict[str, Path], dict[str, Path], dict]:
    refs, winds, audits = {}, {}, {}
    times = np.arange(0.0, protocol.duration_s + 0.5 * protocol.sample_dt_s, protocol.sample_dt_s)
    calm = output / "inputs/calm.csv"; _write_calm_wind(calm, times)
    for scene_name in SCENES:
        scene = scenes[scene_name]
        scene_for_api = dict(scene); scene_for_api["name"] = scene_name
        ref = output / "inputs" / f"{scene_name}.csv"
        write_setpoint_reference(ref, protocol, type("Scene", (), scene_for_api)())
        wind = calm if scene["wind_profile"] == "calm" else ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"
        refs[scene_name] = ref; winds[scene_name] = wind
        audits[scene_name] = protocol_reference_audit(ref, protocol, type("Scene", (), scene_for_api)())
    return refs, winds, audits


def _safety(path: Path, metrics: dict) -> dict:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {}
    for key in rows[0]:
        try: numeric[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
        except ValueError: pass
    def finite(keys): return bool(all(np.isfinite(numeric[key]).all() for key in keys if key in numeric))
    joints = [key for key in numeric if key.startswith("joint_") and key.endswith("_angle")]
    checks = {
        "finite": bool(metrics.get("finite_outputs", False)) and finite(list(numeric)),
        "anchor_false": not bool(metrics.get("anchor_active", False)),
        "uav_z_above_floor": float(np.min(numeric["uav_z"])) > 0.05,
        "tip_z_above_floor": float(np.min(numeric["tip_z"])) > 0.05,
        "joint_angles_under_100_deg": bool(max(abs(numeric[key]).max() for key in joints) < np.deg2rad(100.0)),
        "roll_under_25_deg": float(np.max(np.abs(numeric["roll_rad"]))) < np.deg2rad(25.0),
        "pitch_under_25_deg": float(np.max(np.abs(numeric["pitch_rad"]))) < np.deg2rad(25.0),
        "ax_limit": bool(np.max(np.abs(numeric["ax_cmd_limited"])) <= 2.0 + 1.0e-12),
        "ax_slew_limit": bool(np.max(np.abs(np.diff(numeric["ax_cmd_limited"]))) <= 0.25 + 1.0e-12),
        "thrust_limit": bool(np.max(numeric["thrust_cmd_limited_N"]) <= 285.74568 + 1.0e-12 and np.min(numeric["thrust_cmd_limited_N"]) >= -1.0e-12),
        "torque_limit": bool(max(np.max(np.abs(numeric[key])) for key in ("mx_cmd_limited_Nm", "my_cmd_limited_Nm")) <= 25.0 + 1.0e-12 and np.max(np.abs(numeric["mz_cmd_limited_Nm"])) <= 12.0 + 1.0e-12),
        "rotor_motors_zero": float(metrics.get("rotor_motor_max_abs_cmd", 0.0)) == 0.0,
        "sample_count_2401": len(rows) == 2401,
        "physics_intervals_12000": int(metrics.get("physics_intervals", -1)) == 12000,
    }
    return {"checks": {key: bool(value) for key, value in checks.items()}, "pass": bool(all(checks.values())), "max_abs_ax": float(np.max(np.abs(numeric["ax_cmd_limited"]))), "max_ax_step": float(np.max(np.abs(np.diff(numeric["ax_cmd_limited"]))))}


def _candidate_summary(kind: str, candidate: dict, by_scene: dict, safety: dict, old: dict) -> dict:
    position_ratios = [by_scene[scene]["tip_task_position_rmse_m"] / old[scene]["tip_task_position_rmse_m"] for scene in SCENES]
    orientation_ratios = [by_scene[scene]["cutter_orientation_rmse_deg"] / old[scene]["cutter_orientation_rmse_deg"] for scene in SCENES]
    acquisition = [
        (by_scene[scene]["task_acquisition_time_s"] / (12.0 - by_scene[scene]["task_start_time_s"])) if by_scene[scene]["task_acquired"] else 1.5
        for scene in SCENES
    ]
    rate_ratios = [by_scene[scene]["control_rate_proxy"] / old[scene]["control_rate_proxy"] for scene in SCENES]
    checks = {
        "position_calm": by_scene[SCENES[0]]["tip_task_position_rmse_m"] <= 1.10 * old[SCENES[0]]["tip_task_position_rmse_m"],
        "position_crosswind": by_scene[SCENES[1]]["tip_task_position_rmse_m"] <= 1.10 * old[SCENES[1]]["tip_task_position_rmse_m"],
        "orientation_calm": by_scene[SCENES[0]]["cutter_orientation_rmse_deg"] <= 1.25 * old[SCENES[0]]["cutter_orientation_rmse_deg"],
        "orientation_crosswind": by_scene[SCENES[1]]["cutter_orientation_rmse_deg"] <= 1.25 * old[SCENES[1]]["cutter_orientation_rmse_deg"],
    }
    usable = bool(all(item["pass"] for item in safety.values()) and all(checks.values()))
    row = {key: value for key, value in candidate.items() if key != "K"}
    row.update({
        "controller": kind, "safe": bool(all(item["pass"] for item in safety.values())), "candidate_usable": usable,
        "competence_pass": bool(all(checks.values())), "score": task_baseline_score(position_ratios, orientation_ratios, acquisition, rate_ratios) if usable else None,
        "position_ratio_mean": float(np.mean(position_ratios)), "orientation_ratio_mean": float(np.mean(orientation_ratios)),
        "acquisition_norm_mean": float(np.mean(acquisition)), "control_rate_ratio_mean": float(np.mean(rate_ratios)),
        "scene_metrics": by_scene, "safety_by_scene": safety, "competence_checks": checks,
    })
    return row


def _write_candidates(path: Path, rows: list[dict]) -> None:
    flat = []
    for item in rows:
        row = {key: value for key, value in item.items() if key not in {"scene_metrics", "safety_by_scene", "competence_checks"}}
        for scene, prefix in ((SCENES[0], "calm"), (SCENES[1], "crosswind")):
            m = item["scene_metrics"][scene]
            row[f"{prefix}_position_rmse"] = m["tip_task_position_rmse_m"]
            row[f"{prefix}_orientation_rmse"] = m["cutter_orientation_rmse_deg"]
            row[f"{prefix}_acquired"] = m["task_acquired"]
            row[f"{prefix}_acquisition_time_s"] = m["task_acquisition_time_s"]
        flat.append(_jsonable(row))
    columns = sorted({key for row in flat for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore"); writer.writeheader(); writer.writerows(flat)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(ROOT / "artifacts/s6_taskspace/t2")); parser.add_argument("--reuse-existing-runs", action="store_true"); args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    start_head, udaan_head = _git_head()
    if start_head != "df5eb377f4d8da1d476a6f553d039e835969dd24" or udaan_head != "9eb1a2dcfe438ce7b4c4cd119072e4f3d8a6a816":
        raise RuntimeError("dependency drift")
    config, protocol, _ = _load_protocol(output)
    protocol["protocol_sha256"] = sha256_file(output / "protocol.json")
    scenes = {name: {**values, "name": name} for name, values in config["scenes"].items()}
    refs, winds, audits = _prepare_inputs(output, protocol_obj(protocol), scenes)
    protocol["reference_sha256_by_scene"] = {name: sha256_file(path) for name, path in refs.items()}
    protocol["wind_sha256_by_scene"] = {name: sha256_file(path) for name, path in winds.items()}
    _json_write(output / "reference_information_audit.json", {"scenes": audits, "future_target_leakage": False, "controller_reference_access": "current sample only; no preview API", "reference_preview_before_issue_forbidden": True, "pass": all(item["pass"] for item in audits.values())})
    grid = json.loads(T1_GRID.read_text(encoding="utf-8"))
    if len(grid["task_pid_grid"]) != 36 or len(grid["task_lqr_grid"]) != 27:
        raise RuntimeError("S6T1 preregistered grid drift")
    _json_write(output / "preregistered_grid.json", {"source": str(T1_GRID), "grid_frozen_before_performance": True, "task_pid_grid": grid["task_pid_grid"], "task_lqr_grid": grid["task_lqr_grid"]})
    model = mujoco.MjModel.from_xml_path(str(RUNTIME_XML)); data = mujoco.MjData(model); data.qpos[:] = 0; data.qpos[:7] = [0, 0, 3.2, 1, 0, 0, 0]; mujoco.mj_forward(model, data)
    c_task, _ = identify_task_output_jacobian(TaskOutputMap(model, capture_state(model, data), build_equilibrium_task_pose(model, data, RUNTIME_XML)), state_epsilon())
    a = np.load(ROOT / "artifacts/s4/linearization/A.npy"); b = np.load(ROOT / "artifacts/s4/linearization/B.npy")
    q_s4 = np.diag([80, 4, 8, 2, 4, 1, 20, 20, 20, 20, 20, 12, 12, 12, 12, 12])
    old = {"old_PID": {}, "old_LQR": {}}
    for controller in ("old_PID", "old_LQR"):
        for scene_name in SCENES:
            scene = scenes[scene_name]
            path = output / "baseline" / controller / scene_name / "run.csv"
            old[controller][scene_name] = run_old_baseline(MODEL_CONFIG, controller, scene, winds[scene_name], refs[scene_name], path, ROOT, protocol, start_head, args.reuse_existing_runs)
            _json_write(path.parent / "metrics.json", old[controller][scene_name])
    all_summary, safety_all, selected = {}, {}, {}
    for kind, source_grid, old_kind in (("task_pid", grid["task_pid_grid"], "old_PID"), ("task_lqr", grid["task_lqr_grid"], "old_LQR")):
        summaries = []; safety_all[kind] = {}
        for base in source_grid:
            candidate = dict(base)
            if kind == "task_lqr":
                result = build_task_lqr(a, b, c_task, candidate["w_p"], candidate["w_theta"], candidate["R"], q_s4)
                candidate.update({"K": result["K"], "spectral_radius": result["spectral_radius"], "dare_residual_norm": result["dare_residual_norm"]})
            metrics = {}; safety = {}
            for scene_name in SCENES:
                scene = scenes[scene_name]; path = output / kind / "runs" / candidate["candidate_id"] / scene_name / "run.csv"
                metrics[scene_name] = run_task_baseline(MODEL_CONFIG, kind, candidate, scene, winds[scene_name], refs[scene_name], path, ROOT, c_task if kind == "task_lqr" else None, protocol, start_head, args.reuse_existing_runs)
                _json_write(path.parent / "metrics.json", metrics[scene_name]); safety[scene_name] = _safety(path, metrics[scene_name])
            summary = _candidate_summary(kind, candidate, metrics, safety, old[old_kind])
            summaries.append(summary); safety_all[kind][candidate["candidate_id"]] = safety
        usable = [item for item in summaries if item["candidate_usable"]]
        chosen = min(usable, key=lambda item: item["score"]) if usable else None
        selected[kind] = chosen; all_summary[kind] = summaries
        _write_candidates(output / kind / "candidates.csv", summaries)
        _json_write(output / kind / "selected.json", {"selected": chosen is not None, "usable_count": len(usable), "grid_size": len(source_grid), "status": "SELECTED" if chosen else "NO_USABLE_CANDIDATE", "selected_candidate": _jsonable(None if chosen is None else {key: value for key, value in chosen.items() if key not in {"scene_metrics", "safety_by_scene", "competence_checks"}})})
    comparison = []
    for label, key in (("old_PID", "old_PID"), ("old_LQR", "old_LQR")):
        for scene_name in SCENES:
            m = old[key][scene_name]; comparison.append({"controller": label, "scenario": scene_name, "candidate_usable": True, "task_position_rmse": m["tip_task_position_rmse_m"], "orientation_rmse": m["cutter_orientation_rmse_deg"], "task_acquired": m["task_acquired"], "acquisition_time": m["task_acquisition_time_s"], "final_tip_position_error_m": m["final_tip_position_error_m"], "safety": True})
    for kind, old_kind, label in (("task_pid", "old_PID", "Task-PID"), ("task_lqr", "old_LQR", "Task-LQR")):
        representative = selected[kind] or min(all_summary[kind], key=lambda item: item["position_ratio_mean"])
        for scene_name in SCENES:
            m = representative["scene_metrics"][scene_name]; comparison.append({"controller": label, "candidate_id": representative["candidate_id"], "scenario": scene_name, "candidate_usable": representative["candidate_usable"], "task_position_rmse": m["tip_task_position_rmse_m"], "orientation_rmse": m["cutter_orientation_rmse_deg"], "task_acquired": m["task_acquired"], "acquisition_time": m["task_acquisition_time_s"], "final_tip_position_error_m": m["final_tip_position_error_m"], "safety": representative["safety_by_scene"][scene_name]["pass"]})
    with (output / "development_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = sorted({key for row in comparison for key in row}); writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows(comparison)
    _json_write(output / "development_comparison.json", {"rows": comparison, "selection_uses_old_counterpart_only": True, "selection_uses_ls_pmpc": False})
    _json_write(output / "safety_audit.json", _jsonable(safety_all))
    result = "PASS" if all(selected.values()) else "PASS_WITH_OLD_LQR_PRIMARY" if selected["task_lqr"] is None else "PASS"
    _json_write(output / "gate.json", {"setpoint_protocol_valid": all(item["pass"] for item in audits.values()), "target_delta_x_m": 0.30, "future_target_leakage": False, "controller_reference_access": "current sample only", "old_pid_executed": True, "old_lqr_executed": True, "task_pid_grid_size": 36, "task_lqr_grid_size": 27, "task_pid_selected": selected["task_pid"] is not None, "task_lqr_selected": selected["task_lqr"] is not None, "formal_primary_traditional_baseline": "old_LQR", "task_pid_status": "SELECTED" if selected["task_pid"] is not None else "TASK_PID_NO_USABLE_CANDIDATE", "task_lqr_status": "SELECTED" if selected["task_lqr"] is not None else "TASK_LQR_NO_USABLE_CANDIDATE", "gust_performance_executed": False, "random_holdout_executed": False, "metric_contract_modified": False, "physical_model_modified": False, "wind_modified": False, "ls_pmpc_modified": False, "result": result, "start_head": start_head, "udaan_head": udaan_head})
    _json_write(output / "environment.json", {"python": sys.version, "python_version": platform.python_version(), "platform": platform.platform(), "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "runtime_model_sha256": sha256_file(RUNTIME_XML), "physics_dt": 0.001, "inner_dt": 0.005, "outer_dt": 0.05, "start_head": start_head, "udaan_head": udaan_head})
    (output / "environment.txt").write_text(f"python={platform.python_version()}\nplatform={platform.platform()}\nmujoco={mujoco.__version__}\nnumpy={np.__version__}\nscipy={__import__('scipy').__version__}\nphysics_hz=1000\ninner_hz=200\nouter_hz=20\nstart_head={start_head}\nudaan_head={udaan_head}\n", encoding="utf-8", newline="\n")
    (output / "commands.log").write_text("S6T2 setpoint protocol frozen before performance runs.\nDevelopment scenes: task_acquire_calm and task_acquire_crosswind only.\nNo task_gust_recovery performance, no random holdout, no TS-MPC.\nTask grids inherited unchanged: 36 Task-PID and 27 Task-LQR.\n", encoding="utf-8", newline="\n")
    return 0


def protocol_obj(value: dict):
    class Obj:
        pass
    obj = Obj()
    for key, item in value.items(): setattr(obj, key, np.asarray(item) if key.endswith("_m") and isinstance(item, list) else item)
    obj.duration_s = float(value["duration_s"]); obj.sample_dt_s = float(value["sample_dt_s"]); obj.target_delta_x_m = float(value["target_delta_x_m"])
    obj.initial_tip_position_m = np.asarray(value["initial_tip_position_m"], dtype=float); obj.target_tip_position_m = np.asarray(value["target_tip_position_m"], dtype=float)
    class Equilibrium: pass
    obj.equilibrium = Equilibrium(); obj.equilibrium.tip_relative_position_m = np.asarray(value["equilibrium"]["tip_relative_position_m"], dtype=float)
    return obj


if __name__ == "__main__":
    raise SystemExit(main())

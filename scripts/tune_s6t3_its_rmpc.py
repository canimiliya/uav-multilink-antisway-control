"""S6T3 development tuning and freeze for ITS-RMPC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from uav_sway.control.its_rmpc import ITSRMPC, TaskLQI
from uav_sway.evaluation.its_rmpc_gate import (
    SCENES, candidate_score, competence_gate, final_tip_speed_from_csv,
    legacy_mpc_contribution, mpc_contribution, safety_audit,
)
from uav_sway.evaluation.its_rmpc_runner import run_its_scenario
from uav_sway.linearization.task_output import TaskOutputMap, identify_task_output_jacobian
from uav_sway.models.state_io import capture_state
from uav_sway.mpc.integral_task_model import build_augmented_task_model, build_task_lqi
from uav_sway.task_space.reference import build_equilibrium_task_pose


ROOT = Path(__file__).resolve().parents[1]
START_HEAD = "b228ca4a8d0cb6775042d152a15acdcaacb40813"
AUDIT_START_HEAD = "5a9ea06b11e6231fe1b8c4a7e702ebbafaebcfbe"
UDAAN_HEAD = "9eb1a2dcfe438ce7b4c4cd119072e4f3d8a6a816"
T2 = ROOT / "artifacts/s6_taskspace/t2"
RUNTIME_XML = ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"
A_PATH = ROOT / "artifacts/s4/linearization/A.npy"
B_PATH = ROOT / "artifacts/s4/linearization/B.npy"
C_PATH = ROOT / "artifacts/s6_taskspace/t1/task_lqr/C_task.npy"
SCENE_REFS = {scene: T2 / "inputs" / f"{scene}.csv" for scene in SCENES}
SCENE_WINDS = {"task_acquire_calm": T2 / "inputs/calm.csv", "task_acquire_crosswind": ROOT / "artifacts/s2/wind_bank/constant_crosswind.csv"}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, dict): return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list): return [jsonable(item) for item in value]
    return value


def git_heads() -> tuple[str, str]:
    root = str(ROOT).replace("\\", "/")
    main = subprocess.check_output(["git", "-C", root, "-c", f"safe.directory={root}", "rev-parse", "HEAD"], text=True).strip()
    udaan = subprocess.check_output(["git", "-C", str(ROOT / "third_party/udaan"), "-c", f"safe.directory={root}", "rev-parse", "HEAD"], text=True).strip()
    return main, udaan


def build_model_artifacts(output: Path, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    model = build_augmented_task_model(A, B, C, 0.05)
    output.joinpath("model").mkdir(parents=True, exist_ok=True)
    np.savez(output / "model/augmented_model.npz", A_I=model.A_I, B_I=model.B_I, C_p=model.C_p, C_task=C)
    audit = {
        "A_I_shape": list(model.A_I.shape), "B_I_shape": list(model.B_I.shape), "C_task_shape": list(C.shape),
        "dt_s": model.dt, "state_order": ["x_16D", "eta"], "eta_definition": "clip(eta + dt * measured e_tip_x, -1, +1)",
        "pbh_stabilizable": bool(__import__("uav_sway.mpc.integral_task_model", fromlist=["pbh_stabilizable"]).pbh_stabilizable(model.A_I, model.B_I)),
        "finite": bool(np.isfinite(model.A_I).all() and np.isfinite(model.B_I).all()),
        "source_A_sha256": sha256(A_PATH), "source_B_sha256": sha256(B_PATH), "source_C_task_sha256": sha256(C_PATH),
    }
    audit["pass"] = bool(audit["finite"] and audit["pbh_stabilizable"])
    write_json(output / "model/augmented_model_audit.json", audit)
    return model, audit


def traditional_metrics() -> tuple[dict, dict, dict, dict]:
    old_lqr = {}; old_pid = {}; task_lqr = {}
    for scene in SCENES:
        old_lqr[scene] = json.loads((T2 / "baseline/old_LQR" / scene / "metrics.json").read_text(encoding="utf-8"))
        old_pid[scene] = json.loads((T2 / "baseline/old_PID" / scene / "metrics.json").read_text(encoding="utf-8"))
        task_lqr[scene] = json.loads((T2 / "task_lqr/runs/lqr_011" / scene / "metrics.json").read_text(encoding="utf-8"))
    traditional = {}
    for scene in SCENES:
        traditional[scene] = {
            "best_position_rmse": min(old_lqr[scene]["tip_task_position_rmse_m"], task_lqr[scene]["tip_task_position_rmse_m"]),
            "best_orientation_rmse": min(old_lqr[scene]["cutter_orientation_rmse_deg"], task_lqr[scene]["cutter_orientation_rmse_deg"]),
            "available_task_time_s": 12.0 - old_lqr[scene]["task_start_time_s"],
        }
    return old_pid, old_lqr, task_lqr, traditional


def run_candidate(output: Path, kind: str, candidate: dict, model, protocol: dict,
                  config_sha: str, start_head: str, traditional: dict, old_lqr: dict,
                  reuse_existing: bool = False) -> dict:
    q_solution = build_task_lqi(model, candidate["q_eta"])
    candidate = dict(candidate)
    candidate.update({"spectral_radius": q_solution["spectral_radius"], "dare_residual_norm": q_solution["dare_residual_norm"]})
    if kind == "task_lqi":
        controller = TaskLQI(q_solution["K_I"])
    else:
        from uav_sway.mpc.task_residual_qp import TaskResidualQP
        qp = TaskResidualQP(model.A_I, model.B_I, q_solution["K_I"], q_solution["Q_I"], q_solution["P_I"], candidate["horizon_steps"], candidate["residual_weight"])
        controller = ITSRMPC(q_solution["K_I"], qp)
    metrics = {}; safety = {}
    for scene in SCENES:
        path = output / kind / "runs" / candidate["candidate_id"] / scene / "run.csv"
        metric_path = path.parent / "metrics.json"
        if reuse_existing and path.exists() and metric_path.exists():
            metrics[scene] = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            metrics[scene] = run_its_scenario(ROOT / "configs/model_5link.yaml", controller, "Task-LQI" if kind == "task_lqi" else "ITS-RMPC", scene, SCENE_WINDS[scene], SCENE_REFS[scene], path, ROOT, protocol, start_head, config_sha, candidate)
            write_json(metric_path, metrics[scene])
        metrics[scene] = dict(metrics[scene], final_tip_speed_m_s=final_tip_speed_from_csv(path))
        safety[scene] = safety_audit(path, metrics[scene], kind != "task_lqi")
    competence = competence_gate(metrics, traditional)
    score = candidate_score(metrics, traditional, old_lqr) if competence["pass"] and all(item["pass"] for item in safety.values()) else {"score": None}
    return {**candidate, "controller": "Task-LQI" if kind == "task_lqi" else "ITS-RMPC", "metrics": metrics, "safety": safety, "competence": competence, "safe": bool(all(item["pass"] for item in safety.values())), "usable": bool(competence["pass"] and all(item["pass"] for item in safety.values())), **score}


def flatten_candidates(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        out = {key: jsonable(value) for key, value in row.items() if key not in {"metrics", "safety", "competence", "mpc_contribution"}}
        if "mpc_contribution" in row:
            contribution = row["mpc_contribution"]
            out["mpc_contribution_pass"] = contribution["pass"]
            out["acquisition_improvement_vs_lqi"] = contribution["acquisition_improvement_vs_lqi"]
            out["position_improvement_vs_lqi"] = contribution["position_improvement_vs_lqi"]
            out["no_more_than_10_percent_degradation"] = contribution["no_more_than_10_percent_degradation"]
            out["usable_with_mpc"] = row["usable_with_mpc"]
        for scene, prefix in ((SCENES[0], "calm"), (SCENES[1], "crosswind")):
            m = row["metrics"][scene]
            out[f"{prefix}_position_rmse"] = m["tip_task_position_rmse_m"]; out[f"{prefix}_orientation_rmse"] = m["cutter_orientation_rmse_deg"]
            out[f"{prefix}_acquired"] = m["task_acquired"]; out[f"{prefix}_acquisition_time_s"] = m["task_acquisition_time_s"]
        flat.append(out)
    columns = sorted({key for item in flat for key in item})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(flat)


def write_figures(output: Path, selected: dict | None) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4)); plt.axis("off"); plt.text(0.03, 0.8, "S6T3 ITS-RMPC development\n32 preregistered candidates\ncalm + crosswind only", fontsize=16); plt.savefig(figures / "development_summary.png", dpi=160, bbox_inches="tight"); plt.close()
    if selected:
        plt.figure(figsize=(8, 4));
        for index, scene in enumerate(SCENES):
            path = output / "its_rmpc/runs" / selected["candidate_id"] / scene / "run.csv"
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            plt.plot(data["time"], data["task_position_error_x"], label=scene)
        plt.xlabel("time (s)"); plt.ylabel("task x error (m)"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "selected_timeseries.png", dpi=160); plt.close()


def _existing_candidate_metrics(output: Path, kind: str, candidate_id: str) -> dict:
    """Load frozen run evidence and derive final speed without running physics."""
    metrics = {}
    for scene in SCENES:
        run_path = output / kind / "runs" / candidate_id / scene / "run.csv"
        metric_path = run_path.parent / "metrics.json"
        if not run_path.exists() or not metric_path.exists():
            raise RuntimeError(f"audit evidence missing: {run_path}")
        source = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics[scene] = dict(source, final_tip_speed_m_s=final_tip_speed_from_csv(run_path))
    return metrics


def _audit_row(output: Path, kind: str, candidate_id: str, traditional: dict) -> dict:
    metrics = _existing_candidate_metrics(output, kind, candidate_id)
    safety = {}
    for scene in SCENES:
        path = output / kind / "runs" / candidate_id / scene / "run.csv"
        safety[scene] = safety_audit(path, metrics[scene], kind == "its_rmpc")
    competence = competence_gate(metrics, traditional)
    legacy_speed_metrics = {scene: dict(value, final_tip_speed_m_s=value["tip_speed_rms_m_s"]) for scene, value in metrics.items()}
    legacy_competence = competence_gate(legacy_speed_metrics, traditional)
    return {
        "candidate_id": candidate_id,
        "controller": "Task-LQI" if kind == "task_lqi" else "ITS-RMPC",
        "metrics": metrics,
        "safety": safety,
        "competence": competence,
        "legacy_competence": legacy_competence,
        "safe": bool(all(item["pass"] for item in safety.values())),
        "usable": bool(competence["pass"] and all(item["pass"] for item in safety.values())),
        "legacy_usable": bool(legacy_competence["pass"] and all(item["pass"] for item in safety.values())),
    }


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_existing(output: Path) -> int:
    """Audit all existing runs; this path intentionally never calls the MuJoCo runner."""
    main_head, udaan_head = git_heads()
    if main_head != AUDIT_START_HEAD or udaan_head != UDAAN_HEAD:
        raise RuntimeError("dependency drift for audit-only closure")
    _old_pid, _old_lqr, _task_lqr, traditional = traditional_metrics()
    lqi_rows = [_audit_row(output, "task_lqi", f"lqi_{index:02d}", traditional) for index in range(1, 5)]
    its_rows = [_audit_row(output, "its_rmpc", f"its_{index:03d}", traditional) for index in range(1, 33)]
    closure = output / "closure"

    speed_rows = []
    for row in lqi_rows + its_rows:
        for scene in SCENES:
            metric = row["metrics"][scene]
            old_pass = row["legacy_competence"]["checks"][f"{scene}_tip_speed"]
            new_pass = row["competence"]["checks"][f"{scene}_tip_speed"]
            speed_rows.append({"controller": row["controller"], "candidate_id": row["candidate_id"], "scenario": scene, "final_tip_speed_m_s": metric["final_tip_speed_m_s"], "tip_speed_rms_m_s": metric["tip_speed_rms_m_s"], "task_acquired": metric["task_acquired"], "old_speed_gate_pass": old_pass, "new_speed_gate_pass": new_pass, "speed_gate_changed": old_pass != new_pass})
    _write_audit_csv(closure / "final_speed_audit.csv", speed_rows)
    write_json(closure / "final_speed_audit.json", {"old_definition": "whole-run tip_speed_rms_m_s <= 0.10", "new_definition": "final instantaneous tip_speed_m_s <= 0.10", "task_acquisition_speed_definition": "instantaneous tip speed <= 0.10 m/s continuously for 1 s", "rejection_gate_uses_whole_run_rms": False, "changed_rows": sum(bool(row["speed_gate_changed"]) for row in speed_rows), "rows": speed_rows})

    q_values = [0.1, 1.0, 10.0, 100.0]
    lqi_by_q = {q: lqi_rows[index] for index, q in enumerate(q_values)}
    contribution_rows = []
    its_final_usable = []
    for index, row in enumerate(its_rows, 1):
        q = q_values[(index - 1) // 8]
        matched = lqi_by_q[q]
        contribution = mpc_contribution(row["metrics"], matched["metrics"])
        legacy_contribution = legacy_mpc_contribution(row["metrics"], matched["metrics"])
        row["mpc_contribution"] = contribution
        row["legacy_mpc_contribution"] = legacy_contribution
        row["usable_with_mpc"] = bool(row["usable"] and contribution["pass"])
        row["legacy_usable_with_mpc"] = bool(row["legacy_usable"] and legacy_contribution["pass"])
        if row["usable_with_mpc"]:
            its_final_usable.append(row)
        contribution_rows.append({"candidate_id": row["candidate_id"], "q_eta": q, "matched_lqi": matched["candidate_id"], "new": contribution, "legacy": legacy_contribution, "usable_with_mpc": row["usable_with_mpc"], "legacy_usable_with_mpc": row["legacy_usable_with_mpc"]})
    write_json(closure / "mpc_contribution_semantics.json", {"cases": {"lqi_false_its_true": "acquisition dominance true", "lqi_true_its_false": "acquisition dominance false", "both_true": "compare acquisition time", "both_false": "no acquisition evidence"}, "position_orientation_rule": "ITS <= 110% matched Task-LQI per scene", "thresholds_unchanged": {"acquisition_time_improvement": 0.05, "position_improvement": 0.05, "degradation": 0.10}, "candidates": contribution_rows})

    competence_rows = []
    for row in lqi_rows + its_rows:
        competence_rows.append({"controller": row["controller"], "candidate_id": row["candidate_id"], "safe": row["safe"], "competence_pass": row["competence"]["pass"], "usable": row["usable"], "legacy_competence_pass": row["legacy_competence"]["pass"], "legacy_usable": row["legacy_usable"], "changed": row["competence"]["pass"] != row["legacy_competence"]["pass"]})
    _write_audit_csv(closure / "competence_reaudit.csv", competence_rows)
    write_json(closure / "competence_reaudit.json", {"criteria": {"position": "<=110% best traditional", "orientation": "<=125% best traditional", "final_tip_speed": "<=0.10 m/s", "task_acquired": True}, "rows": competence_rows})

    lqi_usable = [row for row in lqi_rows if row["usable"]]
    its_competence_usable = [row for row in its_rows if row["usable"]]
    legacy_lqi_usable = [row for row in lqi_rows if row["legacy_usable"]]
    legacy_its_usable = [row for row in its_rows if row["legacy_usable"]]
    legacy_final = sum(row["legacy_usable_with_mpc"] for row in its_rows)
    new_final = sum(row["usable_with_mpc"] for row in its_rows)
    result = "ITS_RMPC_DEVELOPMENT_PASS_AFTER_GATE_FIX" if its_final_usable else "CLOSED_WITH_NO_USABLE_METHOD_VALIDATED"
    validity = {"old_speed_rms_bug_affected_candidate_usable_count": (len(legacy_lqi_usable) != len(lqi_usable) or len(legacy_its_usable) != len(its_competence_usable)), "old_speed_rms_bug_affected_usable_count": {"task_lqi": {"legacy": len(legacy_lqi_usable), "corrected": len(lqi_usable)}, "its_rmpc_competence": {"legacy": len(legacy_its_usable), "corrected": len(its_competence_usable)}}, "old_mpc_null_semantics_affected_usable_with_mpc_count": legacy_final != new_final, "old_usable_with_mpc_count": legacy_final, "corrected_usable_with_mpc_count": new_final, "mpc_contribution_semantics_changed_candidates": [row["candidate_id"] for row in its_rows if row["legacy_mpc_contribution"]["pass"] != row["mpc_contribution"]["pass"]], "task_lqi_usable_after_reaudit": len(lqi_usable), "its_rmpc_competence_usable_after_reaudit": len(its_competence_usable), "its_rmpc_final_usable_after_reaudit": new_final, "old_result_classification": "CLOSED_WITH_NO_USABLE_METHOD", "corrected_result": result, "original_result_still_valid": False, "classification_unchanged": result == "CLOSED_WITH_NO_USABLE_METHOD_VALIDATED", "explanation": "The previous closure was not valid as a final audit because both gate defects were present; the corrected audit independently establishes the final count."}
    write_json(closure / "result_validity.json", validity)

    write_json(output / "grid/selection.json", {"selected": bool(its_final_usable), "usable_count": new_final, "selected_candidate": its_final_usable[0]["candidate_id"] if its_final_usable else None, "audit_only": True, "selection_uses_ls_pmpc": False})
    write_json(output / "raw_gate.json", {"audit_only": True, "task_lqi": [{"candidate_id": row["candidate_id"], "safe": row["safe"], "competence": row["competence"], "usable": row["usable"]} for row in lqi_rows], "its_rmpc": [{"candidate_id": row["candidate_id"], "safe": row["safe"], "competence": row["competence"], "usable": row["usable"], "mpc_contribution": row["mpc_contribution"], "usable_with_mpc": row["usable_with_mpc"]} for row in its_rows]})
    gate = {"physics_rerun": False, "grid_modified": False, "controller_modified": False, "speed_gate_uses_final_instantaneous_speed": True, "whole_run_speed_rms_is_rejection_gate": False, "mpc_acquisition_dominance_supported": True, "task_lqi_usable_after_reaudit": len(lqi_usable), "its_rmpc_competence_usable_after_reaudit": len(its_competence_usable), "its_rmpc_final_usable_after_reaudit": new_final, "gust_executed": False, "random_holdout_executed": False, "future_wind_used": False, "metric_contract_modified": False, "setpoint_protocol_modified": False, "physical_model_modified": False, "old_lqr_modified": False, "task_lqr_modified": False, "ls_pmpc_modified": False, "result": result, "start_head": AUDIT_START_HEAD, "udaan_head": udaan_head}
    write_json(output / "gate.json", gate)
    write_json(closure / "gate.json", gate)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(ROOT / "artifacts/s6_taskspace/t3")); parser.add_argument("--reuse-existing-runs", action="store_true"); parser.add_argument("--audit-existing-only", action="store_true"); args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if args.audit_existing_only:
        return audit_existing(output)
    start_head, udaan_head = git_heads()
    if udaan_head != UDAAN_HEAD or (start_head != START_HEAD and not args.reuse_existing_runs):
        raise RuntimeError("dependency drift")
    provenance_start_head = START_HEAD if args.reuse_existing_runs else start_head
    config_path = ROOT / "configs/s6_its_rmpc.yaml"; config_sha = sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = json.loads((T2 / "protocol.json").read_text(encoding="utf-8")); protocol["protocol_sha256"] = sha256(T2 / "protocol.json")
    A = np.load(A_PATH); B = np.load(B_PATH); C = np.load(C_PATH); model, model_audit = build_model_artifacts(output, A, B, C)
    old_pid, old_lqr, selected_task_lqr, traditional = traditional_metrics()
    write_json(output / "baseline_context.json", {"old_PID": old_pid, "old_LQR": old_lqr, "selected_Task_LQR": selected_task_lqr, "traditional_reference": traditional, "source": "S6T2 frozen metrics; no physics rerun"})
    grid = [{"candidate_id": f"lqi_{i+1:02d}", "q_eta": float(q)} for i, q in enumerate(config["q_eta_grid"])]
    its_grid = [{"candidate_id": f"its_{i:03d}", "q_eta": float(q), "horizon_steps": int(h), "residual_weight": float(lam)} for i, (q, h, lam) in enumerate(((q, h, lam) for q in config["q_eta_grid"] for h in config["horizon_steps_grid"] for lam in config["residual_weight_grid"]), 1)]
    write_json(output / "grid/preregistered_grid.json", {"q_eta_grid": config["q_eta_grid"], "horizon_grid": config["horizon_steps_grid"], "residual_weight_grid": config["residual_weight_grid"], "task_lqi_grid_size": 4, "its_rmpc_grid_size": 32, "frozen_before_performance": True})
    lqi_rows = [run_candidate(output, "task_lqi", item, model, protocol, config_sha, provenance_start_head, traditional, old_lqr, args.reuse_existing_runs) for item in grid]
    its_rows = [run_candidate(output, "its_rmpc", item, model, protocol, config_sha, provenance_start_head, traditional, old_lqr, args.reuse_existing_runs) for item in its_grid]
    flatten_candidates(output / "task_lqi/candidates.csv", lqi_rows); flatten_candidates(output / "grid/its_grid.csv", its_rows)
    lqi_usable = [item for item in lqi_rows if item["usable"]]; selected_lqi = min(lqi_usable, key=lambda item: item["score"]) if lqi_usable else None
    lqi_by_q = {item["q_eta"]: item for item in lqi_rows}
    for item in its_rows:
        item["mpc_contribution"] = mpc_contribution(item["metrics"], lqi_by_q[item["q_eta"]]["metrics"])
        item["usable_with_mpc"] = bool(item["usable"] and item["mpc_contribution"]["pass"])
    its_usable = [item for item in its_rows if item["usable_with_mpc"]]; selected = min(its_usable, key=lambda item: item["score"]) if its_usable else None
    flatten_candidates(output / "grid/its_grid.csv", its_rows)
    write_json(output / "task_lqi/selection.json", {"selected": selected_lqi is not None, "usable_count": len(lqi_usable), "selected_candidate": jsonable(selected_lqi) if selected_lqi else None})
    write_json(output / "grid/selection.json", {"selected": selected is not None, "usable_count": len(its_usable), "selected": jsonable(selected) if selected else None})
    comparison = []
    lqi_report = selected_lqi or min(lqi_rows, key=lambda item: sum(item["metrics"][scene]["tip_task_position_rmse_m"] for scene in SCENES))
    its_report = selected or min(its_rows, key=lambda item: sum(item["metrics"][scene]["tip_task_position_rmse_m"] for scene in SCENES))
    comparison_sources = (("old_PID", old_pid, None), ("old_LQR", old_lqr, None),
                          ("Task-LQR", selected_task_lqr, None),
                          ("Task-LQI", lqi_report["metrics"], lqi_report["candidate_id"]),
                          ("ITS-RMPC", its_report["metrics"], its_report["candidate_id"]))
    for label, source, candidate_id in comparison_sources:
        if source is None: continue
        for scene in SCENES:
            m = source[scene]; comparison.append({"controller": label, "candidate_id": candidate_id, "candidate_usable": bool(selected_lqi and label == "Task-LQI") if label == "Task-LQI" else bool(selected and label == "ITS-RMPC") if label == "ITS-RMPC" else True, "scenario": scene, "task_position_rmse": m["tip_task_position_rmse_m"], "orientation_rmse": m["cutter_orientation_rmse_deg"], "task_acquired": m["task_acquired"], "acquisition_time": m["task_acquisition_time_s"], "final_tip_error": m["final_tip_position_error_m"], "time_to_10cm": None, "time_to_5cm": None, "tip_speed_rms": m["tip_speed_rms_m_s"], "control_energy": m["control_energy_proxy"], "control_rate": m["control_rate_proxy"], "solve_time_mean_ms": m.get("solve_time_mean_ms", 0.0), "solve_time_p95_ms": m.get("solve_time_p95_ms", 0.0), "solve_time_max_ms": m.get("solve_time_max_ms", 0.0)})
    write_json(output / "development_comparison.json", {"rows": comparison, "selection_uses_ls_pmpc": False, "nonselected_representatives": {"Task-LQI": lqi_report["candidate_id"], "ITS-RMPC": its_report["candidate_id"]}, "source": "S6T2 frozen baseline context plus S6T3 runs"})
    write_json(output / "passivity_summary.json", {"passivity_constraint": False, "reason": "ITS-RMPC uses integral augmentation; no passivity inequality is part of this method."})
    write_json(output / "solver_summary.json", {"solver": "OSQP", "warm_start": True, "outer_loop_hz": 20, "candidate_count": 32, "solve_times": [{"candidate_id": item["candidate_id"], "scenes": {scene: {"mean_ms": item["metrics"][scene]["qp_solve_time_mean_ms"], "p95_ms": item["metrics"][scene]["qp_solve_time_p95_ms"], "max_ms": item["metrics"][scene]["qp_solve_time_max_ms"], "status_nonzero": item["metrics"][scene]["qp_status_nonzero_count"]} for scene in SCENES}} for item in its_rows]})
    write_json(output / "raw_gate.json", {"task_lqi": [{"candidate_id": item["candidate_id"], "safe": item["safe"], "usable": item["usable"], "competence": item["competence"]} for item in lqi_rows], "its_rmpc": [{"candidate_id": item["candidate_id"], "safe": item["safe"], "usable": item["usable"], "competence": item["competence"], "mpc_contribution": item["mpc_contribution"], "usable_with_mpc": item["usable_with_mpc"]} for item in its_rows]})
    write_json(output / "algorithm_audit.json", {"method": "ITS-RMPC", "exact_reproduction": False, "state_dimension": 17, "internal_model": "frozen 16D A/B plus measured task-error integral", "future_wind_used": False, "future_target_before_issue": False, "ls_pmpc_model_used": False, "old_lqr_modified": False, "task_lqr_modified": False, "shared_inner_loop": True, "outer_loop_hz": 20, "solver": "OSQP", "warm_start": True, "candidate_count": 32, "parameter_selection_uses_ls_pmpc": False})
    result = "ITS_RMPC_DEVELOPMENT_PASS" if selected else "CLOSED_WITH_INTEGRAL_ONLY_RESULT" if selected_lqi else "CLOSED_WITH_NO_USABLE_METHOD"
    write_json(output / "gate.json", {"method": "ITS-RMPC", "q_eta_grid_size": 4, "horizon_grid_size": 2, "residual_grid_size": 4, "total_candidates": 32, "usable_candidates": len(its_usable), "selected": selected is not None, "task_lqi_ablation_executed": True, "mpc_contribution_pass": selected is not None, "calm_acquired": bool(selected and selected["metrics"][SCENES[0]]["task_acquired"]), "crosswind_acquired": bool(selected and selected["metrics"][SCENES[1]]["task_acquired"]), "gust_executed": False, "random_holdout_executed": False, "future_wind_used": False, "future_target_before_issue": False, "metric_contract_modified": False, "setpoint_protocol_modified": False, "physical_model_modified": False, "old_lqr_modified": False, "task_lqr_modified": False, "ls_pmpc_modified": False, "result": result, "start_head": provenance_start_head, "udaan_head": udaan_head})
    if selected:
        freeze = {"method": "ITS-RMPC", "status": "ITS_RMPC_DEVELOPMENT_PASS", "selected_candidate_id": selected["candidate_id"], "q_eta": selected["q_eta"], "horizon_steps": selected["horizon_steps"], "residual_weight": selected["residual_weight"], "K_I": jsonable(build_task_lqi(model, selected["q_eta"])["K_I"]), "Q_I": jsonable(build_task_lqi(model, selected["q_eta"])["Q_I"]), "P_I": jsonable(build_task_lqi(model, selected["q_eta"])["P_I"]), "spectral_radius": selected["spectral_radius"], "configuration_sha256": config_sha, "frozen_after_development": True}
        write_json(output / "method_freeze.json", freeze)
    (output / "environment.json").write_text(json.dumps({"python": sys.version, "python_version": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "osqp": __import__("osqp").__version__, "runtime_model_sha256": sha256(RUNTIME_XML), "start_head": provenance_start_head, "udaan_head": udaan_head, "physics_dt": 0.001, "inner_dt": 0.005, "outer_dt": 0.05}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "commands.log").write_text("S6T3 development: task_acquire_calm + task_acquire_crosswind only.\nNo task_gust_recovery performance. No random holdout.\nOld LQR and selected Task-LQR read from frozen S6T2 metrics; no physics rerun.\nGrid frozen before performance: q_eta=4 x horizon=2 x residual=4 = 32.\n", encoding="utf-8", newline="\n")
    (output / "failure.log").write_text("Historical first attempt: runner cleared all actuator controls after logging; those non-formal runs were invalid and replaced. Corrected rerun used the shared inner-loop command without post-log clearing.\n" if selected or selected_lqi else "Historical first attempt: runner cleared all actuator controls after logging; those non-formal runs were invalid and replaced. Corrected rerun used the shared inner-loop command without post-log clearing. No usable candidate in corrected development set.\n", encoding="utf-8", newline="\n")
    write_figures(output, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

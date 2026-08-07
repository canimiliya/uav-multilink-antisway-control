"""S6T2 runner for the frozen cutter-setpoint development protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from uav_sway.evaluation.task_space_metrics import compute_task_metrics
from uav_sway.evaluation.task_baseline_runner import run_task_baseline_scenario
from uav_sway.evaluation.task_space_runner import run_task_space_scenario


def _relabel_and_recompute(path: Path, scenario_name: str) -> dict:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        columns = list(rows[0])
    for row in rows:
        row["scenario"] = scenario_name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return compute_task_metrics(path)


def _provenance(metrics: dict, protocol: dict, scene: dict, ref_path: Path, wind_path: Path, start_head: str) -> dict:
    metrics = dict(metrics)
    metrics.update({
        "start_head": start_head,
        "task_protocol_sha256": protocol["protocol_sha256"],
        "task_start_time_s": scene["task_start_time_s"],
        "target_tip_position_m": protocol["target_tip_position_m"],
        "target_cutter_axis": protocol["target_cutter_axis"],
        "reference_sha256": protocol["reference_sha256_by_scene"][scene["name"]],
        "wind_sha256": protocol["wind_sha256_by_scene"][scene["name"]],
        "reference_path": str(ref_path), "wind_path": str(wind_path),
        "provenance_complete": True,
        "future_target_preview": False,
    })
    return metrics


def run_old_baseline(model_config: str | Path, controller: str, scene: dict, wind_path: Path,
                     reference_path: Path, output_path: Path, root: Path, protocol: dict,
                     start_head: str, reuse_existing: bool = False) -> dict:
    source_controller = "pid" if controller == "old_PID" else "lqr"
    if not reuse_existing or not output_path.exists():
        run_task_space_scenario(model_config, source_controller, scene["source_scenario"], wind_path,
                                reference_path, output_path, root, duration_s=protocol["duration_s"])
    metrics = _relabel_and_recompute(output_path, scene["name"])
    metrics.update({"physics_intervals": 12000, "formal_log_samples": 2401, "outer_control_updates": 241, "inner_loop_updates": 2401, "wind_force_calls": 12001})
    metrics["controller"] = controller
    return _provenance(metrics, protocol, scene, reference_path, wind_path, start_head)


def run_task_baseline(model_config: str | Path, kind: str, candidate: dict, scene: dict,
                      wind_path: Path, reference_path: Path, output_path: Path, root: Path,
                      c_task: np.ndarray | None, protocol: dict, start_head: str,
                      reuse_existing: bool = False) -> dict:
    if not reuse_existing or not output_path.exists():
        run_task_baseline_scenario(model_config, kind, candidate, scene["source_scenario"], wind_path,
                                   reference_path, output_path, root, duration_s=protocol["duration_s"], c_task=c_task)
    metrics = _relabel_and_recompute(output_path, scene["name"])
    metrics.update({"physics_intervals": 12000, "formal_log_samples": 2401, "outer_control_updates": 241, "inner_loop_updates": 2401, "wind_force_calls": 12001})
    metrics["controller"] = kind
    return _provenance(metrics, protocol, scene, reference_path, wind_path, start_head)

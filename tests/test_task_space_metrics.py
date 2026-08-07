import csv

import numpy as np

from uav_sway.evaluation.metrics import control_rate_proxy as legacy_control_rate_proxy
from uav_sway.evaluation.task_space_metrics import compute_task_metrics


def _write(path, n=41):
    columns = ["time", "scenario", "controller", "wind_x", "x_ref", "z_ref", "uav_x", "uav_z", "tip_displacement", "ax_cmd_limited", "solve_time_ms", "tip_task_x_error_m", "tip_task_z_error_m", "task_position_error_xz_m", "position_error_3d_m", "orientation_error_deg", "tip_speed_m_s"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for i in range(n):
            t = i * 0.05
            writer.writerow({"time": t, "scenario": "approach_stop", "controller": "pid", "wind_x": 0, "x_ref": 0, "z_ref": 0, "uav_x": 0, "uav_z": 0, "tip_displacement": 0, "ax_cmd_limited": 0, "solve_time_ms": 1, "tip_task_x_error_m": 0, "tip_task_z_error_m": 0, "task_position_error_xz_m": 0, "position_error_3d_m": 0, "orientation_error_deg": 0, "tip_speed_m_s": 0})


def test_zero_task_error_metrics_are_finite(tmp_path):
    path = tmp_path / "run.csv"; _write(path)
    metrics = compute_task_metrics(path)
    assert metrics["task_acquired"]
    assert np.isclose(metrics["tip_task_position_rmse_m"], 0.0)
    assert metrics["uav_metrics_secondary"] is True
    assert metrics["finite_outputs"] is True


def test_formal_acquisition_time_is_elapsed_from_task_start(tmp_path):
    path = tmp_path / "acquisition.csv"
    columns = ["time", "scenario", "controller", "reference_event", "wind_x", "x_ref", "z_ref", "uav_x", "uav_z", "tip_displacement", "ax_cmd_limited", "solve_time_ms", "tip_task_x_error_m", "tip_task_z_error_m", "task_position_error_xz_m", "position_error_3d_m", "orientation_error_deg", "tip_speed_m_s"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for i in range(81):
            t = i * 0.05
            ready = t >= 2.5
            writer.writerow({"time": t, "scenario": "approach_stop", "controller": "pid", "reference_event": "active" if t >= 1.0 else "hover", "wind_x": 0, "x_ref": 0, "z_ref": 0, "uav_x": 0, "uav_z": 0, "tip_displacement": 0, "ax_cmd_limited": 0, "solve_time_ms": 1, "tip_task_x_error_m": 0 if ready else 1, "tip_task_z_error_m": 0, "task_position_error_xz_m": 0 if ready else 1, "position_error_3d_m": 0 if ready else 1, "orientation_error_deg": 0, "tip_speed_m_s": 0})
    metrics = compute_task_metrics(path)
    assert np.isclose(metrics["task_start_time_s"], 1.0)
    assert np.isclose(metrics["task_acquisition_timestamp_s"], 2.5)
    assert np.isclose(metrics["task_acquisition_time_s"], 1.5)


def test_task_space_control_rate_matches_legacy_uniform_nonuniform_and_constant(tmp_path):
    cases = (
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.0])),
        (np.array([0.0, 0.5, 2.0]), np.array([0.0, 1.0, 4.0])),
        (np.array([0.0, 1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0, 2.0])),
    )
    for index, (time, command) in enumerate(cases):
        path = tmp_path / f"rate_{index}.csv"
        _write_rate_case(path, time, command)
        metrics = compute_task_metrics(path)
        assert metrics["control_rate_proxy"] == legacy_control_rate_proxy(time, command)


def _write_rate_case(path, time, command):
    columns = ["time", "scenario", "controller", "wind_x", "x_ref", "z_ref", "uav_x", "uav_z", "tip_displacement", "ax_cmd_limited", "solve_time_ms", "tip_task_x_error_m", "tip_task_z_error_m", "task_position_error_xz_m", "position_error_3d_m", "orientation_error_deg", "tip_speed_m_s"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for t, u in zip(time, command):
            writer.writerow({"time": t, "scenario": "approach_stop", "controller": "pid", "wind_x": 0, "x_ref": 0, "z_ref": 0, "uav_x": 0, "uav_z": 0, "tip_displacement": 0, "ax_cmd_limited": u, "solve_time_ms": 1, "tip_task_x_error_m": 0, "tip_task_z_error_m": 0, "task_position_error_xz_m": 0, "position_error_3d_m": 0, "orientation_error_deg": 0, "tip_speed_m_s": 0})

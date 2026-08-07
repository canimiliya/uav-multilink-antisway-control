import csv

import numpy as np

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
    assert metrics["secondary_metric"] is True
    assert metrics["finite_outputs"] is True

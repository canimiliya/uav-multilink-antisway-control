import csv
from pathlib import Path

import numpy as np
import pytest

from uav_sway.evaluation.metrics import compute_metrics
from uav_sway.evaluation.schema import schema_columns


def _write_rows(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=schema_columns(1), lineterminator="\n")
        writer.writeheader()
        for row in values:
            writer.writerow(row)


def _row(time: float, displacement: float, x_ref: float = 0.0, command: float = 0.0, saturated: bool = False, uav_error: float = 0.0) -> dict:
    row = {column: 0 for column in schema_columns(1)}
    row.update({"time": time, "scenario": "test", "seed": 0, "protocol_mode": "anchored_wind_validation", "x_ref": x_ref, "y_ref": 0, "z_ref": 0, "tip_displacement": displacement, "tip_z": 1, "uav_x": x_ref + uav_error, "uav_y": 0, "uav_z": 0, "ax_cmd_limited": command, "ax_saturated": saturated})
    return row


def test_metrics_recompute_rms_rmse_and_control_proxies(tmp_path):
    path = tmp_path / "run.csv"
    _write_rows(path, [_row(0, 0.5, command=1), _row(1, 0.5, x_ref=0.1, command=2, saturated=True, uav_error=0.2), _row(2, 0.0, x_ref=0.1)])
    metrics = compute_metrics(path, settling_start_s=1.0)
    assert metrics["tip_max_abs_m"] == 0.5
    assert metrics["tip_rms_m"] == pytest.approx(np.sqrt(0.375 / 2.0))
    assert metrics["uav_position_rmse_m"] == pytest.approx(np.sqrt(0.04 / 2.0))
    assert metrics["control_energy_proxy"] == 4.5
    assert metrics["control_rate_proxy"] == 2.5
    assert metrics["saturation_rate"] == 1 / 3
    assert metrics["finite_outputs"] is True


def test_metrics_reports_unsettled_signal(tmp_path):
    path = tmp_path / "run.csv"
    _write_rows(path, [_row(float(i), 0.1) for i in range(0, 3)])
    metrics = compute_metrics(path, settling_start_s=0.0)
    assert metrics["settled"] is False
    assert metrics["settling_time_s"] is None

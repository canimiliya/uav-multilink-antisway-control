from pathlib import Path
import json
import numpy as np

from uav_sway.evaluation.controlled_metrics import compute_controlled_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_three_lqr_raw_runs_are_finite_and_free_flight():
    for scenario in ("approach_stop", "crosswind_hover", "gust_micro_adjust"):
        path = ROOT / "artifacts/s4/runs" / scenario / "run.csv"
        metrics = compute_controlled_metrics(path)
        assert metrics["finite_outputs"]
        assert not metrics["anchor_active_any"]
        assert metrics["minimum_uav_height_m"] > 0.05
        assert metrics["minimum_tip_height_m"] > 0.05
        assert metrics["maximum_abs_pitch_rad"] < np.deg2rad(25.0)


def test_s4_selection_records_the_safe_repair_candidate():
    selection = json.loads((ROOT / "artifacts/s4/tuning/lqr_selection.json").read_text(encoding="utf-8"))
    assert selection["selection_status"] == "SELECTED_SAFE_CANDIDATE"
    assert selection["grid_size"] == 64
    assert selection["gust_used_for_selection"] is False
    assert selection["selected"]["safe_gate"] is True

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_three_s3_scenarios_have_finite_raw_gate_results():
    gate = json.loads((ROOT / "artifacts/s3/raw_gate.json").read_text(encoding="utf-8"))
    assert gate["pass"] is True
    assert gate["residual_sway_confirmed"] is True
    for scenario in ("approach_stop", "crosswind_hover", "gust_micro_adjust"):
        metrics = json.loads((ROOT / "artifacts/s3/runs" / scenario / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["finite_outputs"] is True
        assert metrics["anchor_active"] is False
        assert metrics["minimum_tip_height_m"] > 0.05

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_s3_scheduler_counts_are_frozen():
    for scenario in ("approach_stop", "crosswind_hover", "gust_micro_adjust"):
        metrics = json.loads((ROOT / "artifacts/s3/runs" / scenario / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["physics_intervals"] == 12000
        assert metrics["formal_log_samples"] == 2401
        assert metrics["outer_control_updates"] == 241
        assert metrics["inner_loop_updates"] == 2401
        assert metrics["wind_force_calls"] == 12001

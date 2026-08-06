import json
from pathlib import Path

import numpy as np

from uav_sway.evaluation.controlled_metrics import load_controlled_csv


ROOT = Path(__file__).resolve().parents[1]


def test_raw_csv_independently_satisfies_common_limits():
    gate = json.loads((ROOT / "artifacts/s3/raw_gate.json").read_text(encoding="utf-8"))
    assert gate["source"] == "independent_raw_csv_recomputation"
    for scenario in ("approach_stop", "crosswind_hover", "gust_micro_adjust"):
        _, v = load_controlled_csv(ROOT / "artifacts/s3/runs" / scenario / "run.csv")
        assert np.isfinite(v["uav_x"]).all()
        assert np.all(v["anchor_active"] == False)
        assert np.max(np.abs(v["ax_cmd_limited"])) <= 2.0 + 1e-12
        assert np.max(np.abs(np.diff(v["ax_cmd_limited"]))) <= 0.25 + 1e-12
        assert np.min(v["tip_z"]) > 0.05

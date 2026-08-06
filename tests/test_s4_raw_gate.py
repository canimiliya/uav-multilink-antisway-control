from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_raw_gate_is_independent_and_exposes_fairness_block():
    gate = json.loads((ROOT / "artifacts/s4/raw_gate.json").read_text(encoding="utf-8"))
    assert gate["source"] == "independent_raw_csv_recomputation"
    assert gate["pass"] is False
    assert gate["scenarios"]["crosswind_hover"]["checks"]["position_fairness"] is False

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_raw_gate_is_independent_and_records_all_repair_gates():
    gate = json.loads((ROOT / "artifacts/s4/raw_gate.json").read_text(encoding="utf-8"))
    assert gate["source"] == "independent_raw_csv_recomputation"
    assert gate["pass"] is True
    assert gate["local_linearization_pass"] is True
    assert gate["operating_region_validation_reported"] is True
    assert gate["scoring_formula_correct"] is True
    assert gate["grid_size"] == 64
    assert gate["selected_candidate_is_safe"] is True
    assert gate["scenarios"]["crosswind_hover"]["checks"]["position_fairness"] is True

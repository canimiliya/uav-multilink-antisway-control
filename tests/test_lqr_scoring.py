import json
from pathlib import Path

from uav_sway.control.full_state_lqr import lqr_candidate_score


ROOT = Path(__file__).resolve().parents[1]


def test_all_score_terms_are_penalties():
    base = lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0])
    assert lqr_candidate_score([2, 1], [1, 1], [1, 1], [0, 0]) > base
    assert lqr_candidate_score([1, 1], [2, 1], [1, 1], [0, 0]) > base
    assert lqr_candidate_score([1, 1], [1, 1], [2, 1], [0, 0]) > base
    assert lqr_candidate_score([1, 1], [1, 1], [1, 1], [0, 0.5]) > base


def test_production_scoring_audit_passes():
    audit = json.loads((ROOT / "artifacts/s4/repair/scoring_formula_audit.json").read_text(encoding="utf-8"))
    assert audit["pass"] is True
    assert audit["computed_uniform_case"] == 1.3

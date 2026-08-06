from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_controllability_and_pbh_are_saved():
    report = json.loads((ROOT / "artifacts/s4/linearization/controllability.json").read_text(encoding="utf-8"))
    assert 0 < report["rank"] <= 16
    assert len(report["singular_values"]) == 16
    assert report["pbh_stabilizable"] is True

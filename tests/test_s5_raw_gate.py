import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_s5_raw_gate_declares_independent_source_when_available():
    gate = ROOT / "artifacts/s5/raw_gate.json"
    if not gate.exists():
        return
    payload = json.loads(gate.read_text(encoding="utf-8"))
    assert payload["source"] == "independent_raw_csv_recomputation"

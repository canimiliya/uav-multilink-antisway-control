import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_s5_artifacts_have_three_scenarios_when_available():
    gate = ROOT / "artifacts/s5/raw_gate.json"
    if not gate.exists():
        return
    payload = json.loads(gate.read_text(encoding="utf-8"))
    if payload.get("status") == "BLOCKED_NO_SAFE_MPPI":
        return
    assert set(payload["scenarios"]) == {"approach_stop", "crosswind_hover", "gust_micro_adjust"}

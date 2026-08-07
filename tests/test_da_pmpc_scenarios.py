from pathlib import Path

def test_only_development_scenarios_are_declared():
    text=Path("configs/da_pmpc.yaml").read_text(encoding="utf-8")
    assert "approach_stop" in text and "crosswind_hover" in text
    assert "development_scenarios: [approach_stop, crosswind_hover]" in text

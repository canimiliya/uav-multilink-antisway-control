from pathlib import Path
import yaml

def test_only_development_scenarios_are_declared():
    config = yaml.safe_load(Path("configs/da_pmpc.yaml").read_text(encoding="utf-8"))
    assert config["development_scenarios"] == ["approach_stop", "crosswind_hover"]

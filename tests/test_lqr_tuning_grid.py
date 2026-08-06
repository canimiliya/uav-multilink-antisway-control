import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_original_and_repair_grids_are_both_retained():
    assert len(_rows(ROOT / "artifacts/s4/tuning/lqr_grid_original_27.csv")) == 27
    repair = _rows(ROOT / "artifacts/s4/tuning/lqr_grid_repair_64.csv")
    assert len(repair) == 64
    assert len({row["index"] for row in repair}) == 64
    assert all(row["safe_gate"] in {"True", "False"} for row in repair)


def test_selection_uses_only_two_development_scenarios():
    selection = json.loads((ROOT / "artifacts/s4/tuning/lqr_selection.json").read_text(encoding="utf-8"))
    assert selection["development_scenarios"] == ["approach_stop", "crosswind_hover"]
    assert selection["gust_used_for_selection"] is False
    assert selection["grid_size"] == 64

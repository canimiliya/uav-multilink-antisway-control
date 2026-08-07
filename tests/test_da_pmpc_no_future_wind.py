from pathlib import Path

def test_controller_api_has_no_future_wind_parameter():
    text=Path("src/uav_sway/control/da_pmpc.py").read_text(encoding="utf-8")
    assert "wind_future" not in text and "future_wind" not in text and "wind_csv" not in text

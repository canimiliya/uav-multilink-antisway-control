import json
from pathlib import Path

import numpy as np
import pandas as pd

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.passive_sim import simulate_passive


ROOT = Path(__file__).resolve().parents[1]


def test_rms_uses_static_equilibrium_not_endpoint_subtraction():
    values = np.array([0.5, 0.4, 0.6])
    equilibrium = 0.0
    correct = float(np.sqrt(np.mean((values - equilibrium) ** 2)))
    endpoint_subtraction = float(np.sqrt(np.mean((values - values[-1]) ** 2)))
    assert np.isclose(correct, np.sqrt(np.mean(values**2)))
    assert not np.isclose(correct, endpoint_subtraction)


def test_five_link_ten_degree_passive_decay(tmp_path):
    xml_path = tmp_path / "model_5link.xml"
    csv_path = tmp_path / "passive_decay_5link.csv"
    metrics_path = tmp_path / "passive_decay_metrics.json"
    build_planar_chain_model(ROOT / "configs/model_5link.yaml", xml_path)
    result = simulate_passive(
        ROOT / "configs/model_5link.yaml",
        10.0,
        15.0,
        csv_path=csv_path,
        metrics_path=metrics_path,
        model_path=xml_path,
    )
    assert result["finite"]
    assert result["final_total_energy_j"] < result["initial_total_energy_j"]
    assert result["final_tip_rms_m"] < result["initial_tip_rms_m"]
    assert result["decay_ratio"] < 1.0
    assert result["max_abs_joint_angle_rad"] < 1.7453293
    assert result["min_tip_z_m"] > 0.0


def test_csv_rms_recomputes_metrics_independently(tmp_path):
    xml_path = tmp_path / "model_5link.xml"
    csv_path = tmp_path / "passive_decay_5link.csv"
    metrics_path = tmp_path / "passive_decay_metrics.json"
    build_planar_chain_model(ROOT / "configs/model_5link.yaml", xml_path)
    simulate_passive(
        ROOT / "configs/model_5link.yaml",
        10.0,
        15.0,
        csv_path=csv_path,
        metrics_path=metrics_path,
        model_path=xml_path,
    )
    df = pd.read_csv(csv_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    initial = df[(df.time >= 0.0) & (df.time < 2.0)]
    final = df[(df.time >= 13.0) & (df.time <= 15.0)]
    initial_rms = float(np.sqrt(np.mean(initial["tip_displacement"].to_numpy() ** 2)))
    final_rms = float(np.sqrt(np.mean(final["tip_displacement"].to_numpy() ** 2)))
    decay_ratio = final_rms / initial_rms
    assert list(df.columns).count("tip_displacement") == 1
    assert len(initial) == metrics["initial_window_samples"]
    assert len(final) == metrics["final_window_samples"]
    assert abs(initial_rms - metrics["initial_tip_rms"]) < 1e-12
    assert abs(final_rms - metrics["final_tip_rms"]) < 1e-12
    assert abs(decay_ratio - metrics["decay_ratio"]) < 1e-12

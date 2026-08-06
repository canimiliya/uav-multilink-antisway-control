from pathlib import Path

from uav_sway.models.build_planar_chain import build_planar_chain_model
from uav_sway.models.passive_sim import simulate_passive


ROOT = Path(__file__).resolve().parents[1]


def test_five_link_ten_degree_passive_decay(tmp_path):
    xml_path = tmp_path / "model_5link.xml"
    build_planar_chain_model(ROOT / "configs/model_5link.yaml", xml_path)
    result = simulate_passive(ROOT / "configs/model_5link.yaml", 10.0, 15.0, model_path=xml_path)
    assert result["finite"]
    assert result["final_total_energy_j"] < result["initial_total_energy_j"]
    assert result["decay_ratio"] < 0.60
    assert result["max_abs_joint_angle_rad"] < 1.7453293
    assert result["min_tip_z_m"] > 0.0

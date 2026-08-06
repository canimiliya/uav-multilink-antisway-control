from pathlib import Path

import mujoco
import yaml

from uav_sway.linearization.equilibrium import find_equilibrium


ROOT = Path(__file__).resolve().parents[1]


def test_automatic_equilibrium_is_repeatable_and_unanchored():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/s3/runtime/model_5link_controlled.xml"))
    config = yaml.safe_load((ROOT / "configs/lqr.yaml").read_text(encoding="utf-8"))
    result = find_equilibrium(model, config)
    assert result["final_residual"] < 1e-8
    assert result["repeat_error"] < 1e-12
    assert not result["snapshot"].eq_active.any()
